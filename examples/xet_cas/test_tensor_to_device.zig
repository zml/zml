// "tensor-name → device buffer" end-to-end smoke test.
//
// Given an HF model repo URI + tensor name, this binary:
//   1. uses ZML's safetensors machinery (TensorRegistry) to translate the
//      tensor name into [file_uri, offset, byteSize),
//   2. parses the file_uri to get the HF repo coordinates (ns/model/rev/path),
//   3. calls the HF CAS reconstruction API with a Range covering exactly
//      [offset, offset+byteSize) → ReconstructionResponse JSON in memory,
//   4. fetches each needed xorb via HTTPS range GET into a BSS slot,
//   5. decompresses + stitches into a single PJRT device buffer of size
//      byteSize (with head-skip from offset_into_first_range and tail-clip
//      on the last term),
//   6. reads back and verifies.
//
// Usage:
//   test_tensor_to_device --model <uri> --tensor <name>
// Example:
//   --model hf://meta-llama/Llama-3.1-70B \
//   --tensor model.layers.0.self_attn.k_proj.weight

const std = @import("std");
const zml = @import("zml");
const pjrt = @import("pjrt");
const pjrtx = zml.pjrtx;
const xet = @import("io").xet;

const log = std.log.scoped(.test_tensor_to_device);

pub const std_options: std.Options = .{ .log_level = .info };

// Per xorb spec: max 64 MiB compressed. One shared slot reused across xorbs.
const MAX_XORB: usize = 64 * 1024 * 1024;
const MAX_XORBS: usize = 128;
// const MAX_UNCOMP: usize = 64 * 1024 * 1024;
const MAX_UNCOMP: usize = 2560 * 1024 * 1024;
const MAX_CHUNK: usize = 128 * 1024;
const MAX_PLANS: usize = 4096;

var xorb_buf: [MAX_XORB]u8 = undefined;

// Single uncompressed-chunk staging slot. transferData sources its bytes from
// here, then the next chunk overwrites it. No per-tensor host materialization.
var chunk_buf: [MAX_CHUNK]u8 = undefined;

var dec_tmp: [MAX_CHUNK]u8 = undefined;

// Verification only: parallel reference + device readback, both sized for the
// full tensor. Not part of the streaming push path. Allocated on the heap in
// main() — keeping multi-GB arrays in BSS makes dyld refuse to map the binary
// on macOS.
var host_ref: []u8 = &.{};
var readback: []u8 = &.{};

const XorbWork = struct {
    hash: []const u8,
    needed_chunk_start: u32,
    needed_chunk_end: u32,
    fetch_chunk_start: u32,
    fetch_chunk_end: u32,
    fetch_url: []const u8,
    fetch_url_range_start: u64,
    fetch_url_range_end: u64, // inclusive
    fetch_byte_len: u64,
};

var xorbs: [MAX_XORBS]XorbWork = undefined;

const TermPlan = struct {
    xorb_idx: u8,
    dst_off: u32, // offset within device buffer [0, byteSize)
    wanted_len: u32, // bytes to write from this term
    byte_skip: u32, // head bytes to drop from first chunk
    chunk_start: u32,
    chunk_end: u32,
    bytes_written: u32,
};

var plans: [MAX_PLANS]TermPlan = undefined;

const RepoInfo = struct {
    namespace: []const u8,
    model: []const u8,
    rev: []const u8,
    filepath: []const u8,
};

const CasAuth = struct {
    url: []const u8,
    token: []const u8,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args_it = init.minimal.args.iterate();
    _ = args_it.skip();
    var model_arg: []const u8 = "";
    var tensor_name: []const u8 = "";
    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model")) {
            model_arg = args_it.next() orelse return error.MissingModelArg;
        } else if (std.mem.eql(u8, arg, "--tensor")) {
            tensor_name = args_it.next() orelse return error.MissingTensorArg;
        }
    }
    if (model_arg.len == 0 or tensor_name.len == 0) {
        std.debug.print("Usage: test_tensor_to_device --model <uri> --tensor <name>\n", .{});
        std.process.exit(1);
    }

    // ── VFS + io ────────────────────────────────────────────────────────────
    var vfs_file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();

    var http_client: std.http.Client = .{ .allocator = allocator, .io = init.io };
    try http_client.initDefaultProxies(allocator, init.environ_map);
    defer http_client.deinit();

    var hf_vfs: zml.io.VFS.HF = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();

    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();
    try vfs.register("file", vfs_file.io());
    try vfs.register("hf", hf_vfs.io());
    const io = vfs.io();

    // ── Tensor registry & lookup ────────────────────────────────────────────
    log.info("Resolving model repo: {s}", .{model_arg});
    const repo_dir = try zml.safetensors.resolveModelRepo(io, model_arg);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo_dir);
    defer registry.deinit();

    const tensor = registry.tensors.get(tensor_name) orelse {
        log.err("Tensor not found: {s}", .{tensor_name});
        return error.TensorNotFound;
    };
    const tensor_offset = tensor.offset;
    const tensor_size = tensor.byteSize();
    log.info("Tensor {s}: shape={f} offset={d} size={d} file={s}", .{
        tensor.name, tensor.shape, tensor_offset, tensor_size, tensor.file_uri,
    });
    if (tensor_size > MAX_UNCOMP) return error.TensorTooLarge;

    host_ref = try allocator.alloc(u8, @intCast(tensor_size));
    defer allocator.free(host_ref);
    readback = try allocator.alloc(u8, @intCast(tensor_size));
    defer allocator.free(readback);

    // ── Derive RepoInfo from file_uri (strip hf:// then parse) ─────────────
    const repo = try parseFileUri(tensor.file_uri);
    log.info("Repo: {s}/{s}@{s} path={s}", .{ repo.namespace, repo.model, repo.rev, repo.filepath });

    // ── HF auth ─────────────────────────────────────────────────────────────
    const hf_token = init.environ_map.get("HF_TOKEN") orelse {
        log.err("HF_TOKEN env var must be set", .{});
        return error.MissingToken;
    };
    var auth_buf: [1024]u8 = undefined;
    const auth = std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{std.mem.trim(u8, hf_token, " \t\n\r")}) catch return error.TokenTooLong;

    // ── Xet file ID + CAS token ─────────────────────────────────────────────
    const file_id = try getXetFileId(allocator, &http_client, repo, auth);
    defer allocator.free(file_id);
    log.info("Xet file id: {s}", .{file_id});

    const cas = try getCasToken(allocator, &http_client, repo, auth);
    defer {
        allocator.free(cas.url);
        allocator.free(cas.token);
    }
    var cas_auth_buf: [65536]u8 = undefined;
    const cas_auth = std.fmt.bufPrint(&cas_auth_buf, "Bearer {s}", .{cas.token}) catch return error.TokenTooLong;

    // ── CAS reconstruction for [tensor_offset, tensor_offset+size) ──────────
    const recon_body = try callReconstruction(
        allocator,
        &http_client,
        cas.url,
        cas_auth,
        file_id,
        tensor_offset, // Start the window at the tensor's file offset
        tensor_offset + tensor_size, // Request a window covering exactly the tensor's bytes within the file.
    );
    defer allocator.free(recon_body);

    const parsed = try std.json.parseFromSlice(
        xet.ReconstructionResponse,
        allocator,
        recon_body,
        .{ .ignore_unknown_fields = true },
    );
    defer parsed.deinit();
    const resp = parsed.value;

    // ── Local-coordinate window walking: populate `plans` and `xorbs` ───────

    // [win_start, win_end) = [offset_into_first_range, offset_into_first_range + tensor_size).
    const win_start: u64 = resp.offset_into_first_range;
    const win_end: u64 = win_start + tensor_size;

    var n_xorbs: u8 = 0;
    var n_plans: u32 = 0;
    var stream_pos: u64 = 0;
    for (resp.terms) |t| { // Walk terms in order, tracking their offsets in the local (concatenated unpacked) stream and which xorb they require.
        const t_off = stream_pos;
        const t_end = stream_pos + t.unpacked_length;
        stream_pos = t_end;

        const head_skip: u64 = if (t_off < win_start) win_start - t_off else 0;
        const tail_avail: u64 = @min(t_end, win_end) - t_off; // bytes from start-of-term to write boundary
        const wanted: u64 = tail_avail - head_skip;
        const dst_off: u64 = if (t_off < win_start) 0 else t_off - win_start;

        var xi: u8 = 0;
        while (xi < n_xorbs) : (xi += 1) {
            if (std.mem.eql(u8, xorbs[xi].hash, t.hash)) break;
        }
        if (xi == n_xorbs) { // New xorb hash we haven't seen before. Add to xorbs if we have room.
            if (n_xorbs >= MAX_XORBS) return error.TooManyXorbs;
            xorbs[n_xorbs] = .{
                .hash = t.hash,
                .needed_chunk_start = std.math.maxInt(u32),
                .needed_chunk_end = 0,
                .fetch_chunk_start = 0,
                .fetch_chunk_end = 0,
                .fetch_url = "",
                .fetch_url_range_start = 0,
                .fetch_url_range_end = 0,
                .fetch_byte_len = 0,
            };
            n_xorbs += 1;
        }
        const x = &xorbs[xi];
        if (t.range.start < x.needed_chunk_start) x.needed_chunk_start = @intCast(t.range.start);
        if (t.range.end > x.needed_chunk_end) x.needed_chunk_end = @intCast(t.range.end);
        if (n_plans >= plans.len) return error.TooManyPlans;
        plans[n_plans] = .{
            .xorb_idx = xi,
            .dst_off = @intCast(dst_off),
            .wanted_len = @intCast(wanted),
            .byte_skip = @intCast(head_skip),
            .chunk_start = @intCast(t.range.start),
            .chunk_end = @intCast(t.range.end),
            .bytes_written = 0,
        };
        n_plans += 1;
    }
    if (n_plans == 0) return error.NoTermsInWindow;
    log.info("window [{d},{d}) (local) → {d} term(s) across {d} xorb(s)", .{ win_start, win_end, n_plans, n_xorbs });

    // Dedup diagnostic: a chunk range (xorb_idx, chunk_start, chunk_end) appearing
    // in >1 plan with different dst_off means the same compressed bytes land at
    // multiple non-contiguous offsets in the device buffer.
    var dup_chunks: u32 = 0;
    for (plans[0..n_plans], 0..) |p, i| {
        for (plans[0..i]) |q| {
            if (p.xorb_idx == q.xorb_idx and p.chunk_start == q.chunk_start and p.chunk_end == q.chunk_end and p.dst_off != q.dst_off) {
                dup_chunks += 1;
                break;
            }
        }
    }
    log.info("dedup: {d}/{d} plan(s) reuse a chunk range at a different device offset", .{ dup_chunks, n_plans });

    // ── Resolve covering fetch entry per xorb (no I/O yet) ─────────────────
    for (xorbs[0..n_xorbs]) |*x| {
        const fetch_list = resp.fetch_info.map.get(x.hash) orelse return error.XorbNotInFetchInfo;
        var picked: ?xet.FetchUrl = null;
        for (fetch_list) |f| {
            if (f.range.start <= x.needed_chunk_start and f.range.end >= x.needed_chunk_end) {
                picked = f;
                break;
            }
        }
        const fetch = picked orelse return error.NoFetchEntryCoversNeededChunks;
        x.fetch_chunk_start = @intCast(fetch.range.start);
        x.fetch_chunk_end = @intCast(fetch.range.end);
        x.fetch_url = fetch.url;
        x.fetch_url_range_start = fetch.url_range.start;
        x.fetch_url_range_end = fetch.url_range.end;
        x.fetch_byte_len = fetch.url_range.end - fetch.url_range.start + 1;
        if (x.fetch_byte_len > xorb_buf.len) return error.XorbTooLargeForSlot;
    }

    // ── Boot platform & create device buffer ────────────────────────────────
    const platform: *zml.Platform = try .auto(allocator, init.io, .{});
    defer platform.deinit(allocator, init.io);
    if (platform.devices.len == 0) return error.NoDevices;
    const memory = platform.devices[0].memory(.default);

    const dev_size: usize = @intCast(tensor_size);
    const dims: [1]i64 = .{@intCast(dev_size)};
    const shape_spec: pjrt.ShapeSpec = .init(&dims, pjrtx.bufferTypeFromDtype(.u8));
    const tm = try platform.pjrt_client.createBuffersForAsyncHostToDevice(
        platform.pjrt_api,
        .{ .shape_specs = &.{shape_spec}, .memory = memory.pjrt_memory },
    );
    defer tm.deinit(platform.pjrt_api);
    const device_buffer = try tm.retrieveBuffer(platform.pjrt_api, 0);

    // ── Sequential xorb streaming, chunk-sized host footprint ──────────────
    // Each chunk is decompressed once into chunk_buf, then the wanted slice
    // chunk_buf[src_off..src_off+take] is pushed to the device. The look-back
    // pattern flushes the PREVIOUS pending push BEFORE overwriting chunk_buf
    // with the next decompress, so a single chunk-sized staging slot suffices.
    // host_ref is populated in parallel for the final readback comparison only.
    var have_pending = false;
    var pending_src_off: usize = 0;
    var pending_dst_off: usize = 0;
    var pending_len: usize = 0;
    var net_bytes: u64 = 0;
    var net_ns: u64 = 0;
    var decomp_ns: u64 = 0;
    var transfer_ns: u64 = 0;
    var memcpy_ns: u64 = 0;
    var chunkloop_ns: u64 = 0;
    var n_chunks_total: u64 = 0;
    var n_transfers: u64 = 0;
    for (xorbs[0..n_xorbs], 0..) |x, xi_usize| {
        const xi: u8 = @intCast(xi_usize);
        log.info("xorb {s}..: needed=[{d},{d}) fetch=[{d},{d}) bytes={d}", .{
            x.hash[0..16],       x.needed_chunk_start, x.needed_chunk_end,
            x.fetch_chunk_start, x.fetch_chunk_end,    x.fetch_byte_len,
        });
        const t0: std.Io.Timestamp = .now(init.io, .awake);
        try httpRangeGetIntoSlot(
            &http_client,
            x.fetch_url,
            x.fetch_url_range_start,
            x.fetch_url_range_end,
            xorb_buf[0..x.fetch_byte_len],
        );
        const dt_ns = t0.untilNow(init.io, .awake).toNanoseconds();
        net_bytes += x.fetch_byte_len;
        net_ns += @intCast(dt_ns);
        const mbps = (@as(f64, @floatFromInt(x.fetch_byte_len)) / (1024.0 * 1024.0)) / (@as(f64, @floatFromInt(dt_ns)) / 1e9);
        log.info("  fetched {d} bytes in {d:.1} ms ({d:.1} MiB/s)", .{ x.fetch_byte_len, @as(f64, @floatFromInt(dt_ns)) / 1e6, mbps });

        var it: xet.ChunkIterator = .{ .data = xorb_buf[0..x.fetch_byte_len] };
        var chunk_idx: u32 = x.fetch_chunk_start;
        const ts_cl: std.Io.Timestamp = .now(init.io, .awake);
        while (try it.next()) |chunk| : (chunk_idx += 1) {
            if (chunk.uncompressed_size > MAX_CHUNK) return error.ChunkTooLarge;
            n_chunks_total += 1;
            var decompressed = false;
            for (plans[0..n_plans]) |*p| {
                if (p.xorb_idx != xi) continue;
                if (chunk_idx < p.chunk_start or chunk_idx >= p.chunk_end) continue;
                const need_skip = p.byte_skip > 0 and p.bytes_written == 0;
                const src_off: usize = if (need_skip) p.byte_skip else 0;
                const avail: usize = chunk.uncompressed_size - src_off;
                const remaining: usize = p.wanted_len - p.bytes_written;
                const take: usize = @min(avail, remaining);
                const slot_off: usize = p.dst_off + p.bytes_written;
                // Flush previous pending while chunk_buf still holds its source bytes.
                if (have_pending) {
                    const ts_tx: std.Io.Timestamp = .now(init.io, .awake);
                    const ev = try tm.transferData(platform.pjrt_api, 0, chunk_buf[pending_src_off .. pending_src_off + pending_len], @intCast(pending_dst_off), false);
                    defer ev.deinit(platform.pjrt_api);
                    try ev.await(platform.pjrt_api, init.io);
                    transfer_ns += @intCast(ts_tx.untilNow(init.io, .awake).toNanoseconds());
                    n_transfers += 1;
                }
                if (!decompressed) {
                    const ts_d: std.Io.Timestamp = .now(init.io, .awake);
                    _ = try xet.decompressChunk(chunk, chunk_buf[0..chunk.uncompressed_size], &dec_tmp);
                    decomp_ns += @intCast(ts_d.untilNow(init.io, .awake).toNanoseconds());
                    decompressed = true;
                }
                // Verification-only mirror; safe because user carved this out.
                const ts_m: std.Io.Timestamp = .now(init.io, .awake);
                @memcpy(host_ref[slot_off .. slot_off + take], chunk_buf[src_off .. src_off + take]);
                memcpy_ns += @intCast(ts_m.untilNow(init.io, .awake).toNanoseconds());
                pending_src_off = src_off;
                pending_dst_off = slot_off;
                pending_len = take;
                have_pending = true;
                p.bytes_written += @intCast(take);
            }
        }
        chunkloop_ns += @intCast(ts_cl.untilNow(init.io, .awake).toNanoseconds());
    }
    for (plans[0..n_plans]) |p| {
        if (p.bytes_written != p.wanted_len) {
            log.err("plan mismatch: written={d} expected={d}", .{ p.bytes_written, p.wanted_len });
            return error.PlanByteCountMismatch;
        }
    }
    if (!have_pending) return error.NoChunksPushed;
    {
        const ev = try tm.transferData(platform.pjrt_api, 0, chunk_buf[pending_src_off .. pending_src_off + pending_len], @intCast(pending_dst_off), true);
        defer ev.deinit(platform.pjrt_api);
        try ev.await(platform.pjrt_api, init.io);
    }

    // ── Readback & verify ───────────────────────────────────────────────────
    if (try device_buffer.toHostBuffer(platform.pjrt_api, readback[0..dev_size])) |ev| {
        defer ev.deinit(platform.pjrt_api);
        try ev.await(platform.pjrt_api, init.io);
    }
    if (!std.mem.eql(u8, host_ref[0..dev_size], readback[0..dev_size])) {
        for (host_ref[0..dev_size], readback[0..dev_size], 0..) |a, b, j| {
            if (a != b) {
                log.err("mismatch at byte {d}: host=0x{x:0>2} dev=0x{x:0>2}", .{ j, a, b });
                break;
            }
        }
        return error.ReadbackMismatch;
    }
    log.info("OK: tensor {s} ({d} bytes) from {d} term(s) / {d} xorb(s)", .{
        tensor_name, dev_size, n_plans, n_xorbs,
    });
    const net_mib = @as(f64, @floatFromInt(net_bytes)) / (1024.0 * 1024.0);
    const net_s = @as(f64, @floatFromInt(net_ns)) / 1e9;
    log.info("network: {d:.2} MiB in {d:.2} s ({d:.1} MiB/s aggregate)", .{ net_mib, net_s, net_mib / net_s });
    const to_ms = struct {
        fn f(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    }.f;
    const overhead_ns = if (chunkloop_ns > decomp_ns + transfer_ns + memcpy_ns) chunkloop_ns - decomp_ns - transfer_ns - memcpy_ns else 0;
    log.info("chunk-loop breakdown ({d} chunks, {d} transfers):", .{ n_chunks_total, n_transfers });
    log.info("  total      = {d:.1} ms", .{to_ms(chunkloop_ns)});
    log.info("  decompress = {d:.1} ms", .{to_ms(decomp_ns)});
    log.info("  transfer   = {d:.1} ms  (transferData + await)", .{to_ms(transfer_ns)});
    log.info("  memcpy ref = {d:.1} ms  (verification mirror)", .{to_ms(memcpy_ns)});
    log.info("  scan/other = {d:.1} ms", .{to_ms(overhead_ns)});
}

fn parseFileUri(uri: []const u8) !RepoInfo {
    // Expected: hf://{ns}/{model}[@rev]/{filepath...}
    var path = uri;
    if (std.mem.startsWith(u8, path, "hf://")) path = path["hf://".len..];
    var parts = std.mem.splitScalar(u8, path, '/');
    const namespace = parts.next() orelse return error.InvalidFileUri;
    var model = parts.next() orelse return error.InvalidFileUri;
    var rev: []const u8 = "main";
    if (std.mem.indexOfScalar(u8, model, '@')) |at| {
        rev = model[at + 1 ..];
        model = model[0..at];
    }
    const filepath = parts.rest();
    if (filepath.len == 0) return error.InvalidFileUri;
    return .{ .namespace = namespace, .model = model, .rev = rev, .filepath = filepath };
}

// ── HTTP helpers (xorb fetch) ───────────────────────────────────────────────

fn httpRangeGetIntoSlot(
    client: *std.http.Client,
    url: []const u8,
    range_start: u64,
    range_end_inclusive: u64,
    slot: []u8,
) !void {
    var range_buf: [64]u8 = undefined;
    const range_header = std.fmt.bufPrint(&range_buf, "bytes={d}-{d}", .{ range_start, range_end_inclusive }) catch unreachable;
    const uri: std.Uri = try .parse(url);
    var req = try client.request(.GET, uri, .{
        .headers = .{ .accept_encoding = .{ .override = "identity" } },
        .extra_headers = &.{.{ .name = "Range", .value = range_header }},
    });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buffer: [8 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buffer);
    if (res.head.status != .partial_content and res.head.status != .ok) {
        log.err("HTTP range GET failed: status={} url={s}", .{ res.head.status, url });
        return error.HttpRequestFailed;
    }
    try res.reader(&.{}).readSliceAll(slot);
}

// ── CAS helpers (duplicated from examples/xet_cas/main.zig) ─────────────────

fn getXetFileId(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    repo: RepoInfo,
    auth: []const u8,
) ![]const u8 {
    var url_buf: [4096]u8 = undefined;
    const resolve_url = try std.fmt.bufPrint(
        &url_buf,
        "https://huggingface.co/{s}/{s}/resolve/{s}/{s}",
        .{ repo.namespace, repo.model, repo.rev, repo.filepath },
    );
    const uri: std.Uri = try .parse(resolve_url);
    var req = try client.request(.GET, uri, .{
        .redirect_behavior = .unhandled,
        .headers = .{
            .accept_encoding = .{ .override = "identity" },
            .authorization = .{ .override = auth },
        },
    });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buffer: [8 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buffer);
    var header_it = res.head.iterateHeaders();
    while (header_it.next()) |header| {
        if (std.ascii.eqlIgnoreCase(header.name, "X-Xet-Hash")) {
            return try allocator.dupe(u8, header.value);
        }
    }
    log.err("X-Xet-Hash not found (status={}).", .{res.head.status});
    return error.XetHashNotFound;
}

fn getCasToken(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    repo: RepoInfo,
    auth: []const u8,
) !CasAuth {
    var url_buf: [4096]u8 = undefined;
    const token_url = try std.fmt.bufPrint(
        &url_buf,
        "https://huggingface.co/api/models/{s}/{s}/xet-read-token/{s}",
        .{ repo.namespace, repo.model, repo.rev },
    );
    const uri: std.Uri = try .parse(token_url);
    var req = try client.request(.GET, uri, .{
        .headers = .{
            .accept_encoding = .{ .override = "identity" },
            .authorization = .{ .override = auth },
        },
    });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buffer: [4 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buffer);
    if (res.head.status != .ok) {
        log.err("CAS token request failed: status={}", .{res.head.status});
        return error.CasTokenFailed;
    }
    const body = try res.reader(&.{}).readAlloc(allocator, res.head.content_length orelse 128 * 1024);
    defer allocator.free(body);
    const parsed = try std.json.parseFromSlice(struct {
        accessToken: []const u8,
        casUrl: []const u8,
    }, allocator, body, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    return .{
        .url = try allocator.dupe(u8, parsed.value.casUrl),
        .token = try allocator.dupe(u8, parsed.value.accessToken),
    };
}

fn callReconstruction(
    allocator: std.mem.Allocator,
    client: *std.http.Client,
    cas_url: []const u8,
    cas_auth: []const u8,
    file_id: []const u8,
    range_start: u64,
    range_end_exclusive: u64,
) ![]const u8 {
    var url_buf: [4096]u8 = undefined;
    const recon_url = try std.fmt.bufPrint(&url_buf, "{s}/v1/reconstructions/{s}", .{ cas_url, file_id });
    var range_buf: [64]u8 = undefined;
    const range_header = std.fmt.bufPrint(&range_buf, "bytes={}-{}", .{ range_start, range_end_exclusive - 1 }) catch unreachable;
    const uri: std.Uri = try .parse(recon_url);
    var req = try client.request(.GET, uri, .{
        .headers = .{
            .accept_encoding = .{ .override = "identity" },
            .authorization = .{ .override = cas_auth },
        },
        .extra_headers = &.{.{ .name = "Range", .value = range_header }},
    });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buffer: [8 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buffer);
    if (res.head.status != .ok and res.head.status != .partial_content) {
        log.err("Reconstruction failed: status={}", .{res.head.status});
        return error.ReconstructionFailed;
    }
    return try res.reader(&.{}).readAlloc(allocator, res.head.content_length orelse 64 * 1024 * 1024);
}
