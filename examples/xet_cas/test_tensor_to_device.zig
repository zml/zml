// "tensor-name → host sink" end-to-end smoke test.
//
// Given an HF model repo URI + tensor name, this binary:
//   1. uses ZML's safetensors machinery (TensorRegistry) to translate the
//      tensor name into [file_uri, offset, byteSize),
//   2. parses the file_uri to get the HF repo coordinates (ns/model/rev/path),
//   3. calls the HF CAS reconstruction API with a Range covering exactly
//      [offset, offset+byteSize) → ReconstructionResponse JSON in memory,
//   4. fetches each needed xorb via HTTPS range GET into a BSS slot,
//   5. decompresses + stitches into a host `Sink` of size byteSize via
//      std.Io.Writer (with head-skip from offset_into_first_range and
//      tail-clip on the last term),
//   6. SHA-256-verifies the sink against an HF LFS range GET of the same
//      [offset, offset+byteSize) window, both digests computed by streaming
//      through std.Io.Reader.
//
// Usage:
//   test_tensor_to_device --model <uri> --tensor <name>
// Example:
//   --model hf://meta-llama/Llama-3.1-70B \
//   --tensor model.layers.0.self_attn.k_proj.weight

const std = @import("std");
const zml = @import("zml");
const xet = @import("io").xet;
const util = @import("util.zig");

const log = std.log.scoped(.test_tensor_to_device);

pub const std_options: std.Options = .{ .log_level = .info };

// Per xorb spec: max 64 MiB compressed.
const MAX_XORB: usize = 64 * 1024 * 1024;
const MAX_CHUNK: usize = 128 * 1024;
const DEFAULT_WORKERS: usize = 4;

// Host destination: a flat buffer the size of the tensor, written at random
// offsets via a std.Io.Writer, read back via a std.Io.Reader for verification.
const Sink = struct {
    buf: []u8,

    fn writeAt(self: *Sink, off: usize, src: []const u8) !void {
        var w: std.Io.Writer = .fixed(self.buf[off..]);
        try w.writeAll(src);
    }

    fn reader(self: *Sink) std.Io.Reader {
        return .fixed(self.buf);
    }
};

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

const TermPlan = struct {
    xorb_idx: u8,
    dst_off: u32, // offset within device buffer [0, byteSize)
    wanted_len: u32, // bytes to write from this term
    byte_skip: u32, // head bytes to drop from first chunk
    chunk_start: u32,
    chunk_end: u32,
    bytes_written: u32,
};

const Worker = struct {
    counter: *std.atomic.Value(u32),
    xorbs: []const XorbWork,
    plans: []TermPlan,
    sink: *Sink,
    allocator: std.mem.Allocator,
    env: *std.process.Environ.Map,
    io: std.Io,
    net_bytes: u64 = 0,
    net_ns: u64 = 0,
    decomp_ns: u64 = 0,
    write_ns: u64 = 0,
    chunkloop_ns: u64 = 0,
    n_chunks_total: u64 = 0,
    err: ?anyerror = null,

    fn run(self: *Worker) void {
        self.runImpl() catch |e| {
            self.err = e;
        };
    }

    fn runImpl(self: *Worker) !void {
        var client: std.http.Client = .{ .allocator = self.allocator, .io = self.io };
        try client.initDefaultProxies(self.allocator, self.env);
        defer client.deinit();

        const xb = try self.allocator.alloc(u8, MAX_XORB);
        defer self.allocator.free(xb);
        const cb = try self.allocator.alloc(u8, MAX_CHUNK);
        defer self.allocator.free(cb);

        while (true) {
            const next = self.counter.fetchAdd(1, .acq_rel);
            if (next >= self.xorbs.len) break;
            const xi: u8 = @intCast(next);
            const x = self.xorbs[xi];

            const t0: std.Io.Timestamp = .now(self.io, .awake);
            try util.httpRangeGetIntoSlot(&client, x.fetch_url, x.fetch_url_range_start, x.fetch_url_range_end, xb[0..x.fetch_byte_len]);
            self.net_ns += @intCast(t0.untilNow(self.io, .awake).toNanoseconds());
            self.net_bytes += x.fetch_byte_len;

            var it: xet.ChunkIterator = .{ .data = xb[0..x.fetch_byte_len] };
            var chunk_idx: u32 = x.fetch_chunk_start;
            const ts_cl: std.Io.Timestamp = .now(self.io, .awake);
            while (try it.next()) |chunk| : (chunk_idx += 1) {
                if (chunk.uncompressed_size > MAX_CHUNK) return error.ChunkTooLarge;
                self.n_chunks_total += 1;
                var decompressed = false;
                for (self.plans) |*p| {
                    if (p.xorb_idx != xi) continue;
                    if (chunk_idx < p.chunk_start or chunk_idx >= p.chunk_end) continue;
                    const need_skip = p.byte_skip > 0 and p.bytes_written == 0;
                    const src_off: usize = if (need_skip) p.byte_skip else 0;
                    const avail: usize = chunk.uncompressed_size - src_off;
                    const remaining: usize = p.wanted_len - p.bytes_written;
                    const take: usize = @min(avail, remaining);
                    const slot_off: usize = p.dst_off + p.bytes_written;
                    if (!decompressed) {
                        const ts_d: std.Io.Timestamp = .now(self.io, .awake);
                        _ = try xet.decompressChunk(chunk, cb[0..chunk.uncompressed_size]);
                        self.decomp_ns += @intCast(ts_d.untilNow(self.io, .awake).toNanoseconds());
                        decompressed = true;
                    }
                    const ts_w: std.Io.Timestamp = .now(self.io, .awake);
                    try self.sink.writeAt(slot_off, cb[src_off .. src_off + take]);
                    self.write_ns += @intCast(ts_w.untilNow(self.io, .awake).toNanoseconds());
                    p.bytes_written += @intCast(take);
                }
            }
            self.chunkloop_ns += @intCast(ts_cl.untilNow(self.io, .awake).toNanoseconds());
        }
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args_it = init.minimal.args.iterate();
    _ = args_it.skip();
    var model_arg: []const u8 = "";
    var tensor_name: []const u8 = "";
    var n_workers: usize = DEFAULT_WORKERS;
    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model")) {
            model_arg = args_it.next() orelse return error.MissingModelArg;
        } else if (std.mem.eql(u8, arg, "--tensor")) {
            tensor_name = args_it.next() orelse return error.MissingTensorArg;
        } else if (std.mem.eql(u8, arg, "--workers")) {
            const v = args_it.next() orelse return error.MissingWorkersArg;
            n_workers = try std.fmt.parseInt(usize, v, 10);
            if (n_workers == 0) return error.InvalidWorkers;
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
    if (tensor_size > std.math.maxInt(usize)) return error.TensorTooLarge;
    var sink: Sink = .{ .buf = try allocator.alloc(u8, @intCast(tensor_size)) };
    defer allocator.free(sink.buf);

    // ── Derive Repo from file_uri (strip hf:// then parse) ────────────────────
    const hf_path = if (std.mem.startsWith(u8, tensor.file_uri, "hf://")) tensor.file_uri["hf://".len..] else tensor.file_uri;
    const hf_repo = try zml.io.VFS.HF.Repo.parse(hf_path);
    const repo: xet.Client.Repo = .{ .repo = hf_repo.repo, .model = hf_repo.model, .rev = hf_repo.rev, .path = hf_repo.path };
    log.info("Repo: {s}/{s}@{s} path={s}", .{ repo.repo, repo.model, repo.rev, repo.path });

    // ── HF auth (for the LFS oracle path) ─────────────────────────────────
    const hf_token = init.environ_map.get("HF_TOKEN") orelse {
        log.err("HF_TOKEN env var must be set", .{});
        return error.MissingToken;
    };
    var auth_buf: [1024]u8 = undefined;
    const auth = std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{std.mem.trim(u8, hf_token, " \t\n\r")}) catch return error.TokenTooLong;

    // ── CAS reconstruction for [tensor_offset, tensor_offset+size) ──────────
    var xet_client: xet.Client = .init(allocator, &http_client, hf_token);
    defer xet_client.deinit();
    log.info("Xet file id: {s}", .{try xet_client.fileId(repo)});
    const parsed = try xet_client.reconstruct(repo, tensor_offset, tensor_offset + tensor_size);
    defer parsed.deinit();
    const resp = parsed.value;

    // ── Local-coordinate window walking: populate `plans` and `xorbs` ───────

    // [win_start, win_end) = [offset_into_first_range, offset_into_first_range + tensor_size).
    const win_start: u64 = resp.offset_into_first_range;
    const win_end: u64 = win_start + tensor_size;

    // One plan per term; at most one xorb per term (fewer when hashes repeat).
    const xorbs = try allocator.alloc(XorbWork, resp.terms.len);
    defer allocator.free(xorbs);
    const plans = try allocator.alloc(TermPlan, resp.terms.len);
    defer allocator.free(plans);

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
        if (xi == n_xorbs) { // New xorb hash we haven't seen before.
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
        if (x.fetch_byte_len > MAX_XORB) return error.XorbTooLargeForSlot;
    }

    log.info("spawning {d} worker(s) for {d} xorb(s)", .{ n_workers, n_xorbs });
    var counter: std.atomic.Value(u32) = .init(0);
    const workers = try allocator.alloc(Worker, n_workers);
    defer allocator.free(workers);
    const threads = try allocator.alloc(std.Thread, n_workers);
    defer allocator.free(threads);
    const ts_xet_wall: std.Io.Timestamp = .now(init.io, .awake);
    var n_spawned: usize = 0;
    errdefer {
        // Drain any already-spawned workers before unwinding frees their inputs.
        counter.store(@intCast(n_xorbs), .release);
        for (threads[0..n_spawned]) |t| t.join();
    }
    for (workers, threads) |*w, *t| {
        w.* = .{
            .counter = &counter,
            .xorbs = xorbs[0..n_xorbs],
            .plans = plans[0..n_plans],
            .sink = &sink,
            .allocator = allocator,
            .env = init.environ_map,
            .io = init.io,
        };
        t.* = try std.Thread.spawn(.{}, Worker.run, .{w});
        n_spawned += 1;
    }
    for (threads) |t| t.join();
    const xet_wall_ns: u64 = @intCast(ts_xet_wall.untilNow(init.io, .awake).toNanoseconds());
    var net_bytes: u64 = 0;
    var net_ns: u64 = 0;
    var decomp_ns: u64 = 0;
    var write_ns: u64 = 0;
    var chunkloop_ns: u64 = 0;
    var n_chunks_total: u64 = 0;
    for (workers) |w| {
        if (w.err) |e| return e;
        net_bytes += w.net_bytes;
        net_ns += w.net_ns;
        decomp_ns += w.decomp_ns;
        write_ns += w.write_ns;
        chunkloop_ns += w.chunkloop_ns;
        n_chunks_total += w.n_chunks_total;
    }
    for (plans[0..n_plans]) |p| {
        if (p.bytes_written != p.wanted_len) {
            log.err("plan mismatch: written={d} expected={d}", .{ p.bytes_written, p.wanted_len });
            return error.PlanByteCountMismatch;
        }
    }
    // LFS SHA-256 oracle: stream both sides through std.Io.Reader.
    var sink_reader = sink.reader();
    const xet_digest = try util.sha256ReadAll(&sink_reader);
    var lfs_net_ns: u64 = 0;
    const lfs_digest = try util.sha256LfsRange(&http_client, repo, auth, tensor_offset, tensor_offset + tensor_size - 1, init.io, &lfs_net_ns);
    if (!std.mem.eql(u8, &xet_digest, &lfs_digest)) {
        log.err("LFS oracle MISMATCH: xet=0x{x} lfs=0x{x}", .{ xet_digest, lfs_digest });
        return error.LfsOracleMismatch;
    }
    log.info("OK: tensor {s} ({d} bytes) from {d} term(s) / {d} xorb(s)", .{
        tensor_name, sink.buf.len, n_plans, n_xorbs,
    });
    log.info("OK: xet sha256 == lfs sha256 = 0x{x}", .{xet_digest});
    const net_mib = @as(f64, @floatFromInt(net_bytes)) / (1024.0 * 1024.0);
    const net_s = @as(f64, @floatFromInt(net_ns)) / 1e9;
    log.info("network: {d:.2} MiB in {d:.2} s ({d:.1} MiB/s aggregate)", .{ net_mib, net_s, net_mib / net_s });
    // Apples-to-apples: wall time to land tensor_size bytes into a host Sink.
    const tensor_mib = @as(f64, @floatFromInt(tensor_size)) / (1024.0 * 1024.0);
    const xet_wall_s = @as(f64, @floatFromInt(xet_wall_ns)) / 1e9;
    const lfs_wall_s = @as(f64, @floatFromInt(lfs_net_ns)) / 1e9;
    log.info("COMPARE ({d:.2} MiB): xet={d:.2}s ({d:.1} MiB/s)  lfs={d:.2}s ({d:.1} MiB/s)", .{
        tensor_mib, xet_wall_s, tensor_mib / xet_wall_s, lfs_wall_s, tensor_mib / lfs_wall_s,
    });
    const overhead_ns = if (chunkloop_ns > decomp_ns + write_ns) chunkloop_ns - decomp_ns - write_ns else 0;
    log.info("chunk-loop breakdown ({d} chunks):", .{n_chunks_total});
    log.info("  total      = {d:.1} ms", .{nsToMs(chunkloop_ns)});
    log.info("  decompress = {d:.1} ms", .{nsToMs(decomp_ns)});
    log.info("  sink write = {d:.1} ms", .{nsToMs(write_ns)});
    log.info("  scan/other = {d:.1} ms", .{nsToMs(overhead_ns)});
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}
