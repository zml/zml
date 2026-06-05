// "all-tensors-of-one-file → host sinks" end-to-end smoke test.
//
// Generalization of test_tensor_to_device: loads ALL tensors sharing the
// selected file in a single driver pass with ONE CAS reconstruction call.
// Each xorb is fetched at most once and its chunks fan out to every tensor
// sink that needs them.
//
// Verification: one LFS GET of [0, file_size) into a reference buffer,
// then per-tensor SHA-256 compare(sink, ref[offset..offset+size]).
//
// Usage:
//   test_file_to_device --model <uri> --file <repo-relative-path> [--workers N]
// `--file` matches any tensor whose file_uri ends with that suffix
// (e.g. `model-00001-of-00002.safetensors`).

const std = @import("std");
const zml = @import("zml");
const xet = @import("io").xet;
const xet_stats = @import("io").xet_stats;
const util = @import("util.zig");

const log = std.log.scoped(.test_file_to_device);

pub const std_options: std.Options = .{ .log_level = .info };

const MAX_CHUNK: usize = 128 * 1024;
const DEFAULT_WORKERS: usize = 4;

const Sink = struct {
    buf: []u8,

    fn writeAt(self: *Sink, off: usize, src: []const u8) !void {
        var w: std.Io.Writer = .fixed(self.buf[off..]);
        try w.writeAll(src);
    }
};

const FetchTask = struct {
    url: []const u8,
    url_range_start: u64,
    url_range_end: u64, // inclusive
    chunk_start: u32, // chunk index of the first chunk in this fetch
    byte_len: u64,
};

const XorbWork = struct {
    hash: []const u8,
    needed_chunk_start: u32,
    needed_chunk_end: u32,
    // One or more FetchUrl ranges that together cover [needed_chunk_start, needed_chunk_end).
    fetches: []FetchTask,
};

const TermPlan = struct {
    xorb_idx: u16,
    tensor_idx: u16, // index into sinks[]
    dst_off: u32, // offset within the destination sink [0, tensor_size)
    wanted_len: u32,
    byte_skip: u32,
    chunk_start: u32,
    chunk_end: u32,
    bytes_written: u32,
    bytes_consumed: u32 = 0,
};

const Worker = struct {
    counter: *std.atomic.Value(u32),
    xorbs: []const XorbWork,
    plans: []TermPlan,
    sinks: []Sink,
    allocator: std.mem.Allocator,
    env: *std.process.Environ.Map,
    io: std.Io,
    stats: *xet_stats.AtomicStats,
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

        // xorb buffer is grown on demand: whole-file responses can produce
        // FetchUrl ranges larger than a single xorb's 64 MiB max.
        var xb: []u8 = &.{};
        defer if (xb.len != 0) self.allocator.free(xb);
        const cb = try self.allocator.alloc(u8, MAX_CHUNK);
        defer self.allocator.free(cb);

        while (true) {
            const next = self.counter.fetchAdd(1, .acq_rel);
            if (next >= self.xorbs.len) break;
            const xi: u16 = @intCast(next);
            const x = self.xorbs[xi];

            // A single xorb may be backed by several non-overlapping FetchUrl
            // entries (the server splits large CAS shards). Process them in
            // chunk-index order; plan state (bytes_consumed, bytes_written)
            // persists naturally across fetches.
            for (x.fetches) |f| {
                if (xb.len < f.byte_len) {
                    if (xb.len != 0) self.allocator.free(xb);
                    xb = try self.allocator.alloc(u8, @intCast(f.byte_len));
                }

                const t0: std.Io.Timestamp = .now(self.io, .awake);
                try util.httpRangeGetIntoSlot(&client, f.url, f.url_range_start, f.url_range_end, xb[0..f.byte_len]);
                self.stats.add("xorb_http_ns", @intCast(t0.untilNow(self.io, .awake).toNanoseconds()));
                self.stats.add("xorb_bytes_read", f.byte_len);

                var it: xet.ChunkIterator = .{ .data = xb[0..f.byte_len] };
                var chunk_idx: u32 = f.chunk_start;
                while (try it.next()) |chunk| : (chunk_idx += 1) {
                    if (chunk.uncompressed_size > MAX_CHUNK) return error.ChunkTooLarge;
                    self.stats.add("xorb_chunks", 1);
                    const cu: u32 = @intCast(chunk.uncompressed_size);
                    var decompressed = false;
                    for (self.plans) |*p| {
                        if (p.xorb_idx != xi) continue;
                        if (chunk_idx < p.chunk_start or chunk_idx >= p.chunk_end) continue;
                        if (p.bytes_consumed + cu <= p.byte_skip) {
                            p.bytes_consumed += cu;
                            continue;
                        }
                        const src_off: u32 = if (p.byte_skip > p.bytes_consumed) p.byte_skip - p.bytes_consumed else 0;
                        const avail: u32 = cu - src_off;
                        const remaining: u32 = p.wanted_len - p.bytes_written;
                        const take: u32 = @min(avail, remaining);
                        const slot_off: usize = p.dst_off + p.bytes_written;
                        if (!decompressed) {
                            const ts_d: std.Io.Timestamp = .now(self.io, .awake);
                            _ = try xet.decompressChunk(chunk, cb[0..chunk.uncompressed_size]);
                            self.stats.add("xorb_decode_ns", @intCast(ts_d.untilNow(self.io, .awake).toNanoseconds()));
                            decompressed = true;
                        }
                        try self.sinks[p.tensor_idx].writeAt(slot_off, cb[src_off .. src_off + take]);
                        p.bytes_written += take;
                        p.bytes_consumed += src_off + take;
                    }
                }
            }
        }
    }
};

// Per-tensor handle to its place in the file + its destination sink.
const TensorJob = struct {
    name: []const u8,
    file_offset: u64,
    size: u64,
    sink_idx: u16,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args_it = init.minimal.args.iterate();
    _ = args_it.skip();
    var model_arg: []const u8 = "";
    var file_arg: []const u8 = "";
    var n_workers: usize = DEFAULT_WORKERS;
    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model")) {
            model_arg = args_it.next() orelse return error.MissingModelArg;
        } else if (std.mem.eql(u8, arg, "--file")) {
            file_arg = args_it.next() orelse return error.MissingFileArg;
        } else if (std.mem.eql(u8, arg, "--workers")) {
            n_workers = try std.fmt.parseInt(usize, args_it.next() orelse return error.MissingWorkersArg, 10);
            if (n_workers == 0) return error.InvalidWorkers;
        }
    }
    if (model_arg.len == 0 or file_arg.len == 0) {
        std.debug.print("Usage: test_file_to_device --model <uri> --file <repo-relative-path> [--workers N]\n", .{});
        std.process.exit(1);
    }

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

    log.info("Resolving model repo: {s}", .{model_arg});
    const repo_dir = try zml.safetensors.resolveModelRepo(io, model_arg);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo_dir);
    defer registry.deinit();

    // Resolve --file by matching any tensor whose file_uri ends with the
    // given suffix (handles both bare paths and full hf://... URIs).
    var file_uri: []const u8 = "";
    for (registry.tensors.values()) |t| {
        if (std.mem.endsWith(u8, t.file_uri, file_arg)) {
            file_uri = t.file_uri;
            break;
        }
    }
    if (file_uri.len == 0) {
        log.err("No tensor has file_uri matching: {s}", .{file_arg});
        return error.FileNotFound;
    }
    log.info("File: {s}", .{file_uri});

    // ── Collect every tensor sharing this file ─────────────────────────────
    var jobs = std.ArrayList(TensorJob).empty;
    defer jobs.deinit(allocator);
    var file_size: u64 = 0;
    for (registry.tensors.values()) |t| {
        if (!std.mem.eql(u8, t.file_uri, file_uri)) continue;
        const end = t.offset + t.byteSize();
        if (end > file_size) file_size = end;
        try jobs.append(allocator, .{
            .name = t.name,
            .file_offset = t.offset,
            .size = t.byteSize(),
            .sink_idx = @intCast(jobs.items.len),
        });
    }
    log.info("file_size={d} tensors_in_file={d}", .{ file_size, jobs.items.len });
    if (jobs.items.len == 0) return error.NoTensorsInFile;

    // ── Allocate one host sink per tensor ─────────────────────────────────
    const sinks = try allocator.alloc(Sink, jobs.items.len);
    defer {
        for (sinks) |s| allocator.free(s.buf);
        allocator.free(sinks);
    }
    var total_tensor_bytes: u64 = 0;
    for (jobs.items, sinks) |j, *s| {
        if (j.size > std.math.maxInt(usize)) return error.TensorTooLarge;
        s.* = .{ .buf = try allocator.alloc(u8, @intCast(j.size)) };
        total_tensor_bytes += j.size;
    }

    // ── Repo + auth ────────────────────────────────────────────────────────
    const hf_path = if (std.mem.startsWith(u8, file_uri, "hf://")) file_uri["hf://".len..] else file_uri;
    const hf_repo = try zml.io.VFS.HF.Repo.parse(hf_path);
    const repo: xet.Client.Repo = .{ .repo = hf_repo.repo, .model = hf_repo.model, .rev = hf_repo.rev, .path = hf_repo.path };
    const hf_token = try loadHfToken(allocator, init.io, init.environ_map);
    defer allocator.free(hf_token);
    var auth_buf: [1024]u8 = undefined;
    const auth = std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{std.mem.trim(u8, hf_token, " \t\n\r")}) catch return error.TokenTooLong;

    // ── ONE reconstruction call for the whole file ────────────────────────
    var xet_client: xet.Client = .init(allocator, &http_client, hf_token);
    defer xet_client.deinit();
    // Total xet wall: covers reconstruct RTT + plan build + worker fetch/decompress.
    const ts_xet_total: std.Io.Timestamp = .now(init.io, .awake);
    const ts_recon: std.Io.Timestamp = .now(init.io, .awake);
    const parsed = try xet_client.reconstruct(repo, 0, file_size);
    defer parsed.deinit();
    const recon_ns: u64 = @intCast(ts_recon.untilNow(init.io, .awake).toNanoseconds());
    const resp = parsed.value;
    log.info("reconstruct: 1 call, {d} terms, {d} ms", .{ resp.terms.len, recon_ns / std.time.ns_per_ms });

    const ts_plan: std.Io.Timestamp = .now(init.io, .awake);

    // ── Build a single (xorbs, plans) shared across all tensors ───────────
    // Upper bound: each tensor contributes ≤ resp.terms.len plans.
    const max_plans: usize = resp.terms.len * jobs.items.len;
    const plans = try allocator.alloc(TermPlan, max_plans);
    defer allocator.free(plans);
    const xorbs = try allocator.alloc(XorbWork, resp.terms.len);
    defer allocator.free(xorbs);

    // Pre-compute (term_off, term_end) once; we'll scan it once per tensor.
    const term_offs = try allocator.alloc(u64, resp.terms.len + 1);
    defer allocator.free(term_offs);
    term_offs[0] = 0;
    for (resp.terms, 0..) |t, i| term_offs[i + 1] = term_offs[i] + t.unpacked_length;

    var n_xorbs: u16 = 0;
    var n_plans: u32 = 0;
    for (jobs.items, 0..) |j, ji| {
        const win_start: u64 = resp.offset_into_first_range + j.file_offset;
        const win_end: u64 = win_start + j.size;
        // Binary-search the first term overlapping the window. Linear scan
        // is fine here (few thousand terms, hot path is the worker loop).
        var first: usize = 0;
        while (first < resp.terms.len and term_offs[first + 1] <= win_start) : (first += 1) {}
        var ti: usize = first;
        while (ti < resp.terms.len and term_offs[ti] < win_end) : (ti += 1) {
            const t = resp.terms[ti];
            const t_off = term_offs[ti];
            const t_end = term_offs[ti + 1];

            const head_skip: u64 = if (t_off < win_start) win_start - t_off else 0;
            const tail_avail: u64 = @min(t_end, win_end) - t_off;
            const wanted: u64 = tail_avail - head_skip;
            const dst_off: u64 = if (t_off < win_start) 0 else t_off - win_start;

            // Find-or-insert the xorb. Linear scan: n_xorbs is small (10s-100s).
            var xi: u16 = 0;
            while (xi < n_xorbs) : (xi += 1) {
                if (std.mem.eql(u8, xorbs[xi].hash, t.hash)) break;
            }
            if (xi == n_xorbs) {
                xorbs[n_xorbs] = .{
                    .hash = t.hash,
                    .needed_chunk_start = std.math.maxInt(u32),
                    .needed_chunk_end = 0,
                    .fetches = &.{},
                };
                n_xorbs += 1;
            }
            const x = &xorbs[xi];
            if (t.range.start < x.needed_chunk_start) x.needed_chunk_start = @intCast(t.range.start);
            if (t.range.end > x.needed_chunk_end) x.needed_chunk_end = @intCast(t.range.end);
            plans[n_plans] = .{
                .xorb_idx = xi,
                .tensor_idx = @intCast(ji),
                .dst_off = @intCast(dst_off),
                .wanted_len = @intCast(wanted),
                .byte_skip = @intCast(head_skip),
                .chunk_start = @intCast(t.range.start),
                .chunk_end = @intCast(t.range.end),
                .bytes_written = 0,
            };
            n_plans += 1;
        }
    }
    log.info("plans: {d} across {d} unique xorbs", .{ n_plans, n_xorbs });

    // ── Resolve covering FetchUrl entries per xorb ──────────────────────────
    // A xorb may be backed by 1..N FetchUrl entries whose chunk ranges
    // are non-overlapping. Collect every entry that intersects the needed
    // range, then sort by chunk_start so the worker walks chunks in order.
    var all_fetches = std.ArrayList(FetchTask).empty;
    defer all_fetches.deinit(allocator);
    // Stash (offset, len) while appending; resolve to slices after the list
    // stops growing, because realloc invalidates earlier slice pointers.
    const ranges = try allocator.alloc(struct { off: usize, len: usize }, n_xorbs);
    defer allocator.free(ranges);
    for (xorbs[0..n_xorbs], 0..) |*x, xi_| {
        const fetch_list = resp.fetch_info.map.get(x.hash) orelse return error.XorbNotInFetchInfo;
        const off = all_fetches.items.len;
        for (fetch_list) |f| {
            // Half-open intersection: [f.range.start, f.range.end) ∩ [needed_start, needed_end)
            if (f.range.end <= x.needed_chunk_start) continue;
            if (f.range.start >= x.needed_chunk_end) continue;
            try all_fetches.append(allocator, .{
                .url = f.url,
                .url_range_start = f.url_range.start,
                .url_range_end = f.url_range.end,
                .chunk_start = @intCast(f.range.start),
                .byte_len = f.url_range.end - f.url_range.start + 1,
            });
        }
        const len = all_fetches.items.len - off;
        if (len == 0) return error.NoFetchEntryCoversNeededChunks;
        ranges[xi_] = .{ .off = off, .len = len };
    }
    for (xorbs[0..n_xorbs], ranges) |*x, r| {
        const my_fetches = all_fetches.items[r.off .. r.off + r.len];
        std.mem.sort(FetchTask, my_fetches, {}, struct {
            fn lt(_: void, a: FetchTask, b: FetchTask) bool {
                return a.chunk_start < b.chunk_start;
            }
        }.lt);
        if (my_fetches[0].chunk_start > x.needed_chunk_start) return error.NoFetchEntryCoversNeededChunks;
        x.fetches = my_fetches;
    }

    // ── Run workers ───────────────────────────────────────────────────────
    const plan_ns: u64 = @intCast(ts_plan.untilNow(init.io, .awake).toNanoseconds());
    log.info("plan build: {d} ms ({d} plans, {d} xorbs, {d} fetch ranges)", .{ plan_ns / std.time.ns_per_ms, n_plans, n_xorbs, all_fetches.items.len });
    log.info("spawning {d} worker(s) for {d} xorb(s)", .{ n_workers, n_xorbs });
    var counter: std.atomic.Value(u32) = .init(0);
    var stats: xet_stats.AtomicStats = .{};
    const workers = try allocator.alloc(Worker, n_workers);
    defer allocator.free(workers);
    const threads = try allocator.alloc(std.Thread, n_workers);
    defer allocator.free(threads);
    const ts_workers: std.Io.Timestamp = .now(init.io, .awake);
    var n_spawned: usize = 0;
    errdefer {
        counter.store(@intCast(n_xorbs), .release);
        for (threads[0..n_spawned]) |t| t.join();
    }
    for (workers, threads) |*w, *t| {
        w.* = .{
            .counter = &counter,
            .xorbs = xorbs[0..n_xorbs],
            .plans = plans[0..n_plans],
            .sinks = sinks,
            .allocator = allocator,
            .env = init.environ_map,
            .io = init.io,
            .stats = &stats,
        };
        t.* = try std.Thread.spawn(.{}, Worker.run, .{w});
        n_spawned += 1;
    }
    for (threads) |t| t.join();
    const workers_ns: u64 = @intCast(ts_workers.untilNow(init.io, .awake).toNanoseconds());
    const xet_total_ns: u64 = @intCast(ts_xet_total.untilNow(init.io, .awake).toNanoseconds());

    var net_bytes: u64 = 0;
    for (workers) |w| {
        if (w.err) |e| return e;
    }
    net_bytes = stats.snapshot().xorb_bytes_read;
    for (plans[0..n_plans]) |p| {
        if (p.bytes_written != p.wanted_len) {
            log.err("plan mismatch tensor_idx={d} written={d} expected={d}", .{ p.tensor_idx, p.bytes_written, p.wanted_len });
            return error.PlanByteCountMismatch;
        }
    }

    // ── Verify: one LFS GET of [0, file_size), SHA per tensor slot ────────
    log.info("verifying {d} tensors via single LFS reference download...", .{jobs.items.len});
    if (file_size > std.math.maxInt(usize)) return error.FileTooLargeForRef;
    const ref = try allocator.alloc(u8, @intCast(file_size));
    defer allocator.free(ref);
    var lfs_net_ns: u64 = 0;
    try lfsDownloadAll(&http_client, repo, auth, init.io, ref, &lfs_net_ns);
    var mismatches: u32 = 0;
    for (jobs.items, sinks) |j, s| {
        const ref_slice = ref[@intCast(j.file_offset)..@intCast(j.file_offset + j.size)];
        var sink_hasher = std.crypto.hash.sha2.Sha256.init(.{});
        sink_hasher.update(s.buf);
        var sink_d: [32]u8 = undefined;
        sink_hasher.final(&sink_d);
        var ref_hasher = std.crypto.hash.sha2.Sha256.init(.{});
        ref_hasher.update(ref_slice);
        var ref_d: [32]u8 = undefined;
        ref_hasher.final(&ref_d);
        if (!std.mem.eql(u8, &sink_d, &ref_d)) {
            log.err("MISMATCH {s}: xet=0x{x} lfs=0x{x}", .{ j.name, sink_d, ref_d });
            mismatches += 1;
        }
    }
    if (mismatches > 0) {
        log.err("{d}/{d} tensors mismatch", .{ mismatches, jobs.items.len });
        return error.LfsOracleMismatch;
    }
    log.info("OK: all {d} tensors verified against LFS reference", .{jobs.items.len});

    const workers_s = @as(f64, @floatFromInt(workers_ns)) / 1e9;
    const xet_total_s = @as(f64, @floatFromInt(xet_total_ns)) / 1e9;
    const lfs_s = @as(f64, @floatFromInt(lfs_net_ns)) / 1e9;
    const data_mib = @as(f64, @floatFromInt(total_tensor_bytes)) / (1024.0 * 1024.0);
    const net_mib = @as(f64, @floatFromInt(net_bytes)) / (1024.0 * 1024.0);
    log.info("xet total: {d:.2} s wall (reconstruct+plan+workers) → {d:.1} MiB/s on {d:.2} MiB tensor data", .{
        xet_total_s, data_mib / xet_total_s, data_mib,
    });
    log.info("  workers only: {d:.2} MiB net in {d:.2} s ({d:.1} MiB/s)", .{
        net_mib, workers_s, data_mib / workers_s,
    });
    log.info("lfs full-file: {d:.2} MiB in {d:.2} s ({d:.1} MiB/s)", .{
        @as(f64, @floatFromInt(file_size)) / (1024.0 * 1024.0), lfs_s, @as(f64, @floatFromInt(file_size)) / (1024.0 * 1024.0) / lfs_s,
    });
    log.info("reconstruct RTT: {d} ms (1 call vs {d} per-tensor calls saved)", .{
        recon_ns / std.time.ns_per_ms, jobs.items.len,
    });
}

fn lfsDownloadAll(
    client: *std.http.Client,
    repo: xet.Client.Repo,
    auth: []const u8,
    io: std.Io,
    out: []u8,
    out_net_ns: *u64,
) !void {
    var url_buf: [4096]u8 = undefined;
    const lfs_url = try std.fmt.bufPrint(
        &url_buf,
        "https://huggingface.co/{s}/{s}/resolve/{s}/{s}",
        .{ repo.repo, repo.model, repo.rev, repo.path },
    );
    const uri: std.Uri = try .parse(lfs_url);
    var req = try client.request(.GET, uri, .{
        .headers = .{
            .accept_encoding = .{ .override = "identity" },
            .authorization = .{ .override = auth },
        },
    });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buffer: [8 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buffer);
    if (res.head.status != .ok) {
        log.err("LFS full-file GET failed: status={}", .{res.head.status});
        return error.HttpRequestFailed;
    }
    var transfer_buf: [64 * 1024]u8 = undefined;
    const r = res.reader(&transfer_buf);
    const ts: std.Io.Timestamp = .now(io, .awake);
    try r.readSliceAll(out);
    out_net_ns.* = @intCast(ts.untilNow(io, .awake).toNanoseconds());
}

// Mirrors `zml.io.VFS.HF.auto`: env var first, then ~/.cache/huggingface/token.
fn loadHfToken(allocator: std.mem.Allocator, io: std.Io, env: *std.process.Environ.Map) ![]u8 {
    if (env.get("HF_TOKEN")) |t| return allocator.dupe(u8, t);
    const home = env.get("HOME") orelse return error.MissingToken;
    var path_buf: [256]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "{s}/.cache/huggingface/token", .{home}) catch return error.MissingToken;
    var file = std.Io.Dir.openFileAbsolute(io, path, .{ .mode = .read_only }) catch return error.MissingToken;
    defer file.close(io);
    const size = file.stat(io) catch return error.MissingToken;
    var reader = file.reader(io, &.{});
    return reader.interface.readAlloc(allocator, size.size) catch error.MissingToken;
}
