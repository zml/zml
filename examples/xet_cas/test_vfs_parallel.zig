// Step 6 acceptance driver: read 4 different shards of the same XET-backed
// model concurrently through one shared `hf_vfs` (one shared `xet.State`,
// one shared `WindowCache`). Verifies that `acquire`/`release`, the per-
// entry pin, and the atomic `bytes_fetched` counter survive multi-threaded
// pull-driven reads. SHA-256s are compared against the LFS oracle.
//
// Caveat: the lazy caches on `xet.State` (file_id_cache, cas_cache,
// plan_cache) are not yet locked; this driver pre-warms them in the main
// thread before spawning workers so each worker only hits read-only paths.
//
// Usage:
//   HF_TOKEN=... bazel run //examples/xet_cas:test_vfs_parallel -- \
//     --model hf://meta-llama/Meta-Llama-3-70B \
//     [--shards model-00001-of-00030.safetensors,model-00002-of-00030.safetensors,...]

const std = @import("std");
const zml = @import("zml");
const xet = @import("io").xet;
const util = @import("util.zig");

const log = std.log.scoped(.test_vfs_parallel);

const num_workers = 4;

const default_shards = [num_workers][]const u8{
    "model-00001-of-00030.safetensors",
    "model-00010-of-00030.safetensors",
    "model-00020-of-00030.safetensors",
    "model-00030-of-00030.safetensors",
};

const Result = struct {
    sha: [32]u8 = undefined,
    bytes: u64 = 0,
    ms: u64 = 0,
    err: ?anyerror = null,
};

const WorkerCtx = struct {
    io: std.Io,
    vfs_io: std.Io,
    file_uri: []const u8,
    result: *Result,
};

fn worker(ctx: *WorkerCtx) void {
    const f = std.Io.Dir.openFile(.cwd(), ctx.vfs_io, ctx.file_uri, .{ .mode = .read_only }) catch |e| {
        ctx.result.err = e;
        return;
    };
    defer f.close(ctx.vfs_io);

    const stat = f.stat(ctx.vfs_io) catch |e| {
        ctx.result.err = e;
        return;
    };
    var reader = f.reader(ctx.vfs_io, &.{});
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [64 * 1024]u8 = undefined;
    var remaining: u64 = stat.size;
    const ts: std.Io.Timestamp = .now(ctx.io, .awake);
    while (remaining > 0) {
        const take: usize = @intCast(@min(@as(u64, buf.len), remaining));
        const n = reader.interface.readSliceShort(buf[0..take]) catch |e| {
            ctx.result.err = e;
            return;
        };
        if (n == 0) break;
        hasher.update(buf[0..n]);
        remaining -= n;
    }
    ctx.result.ms = @intCast(@as(u64, @intCast(ts.untilNow(ctx.io, .awake).toNanoseconds())) / std.time.ns_per_ms);
    ctx.result.bytes = stat.size - remaining;
    hasher.final(&ctx.result.sha);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args_it = init.minimal.args.iterate();
    _ = args_it.skip();
    var model_arg: []const u8 = "";
    var shards_arg: ?[]const u8 = null;
    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model")) {
            model_arg = args_it.next() orelse return error.MissingModelArg;
        } else if (std.mem.eql(u8, arg, "--shards")) {
            shards_arg = args_it.next() orelse return error.MissingShardsArg;
        }
    }
    if (model_arg.len == 0) {
        std.debug.print("Usage: test_vfs_parallel --model <uri> [--shards a,b,c,d]\n", .{});
        std.process.exit(1);
    }

    var shards: [num_workers][]const u8 = default_shards;
    if (shards_arg) |s| {
        var it = std.mem.splitScalar(u8, s, ',');
        var i: usize = 0;
        while (it.next()) |part| : (i += 1) {
            if (i >= num_workers) return error.TooManyShards;
            shards[i] = part;
        }
        if (i != num_workers) return error.NotEnoughShards;
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
    const vfs_io = vfs.io();

    const model_path = if (std.mem.startsWith(u8, model_arg, "hf://")) model_arg["hf://".len..] else model_arg;

    var uri_bufs: [num_workers][4096]u8 = undefined;
    var uris: [num_workers][]const u8 = undefined;
    for (shards, 0..) |s, i| {
        uris[i] = try std.fmt.bufPrint(&uri_bufs[i], "hf://{s}/{s}", .{ model_path, s });
    }

    // ── Pre-warm xet.State lazy caches (file_id, cas_auth, plan) by
    //    reading 1 byte from each shard in the main thread. After this,
    //    workers will only hit read-only paths on those caches.
    log.info("pre-warming caches for {d} shards", .{num_workers});
    for (uris) |u| {
        const f = try std.Io.Dir.openFile(.cwd(), vfs_io, u, .{ .mode = .read_only });
        defer f.close(vfs_io);
        var rdr = f.reader(vfs_io, &.{});
        var one: [1]u8 = undefined;
        _ = try rdr.interface.readSliceShort(&one);
    }

    // ── Spawn workers ────────────────────────────────────────────────────
    var results: [num_workers]Result = @splat(.{});
    var ctxs: [num_workers]WorkerCtx = undefined;
    for (&ctxs, 0..) |*c, i| {
        c.* = .{ .io = init.io, .vfs_io = vfs_io, .file_uri = uris[i], .result = &results[i] };
    }

    const bytes_before = hf_vfs.xet_state.bytes_fetched.load(.monotonic);
    const ts_par: std.Io.Timestamp = .now(init.io, .awake);

    var threads: [num_workers]std.Thread = undefined;
    for (&threads, 0..) |*t, i| {
        t.* = try std.Thread.spawn(.{}, worker, .{&ctxs[i]});
    }
    for (threads) |t| t.join();

    const par_ms: u64 = @intCast(@as(u64, @intCast(ts_par.untilNow(init.io, .awake).toNanoseconds())) / std.time.ns_per_ms);
    const bytes_delta = hf_vfs.xet_state.bytes_fetched.load(.monotonic) - bytes_before;

    // ── Oracle SHA per shard (sequential in main thread) ────────────────
    // Use a fresh `std.http.Client` so we don't reuse keep-alive
    // connections from the parallel phase, which sit idle long enough to
    // be silently half-closed by the server (causes indefinite blocking
    // reads). Real production code should add socket read deadlines; the
    // test driver sidesteps that by isolating phases.
    var oracle_client: std.http.Client = .{ .allocator = allocator, .io = init.io };
    try oracle_client.initDefaultProxies(allocator, init.environ_map);
    defer oracle_client.deinit();

    const hf_token = try loadHfToken(allocator, init.io, init.environ_map);
    defer allocator.free(hf_token);
    var auth_buf: [1024]u8 = undefined;
    const auth = std.fmt.bufPrint(&auth_buf, "Bearer {s}", .{std.mem.trim(u8, hf_token, " \t\n\r")}) catch return error.TokenTooLong;

    var any_err = false;
    var total_bytes: u64 = 0;
    for (uris, 0..) |u, i| {
        const r = &results[i];
        if (r.err) |e| {
            log.err("shard {d} ({s}) worker error: {s}", .{ i, shards[i], @errorName(e) });
            any_err = true;
            continue;
        }
        total_bytes += r.bytes;
        const hf_path = u["hf://".len..];
        const hf_repo = try zml.io.VFS.HF.Repo.parse(hf_path);
        const repo: xet.State.Repo = .{ .repo = hf_repo.repo, .model = hf_repo.model, .rev = hf_repo.rev, .path = hf_repo.path };
        var oracle_ns: u64 = 0;
        const oracle = util.sha256LfsRange(&oracle_client, repo, auth, 0, r.bytes - 1, init.io, &oracle_ns) catch |e| {
            log.err("shard {d} {s} oracle error: {s} (worker sha256={x} bytes={d})", .{ i, shards[i], @errorName(e), r.sha, r.bytes });
            any_err = true;
            continue;
        };
        const ok = std.mem.eql(u8, &r.sha, &oracle);
        log.info("shard {d} {s} bytes={d} vfs_ms={d} sha256={x} {s}", .{ i, shards[i], r.bytes, r.ms, r.sha, if (ok) "OK" else "MISMATCH" });
        if (!ok) any_err = true;
    }

    log.info("parallel total: workers={d} elapsed_ms={d} bytes_fetched={d} bytes_read={d} ratio={d:.3}", .{
        num_workers,
        par_ms,
        bytes_delta,
        total_bytes,
        @as(f64, @floatFromInt(bytes_delta)) / @as(f64, @floatFromInt(total_bytes)),
    });

    if (any_err) return error.ParallelReadFailed;
    log.info("OK — {d} shards read in parallel, all SHA-256 match", .{num_workers});
}

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
