// Minimal reproducer: do `std.http.Client` range GETs from N threads
// against a public HTTPS URL, sharing ONE client. No XET, no VFS, no
// `WindowCache`, no `xet.State`. If this hangs, the issue is purely in
// `std.http.Client`'s connection pool. If it succeeds, the hang in
// `test_vfs_parallel` lives in our shared mutable state.
//
// Usage:
//   bazel run //examples/xet_cas:test_http_parallel -- \
//     [--url https://...] [--threads 4] [--reqs 8] [--bytes 65536]

const std = @import("std");

const log = std.log.scoped(.test_http_parallel);

const default_url = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/model.safetensors";

const Worker = struct {
    id: u32,
    client: *std.http.Client,
    url: []const u8,
    reqs: u32,
    bytes: u64,
    done: std.atomic.Value(u32) = .init(0),
    last_err: std.atomic.Value(u32) = .init(0), // 0 = ok, 1 = had error

    fn run(self: *@This()) void {
        var i: u32 = 0;
        while (i < self.reqs) : (i += 1) {
            const start: u64 = @as(u64, i) * self.bytes;
            const end: u64 = start + self.bytes - 1;
            log.info("worker {d} req {d}/{d} start", .{ self.id, i + 1, self.reqs });
            doRange(self.client, self.url, start, end) catch |e| {
                log.err("worker {d} req {d}/{d} ERR: {s}", .{ self.id, i + 1, self.reqs, @errorName(e) });
                self.last_err.store(1, .monotonic);
                continue;
            };
            log.info("worker {d} req {d}/{d} OK", .{ self.id, i + 1, self.reqs });
            _ = self.done.fetchAdd(1, .monotonic);
        }
    }
};

fn doRange(client: *std.http.Client, url: []const u8, range_start: u64, range_end: u64) !void {
    var range_buf: [64]u8 = undefined;
    const range_header = std.fmt.bufPrint(&range_buf, "bytes={d}-{d}", .{ range_start, range_end }) catch unreachable;
    const uri = try std.Uri.parse(url);
    var req = try client.request(.GET, uri, .{
        .headers = .{ .accept_encoding = .{ .override = "identity" } },
        .extra_headers = &.{.{ .name = "Range", .value = range_header }},
    });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buffer: [8 * 1024]u8 = undefined;
    var res = try req.receiveHead(&redirect_buffer);
    if (res.head.status != .partial_content and res.head.status != .ok) {
        log.err("status={} url={s}", .{ res.head.status, url });
        return error.HttpRequestFailed;
    }
    const len: usize = @intCast(range_end - range_start + 1);
    const buf = try std.heap.page_allocator.alloc(u8, len);
    defer std.heap.page_allocator.free(buf);
    try res.reader(&.{}).readSliceAll(buf);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args_it = init.minimal.args.iterate();
    _ = args_it.skip();
    var url: []const u8 = default_url;
    var threads_n: u32 = 4;
    var reqs: u32 = 8;
    var bytes: u64 = 64 * 1024;
    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--url")) {
            url = args_it.next() orelse return error.MissingArg;
        } else if (std.mem.eql(u8, arg, "--threads")) {
            threads_n = try std.fmt.parseInt(u32, args_it.next() orelse return error.MissingArg, 10);
        } else if (std.mem.eql(u8, arg, "--reqs")) {
            reqs = try std.fmt.parseInt(u32, args_it.next() orelse return error.MissingArg, 10);
        } else if (std.mem.eql(u8, arg, "--bytes")) {
            bytes = try std.fmt.parseInt(u64, args_it.next() orelse return error.MissingArg, 10);
        }
    }

    log.info("config: url={s} threads={d} reqs_per_thread={d} bytes_per_req={d}", .{ url, threads_n, reqs, bytes });

    var http_client: std.http.Client = .{ .allocator = allocator, .io = init.io };
    try http_client.initDefaultProxies(allocator, init.environ_map);
    defer http_client.deinit();

    // Pre-warm: one sequential GET to populate any DNS / TLS state, mirroring
    // what the real driver does (open one shard before forking workers).
    log.info("pre-warm: 1 sequential range GET", .{});
    try doRange(&http_client, url, 0, bytes - 1);
    log.info("pre-warm OK; spawning {d} workers", .{threads_n});

    const workers = try allocator.alloc(Worker, threads_n);
    defer allocator.free(workers);
    for (workers, 0..) |*w, i| w.* = .{
        .id = @intCast(i),
        .client = &http_client,
        .url = url,
        .reqs = reqs,
        .bytes = bytes,
    };

    const threads = try allocator.alloc(std.Thread, threads_n);
    defer allocator.free(threads);

    const ts: std.Io.Timestamp = .now(init.io, .awake);
    for (threads, workers) |*t, *w| t.* = try std.Thread.spawn(.{}, Worker.run, .{w});
    for (threads) |t| t.join();
    const elapsed_ns: u64 = @intCast(ts.untilNow(init.io, .awake).toNanoseconds());

    var total_done: u32 = 0;
    var total_err: u32 = 0;
    for (workers) |*w| {
        total_done += w.done.load(.monotonic);
        total_err += w.last_err.load(.monotonic);
    }
    log.info("done: total_ok={d}/{d} workers_with_errors={d} elapsed_ms={d}", .{
        total_done, threads_n * reqs, total_err, elapsed_ns / std.time.ns_per_ms,
    });
    if (total_done != threads_n * reqs) {
        log.err("FAIL: not all requests completed", .{});
        std.process.exit(1);
    }
    log.info("OK", .{});
}
