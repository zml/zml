const std = @import("std");

const HTTP = @import("http.zig").HTTP;
const MockServer = @import("mock_server.zig").MockServer;
const range_read = @import("range_read.zig");
const vfs_request = @import("request.zig");
const S3 = @import("s3.zig").S3;

/// A retry configuration that never sleeps, for the tests that count
/// attempts rather than time.
fn instantRetries(max_retries: usize) vfs_request.Retry {
    return .{
        .max_retries = max_retries,
        .initial_delay = .fromNanoseconds(0),
        .max_delay = .fromNanoseconds(0),
        .max_hold = .fromNanoseconds(0),
        .throttle_budget = .fromSeconds(300),
    };
}

/// A retry configuration whose retries and holds are short enough for a test.
fn fastRetry(initial_ms: i64, budget_ms: i64) vfs_request.Retry {
    return .{
        .max_retries = 5,
        .initial_delay = .fromMilliseconds(initial_ms),
        .max_delay = .fromMilliseconds(initial_ms),
        .max_hold = .fromSeconds(30),
        .throttle_budget = .fromMilliseconds(budget_ms),
    };
}

/// A `MockServer` serving `object`, an HTTP backend with a client of its
/// own, and `/object` open through that backend. Initialized in place: the
/// backend points at the client and the server must not move once started.
const Fixture = struct {
    server: MockServer,
    client: std.http.Client,
    http: HTTP,
    file: std.Io.File,

    fn init(self: *Fixture, object: []const u8, server_opts: MockServer.Options, retry: vfs_request.Retry) !void {
        const allocator = std.testing.allocator;
        const io = std.testing.io;
        self.server = try .init(io, object, server_opts);
        errdefer self.server.deinit(io);
        try self.server.start(io);
        self.client = .{ .allocator = allocator, .io = io };
        errdefer self.client.deinit();
        self.http = try .init(allocator, io, &self.client, .{ .protocol = .http, .retry = retry });
        errdefer self.http.deinit();
        var path_buffer: [128]u8 = undefined;
        const path = try self.server.path(&path_buffer);
        self.file = try std.Io.Dir.openFile(.cwd(), self.http.io(), path, .{ .mode = .read_only });
    }

    fn deinit(self: *Fixture) void {
        self.file.close(self.http.io());
        self.http.deinit();
        self.client.deinit();
        self.server.deinit(std.testing.io);
    }
};

/// Reads `object` back through the fixture's file with `readers` concurrent
/// positional reads of `read_size` bytes, reader `i` at `i * read_size`,
/// all released at once. Fails on the first reader that failed or read
/// other bytes.
fn expectConcurrentReads(fixture: *Fixture, object: []const u8, comptime readers: usize, comptime read_size: usize) !void {
    const Reader = struct {
        fn run(file: std.Io.File, file_io: std.Io, output: []u8, offset: u64, result: *anyerror!usize, start: *std.Io.Event) void {
            start.wait(file_io) catch |err| {
                result.* = err;
                return;
            };
            result.* = file.readPositional(file_io, &.{output}, offset);
        }
    };
    const io = std.testing.io;
    var outputs: [readers][read_size]u8 = undefined;
    var results: [readers]anyerror!usize = undefined;
    var released: std.Io.Event = .unset;
    var group: std.Io.Group = .init;
    defer group.cancel(io);
    for (&outputs, &results, 0..) |*output, *result, index| {
        try group.concurrent(io, Reader.run, .{ fixture.file, fixture.http.io(), output, index * read_size, result, &released });
    }
    released.set(io);
    try group.await(io);

    for (&outputs, results, 0..) |*output, result, index| {
        try std.testing.expectEqual(read_size, try result);
        try std.testing.expectEqualSlices(u8, object[index * read_size ..][0..read_size], output);
    }
}

test "generic HTTP keeps a large scattered positional read to one GET" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const offset = 13;
    const request_size = 17 * 1024 * 1024 + 257;

    const object = try allocator.alloc(u8, offset + request_size);
    defer allocator.free(object);
    @memset(object, 0xa5);
    const output = try allocator.alloc(u8, request_size);
    defer allocator.free(output);

    var fixture: Fixture = undefined;
    try fixture.init(object, .{}, .{});
    defer fixture.deinit();
    const server = &fixture.server;
    const stats = &fixture.http.governor.stats;

    // Fresh stats for the read: the open's HEAD is a request too.
    stats.* = .{};
    const buffers = [_][]u8{
        output[0 .. 4 * 1024 * 1024],
        output[4 * 1024 * 1024 .. 10 * 1024 * 1024],
        output[10 * 1024 * 1024 ..],
    };
    try std.testing.expectEqual(
        request_size,
        try fixture.file.readPositional(fixture.http.io(), &buffers, offset),
    );
    try std.testing.expectEqualSlices(u8, object[offset..], output);

    try server.stop(io);

    try std.testing.expectEqual(@as(usize, 1), server.head_requests.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), server.get_requests.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), stats.physical_requests.load(.acquire));
    try std.testing.expectEqual(@as(u64, request_size), stats.physical_bytes.load(.acquire));
}

test "generic HTTP follows a relative redirect chain at open and on every read" {
    const io = std.testing.io;

    var object: [4096]u8 = undefined;
    for (&object, 0..) |*byte, i| byte.* = @truncate(i);
    var output: [1000]u8 = undefined;

    // `/object` redirects twice, and the second location resolves against
    // the first hop's path.
    var fixture: Fixture = undefined;
    try fixture.init(&object, .{
        .redirects = &.{
            .{ .path = "/object", .location = "/hop/next" },
            .{ .path = "/hop/next", .location = "../moved" },
        },
        .objects = &.{.{ .path = "/moved", .bytes = &object }},
    }, .{});
    defer fixture.deinit();
    const server = &fixture.server;
    const http_io = fixture.http.io();

    try std.testing.expectEqual(@as(u64, object.len), (try fixture.file.stat(http_io)).size);
    const buffers = [_][]u8{&output};
    try std.testing.expectEqual(output.len, try fixture.file.readPositional(http_io, &buffers, 77));
    try std.testing.expectEqualSlices(u8, object[77..][0..output.len], &output);

    try server.stop(io);
    // Only the object answers with bytes; the hops before it redirect.
    try std.testing.expectEqual(@as(usize, 1), server.head_requests.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), server.get_requests.load(.acquire));
}

test "generic HTTP retries serially and reports aggregate retry counters" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const request_size = 2 * 1024 * 1024;

    const object = try allocator.alloc(u8, request_size);
    defer allocator.free(object);
    @memset(object, 0x3c);
    const output = try allocator.alloc(u8, request_size);
    defer allocator.free(output);

    var fixture: Fixture = undefined;
    try fixture.init(object, .{ .fail_first_gets = 1 }, .{
        .max_retries = 1,
        .initial_delay = .fromNanoseconds(0),
        .max_delay = .fromNanoseconds(0),
    });
    defer fixture.deinit();
    const server = &fixture.server;
    const http_io = fixture.http.io();
    const stats = &fixture.http.governor.stats;

    stats.* = .{};
    const one_buffer = [_][]u8{output};
    try std.testing.expectEqual(
        request_size,
        try fixture.file.readPositional(http_io, &one_buffer, 0),
    );
    try std.testing.expectEqual(@as(u64, 2), stats.physical_requests.load(.acquire));
    try std.testing.expectEqual(@as(u64, request_size), stats.physical_bytes.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), stats.retries.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), stats.server_failures.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), server.peak_gets.load(.acquire));

    stats.* = .{};
    try std.testing.expectEqual(
        request_size,
        try fixture.file.readPositional(http_io, &one_buffer, 0),
    );
    try std.testing.expectEqual(@as(u64, 1), stats.physical_requests.load(.acquire));
    try std.testing.expectEqual(@as(u64, request_size), stats.physical_bytes.load(.acquire));
    try std.testing.expectEqual(@as(u64, 0), stats.retries.load(.acquire));
    try std.testing.expectEqualSlices(u8, object, output);

    try server.stop(io);
    try std.testing.expectEqual(@as(usize, 1), server.head_requests.load(.acquire));
    try std.testing.expectEqual(@as(usize, 3), server.get_requests.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), server.peak_gets.load(.acquire));
}

test "generic HTTP physical concurrency does not exceed caller admission" {
    const io = std.testing.io;
    const admission = 3;
    const read_size = 1024;

    var object: [admission * read_size]u8 = undefined;
    for (&object, 0..) |*byte, index| byte.* = @truncate(index);

    var fixture: Fixture = undefined;
    try fixture.init(&object, .{ .barrier_gets = admission }, .{});
    defer fixture.deinit();
    const server = &fixture.server;
    const stats = &fixture.http.governor.stats;

    stats.* = .{};
    try expectConcurrentReads(&fixture, &object, admission, read_size);

    try server.stop(io);

    const peak = server.peak_gets.load(.acquire);
    try std.testing.expectEqual(@as(usize, 1), server.head_requests.load(.acquire));
    try std.testing.expectEqual(@as(usize, admission), peak);
    try std.testing.expect(peak <= admission);
    try std.testing.expectEqual(
        @as(usize, admission),
        server.get_requests.load(.acquire),
    );
    try std.testing.expectEqual(@as(u64, admission), stats.physical_requests.load(.acquire));
}

test "the shared range loop prepares the request once per attempt" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const request_size = 3 * 1024;

    var object: [request_size]u8 = undefined;
    for (&object, 0..) |*byte, index| byte.* = @truncate(index *% 7);
    var output: [request_size]u8 = undefined;

    var server = try MockServer.init(io, &object, .{ .fail_first_gets = 2 });
    defer server.deinit(io);
    try server.start(io);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();

    // An S3-style request: as a SigV4 signature must be, it is prepared
    // again by the hook for every attempt.
    const Signed = struct {
        uri: std.Uri,
        prepared: usize = 0,

        fn prepare(self: *@This(), _: vfs_request.Attempt, range: std.http.Header) anyerror!range_read.PreparedRequest {
            try std.testing.expectEqualStrings("Range", range.name);
            self.prepared += 1;
            return .{ .uri = self.uri };
        }
    };
    var url_buffer: [160]u8 = undefined;
    const url = try std.fmt.bufPrint(&url_buffer, "http://127.0.0.1:{d}/object", .{server.port()});
    var signed: Signed = .{ .uri = try .parse(url) };
    const spec: range_read.RangeSpec = .{
        .request = .{ .target = url, .key = vfs_request.authorityOf(signed.uri) },
    };
    const buffers = [_][]u8{ output[0..1024], output[1024..] };

    // No retry budget: one attempt, one hook call, the failure is counted.
    var no_retry: vfs_request.Governor = .init(io, &client, "test", .throttle, instantRetries(0));
    try std.testing.expectError(
        error.RetriesExhausted,
        range_read.performRangeRead(&no_retry, spec, &signed, Signed.prepare, &buffers, 0, object.len),
    );
    try std.testing.expectEqual(@as(usize, 1), signed.prepared);
    try std.testing.expectEqual(@as(u64, 1), no_retry.stats.server_failures.load(.acquire));

    signed.prepared = 0;
    var one_retry: vfs_request.Governor = .init(io, &client, "test", .throttle, instantRetries(1));
    try std.testing.expectEqual(
        object.len,
        try range_read.performRangeRead(&one_retry, spec, &signed, Signed.prepare, &buffers, 0, object.len),
    );
    try std.testing.expectEqual(@as(usize, 2), signed.prepared);
    try std.testing.expectEqualSlices(u8, &object, &output);

    try server.stop(io);

    // One GET per hook call.
    try std.testing.expectEqual(@as(usize, 3), server.get_requests.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), server.peak_gets.load(.acquire));
    const stats = &one_retry.stats;
    try std.testing.expectEqual(@as(u64, 2), stats.physical_requests.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), stats.retries.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), stats.server_failures.load(.acquire));
    try std.testing.expectEqual(@as(u64, request_size), stats.physical_bytes.load(.acquire));
}

test "one throttled GET holds every reader of the backend" {
    const io = std.testing.io;
    const readers = 8;
    const read_size = 512;

    var object: [readers * read_size]u8 = undefined;
    for (&object, 0..) |*byte, index| byte.* = @truncate(index);

    // Every reader's first GET is throttled, all eight answered together (the
    // barrier), with a one second reset.
    var fixture: Fixture = undefined;
    try fixture.init(&object, .{
        .barrier_gets = readers,
        .throttle = .{ .first_gets = readers, .retry_after_s = 1 },
    }, fastRetry(50, 300_000));
    defer fixture.deinit();
    const server = &fixture.server;
    const stats = &fixture.http.governor.stats;

    try expectConcurrentReads(&fixture, &object, readers, read_size);

    try server.stop(io);

    // One hold covered every reader, the server-named second was honoured,
    // and no reader burned a retry for it.
    try std.testing.expectEqual(@as(u64, 1), stats.holds.load(.acquire));
    try std.testing.expectEqual(@as(u64, readers), stats.throttles.load(.acquire));
    try std.testing.expectEqual(@as(u64, 0), stats.retries.load(.acquire));
    try std.testing.expectEqual(@as(usize, 2 * readers), server.get_requests.load(.acquire));
    const throttled_at = server.first_throttle_ns.load(.acquire);
    const resumed_at = server.first_get_after_throttle_ns.load(.acquire);
    try std.testing.expect(throttled_at != 0 and resumed_at != 0);
    try std.testing.expect(resumed_at - throttled_at >= std.time.ns_per_s);
}

test "a rate window is absorbed by the hold instead of the retry budget" {
    const io = std.testing.io;
    const readers = 12;
    const read_size = 256;

    var object: [readers * read_size]u8 = undefined;
    for (&object, 0..) |*byte, index| byte.* = @truncate(index *% 3);

    // Four GETs per 100 ms; twelve readers need three windows at best.
    var fixture: Fixture = undefined;
    try fixture.init(&object, .{
        .throttle = .{ .window = .{ .gets = 4, .per_ms = 100 } },
    }, fastRetry(50, 300_000));
    defer fixture.deinit();
    const stats = &fixture.http.governor.stats;

    try expectConcurrentReads(&fixture, &object, readers, read_size);

    try fixture.server.stop(io);

    try std.testing.expect(stats.holds.load(.acquire) > 0);
    try std.testing.expect(stats.throttles.load(.acquire) > 0);
    // A limited server never exhausts a reader's retries: the hold carries
    // the whole episode.
    try std.testing.expectEqual(@as(u64, 0), stats.retries.load(.acquire));
    try std.testing.expect(stats.hold_wait_ns.load(.acquire) > 0);
}

test "a throttled HEAD at open waits the hold out and succeeds" {
    const io = std.testing.io;

    var object: [64]u8 = undefined;
    @memset(&object, 0x11);

    var fixture: Fixture = undefined;
    try fixture.init(&object, .{
        .throttle = .{ .first_heads = 1, .retry_after_s = 0 },
    }, fastRetry(20, 300_000));
    defer fixture.deinit();
    const server = &fixture.server;
    const stats = &fixture.http.governor.stats;
    try std.testing.expectEqual(@as(u64, object.len), try fixture.file.length(fixture.http.io()));

    try server.stop(io);

    // The open retried its HEAD after the hold, and the hold is the
    // backend's, not the read path's.
    try std.testing.expectEqual(@as(usize, 2), server.head_requests.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), stats.holds.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), stats.throttles.load(.acquire));
    try std.testing.expectEqual(@as(u64, 0), stats.retries.load(.acquire));
}

test "a 503 holds an object store and only retries a plain HTTP server" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const read_size = 512;

    var object: [read_size]u8 = undefined;
    @memset(&object, 0x7e);
    var output: [read_size]u8 = undefined;

    // Each backend's first two GETs are answered 503.
    var server = try MockServer.init(io, &object, .{
        .throttle = .{ .first_gets = 2, .status = .service_unavailable },
        .objects = &.{
            .{ .path = "/bucket/object", .bytes = &object },
        },
    });
    defer server.deinit(io);
    try server.start(io);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var endpoint_buffer: [64]u8 = undefined;
    const endpoint = try std.fmt.bufPrint(&endpoint_buffer, "http://127.0.0.1:{d}", .{server.port()});
    var s3 = try S3.init(allocator, io, &client, .{ .endpoint_url = endpoint, .region = "us-east-1", .retry = instantRetries(5) });
    defer s3.deinit();
    var http = try HTTP.init(allocator, io, &client, .{ .protocol = .http, .retry = instantRetries(5) });
    defer http.deinit();

    var path_buffer: [128]u8 = undefined;
    const backends = [_]struct { std.Io, *const vfs_request.Governor, []const u8, bool }{
        .{ s3.io(), &s3.governor, "bucket/object", true },
        .{ http.io(), &http.governor, try server.path(&path_buffer), false },
    };
    for (backends) |backend| {
        const backend_io, const governor, const path, const object_store = backend;
        const file = try std.Io.Dir.openFile(.cwd(), backend_io, path, .{});
        defer file.close(backend_io);
        server.get_requests.store(0, .release);
        try std.testing.expectEqual(read_size, try file.readPositional(backend_io, &.{&output}, 0));

        const stats = &governor.stats;
        if (object_store) {
            // A 503 is `SlowDown`. One reader, two throttles in a row: each
            // arms its own hold once the previous one has passed, and
            // neither is charged a retry.
            try std.testing.expectEqual(@as(u64, 2), stats.holds.load(.acquire));
            try std.testing.expectEqual(@as(u64, 2), stats.throttles.load(.acquire));
            try std.testing.expectEqual(@as(u64, 0), stats.retries.load(.acquire));
        } else {
            // A 503 is a server failure: no hold, one charged retry per
            // attempt.
            try std.testing.expectEqual(@as(u64, 0), stats.holds.load(.acquire));
            try std.testing.expectEqual(@as(u64, 2), stats.server_failures.load(.acquire));
            try std.testing.expectEqual(@as(u64, 2), stats.retries.load(.acquire));
        }
    }

    try server.stop(io);
}

test "a server that never lets up fails the request with RateLimited" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const read_size = 256;

    var object: [read_size]u8 = undefined;
    @memset(&object, 0x33);
    var output: [read_size]u8 = undefined;

    var server = try MockServer.init(io, &object, .{
        .throttle = .{ .always = true, .retry_after_s = 0 },
    });
    defer server.deinit(io);
    try server.start(io);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();

    var url_buffer: [160]u8 = undefined;
    const url = try std.fmt.bufPrint(&url_buffer, "http://127.0.0.1:{d}/object", .{server.port()});
    const uri: std.Uri = try .parse(url);
    const buffers = [_][]u8{&output};

    var governor: vfs_request.Governor = .init(io, &client, "test", .throttle, .{
        .max_retries = 5,
        .initial_delay = .fromMilliseconds(20),
        .max_delay = .fromMilliseconds(20),
        .max_hold = .fromMilliseconds(50),
        .throttle_budget = .fromMilliseconds(300),
    });
    const started: std.Io.Timestamp = .now(io, .awake);
    // The read never runs: the backend is rate limited past its budget.
    // `fileReadPositional` maps this to `Unexpected` with its own log line.
    try std.testing.expectError(error.RateLimited, range_read.performRangeRead(&governor, .{
        .request = .{ .target = url },
    }, &uri, range_read.static, &buffers, 0, object.len));
    const elapsed = started.durationTo(.now(io, .awake)).nanoseconds;
    try std.testing.expect(elapsed >= 300 * std.time.ns_per_ms);
    try std.testing.expect(elapsed < 5 * std.time.ns_per_s);
    try std.testing.expect(governor.stats.holds.load(.acquire) > 1);

    try server.stop(io);
}

test "a reader cancelled while the backend is held returns Canceled" {
    const io = std.testing.io;
    const read_size = 256;

    var object: [read_size]u8 = undefined;
    @memset(&object, 0x21);
    var output: [read_size]u8 = undefined;

    var fixture: Fixture = undefined;
    try fixture.init(&object, .{
        .throttle = .{ .first_gets = 1, .retry_after_s = 30 },
    }, fastRetry(20, 300_000));
    defer fixture.deinit();

    var failure: std.atomic.Value(u16) = .init(0);
    var readers: std.Io.Group = .init;
    try readers.concurrent(io, struct {
        fn run(file_: std.Io.File, io_: std.Io, out: []u8, failure_: *std.atomic.Value(u16)) void {
            const buffers = [_][]u8{out};
            _ = file_.readPositional(io_, &buffers, 0) catch |err| {
                failure_.store(@intFromError(err), .release);
                return;
            };
        }
    }.run, .{ fixture.file, fixture.http.io(), output[0..], &failure });

    // Let the first GET be throttled and the reader settle into the hold.
    try io.sleep(.fromMilliseconds(50), .awake);
    readers.cancel(io);
    try std.testing.expectEqual(@intFromError(error.Canceled), failure.load(.acquire));

    try fixture.server.stop(io);
}

test "generic HTTP cancels a GET that receives nothing and retries it as a timeout" {
    const io = std.testing.io;

    var object: [4096]u8 = undefined;
    for (&object, 0..) |*byte, i| byte.* = @truncate(i);
    var output: [1000]u8 = undefined;

    // The first GET waits unanswered for the second: a peer that receives
    // the request and sends nothing.
    var fixture: Fixture = undefined;
    try fixture.init(&object, .{ .barrier_gets = 2 }, .{
        .initial_delay = .fromNanoseconds(0),
        .max_delay = .fromNanoseconds(0),
        .stall_timeout = .fromMilliseconds(100),
    });
    defer fixture.deinit();
    const server = &fixture.server;
    const stats = &fixture.http.governor.stats;

    stats.* = .{};
    const buffers = [_][]u8{&output};
    try std.testing.expectEqual(output.len, try fixture.file.readPositional(fixture.http.io(), &buffers, 77));
    try std.testing.expectEqualSlices(u8, object[77..][0..output.len], &output);
    try std.testing.expectEqual(@as(u64, 1), stats.timeouts.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), stats.retries.load(.acquire));

    // The first GET answers late, to a connection the client closed: that
    // write may fail.
    server.stop(io) catch |err| if (err != error.WriteFailed) return err;
    try std.testing.expectEqual(@as(usize, 2), server.get_requests.load(.acquire));
}

test "http and s3 fail open, stat and access of an absent or denied object alike" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const Case = struct { name: []const u8, status: std.http.Status, expected: anyerror };
    const cases = [_]Case{
        .{ .name = "missing", .status = .not_found, .expected = error.FileNotFound },
        .{ .name = "gone", .status = .gone, .expected = error.FileNotFound },
        .{ .name = "unauthorized", .status = .unauthorized, .expected = error.PermissionDenied },
        .{ .name = "forbidden", .status = .forbidden, .expected = error.PermissionDenied },
    };
    // The target each backend sends for an object: HTTP its path, S3
    // `/<bucket>/<key>`.
    const targets = [_][]const u8{ "/", "/bucket/" };
    const refusals = comptime refusals: {
        var list: [cases.len * targets.len]MockServer.Refusal = undefined;
        for (cases, 0..) |case, i| {
            for (targets, 0..) |target, j| {
                list[i * targets.len + j] = .{ .path = target ++ case.name, .status = case.status };
            }
        }
        break :refusals list;
    };

    var object: [32]u8 = undefined;
    @memset(&object, 0x6b);
    var server = try MockServer.init(io, &object, .{
        .objects = &.{
            .{ .path = "/bucket/object", .bytes = &object },
        },
        .refusals = &refusals,
    });
    defer server.deinit(io);
    try server.start(io);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var endpoint_buffer: [64]u8 = undefined;
    const endpoint = try std.fmt.bufPrint(&endpoint_buffer, "http://127.0.0.1:{d}", .{server.port()});
    var http = try HTTP.init(allocator, io, &client, .{ .protocol = .http });
    defer http.deinit();
    var s3 = try S3.init(allocator, io, &client, .{ .endpoint_url = endpoint, .region = "us-east-1" });
    defer s3.deinit();

    var authority_buffer: [64]u8 = undefined;
    const backends = [_]struct { std.Io, []const u8 }{
        .{ http.io(), try std.fmt.bufPrint(&authority_buffer, "{s}/", .{endpoint["http://".len..]}) },
        .{ s3.io(), "bucket/" },
    };
    for (backends) |backend| {
        const backend_io, const prefix = backend;
        var path_buffer: [128]u8 = undefined;
        const present = try std.fmt.bufPrint(&path_buffer, "{s}object", .{prefix});
        try std.Io.Dir.access(.cwd(), backend_io, present, .{});
        try std.testing.expectEqual(@as(u64, object.len), (try std.Io.Dir.statFile(.cwd(), backend_io, present, .{})).size);
        for (cases) |case| {
            const path = try std.fmt.bufPrint(&path_buffer, "{s}{s}", .{ prefix, case.name });
            try std.testing.expectError(case.expected, std.Io.Dir.openFile(.cwd(), backend_io, path, .{}));
            try std.testing.expectError(case.expected, std.Io.Dir.statFile(.cwd(), backend_io, path, .{}));
            try std.testing.expectError(case.expected, std.Io.Dir.access(.cwd(), backend_io, path, .{}));
        }
    }

    try server.stop(io);
}

test "http and s3 fail a refused streaming read with InputOutput and seek back no further than 0" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var object: [64]u8 = undefined;
    for (&object, 0..) |*byte, index| byte.* = @truncate(index *% 5 +% 1);
    // Each backend's first GET is answered 500, and none has a retry left.
    var server = try MockServer.init(io, &object, .{
        .fail_first_gets = 2,
        .objects = &.{
            .{ .path = "/bucket/object", .bytes = &object },
        },
    });
    defer server.deinit(io);
    try server.start(io);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var endpoint_buffer: [64]u8 = undefined;
    const endpoint = try std.fmt.bufPrint(&endpoint_buffer, "http://127.0.0.1:{d}", .{server.port()});
    var http = try HTTP.init(allocator, io, &client, .{ .protocol = .http, .retry = instantRetries(0) });
    defer http.deinit();
    var s3 = try S3.init(allocator, io, &client, .{ .endpoint_url = endpoint, .region = "us-east-1", .retry = instantRetries(0) });
    defer s3.deinit();

    var path_buffer: [128]u8 = undefined;
    const backend_ios = [_]std.Io{ http.io(), s3.io() };
    const paths = [_][]const u8{ try server.path(&path_buffer), "bucket/object" };
    var files: [backend_ios.len]std.Io.File = undefined;
    var opened: usize = 0;
    defer for (files[0..opened], backend_ios[0..opened]) |file, backend_io| file.close(backend_io);
    for (&files, backend_ios, paths) |*file, backend_io, path| {
        file.* = try std.Io.Dir.openFile(.cwd(), backend_io, path, .{});
        opened += 1;
    }

    var head: [8]u8 = undefined;
    const head_slices = [_][]u8{&head};
    for (files, backend_ios) |file, backend_io| {
        try std.testing.expectError(error.InputOutput, file.readStreaming(backend_io, &head_slices));
    }
    for (files, backend_ios) |file, backend_io| {
        // Forward, then back by more than the position: at 0.
        try backend_io.vtable.fileSeekBy(backend_io.userdata, file, 5);
        try backend_io.vtable.fileSeekBy(backend_io.userdata, file, -9);
        try std.testing.expectEqual(head.len, try file.readStreaming(backend_io, &head_slices));
        try std.testing.expectEqualSlices(u8, object[0..head.len], &head);
        // `minInt(i64)`, whose negation overflows, stops at 0 too.
        try backend_io.vtable.fileSeekBy(backend_io.userdata, file, std.math.minInt(i64));
        try std.testing.expectEqual(head.len, try file.readStreaming(backend_io, &head_slices));
        try std.testing.expectEqualSlices(u8, object[0..head.len], &head);
    }

    try server.stop(io);
    try std.testing.expectEqual(@as(usize, 6), server.get_requests.load(.acquire));
}
