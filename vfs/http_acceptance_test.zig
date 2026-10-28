const std = @import("std");

const GCS = @import("gcs.zig").GCS;
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

    var server = try MockServer.init(io, object, .{});
    defer server.deinit(io);
    try server.start(io);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http = try HTTP.init(allocator, io, &client, .{ .protocol = .http });
    defer http.deinit();
    const http_io = http.io();
    const stats = &http.governor.stats;

    var path_buffer: [128]u8 = undefined;
    const path = try server.path(&path_buffer);
    const file = try std.Io.Dir.openFile(.cwd(), http_io, path, .{ .mode = .read_only });
    defer file.close(http_io);

    // Fresh stats for the read: the open's HEAD is a request too.
    stats.* = .{};
    const buffers = [_][]u8{
        output[0 .. 4 * 1024 * 1024],
        output[4 * 1024 * 1024 .. 10 * 1024 * 1024],
        output[10 * 1024 * 1024 ..],
    };
    try std.testing.expectEqual(
        request_size,
        try file.readPositional(http_io, &buffers, offset),
    );
    try std.testing.expectEqualSlices(u8, object[offset..], output);

    try server.stop(io);

    try std.testing.expectEqual(@as(usize, 1), server.head_requests.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), server.get_requests.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), stats.physical_requests.load(.acquire));
    try std.testing.expectEqual(@as(u64, request_size), stats.physical_bytes.load(.acquire));
}

test "generic HTTP follows a relative redirect chain at open and on every read" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var object: [4096]u8 = undefined;
    for (&object, 0..) |*byte, i| byte.* = @truncate(i);
    var output: [1000]u8 = undefined;

    // The second location resolves against the first hop's path.
    var server = try MockServer.init(io, &object, .{ .redirects = &.{
        .{ .path = "/start", .location = "/hop/next" },
        .{ .path = "/hop/next", .location = "../object" },
    } });
    defer server.deinit(io);
    try server.start(io);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http = try HTTP.init(allocator, io, &client, .{ .protocol = .http });
    defer http.deinit();
    const http_io = http.io();

    var path_buffer: [128]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buffer, "127.0.0.1:{d}/start", .{server.port()});
    const file = try std.Io.Dir.openFile(.cwd(), http_io, path, .{ .mode = .read_only });
    defer file.close(http_io);
    try std.testing.expectEqual(@as(u64, object.len), (try file.stat(http_io)).size);
    const buffers = [_][]u8{&output};
    try std.testing.expectEqual(output.len, try file.readPositional(http_io, &buffers, 77));
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

    var server = try MockServer.init(io, object, .{ .fail_first_gets = 1 });
    defer server.deinit(io);
    try server.start(io);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http = try HTTP.init(allocator, io, &client, .{ .protocol = .http, .retry = .{
        .max_retries = 1,
        .initial_delay = .fromNanoseconds(0),
        .max_delay = .fromNanoseconds(0),
    } });
    defer http.deinit();
    const http_io = http.io();
    const stats = &http.governor.stats;

    var path_buffer: [128]u8 = undefined;
    const path = try server.path(&path_buffer);
    const file = try std.Io.Dir.openFile(.cwd(), http_io, path, .{ .mode = .read_only });
    defer file.close(http_io);

    stats.* = .{};
    const one_buffer = [_][]u8{output};
    try std.testing.expectEqual(
        request_size,
        try file.readPositional(http_io, &one_buffer, 0),
    );
    try std.testing.expectEqual(@as(u64, 2), stats.physical_requests.load(.acquire));
    try std.testing.expectEqual(@as(u64, request_size), stats.physical_bytes.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), stats.retries.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), stats.server_failures.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), server.peak_gets.load(.acquire));

    stats.* = .{};
    try std.testing.expectEqual(
        request_size,
        try file.readPositional(http_io, &one_buffer, 0),
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
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const admission = 3;
    const read_size = 1024;

    var object: [admission * read_size]u8 = undefined;
    for (&object, 0..) |*byte, index| byte.* = @truncate(index);

    var server = try MockServer.init(io, &object, .{
        .barrier_gets = admission,
    });
    defer server.deinit(io);
    try server.start(io);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http = try HTTP.init(allocator, io, &client, .{ .protocol = .http });
    defer http.deinit();
    const http_io = http.io();
    const stats = &http.governor.stats;

    var path_buffer: [128]u8 = undefined;
    const path = try server.path(&path_buffer);
    const file = try std.Io.Dir.openFile(.cwd(), http_io, path, .{ .mode = .read_only });
    defer file.close(http_io);

    var outputs: [admission][read_size]u8 = undefined;
    var results: [admission]usize = @splat(0);
    var first_error: std.atomic.Value(u16) = .init(0);
    var start: std.Io.Event = .unset;
    var readers: std.Io.Group = .init;
    const Reader = struct {
        fn run(
            file_: std.Io.File,
            io_: std.Io,
            output: []u8,
            offset: u64,
            result: *usize,
            first_error_: *std.atomic.Value(u16),
            start_: *std.Io.Event,
        ) void {
            start_.wait(io_) catch |err| {
                _ = first_error_.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
                return;
            };
            const buffers = [_][]u8{output};
            result.* = file_.readPositional(io_, &buffers, offset) catch |err| {
                _ = first_error_.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
                return;
            };
        }
    };
    stats.* = .{};
    for (0..admission) |index| {
        try readers.concurrent(io, Reader.run, .{
            file,
            http_io,
            &outputs[index],
            @as(u64, @intCast(index * read_size)),
            &results[index],
            &first_error,
            &start,
        });
    }
    start.set(io);
    try readers.await(io);

    const reader_error = first_error.load(.acquire);
    if (reader_error != 0) return @errorFromInt(reader_error);
    for (0..admission) |index| {
        try std.testing.expectEqual(read_size, results[index]);
        try std.testing.expectEqualSlices(
            u8,
            object[index * read_size ..][0..read_size],
            &outputs[index],
        );
    }

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

        fn prepare(context: *anyopaque, _: vfs_request.Attempt, range: std.http.Header) anyerror!range_read.PreparedRequest {
            const self: *@This() = @ptrCast(@alignCast(context));
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
        .context = &signed,
        .prepare = Signed.prepare,
    };
    const buffers = [_][]u8{ output[0..1024], output[1024..] };
    const size = range_read.readSize(object.len, 0, &buffers);

    // No retry budget: one attempt, one hook call, the failure is counted.
    var no_retry: vfs_request.Governor = .init(io, &client, "test", .throttle, instantRetries(0));
    try std.testing.expectError(error.RetriesExhausted, range_read.performRangeRead(&no_retry, spec, &buffers, 0, size));
    try std.testing.expectEqual(@as(usize, 1), signed.prepared);
    try std.testing.expectEqual(@as(u64, 1), no_retry.stats.server_failures.load(.acquire));

    signed.prepared = 0;
    var one_retry: vfs_request.Governor = .init(io, &client, "test", .throttle, instantRetries(1));
    try std.testing.expectEqual(size, try range_read.performRangeRead(&one_retry, spec, &buffers, 0, size));
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

/// Options whose retries and holds are short enough for a test.
fn fastOptions(initial_ms: i64, budget_ms: i64) HTTP.Options {
    return .{ .protocol = .http, .retry = .{
        .max_retries = 5,
        .initial_delay = .fromMilliseconds(initial_ms),
        .max_delay = .fromMilliseconds(initial_ms),
        .max_hold = .fromSeconds(30),
        .throttle_budget = .fromMilliseconds(budget_ms),
    } };
}

/// Runs one concurrent positional read per output slice, each at its own
/// offset, and reports what every reader got.
const ConcurrentReads = struct {
    results: []usize,
    errors: []u16,

    fn run(io: std.Io, file_io: std.Io, file: std.Io.File, outputs: []const []u8, self: ConcurrentReads) !void {
        var start: std.Io.Event = .unset;
        var readers: std.Io.Group = .init;
        const Reader = struct {
            fn run(
                file_: std.Io.File,
                io_: std.Io,
                output: []u8,
                offset: u64,
                result: *usize,
                failure: *u16,
                start_: *std.Io.Event,
            ) void {
                start_.wait(io_) catch |err| {
                    failure.* = @intFromError(err);
                    return;
                };
                const buffers = [_][]u8{output};
                result.* = file_.readPositional(io_, &buffers, offset) catch |err| {
                    failure.* = @intFromError(err);
                    return;
                };
            }
        };
        for (outputs, self.results, self.errors, 0..) |output, *result, *failure, index| {
            try readers.concurrent(io, Reader.run, .{
                file,
                file_io,
                output,
                @as(u64, @intCast(index * output.len)),
                result,
                failure,
                &start,
            });
        }
        start.set(io);
        try readers.await(io);
    }
};

test "one throttled GET holds every reader of the backend" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const readers = 8;
    const read_size = 512;

    var object: [readers * read_size]u8 = undefined;
    for (&object, 0..) |*byte, index| byte.* = @truncate(index);

    // Every reader's first GET is throttled, all eight answered together (the
    // barrier), with a one second reset.
    var server = try MockServer.init(io, &object, .{
        .barrier_gets = readers,
        .throttle = .{ .first_gets = readers, .retry_after_s = 1 },
    });
    defer server.deinit(io);
    try server.start(io);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http = try HTTP.init(allocator, io, &client, fastOptions(50, 300_000));
    defer http.deinit();
    const http_io = http.io();
    const stats = &http.governor.stats;

    var path_buffer: [128]u8 = undefined;
    const path = try server.path(&path_buffer);
    const file = try std.Io.Dir.openFile(.cwd(), http_io, path, .{ .mode = .read_only });
    defer file.close(http_io);

    var outputs: [readers][read_size]u8 = undefined;
    var slices: [readers][]u8 = undefined;
    for (&outputs, &slices) |*output, *slice| slice.* = output;
    var results: [readers]usize = @splat(0);
    var errors: [readers]u16 = @splat(0);
    try ConcurrentReads.run(io, http_io, file, &slices, .{ .results = &results, .errors = &errors });

    for (results, errors, 0..) |result, failure, index| {
        if (failure != 0) return @errorFromInt(failure);
        try std.testing.expectEqual(read_size, result);
        try std.testing.expectEqualSlices(u8, object[index * read_size ..][0..read_size], &outputs[index]);
    }

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
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const readers = 12;
    const read_size = 256;

    var object: [readers * read_size]u8 = undefined;
    for (&object, 0..) |*byte, index| byte.* = @truncate(index *% 3);

    // Four GETs per 100 ms; twelve readers need three windows at best.
    var server = try MockServer.init(io, &object, .{
        .throttle = .{ .window = .{ .gets = 4, .per_ms = 100 } },
    });
    defer server.deinit(io);
    try server.start(io);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http = try HTTP.init(allocator, io, &client, fastOptions(50, 300_000));
    defer http.deinit();
    const http_io = http.io();
    const stats = &http.governor.stats;

    var path_buffer: [128]u8 = undefined;
    const path = try server.path(&path_buffer);
    const file = try std.Io.Dir.openFile(.cwd(), http_io, path, .{ .mode = .read_only });
    defer file.close(http_io);

    var outputs: [readers][read_size]u8 = undefined;
    var slices: [readers][]u8 = undefined;
    for (&outputs, &slices) |*output, *slice| slice.* = output;
    var results: [readers]usize = @splat(0);
    var errors: [readers]u16 = @splat(0);
    try ConcurrentReads.run(io, http_io, file, &slices, .{ .results = &results, .errors = &errors });

    for (results, errors, 0..) |result, failure, index| {
        if (failure != 0) return @errorFromInt(failure);
        try std.testing.expectEqual(read_size, result);
        try std.testing.expectEqualSlices(u8, object[index * read_size ..][0..read_size], &outputs[index]);
    }

    try server.stop(io);

    try std.testing.expect(stats.holds.load(.acquire) > 0);
    try std.testing.expect(stats.throttles.load(.acquire) > 0);
    // A limited server never exhausts a reader's retries: the hold carries
    // the whole episode.
    try std.testing.expectEqual(@as(u64, 0), stats.retries.load(.acquire));
    try std.testing.expect(stats.hold_wait_ns.load(.acquire) > 0);
}

test "a zero reset still holds for the initial delay instead of hot looping" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const readers = 4;
    const read_size = 256;
    const initial_ms = 100;

    var object: [readers * read_size]u8 = undefined;
    @memset(&object, 0x5a);

    // `RateLimit; t=0`, which parses to a zero delay.
    var server = try MockServer.init(io, &object, .{
        .throttle = .{ .first_gets = readers, .rate_limit_reset_s = 0 },
    });
    defer server.deinit(io);
    try server.start(io);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http = try HTTP.init(allocator, io, &client, fastOptions(initial_ms, 300_000));
    defer http.deinit();
    const http_io = http.io();

    var path_buffer: [128]u8 = undefined;
    const path = try server.path(&path_buffer);
    const file = try std.Io.Dir.openFile(.cwd(), http_io, path, .{ .mode = .read_only });
    defer file.close(http_io);

    var outputs: [readers][read_size]u8 = undefined;
    var slices: [readers][]u8 = undefined;
    for (&outputs, &slices) |*output, *slice| slice.* = output;
    var results: [readers]usize = @splat(0);
    var errors: [readers]u16 = @splat(0);
    try ConcurrentReads.run(io, http_io, file, &slices, .{ .results = &results, .errors = &errors });
    for (results, errors) |result, failure| {
        if (failure != 0) return @errorFromInt(failure);
        try std.testing.expectEqual(read_size, result);
    }

    try server.stop(io);

    const throttled_at = server.first_throttle_ns.load(.acquire);
    const resumed_at = server.first_get_after_throttle_ns.load(.acquire);
    try std.testing.expect(resumed_at - throttled_at >= initial_ms * std.time.ns_per_ms);
    // Two rounds of GETs, not a spin.
    try std.testing.expectEqual(@as(usize, 2 * readers), server.get_requests.load(.acquire));
}

test "a throttled HEAD at open waits the hold out and succeeds" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var object: [64]u8 = undefined;
    @memset(&object, 0x11);

    var server = try MockServer.init(io, &object, .{
        .throttle = .{ .first_heads = 1, .retry_after_s = 0 },
    });
    defer server.deinit(io);
    try server.start(io);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http = try HTTP.init(allocator, io, &client, fastOptions(20, 300_000));
    defer http.deinit();
    const http_io = http.io();
    const stats = &http.governor.stats;

    var path_buffer: [128]u8 = undefined;
    const path = try server.path(&path_buffer);
    const file = try std.Io.Dir.openFile(.cwd(), http_io, path, .{ .mode = .read_only });
    defer file.close(http_io);
    try std.testing.expectEqual(@as(u64, object.len), try file.length(http_io));

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
            .{ .path = "/bucket/object?alt=media", .bytes = &object },
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
    var gcs = try GCS.init(allocator, io, &client, .{ .endpoint_url = endpoint, .retry = instantRetries(5) });
    defer gcs.deinit();
    var http = try HTTP.init(allocator, io, &client, .{ .protocol = .http, .retry = instantRetries(5) });
    defer http.deinit();

    var path_buffer: [128]u8 = undefined;
    const backends = [_]struct { std.Io, *const vfs_request.Governor, []const u8, bool }{
        .{ s3.io(), &s3.governor, "bucket/object", true },
        .{ gcs.io(), &gcs.governor, "bucket/object", true },
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
    var prepared: range_read.PreparedRequest = .{ .uri = try .parse(url) };
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
        .context = &prepared,
        .prepare = range_read.prepareStatic,
    }, &buffers, 0, read_size));
    const elapsed = started.durationTo(.now(io, .awake)).nanoseconds;
    try std.testing.expect(elapsed >= 300 * std.time.ns_per_ms);
    try std.testing.expect(elapsed < 5 * std.time.ns_per_s);
    try std.testing.expect(governor.stats.holds.load(.acquire) > 1);

    try server.stop(io);
}

test "a reader cancelled while the backend is held returns Canceled" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const read_size = 256;

    var object: [read_size]u8 = undefined;
    @memset(&object, 0x21);
    var output: [read_size]u8 = undefined;

    var server = try MockServer.init(io, &object, .{
        .throttle = .{ .first_gets = 1, .retry_after_s = 30 },
    });
    defer server.deinit(io);
    try server.start(io);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http = try HTTP.init(allocator, io, &client, fastOptions(20, 300_000));
    defer http.deinit();
    const http_io = http.io();

    var path_buffer: [128]u8 = undefined;
    const path = try server.path(&path_buffer);
    const file = try std.Io.Dir.openFile(.cwd(), http_io, path, .{ .mode = .read_only });
    defer file.close(http_io);

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
    }.run, .{ file, http_io, output[0..], &failure });

    // Let the first GET be throttled and the reader settle into the hold.
    try io.sleep(.fromMilliseconds(50), .awake);
    readers.cancel(io);
    try std.testing.expectEqual(@intFromError(error.Canceled), failure.load(.acquire));

    try server.stop(io);
}

test "generic HTTP cancels a GET that receives nothing and retries it as a timeout" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var object: [4096]u8 = undefined;
    for (&object, 0..) |*byte, i| byte.* = @truncate(i);
    var output: [1000]u8 = undefined;

    // The first GET waits unanswered for the second: a peer that receives
    // the request and sends nothing.
    var server = try MockServer.init(io, &object, .{ .barrier_gets = 2 });
    defer server.deinit(io);
    try server.start(io);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http = try HTTP.init(allocator, io, &client, .{ .protocol = .http, .retry = .{
        .initial_delay = .fromNanoseconds(0),
        .max_delay = .fromNanoseconds(0),
        .stall_timeout = .fromMilliseconds(100),
    } });
    defer http.deinit();
    const http_io = http.io();
    const stats = &http.governor.stats;

    var path_buffer: [128]u8 = undefined;
    const path = try server.path(&path_buffer);
    const file = try std.Io.Dir.openFile(.cwd(), http_io, path, .{ .mode = .read_only });
    defer file.close(http_io);

    stats.* = .{};
    const buffers = [_][]u8{&output};
    try std.testing.expectEqual(output.len, try file.readPositional(http_io, &buffers, 77));
    try std.testing.expectEqualSlices(u8, object[77..][0..output.len], &output);
    try std.testing.expectEqual(@as(u64, 1), stats.timeouts.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), stats.retries.load(.acquire));

    // The first GET answers late, to a connection the client closed: that
    // write may fail.
    server.stop(io) catch |err| if (err != error.WriteFailed) return err;
    try std.testing.expectEqual(@as(usize, 2), server.get_requests.load(.acquire));
}

test "http, s3 and gs fail open, stat and access of an absent or denied object alike" {
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
    // `/<bucket>/<key>`, GCS the same with `?alt=media`.
    const targets = [_]struct { []const u8, []const u8 }{ .{ "/", "" }, .{ "/bucket/", "" }, .{ "/bucket/", "?alt=media" } };
    const refusals = comptime refusals: {
        var list: [cases.len * targets.len]MockServer.Refusal = undefined;
        for (cases, 0..) |case, i| {
            for (targets, 0..) |target, j| {
                list[i * targets.len + j] = .{ .path = target[0] ++ case.name ++ target[1], .status = case.status };
            }
        }
        break :refusals list;
    };

    var object: [32]u8 = undefined;
    @memset(&object, 0x6b);
    var server = try MockServer.init(io, &object, .{
        .objects = &.{
            .{ .path = "/bucket/object", .bytes = &object },
            .{ .path = "/bucket/object?alt=media", .bytes = &object },
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
    var gcs = try GCS.init(allocator, io, &client, .{ .endpoint_url = endpoint });
    defer gcs.deinit();

    var authority_buffer: [64]u8 = undefined;
    const backends = [_]struct { std.Io, []const u8 }{
        .{ http.io(), try std.fmt.bufPrint(&authority_buffer, "{s}/", .{endpoint["http://".len..]}) },
        .{ s3.io(), "bucket/" },
        .{ gcs.io(), "bucket/" },
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

test "http, s3 and gs fail a refused streaming read with InputOutput and seek back no further than 0" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var object: [64]u8 = undefined;
    for (&object, 0..) |*byte, index| byte.* = @truncate(index *% 5 +% 1);
    // Each backend's first GET is answered 500, and none has a retry left.
    var server = try MockServer.init(io, &object, .{
        .fail_first_gets = 3,
        .objects = &.{
            .{ .path = "/bucket/object", .bytes = &object },
            .{ .path = "/bucket/object?alt=media", .bytes = &object },
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
    var gcs = try GCS.init(allocator, io, &client, .{ .endpoint_url = endpoint, .retry = instantRetries(0) });
    defer gcs.deinit();

    var path_buffer: [128]u8 = undefined;
    const backend_ios = [_]std.Io{ http.io(), s3.io(), gcs.io() };
    const paths = [_][]const u8{ try server.path(&path_buffer), "bucket/object", "bucket/object" };
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
    try std.testing.expectEqual(@as(usize, 9), server.get_requests.load(.acquire));
}
