const std = @import("std");

const AtomicReadStats = @import("base.zig").AtomicReadStats;
const HTTP = @import("http.zig").HTTP;
const mock_server = @import("mock_server.zig");
const MockServer = mock_server.MockServer;
const startMockServer = mock_server.startMockServer;
const cleanupMockServer = mock_server.cleanupMockServer;
const serverPath = mock_server.serverPath;
const range_read = @import("range_read.zig");
const vfs_request = @import("request.zig");

/// A retry configuration that never sleeps, for the tests that count
/// attempts rather than time.
fn instantRetries(max_retries: usize) vfs_request.RetryConfig {
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
    var server_group: std.Io.Group = .init;
    try startMockServer(&server, &server_group, io);
    var server_joined = false;
    defer cleanupMockServer(&server, &server_group, io, server_joined);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http = try HTTP.initWithOptions(allocator, io, &client, .http, .{});
    defer http.deinit();
    const http_io = http.io();
    const stats = http.backend().read_stats.?;

    var path_buffer: [128]u8 = undefined;
    const path = try serverPath(&path_buffer, &server);
    const file = try std.Io.Dir.openFile(.cwd(), http_io, path, .{ .mode = .read_only });
    defer file.close(http_io);

    const before = stats.snapshot();
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

    server_group.cancel(io);
    server_joined = true;
    try server.check();

    const delta = stats.snapshot().sub(before);
    try std.testing.expectEqual(@as(usize, 1), server.head_requests.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), server.get_requests.load(.acquire));
    try std.testing.expectEqual(@as(u64, 1), delta.physical_requests);
    try std.testing.expectEqual(@as(u64, request_size), delta.physical_bytes);
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
    var server_group: std.Io.Group = .init;
    try startMockServer(&server, &server_group, io);
    var server_joined = false;
    defer cleanupMockServer(&server, &server_group, io, server_joined);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http = try HTTP.initWithOptions(allocator, io, &client, .http, .{
        .max_retries = 1,
        .retry_initial_delay = .fromNanoseconds(0),
        .retry_max_delay = .fromNanoseconds(0),
    });
    defer http.deinit();
    const http_io = http.io();
    const stats = http.backend().read_stats.?;

    var path_buffer: [128]u8 = undefined;
    const path = try serverPath(&path_buffer, &server);
    const file = try std.Io.Dir.openFile(.cwd(), http_io, path, .{ .mode = .read_only });
    defer file.close(http_io);

    const before = stats.snapshot();
    const one_buffer = [_][]u8{output};
    try std.testing.expectEqual(
        request_size,
        try file.readPositional(http_io, &one_buffer, 0),
    );
    const retried = stats.snapshot().sub(before);
    try std.testing.expectEqual(@as(u64, 2), retried.physical_requests);
    try std.testing.expectEqual(@as(u64, request_size), retried.physical_bytes);
    try std.testing.expectEqual(@as(u64, 1), retried.retries);
    try std.testing.expectEqual(@as(u64, 1), retried.server_failures);
    try std.testing.expectEqual(@as(usize, 1), server.peak_gets.load(.acquire));

    const before_clean = stats.snapshot();
    try std.testing.expectEqual(
        request_size,
        try file.readPositional(http_io, &one_buffer, 0),
    );
    const clean = stats.snapshot().sub(before_clean);
    try std.testing.expectEqual(@as(u64, 1), clean.physical_requests);
    try std.testing.expectEqual(@as(u64, request_size), clean.physical_bytes);
    try std.testing.expectEqual(@as(u64, 0), clean.retries);
    try std.testing.expectEqualSlices(u8, object, output);

    server_group.cancel(io);
    server_joined = true;
    try server.check();
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
    var server_group: std.Io.Group = .init;
    try startMockServer(&server, &server_group, io);
    var server_joined = false;
    defer cleanupMockServer(&server, &server_group, io, server_joined);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http = try HTTP.initWithOptions(allocator, io, &client, .http, .{});
    defer http.deinit();
    const http_io = http.io();
    const stats = http.backend().read_stats.?;

    var path_buffer: [128]u8 = undefined;
    const path = try serverPath(&path_buffer, &server);
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
    const before = stats.snapshot();
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

    server_group.cancel(io);
    server_joined = true;
    try server.check();

    const peak = server.peak_gets.load(.acquire);
    try std.testing.expectEqual(@as(usize, 1), server.head_requests.load(.acquire));
    try std.testing.expectEqual(@as(usize, admission), peak);
    try std.testing.expect(peak <= admission);
    try std.testing.expectEqual(
        @as(usize, admission),
        server.get_requests.load(.acquire),
    );
    const delta = stats.snapshot().sub(before);
    try std.testing.expectEqual(@as(u64, admission), delta.physical_requests);
}

test "the shared range loop prepares the request once per attempt" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const request_size = 3 * 1024;

    var object: [request_size]u8 = undefined;
    for (&object, 0..) |*byte, index| byte.* = @truncate(index *% 7);
    var output: [request_size]u8 = undefined;

    var server = try MockServer.init(io, &object, .{ .fail_first_gets = 2 });
    var server_group: std.Io.Group = .init;
    try startMockServer(&server, &server_group, io);
    var server_joined = false;
    defer cleanupMockServer(&server, &server_group, io, server_joined);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();

    // An S3-style request: like `x-amz-date` under a SigV4 signature, the
    // header is recomputed by the hook for every attempt.
    const Signed = struct {
        uri: std.Uri,
        prepared: usize = 0,
        attempt_value: [8]u8 = undefined,
        headers: [1]std.http.Header = undefined,

        fn prepare(context: *anyopaque, attempt: vfs_request.Attempt, range: std.http.Header) anyerror!range_read.PreparedRequest {
            const self: *@This() = @ptrCast(@alignCast(context));
            try std.testing.expectEqual(self.prepared, attempt.ordinal);
            try std.testing.expectEqualStrings("Range", range.name);
            self.prepared += 1;
            self.headers = .{.{
                .name = "x-attempt",
                .value = try std.fmt.bufPrint(&self.attempt_value, "{d}", .{attempt.ordinal}),
            }};
            return .{ .uri = self.uri, .extra_headers = &self.headers };
        }
    };
    var url_buffer: [160]u8 = undefined;
    const url = try std.fmt.bufPrint(&url_buffer, "http://127.0.0.1:{d}/object", .{server.port()});
    var signed: Signed = .{ .uri = try .parse(url) };
    const spec: range_read.RangeSpec = .{
        .request = .{
            .backend = "test",
            .target = url,
            .unavailable = .throttle,
            .key = vfs_request.authorityOf(signed.uri),
        },
        .context = &signed,
        .prepare = Signed.prepare,
    };
    var stats: AtomicReadStats = .{};
    const buffers = [_][]u8{ output[0..1024], output[1024..] };
    const size = range_read.readSize(object.len, 0, &buffers);

    // No retry budget: one attempt, one hook call, the failure is counted.
    var no_retry: vfs_request.Governor = .init(instantRetries(0));
    try std.testing.expectError(
        error.RetriesExhausted,
        range_read.performRangeRead(.{
            .io = io,
            .client = &client,
            .governor = &no_retry,
            .stats = &stats,
        }, spec, &buffers, 0, size),
    );
    try std.testing.expectEqual(@as(usize, 1), signed.prepared);

    signed.prepared = 0;
    var one_retry: vfs_request.Governor = .init(instantRetries(1));
    try std.testing.expectEqual(size, try range_read.performRangeRead(.{
        .io = io,
        .client = &client,
        .governor = &one_retry,
        .stats = &stats,
    }, spec, &buffers, 0, size));
    try std.testing.expectEqual(@as(usize, 2), signed.prepared);
    try std.testing.expectEqualSlices(u8, &object, &output);

    server_group.cancel(io);
    server_joined = true;
    try server.check();

    const snapshot = stats.snapshot();
    try std.testing.expectEqual(@as(usize, 3), server.get_requests.load(.acquire));
    try std.testing.expectEqual(@as(usize, 3), server.attempt_header_gets.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), server.peak_attempt.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), server.peak_gets.load(.acquire));
    try std.testing.expectEqual(@as(u64, 3), snapshot.physical_requests);
    try std.testing.expectEqual(@as(u64, 1), snapshot.retries);
    try std.testing.expectEqual(@as(u64, 2), snapshot.server_failures);
    try std.testing.expectEqual(@as(u64, request_size), snapshot.physical_bytes);
}

/// Options whose retries and holds are short enough for a test.
fn fastOptions(initial_ms: i64, budget_ms: i64) HTTP.InitOpts {
    return .{
        .max_retries = 5,
        .retry_initial_delay = .fromMilliseconds(initial_ms),
        .retry_max_delay = .fromMilliseconds(initial_ms),
        .max_hold = .fromSeconds(30),
        .throttle_budget = .fromMilliseconds(budget_ms),
    };
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

    // Every reader's first GET is throttled, with a one second reset.
    var server = try MockServer.init(io, &object, .{
        .throttle = .{ .first_gets = readers, .retry_after_s = 1 },
    });
    var server_group: std.Io.Group = .init;
    try startMockServer(&server, &server_group, io);
    var server_joined = false;
    defer cleanupMockServer(&server, &server_group, io, server_joined);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http = try HTTP.initWithOptions(allocator, io, &client, .http, fastOptions(50, 300_000));
    defer http.deinit();
    const http_io = http.io();
    const stats = http.backend().read_stats.?;

    var path_buffer: [128]u8 = undefined;
    const path = try serverPath(&path_buffer, &server);
    const file = try std.Io.Dir.openFile(.cwd(), http_io, path, .{ .mode = .read_only });
    defer file.close(http_io);

    var outputs: [readers][read_size]u8 = undefined;
    var slices: [readers][]u8 = undefined;
    for (&outputs, &slices) |*output, *slice| slice.* = output;
    var results: [readers]usize = @splat(0);
    var errors: [readers]u16 = @splat(0);
    const before = stats.snapshot();
    try ConcurrentReads.run(io, http_io, file, &slices, .{ .results = &results, .errors = &errors });

    for (results, errors, 0..) |result, failure, index| {
        if (failure != 0) return @errorFromInt(failure);
        try std.testing.expectEqual(read_size, result);
        try std.testing.expectEqualSlices(u8, object[index * read_size ..][0..read_size], &outputs[index]);
    }

    server_group.cancel(io);
    server_joined = true;
    try server.check();

    // One hold covered every reader, the server-named second was honoured,
    // and no reader burned a retry for it.
    const delta = stats.snapshot().sub(before);
    try std.testing.expectEqual(@as(u64, 1), delta.holds);
    try std.testing.expectEqual(@as(u64, readers), delta.throttles);
    try std.testing.expectEqual(@as(u64, 0), delta.retries);
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
    var server_group: std.Io.Group = .init;
    try startMockServer(&server, &server_group, io);
    var server_joined = false;
    defer cleanupMockServer(&server, &server_group, io, server_joined);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http = try HTTP.initWithOptions(allocator, io, &client, .http, fastOptions(50, 300_000));
    defer http.deinit();
    const http_io = http.io();
    const stats = http.backend().read_stats.?;

    var path_buffer: [128]u8 = undefined;
    const path = try serverPath(&path_buffer, &server);
    const file = try std.Io.Dir.openFile(.cwd(), http_io, path, .{ .mode = .read_only });
    defer file.close(http_io);

    var outputs: [readers][read_size]u8 = undefined;
    var slices: [readers][]u8 = undefined;
    for (&outputs, &slices) |*output, *slice| slice.* = output;
    var results: [readers]usize = @splat(0);
    var errors: [readers]u16 = @splat(0);
    const before = stats.snapshot();
    try ConcurrentReads.run(io, http_io, file, &slices, .{ .results = &results, .errors = &errors });

    for (results, errors, 0..) |result, failure, index| {
        if (failure != 0) return @errorFromInt(failure);
        try std.testing.expectEqual(read_size, result);
        try std.testing.expectEqualSlices(u8, object[index * read_size ..][0..read_size], &outputs[index]);
    }

    server_group.cancel(io);
    server_joined = true;
    try server.check();

    const delta = stats.snapshot().sub(before);
    try std.testing.expect(delta.holds > 0);
    try std.testing.expect(delta.throttles > 0);
    // A limited server never exhausts a reader's retries: the hold carries
    // the whole episode.
    try std.testing.expectEqual(@as(u64, 0), delta.retries);
    try std.testing.expect(delta.hold_wait_ns > 0);
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
    var server_group: std.Io.Group = .init;
    try startMockServer(&server, &server_group, io);
    var server_joined = false;
    defer cleanupMockServer(&server, &server_group, io, server_joined);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http = try HTTP.initWithOptions(allocator, io, &client, .http, fastOptions(initial_ms, 300_000));
    defer http.deinit();
    const http_io = http.io();

    var path_buffer: [128]u8 = undefined;
    const path = try serverPath(&path_buffer, &server);
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

    server_group.cancel(io);
    server_joined = true;
    try server.check();

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
    var server_group: std.Io.Group = .init;
    try startMockServer(&server, &server_group, io);
    var server_joined = false;
    defer cleanupMockServer(&server, &server_group, io, server_joined);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http = try HTTP.initWithOptions(allocator, io, &client, .http, fastOptions(20, 300_000));
    defer http.deinit();
    const http_io = http.io();
    const stats = http.backend().read_stats.?;

    var path_buffer: [128]u8 = undefined;
    const path = try serverPath(&path_buffer, &server);
    const file = try std.Io.Dir.openFile(.cwd(), http_io, path, .{ .mode = .read_only });
    defer file.close(http_io);
    try std.testing.expectEqual(@as(u64, object.len), try file.length(http_io));

    server_group.cancel(io);
    server_joined = true;
    try server.check();

    // The open retried its HEAD after the hold, and the hold is the
    // backend's, not the read path's.
    try std.testing.expectEqual(@as(usize, 2), server.head_requests.load(.acquire));
    const snapshot = stats.snapshot();
    try std.testing.expectEqual(@as(u64, 1), snapshot.holds);
    try std.testing.expectEqual(@as(u64, 1), snapshot.throttles);
    try std.testing.expectEqual(@as(u64, 0), snapshot.retries);
}

test "a 503 holds an object store and only retries a plain HTTP server" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const read_size = 512;

    var object: [read_size]u8 = undefined;
    @memset(&object, 0x7e);
    var output: [read_size]u8 = undefined;

    var server = try MockServer.init(io, &object, .{
        .throttle = .{ .first_gets = 2, .status = .service_unavailable },
    });
    var server_group: std.Io.Group = .init;
    try startMockServer(&server, &server_group, io);
    var server_joined = false;
    defer cleanupMockServer(&server, &server_group, io, server_joined);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();

    var url_buffer: [160]u8 = undefined;
    const url = try std.fmt.bufPrint(&url_buffer, "http://127.0.0.1:{d}/object", .{server.port()});
    var prepared: range_read.PreparedRequest = .{ .uri = try .parse(url) };
    const buffers = [_][]u8{&output};

    // On an object store a 503 is `SlowDown`: it holds the backend and
    // charges no retry.
    var throttling: vfs_request.Governor = .init(instantRetries(5));
    var throttling_stats: AtomicReadStats = .{};
    try std.testing.expectEqual(read_size, try range_read.performRangeRead(.{
        .io = io,
        .client = &client,
        .governor = &throttling,
        .stats = &throttling_stats,
    }, .{
        .request = .{ .backend = "test", .target = url, .unavailable = .throttle },
        .context = &prepared,
        .prepare = range_read.prepareStatic,
    }, &buffers, 0, read_size));
    const throttled = throttling_stats.snapshot();
    // One reader, two throttles in a row: each arms its own hold once the
    // previous one has passed, and neither is charged a retry.
    try std.testing.expectEqual(@as(u64, 2), throttled.holds);
    try std.testing.expectEqual(@as(u64, 2), throttled.throttles);
    try std.testing.expectEqual(@as(u64, 0), throttled.retries);

    // On a plain HTTP server the same status is a server failure: no hold,
    // one charged retry per attempt.
    server.first_throttle_ns.store(0, .release);
    server.get_requests.store(0, .release);
    server.throttle.first_gets = 2;
    var failing: vfs_request.Governor = .init(instantRetries(5));
    var failing_stats: AtomicReadStats = .{};
    try std.testing.expectEqual(read_size, try range_read.performRangeRead(.{
        .io = io,
        .client = &client,
        .governor = &failing,
        .stats = &failing_stats,
    }, .{
        .request = .{ .backend = "test", .target = url, .unavailable = .server_failure },
        .context = &prepared,
        .prepare = range_read.prepareStatic,
    }, &buffers, 0, read_size));
    const failed = failing_stats.snapshot();
    try std.testing.expectEqual(@as(u64, 0), failed.holds);
    try std.testing.expectEqual(@as(u64, 2), failed.server_failures);
    try std.testing.expectEqual(@as(u64, 2), failed.retries);

    server_group.cancel(io);
    server_joined = true;
    try server.check();
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
    var server_group: std.Io.Group = .init;
    try startMockServer(&server, &server_group, io);
    var server_joined = false;
    defer cleanupMockServer(&server, &server_group, io, server_joined);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();

    var url_buffer: [160]u8 = undefined;
    const url = try std.fmt.bufPrint(&url_buffer, "http://127.0.0.1:{d}/object", .{server.port()});
    var prepared: range_read.PreparedRequest = .{ .uri = try .parse(url) };
    const buffers = [_][]u8{&output};

    var governor: vfs_request.Governor = .init(.{
        .max_retries = 5,
        .initial_delay = .fromMilliseconds(20),
        .max_delay = .fromMilliseconds(20),
        .max_hold = .fromMilliseconds(50),
        .throttle_budget = .fromMilliseconds(300),
    });
    var stats: AtomicReadStats = .{};
    const started: std.Io.Timestamp = .now(io, .awake);
    // The read never runs: the backend is rate limited past its budget.
    // `fileReadPositional` maps this to `Unexpected` with its own log line.
    try std.testing.expectError(error.RateLimited, range_read.performRangeRead(.{
        .io = io,
        .client = &client,
        .governor = &governor,
        .stats = &stats,
    }, .{
        .request = .{ .backend = "test", .target = url, .unavailable = .throttle },
        .context = &prepared,
        .prepare = range_read.prepareStatic,
    }, &buffers, 0, read_size));
    const elapsed = started.durationTo(.now(io, .awake)).nanoseconds;
    try std.testing.expect(elapsed >= 300 * std.time.ns_per_ms);
    try std.testing.expect(elapsed < 5 * std.time.ns_per_s);
    try std.testing.expect(stats.snapshot().holds > 1);

    server_group.cancel(io);
    server_joined = true;
    try server.check();
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
    var server_group: std.Io.Group = .init;
    try startMockServer(&server, &server_group, io);
    var server_joined = false;
    defer cleanupMockServer(&server, &server_group, io, server_joined);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http = try HTTP.initWithOptions(allocator, io, &client, .http, fastOptions(20, 300_000));
    defer http.deinit();
    const http_io = http.io();

    var path_buffer: [128]u8 = undefined;
    const path = try serverPath(&path_buffer, &server);
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

    server_group.cancel(io);
    server_joined = true;
    try server.check();
}
