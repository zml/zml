const std = @import("std");

const AtomicReadStats = @import("base.zig").AtomicReadStats;
const HTTP = @import("http.zig").HTTP;
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

const MockServer = struct {
    /// How the server rate limits. A throttled answer carries the status
    /// and whichever of the two delay headers the test asked for.
    const Throttle = struct {
        /// The first N GETs are throttled.
        first_gets: usize = 0,
        /// The first N HEADs are throttled.
        first_heads: usize = 0,
        /// Every GET is throttled, forever.
        always: bool = false,
        status: std.http.Status = .too_many_requests,
        retry_after_s: ?u32 = null,
        rate_limit_reset_s: ?u32 = null,
        /// At most `gets` GETs per `per_ms` milliseconds; the rest are
        /// throttled.
        window: ?struct { gets: usize, per_ms: i64 } = null,

        fn any(self: Throttle) bool {
            return self.first_gets != 0 or self.first_heads != 0 or self.always or self.window != null;
        }
    };

    /// One extra path the server serves beside `/object`.
    const Object = struct { path: []const u8, bytes: []const u8 };

    const Options = struct {
        fail_first_gets: usize = 0,
        barrier_gets: usize = 0,
        throttle: Throttle = .{},
        objects: []const Object = &.{},
        /// Paths answered with 404 on purpose.
        missing: []const []const u8 = &.{},
    };

    tcp: std.Io.net.Server,
    object: []const u8,
    fail_first_gets: usize,
    barrier_gets: usize,
    throttle: Throttle,
    objects: []const Object,
    missing: []const []const u8,
    get_barrier: std.Io.Event = .unset,
    head_requests: std.atomic.Value(usize) = .init(0),
    get_requests: std.atomic.Value(usize) = .init(0),
    throttled_requests: std.atomic.Value(usize) = .init(0),
    active_gets: std.atomic.Value(usize) = .init(0),
    peak_gets: std.atomic.Value(usize) = .init(0),
    attempt_header_gets: std.atomic.Value(usize) = .init(0),
    peak_attempt: std.atomic.Value(usize) = .init(0),
    first_error: std.atomic.Value(u16) = .init(0),
    /// Awake nanoseconds of the first throttled answer and of the first
    /// answer served after one; zero until they happen.
    first_throttle_ns: std.atomic.Value(i64) = .init(0),
    first_get_after_throttle_ns: std.atomic.Value(i64) = .init(0),
    /// Guards the rate window.
    window_mutex: std.Io.Mutex = .init,
    window_started_ns: i64 = 0,
    window_gets: usize = 0,

    fn init(
        io: std.Io,
        object: []const u8,
        opts: Options,
    ) !MockServer {
        const address: std.Io.net.IpAddress = .{ .ip4 = .loopback(0) };
        return .{
            .tcp = try address.listen(io, .{ .reuse_address = true }),
            .object = object,
            .fail_first_gets = opts.fail_first_gets,
            .barrier_gets = opts.barrier_gets,
            .throttle = opts.throttle,
            .objects = opts.objects,
            .missing = opts.missing,
        };
    }

    /// `peak_gets` is a monotone maximum; a test that measures two phases
    /// clears it between them.
    fn resetPeak(self: *MockServer) void {
        self.peak_gets.store(0, .release);
    }

    fn bodyFor(self: *const MockServer, target: []const u8) ?[]const u8 {
        if (std.mem.eql(u8, target, "/object")) return self.object;
        for (self.objects) |object| {
            if (std.mem.eql(u8, object.path, target)) return object.bytes;
        }
        return null;
    }

    fn isMissing(self: *const MockServer, target: []const u8) bool {
        for (self.missing) |path| {
            if (std.mem.eql(u8, path, target)) return true;
        }
        return false;
    }

    fn nowNs(io: std.Io) i64 {
        return @intCast(std.Io.Timestamp.now(io, .awake).nanoseconds);
    }

    /// Whether this GET is over the rate window. Counts only the requests
    /// it lets through, so a throttled one does not consume the budget.
    fn overWindow(self: *MockServer, io: std.Io) bool {
        const window = self.throttle.window orelse return false;
        self.window_mutex.lockUncancelable(io);
        defer self.window_mutex.unlock(io);
        const now = nowNs(io);
        if (now - self.window_started_ns > window.per_ms * std.time.ns_per_ms) {
            self.window_started_ns = now;
            self.window_gets = 0;
        }
        if (self.window_gets >= window.gets) return true;
        self.window_gets += 1;
        return false;
    }

    fn respondThrottled(self: *MockServer, io: std.Io, request: *std.http.Server.Request) !void {
        _ = self.throttled_requests.fetchAdd(1, .monotonic);
        _ = self.first_throttle_ns.cmpxchgStrong(0, nowNs(io), .release, .monotonic);
        var headers: [2]std.http.Header = undefined;
        var header_count: usize = 0;
        var retry_after_buffer: [16]u8 = undefined;
        var reset_buffer: [48]u8 = undefined;
        if (self.throttle.retry_after_s) |seconds| {
            headers[header_count] = .{
                .name = "Retry-After",
                .value = try std.fmt.bufPrint(&retry_after_buffer, "{d}", .{seconds}),
            };
            header_count += 1;
        }
        if (self.throttle.rate_limit_reset_s) |seconds| {
            headers[header_count] = .{
                .name = "RateLimit",
                .value = try std.fmt.bufPrint(&reset_buffer, "\"default\"; r=0; t={d}", .{seconds}),
            };
            header_count += 1;
        }
        try request.respond("slow down", .{
            .status = self.throttle.status,
            .keep_alive = false,
            .extra_headers = headers[0..header_count],
        });
    }

    fn port(self: *const MockServer) u16 {
        return self.tcp.socket.address.getPort();
    }

    fn recordError(self: *MockServer, err: anyerror) void {
        _ = self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
    }

    fn check(self: *const MockServer) !void {
        const error_code = self.first_error.load(.acquire);
        if (error_code != 0) return @errorFromInt(error_code);
    }

    fn run(self: *MockServer, io: std.Io) std.Io.Cancelable!void {
        var connections: std.Io.Group = .init;
        defer connections.cancel(io);

        while (true) {
            const stream = self.tcp.accept(io) catch |err| switch (err) {
                error.Canceled => return error.Canceled,
                else => {
                    self.recordError(err);
                    return;
                },
            };
            connections.concurrent(io, onConnection, .{ self, stream, io }) catch |err| {
                stream.close(io);
                self.recordError(err);
                return;
            };
        }
    }

    fn onConnection(
        self: *MockServer,
        stream: std.Io.net.Stream,
        io: std.Io,
    ) std.Io.Cancelable!void {
        defer stream.close(io);
        self.handleConnection(stream, io) catch |err| switch (err) {
            error.Canceled => return error.Canceled,
            else => self.recordError(err),
        };
    }

    fn handleConnection(self: *MockServer, stream: std.Io.net.Stream, io: std.Io) !void {
        var read_buffer: [8 * 1024]u8 = undefined;
        var reader = stream.reader(io, &read_buffer);
        var write_buffer: [8 * 1024]u8 = undefined;
        var writer = stream.writer(io, &write_buffer);
        var server: std.http.Server = .init(&reader.interface, &writer.interface);
        var request = try server.receiveHead();

        const body = self.bodyFor(request.head.target) orelse {
            try request.respond("not found", .{
                .status = .not_found,
                .keep_alive = false,
            });
            if (self.isMissing(request.head.target)) return;
            return error.UnexpectedTarget;
        };

        switch (request.head.method) {
            .HEAD => {
                const head_ordinal = self.head_requests.fetchAdd(1, .acq_rel) + 1;
                if (head_ordinal <= self.throttle.first_heads) {
                    try self.respondThrottled(io, &request);
                    return;
                }
                try request.respond(body, .{ .keep_alive = false });
            },
            .GET => {
                const range = try parseRequestRange(&request, body.len);
                const get_ordinal = self.get_requests.fetchAdd(1, .acq_rel) + 1;
                const active = self.active_gets.fetchAdd(1, .acq_rel) + 1;
                _ = self.peak_gets.fetchMax(active, .acq_rel);
                defer _ = self.active_gets.fetchSub(1, .release);
                if (requestHeader(&request, "x-attempt")) |value| {
                    _ = self.attempt_header_gets.fetchAdd(1, .monotonic);
                    _ = self.peak_attempt.fetchMax(try std.fmt.parseInt(usize, value, 10), .monotonic);
                }

                if (self.barrier_gets != 0 and get_ordinal <= self.barrier_gets) {
                    if (get_ordinal == self.barrier_gets) self.get_barrier.set(io);
                    try self.get_barrier.wait(io);
                }

                if (get_ordinal <= self.fail_first_gets) {
                    try request.respond("retry", .{
                        .status = .internal_server_error,
                        .keep_alive = false,
                    });
                    return;
                }

                if (self.throttle.always or get_ordinal <= self.throttle.first_gets or self.overWindow(io)) {
                    try self.respondThrottled(io, &request);
                    return;
                }
                if (self.first_throttle_ns.load(.acquire) != 0) {
                    _ = self.first_get_after_throttle_ns.cmpxchgStrong(0, nowNs(io), .release, .monotonic);
                }

                var content_range_buffer: [96]u8 = undefined;
                const content_range = try std.fmt.bufPrint(
                    &content_range_buffer,
                    "bytes {d}-{d}/{d}",
                    .{ range.start, range.end, body.len },
                );
                try request.respond(body[range.start .. range.end + 1], .{
                    .status = .partial_content,
                    .keep_alive = false,
                    .extra_headers = &.{
                        .{ .name = "Content-Range", .value = content_range },
                    },
                });
            },
            else => {
                try request.respond("method not allowed", .{
                    .status = .method_not_allowed,
                    .keep_alive = false,
                });
                return error.UnexpectedMethod;
            },
        }
    }

    const Range = struct {
        start: usize,
        end: usize,
    };

    fn requestHeader(request: *const std.http.Server.Request, name: []const u8) ?[]const u8 {
        var headers = request.iterateHeaders();
        while (headers.next()) |header| {
            if (std.ascii.eqlIgnoreCase(header.name, name)) return header.value;
        }
        return null;
    }

    fn parseRequestRange(request: *const std.http.Server.Request, object_len: usize) !Range {
        var value: ?[]const u8 = null;
        var headers = request.iterateHeaders();
        while (headers.next()) |header| {
            if (!std.ascii.eqlIgnoreCase(header.name, "Range")) continue;
            if (value != null) return error.DuplicateRange;
            value = header.value;
        }

        const range_value = value orelse return error.MissingRange;
        const prefix = "bytes=";
        if (range_value.len <= prefix.len or
            !std.ascii.eqlIgnoreCase(range_value[0..prefix.len], prefix))
        {
            return error.InvalidRange;
        }
        const dash = std.mem.indexOfScalar(u8, range_value[prefix.len..], '-') orelse
            return error.InvalidRange;
        const absolute_dash = prefix.len + dash;
        if (absolute_dash == prefix.len or absolute_dash + 1 == range_value.len)
            return error.InvalidRange;

        const result: Range = .{
            .start = try std.fmt.parseInt(usize, range_value[prefix.len..absolute_dash], 10),
            .end = try std.fmt.parseInt(usize, range_value[absolute_dash + 1 ..], 10),
        };
        if (result.end < result.start or result.end >= object_len) return error.InvalidRange;
        return result;
    }
};

fn startMockServer(
    server: *MockServer,
    group: *std.Io.Group,
    io: std.Io,
) !void {
    try group.concurrent(io, MockServer.run, .{ server, io });
}

fn cleanupMockServer(
    server: *MockServer,
    group: *std.Io.Group,
    io: std.Io,
    joined: bool,
) void {
    if (!joined) group.cancel(io);
    server.tcp.deinit(io);
}

fn serverPath(buffer: []u8, server: *const MockServer) ![]const u8 {
    return std.fmt.bufPrint(buffer, "127.0.0.1:{d}/object", .{server.port()});
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
