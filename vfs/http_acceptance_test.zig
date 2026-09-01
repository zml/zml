const std = @import("std");

const HTTP = @import("http.zig").HTTP;
const base = @import("base.zig");

const MockServer = struct {
    const Options = struct {
        fail_first_gets: usize = 0,
        barrier_gets: usize = 0,
    };

    tcp: std.Io.net.Server,
    object: []const u8,
    fail_first_gets: usize,
    barrier_gets: usize,
    get_barrier: std.Io.Event = .unset,
    head_requests: std.atomic.Value(usize) = .init(0),
    get_requests: std.atomic.Value(usize) = .init(0),
    active_gets: std.atomic.Value(usize) = .init(0),
    peak_gets: std.atomic.Value(usize) = .init(0),
    first_error: std.atomic.Value(u16) = .init(0),

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
        };
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

        if (!std.mem.eql(u8, request.head.target, "/object")) {
            try request.respond("not found", .{
                .status = .not_found,
                .keep_alive = false,
            });
            return error.UnexpectedTarget;
        }

        switch (request.head.method) {
            .HEAD => {
                _ = self.head_requests.fetchAdd(1, .monotonic);
                try request.respond(self.object, .{ .keep_alive = false });
            },
            .GET => {
                const range = try parseRequestRange(&request, self.object.len);
                const get_ordinal = self.get_requests.fetchAdd(1, .acq_rel) + 1;
                const active = self.active_gets.fetchAdd(1, .acq_rel) + 1;
                _ = self.peak_gets.fetchMax(active, .acq_rel);
                defer _ = self.active_gets.fetchSub(1, .release);

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

                var content_range_buffer: [96]u8 = undefined;
                const content_range = try std.fmt.bufPrint(
                    &content_range_buffer,
                    "bytes {d}-{d}/{d}",
                    .{ range.start, range.end, self.object.len },
                );
                try request.respond(self.object[range.start .. range.end + 1], .{
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
    var http = try HTTP.initWithOptions(allocator, io, &client, .http, .{
        .minimum_request_size = request_size,
    });
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

test "generic HTTP retries serially and reports typed timing counters" {
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
        .minimum_request_size = request_size,
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
    const bucket_index = base.timingBucketIndex(request_size).?;
    const retried_bucket = retried.timing[bucket_index];

    try std.testing.expectEqual(@as(u64, 2), retried.physical_requests);
    try std.testing.expectEqual(@as(u64, request_size), retried.physical_bytes);
    try std.testing.expectEqual(@as(u64, 1), retried.retries);
    try std.testing.expectEqual(@as(u64, 1), retried.server_failures);
    try std.testing.expectEqual(@as(u64, 2), retried_bucket.attempts);
    try std.testing.expectEqual(@as(u64, 0), retried_bucket.successes);
    try std.testing.expectEqual(@as(u64, 1), retried_bucket.server_failures);
    try std.testing.expectEqual(@as(usize, 1), server.peak_gets.load(.acquire));

    const before_clean = stats.snapshot();
    try std.testing.expectEqual(
        request_size,
        try file.readPositional(http_io, &one_buffer, 0),
    );
    const clean = stats.snapshot().sub(before_clean);
    const clean_bucket = clean.timing[bucket_index];
    try std.testing.expectEqual(@as(u64, 1), clean.physical_requests);
    try std.testing.expectEqual(@as(u64, request_size), clean.physical_bytes);
    try std.testing.expectEqual(@as(u64, 0), clean.retries);
    try std.testing.expectEqual(@as(u64, 1), clean_bucket.attempts);
    try std.testing.expectEqual(@as(u64, 1), clean_bucket.successes);
    try std.testing.expectEqual(@as(u64, request_size), clean_bucket.successful_bytes);
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
    var http = try HTTP.initWithOptions(allocator, io, &client, .http, .{
        .minimum_request_size = read_size,
    });
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
