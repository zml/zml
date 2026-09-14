//! A minimal HTTP origin for the VFS and loader tests: one object per path,
//! range reads, and injectable failures and rate limiting. Exported as
//! `VFS.MockServer` so the loader's own tests can serve a fixture over the
//! `http` backend; nothing outside a test build references it.

const std = @import("std");

pub const MockServer = struct {
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

    pub fn init(
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
    pub fn resetPeak(self: *MockServer) void {
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

    pub fn port(self: *const MockServer) u16 {
        return self.tcp.socket.address.getPort();
    }

    fn recordError(self: *MockServer, err: anyerror) void {
        _ = self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
    }

    pub fn check(self: *const MockServer) !void {
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

pub fn startMockServer(
    server: *MockServer,
    group: *std.Io.Group,
    io: std.Io,
) !void {
    try group.concurrent(io, MockServer.run, .{ server, io });
}

pub fn cleanupMockServer(
    server: *MockServer,
    group: *std.Io.Group,
    io: std.Io,
    joined: bool,
) void {
    if (!joined) group.cancel(io);
    server.tcp.deinit(io);
}

pub fn serverPath(buffer: []u8, server: *const MockServer) ![]const u8 {
    return std.fmt.bufPrint(buffer, "127.0.0.1:{d}/object", .{server.port()});
}
