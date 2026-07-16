const std = @import("std");

pub const Dir = struct {
    pub const path = struct {
        pub fn bufJoin(buf: []u8, paths: []const []const u8) ![]u8 {
            var fa: std.heap.FixedBufferAllocator = .init(buf);
            return try std.Io.Dir.path.join(fa.allocator(), paths);
        }

        pub fn bufJoinZ(buf: []u8, paths: []const []const u8) ![:0]u8 {
            var fa: std.heap.FixedBufferAllocator = .init(buf);
            return try std.Io.Dir.path.joinZ(fa.allocator(), paths);
        }
    };

    pub fn readFileAlloc(dir: std.Io.Dir, io: std.Io, sub_path: []const u8, gpa: std.mem.Allocator, limit: std.Io.Limit) ![]u8 {
        const stat = try std.Io.Dir.statFile(dir, io, sub_path, .{});
        const buffer = try gpa.alloc(u8, limit.minInt64(stat.size));
        errdefer gpa.free(buffer);
        _ = try std.Io.Dir.readFile(.cwd(), io, sub_path, buffer);
        return buffer;
    }
};

pub const LimitedGroup = struct {
    limit: std.atomic.Value(usize),
    in_flight: std.atomic.Value(usize) = .init(0),
    group: std.Io.Group = .init,
    cond: std.Io.Condition = .init,
    mutex: std.Io.Mutex = .init,

    fn Wrapper(comptime function: anytype) type {
        return struct {
            fn wrapper(self: *LimitedGroup, io: std.Io, args: std.meta.ArgsTuple(@TypeOf(function))) std.Io.Cancelable!void {
                while (true) {
                    var in_flight = self.in_flight.load(.acquire);
                    while (in_flight < self.limit.load(.acquire)) {
                        if (self.in_flight.cmpxchgWeak(in_flight, in_flight + 1, .release, .acquire)) |actual| {
                            in_flight = actual;
                            continue;
                        }

                        defer {
                            self.mutex.lockUncancelable(io);
                            defer self.mutex.unlock(io);
                            _ = self.in_flight.fetchSub(1, .release);
                            self.cond.signal(io);
                        }
                        return @call(.auto, function, args);
                    }

                    try self.mutex.lock(io);
                    defer self.mutex.unlock(io);
                    while (self.in_flight.load(.acquire) >= self.limit.load(.acquire)) {
                        try self.cond.wait(io, &self.mutex);
                    }
                }
            }
        };
    }

    pub fn init(limit: usize) LimitedGroup {
        std.debug.assert(limit > 0);
        return .{ .limit = .init(limit) };
    }

    pub fn currentLimit(self: *const LimitedGroup) usize {
        return self.limit.load(.acquire);
    }

    pub fn inFlight(self: *const LimitedGroup) usize {
        return self.in_flight.load(.acquire);
    }

    pub fn setLimit(self: *LimitedGroup, io: std.Io, new_limit: usize) void {
        std.debug.assert(new_limit > 0);

        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);

        const old_limit = self.limit.swap(new_limit, .acq_rel);
        if (new_limit > old_limit) {
            for (old_limit..new_limit) |_| {
                self.cond.signal(io);
            }
        }
    }

    pub fn async(self: *LimitedGroup, io: std.Io, comptime function: anytype, args: std.meta.ArgsTuple(@TypeOf(function))) void {
        self.group.async(io, Wrapper(function).wrapper, .{ self, io, args });
    }

    pub fn concurrent(self: *LimitedGroup, io: std.Io, comptime function: anytype, args: std.meta.ArgsTuple(@TypeOf(function))) std.Io.ConcurrentError!void {
        try self.group.concurrent(io, Wrapper(function).wrapper, .{ self, io, args });
    }

    pub fn await(self: *LimitedGroup, io: std.Io) std.Io.Cancelable!void {
        try self.group.await(io);
    }

    pub fn cancel(self: *LimitedGroup, io: std.Io) void {
        self.group.cancel(io);
    }
};

test "LimitedGroup increases its runtime limit" {
    const io = std.testing.io;
    var group: LimitedGroup = .init(1);
    var release: std.Io.Event = .unset;
    var started: std.atomic.Value(usize) = .init(0);

    const Worker = struct {
        fn run(started_: *std.atomic.Value(usize), release_: *std.Io.Event, io_: std.Io) std.Io.Cancelable!void {
            _ = started_.fetchAdd(1, .release);
            try release_.wait(io_);
        }
    };

    try group.concurrent(io, Worker.run, .{ &started, &release, io });
    try group.concurrent(io, Worker.run, .{ &started, &release, io });

    while (started.load(.acquire) < 1) try io.sleep(.fromMilliseconds(1), .awake);
    try std.testing.expectEqual(1, started.load(.acquire));

    group.setLimit(io, 2);
    while (started.load(.acquire) < 2) try io.sleep(.fromMilliseconds(1), .awake);
    try std.testing.expectEqual(2, group.currentLimit());

    release.set(io);
    try group.await(io);
}

test "LimitedGroup decreases without cancelling in-flight work" {
    const io = std.testing.io;
    var group: LimitedGroup = .init(2);
    var releases: [3]std.Io.Event = @splat(.unset);
    var started: std.atomic.Value(usize) = .init(0);

    const Worker = struct {
        fn run(id: usize, started_: *std.atomic.Value(usize), releases_: *[3]std.Io.Event, io_: std.Io) std.Io.Cancelable!void {
            _ = started_.fetchAdd(1, .release);
            try releases_[id].wait(io_);
        }
    };

    try group.concurrent(io, Worker.run, .{ 0, &started, &releases, io });
    try group.concurrent(io, Worker.run, .{ 1, &started, &releases, io });
    while (started.load(.acquire) < 2) try io.sleep(.fromMilliseconds(1), .awake);

    group.setLimit(io, 1);
    try group.concurrent(io, Worker.run, .{ 2, &started, &releases, io });
    releases[0].set(io);
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expectEqual(2, started.load(.acquire));

    releases[1].set(io);
    while (started.load(.acquire) < 3) try io.sleep(.fromMilliseconds(1), .awake);
    releases[2].set(io);
    try group.await(io);
}

pub const AllocatingLimitedConcurrentGroup = struct {
    allocator: std.mem.Allocator,

    queue_buffer: []bool = &[_]bool{},
    limit: usize,

    group: std.Io.Group = .init,
    queue: std.Io.Queue(bool) = undefined,

    pub fn init(allocator: std.mem.Allocator, limit: usize) std.mem.Allocator.Error!AllocatingLimitedConcurrentGroup {
        const buffer = try allocator.alloc(bool, limit);
        @memset(buffer, false);
        var self: AllocatingLimitedConcurrentGroup = .{
            .allocator = allocator,
            .queue_buffer = buffer,
            .limit = limit,
            .group = .init,
            .queue = undefined,
        };
        self.queue = std.Io.Queue(bool).init(self.queue_buffer);
        return self;
    }

    pub fn deinit(self: *AllocatingLimitedConcurrentGroup) void {
        self.allocator.free(self.queue_buffer);
    }

    pub fn concurrent(
        self: *AllocatingLimitedConcurrentGroup,
        io: std.Io,
        comptime function: anytype,
        args: std.meta.ArgsTuple(@TypeOf(function)),
    ) !void {
        self.queue.putOneUncancelable(io, true) catch unreachable;

        try self.group.concurrent(io, struct {
            fn wrapper(
                io_: std.Io,
                queue: *std.Io.Queue(bool),
                inner_args: std.meta.ArgsTuple(@TypeOf(function)),
            ) !void {
                defer _ = queue.getOneUncancelable(io_) catch unreachable;
                return @call(.auto, function, inner_args);
            }
        }.wrapper, .{ io, &self.queue, args });
    }

    pub fn await(self: *AllocatingLimitedConcurrentGroup, io: std.Io) std.Io.Cancelable!void {
        try self.group.await(io);
    }

    pub fn cancel(self: *AllocatingLimitedConcurrentGroup, io: std.Io) void {
        self.group.cancel(io);
    }
};

pub const Duration = struct {
    pub fn div(self: std.Io.Duration, rhs: u64) std.Io.Duration {
        return .fromNanoseconds(@divTrunc(self.nanoseconds, @as(i96, @intCast(rhs))));
    }

    pub fn hz(self: std.Io.Duration) u64 {
        return @intCast(std.time.ns_per_s / self.nanoseconds);
    }

    pub fn hzFloat(self: std.Io.Duration) f64 {
        return (1 * std.time.ns_per_s) / @as(f64, @floatFromInt(self.nanoseconds));
    }
};
