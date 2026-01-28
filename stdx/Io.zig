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
    limit: usize,
    in_flight: std.atomic.Value(usize) = .init(0),
    group: std.Io.Group = .init,
    cond: std.Io.Condition = .init,
    mutex: std.Io.Mutex = .init,

    fn Wrapper(comptime function: anytype) type {
        return struct {
            fn wrapper(self: *LimitedGroup, io: std.Io, args: std.meta.ArgsTuple(@TypeOf(function))) std.Io.Cancelable!void {
                while (true) {
                    var in_flight = self.in_flight.load(.acquire);
                    while (in_flight < self.limit) {
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
                    while (self.in_flight.load(.acquire) >= self.limit) {
                        try self.cond.wait(io, &self.mutex);
                    }
                }
            }
        };
    }

    pub fn init(limit: usize) LimitedGroup {
        return .{ .limit = limit };
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
