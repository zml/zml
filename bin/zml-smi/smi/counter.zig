const std = @import("std");

pub const Counter = struct {
    last: ?Reading = null,

    pub const Reading = struct { total: u64, at: std.Io.Timestamp };

    pub fn advance(self: *Counter, io: std.Io, total: u64) ?Growth {
        return self.step(.{ .total = total, .at = .now(io, .awake) });
    }

    fn step(self: *Counter, reading: Reading) ?Growth {
        const last = self.last;
        self.last = reading;
        const previous = last orelse return null;
        if (reading.total < previous.total or reading.at.nanoseconds <= previous.at.nanoseconds) return null;
        return .{ .amount = reading.total - previous.total, .elapsed = previous.at.durationTo(reading.at) };
    }
};

pub const Growth = struct {
    amount: u64,
    elapsed: std.Io.Duration,

    pub fn perSecond(self: Growth) u64 {
        const per_second = @as(u128, self.amount) * std.time.ns_per_s / @as(u128, @intCast(self.elapsed.nanoseconds));
        return std.math.cast(u64, per_second) orelse std.math.maxInt(u64);
    }

    pub fn percentOfTime(self: Growth) u8 {
        return percent(self.amount, @intCast(self.elapsed.nanoseconds));
    }

    pub fn percentOf(self: Growth, whole: Growth) u8 {
        return percent(self.amount, whole.amount);
    }

    pub fn add(sum: *?Growth, growth: Growth) void {
        if (sum.*) |*total| total.amount +|= growth.amount else sum.* = growth;
    }
};

fn percent(part: u128, whole: u128) u8 {
    if (whole == 0) return 0;
    return @intCast(@min(part * 100 / whole, 100));
}

pub fn Counters(comptime Key: type) type {
    return struct {
        const Self = @This();

        map: std.AutoHashMapUnmanaged(Key, Entry) = .empty,
        gpa: std.mem.Allocator,

        const Entry = struct { counter: Counter = .{}, seen: bool = false };

        pub fn init(gpa: std.mem.Allocator) Self {
            return .{ .gpa = gpa };
        }

        pub fn deinit(self: *Self) void {
            self.map.deinit(self.gpa);
        }

        pub fn advance(self: *Self, key: Key, io: std.Io, total: u64) error{OutOfMemory}!?Growth {
            const entry = try self.map.getOrPutValue(self.gpa, key, .{});
            entry.value_ptr.seen = true;
            return entry.value_ptr.counter.advance(io, total);
        }

        pub fn sweep(self: *Self) void {
            var it = self.map.iterator();
            while (it.next()) |entry| {
                if (entry.value_ptr.seen) {
                    entry.value_ptr.seen = false;
                } else {
                    self.map.removeByPtr(entry.key_ptr);
                }
            }
        }
    };
}

fn at(ms: i64) std.Io.Timestamp {
    return .fromNanoseconds(@as(i96, ms) * std.time.ns_per_ms);
}

test Counter {
    var counter: Counter = .{};
    try std.testing.expectEqual(null, counter.step(.{ .total = 100, .at = at(0) }));

    const growth = counter.step(.{ .total = 600, .at = at(500) }).?;
    try std.testing.expectEqual(500, growth.amount);
    try std.testing.expectEqual(500, growth.elapsed.toMilliseconds());

    try std.testing.expectEqual(null, counter.step(.{ .total = 10, .at = at(1000) }));
    try std.testing.expectEqual(null, counter.step(.{ .total = 20, .at = at(1000) }));
}

test Growth {
    const half_second: std.Io.Duration = .fromMilliseconds(500);
    try std.testing.expectEqual(20_000, (Growth{ .amount = 10_000, .elapsed = half_second }).perSecond());
    try std.testing.expectEqual(std.math.maxInt(u64), (Growth{ .amount = std.math.maxInt(u64), .elapsed = .fromNanoseconds(1) }).perSecond());

    try std.testing.expectEqual(50, (Growth{ .amount = 250_000_000, .elapsed = half_second }).percentOfTime());
    try std.testing.expectEqual(100, (Growth{ .amount = 2_000_000_000, .elapsed = half_second }).percentOfTime());

    const cycles: Growth = .{ .amount = 30, .elapsed = half_second };
    try std.testing.expectEqual(30, cycles.percentOf(.{ .amount = 100, .elapsed = half_second }));
    try std.testing.expectEqual(0, cycles.percentOf(.{ .amount = 0, .elapsed = half_second }));
}

test Counters {
    const io = std.testing.io;
    var counters: Counters(u32) = .init(std.testing.allocator);
    defer counters.deinit();

    try std.testing.expectEqual(null, try counters.advance(1, io, 10));
    try std.testing.expectEqual(null, try counters.advance(2, io, 10));
    counters.sweep();

    _ = try counters.advance(1, io, 20);
    counters.sweep();
    try std.testing.expect(counters.map.contains(1));
    try std.testing.expect(!counters.map.contains(2));
}
