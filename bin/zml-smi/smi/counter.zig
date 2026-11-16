const std = @import("std");

/// A running total (energy, busy time, CPU ticks) read at two polls.
pub const Counter = struct {
    last: ?Reading = null,

    pub const Reading = struct { total: u64, at: std.Io.Timestamp };

    /// Reads the monotonic clock, so the time is that of the read.
    /// Null on the first reading, if the total went backwards, or if no time passed.
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

    /// Growth per second: microjoules become microwatts. A result that does not fit is the largest u64.
    pub fn perSecond(self: Growth) u64 {
        const per_second = @as(u128, self.amount) * std.time.ns_per_s / @as(u128, @intCast(self.elapsed.nanoseconds));
        return std.math.cast(u64, per_second) orelse std.math.maxInt(u64);
    }

    /// For a total of busy nanoseconds: its share of the elapsed time, at most 100.
    pub fn percentOfTime(self: Growth) u8 {
        return percent(self.amount, @intCast(self.elapsed.nanoseconds));
    }

    /// Its share of another growth, at most 100: busy cycles over total cycles. 0 when the other did not grow.
    pub fn percentOf(self: Growth, whole: Growth) u8 {
        return percent(self.amount, whole.amount);
    }

    /// Adds a growth to `sum`, which starts null: the busy time of several processes over one poll.
    pub fn add(sum: *?Growth, growth: Growth) void {
        if (sum.*) |*total| total.amount +|= growth.amount else sum.* = growth;
    }
};

fn percent(part: u128, whole: u128) u8 {
    if (whole == 0) return 0;
    return @intCast(@min(part * 100 / whole, 100));
}

/// One counter per key, for totals kept per process.
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

        /// Advances the counter of `key` and marks the key as seen.
        pub fn advance(self: *Self, key: Key, io: std.Io, total: u64) error{OutOfMemory}!?Growth {
            const entry = try self.map.getOrPutValue(self.gpa, key, .{});
            entry.value_ptr.seen = true;
            return entry.value_ptr.counter.advance(io, total);
        }

        /// Forgets the keys not advanced since the last sweep. Called once per poll.
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

pub const testing = if (@import("builtin").is_test) struct {
    pub const step = Counter.step;
} else struct {};
