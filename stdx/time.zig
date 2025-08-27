const std = @import("std");

pub const Duration = struct {
    ns: u64 = 0,

    pub fn div(self: Duration, rhs: u64) Duration {
        return .{ .ns = self.ns / rhs };
    }

    pub fn hz(self: Duration) u64 {
        return (1 * std.time.ns_per_s) / self.ns;
    }

    pub fn formatDuration(duration: Duration, writer: *std.io.Writer) std.io.Writer.Error!void {
        try writer.printDuration(duration.ns, .{});
    }

    pub const format = formatDuration;
};

pub const Timer = struct {
    inner: std.time.Timer,

    pub fn start() !Timer {
        return .{ .inner = try std.time.Timer.start() };
    }

    pub fn lap(self: *Timer) Duration {
        return .{ .ns = self.inner.lap() };
    }

    pub fn read(self: *Timer) Duration {
        return .{ .ns = self.inner.read() };
    }

    pub fn reset(self: *Timer) void {
        self.inner.reset();
    }
};
