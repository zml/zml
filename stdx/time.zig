const std = @import("std");

pub const Duration = struct {
    ns: u64,

    pub fn format(
        self: Duration,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) @TypeOf(writer).Error!void {
        return try std.fmt.fmtDuration(self.ns).format(fmt, options, writer);
    }
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
