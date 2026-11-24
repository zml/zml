const std = @import("std");

pub fn DoubleBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        values: [2]T,
        front_index: u1 = 0,
        mutex: std.Io.Mutex = .init,

        pub fn back(self: *Self) *T {
            return &self.values[~self.front_index];
        }

        pub fn front(self: *Self) *T {
            return &self.values[self.front_index];
        }

        pub fn swap(self: *Self, io: std.Io) void {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            self.front_index = ~self.front_index;
        }

        pub fn lock(self: *Self, io: std.Io) *const T {
            self.mutex.lockUncancelable(io);
            return &self.values[self.front_index];
        }

        pub fn unlock(self: *Self, io: std.Io) void {
            self.mutex.unlock(io);
        }

        pub fn read(self: *Self, io: std.Io) T {
            defer self.unlock(io);
            return self.lock(io).*;
        }
    };
}

test "swap publishes the back" {
    const io = std.testing.io;
    var buffer: DoubleBuffer(u32) = .{ .values = .{ 1, 2 } };

    buffer.back().* = 3;
    try std.testing.expectEqual(1, buffer.lock(io).*);
    buffer.unlock(io);

    buffer.swap(io);
    try std.testing.expectEqual(3, buffer.lock(io).*);
    buffer.unlock(io);
}
