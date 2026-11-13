const std = @import("std");

/// Two copies of a value: one writer fills the back while readers copy the front.
/// The writer holds the lock only to swap; a reader holds it while it copies.
pub fn DoubleBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        values: [2]T,
        front_index: u1 = 0,
        mutex: std.Io.Mutex = .init,

        /// The side the writer fills. Only the writer calls it.
        pub fn back(self: *Self) *T {
            return &self.values[~self.front_index];
        }

        /// The side readers see. Only the writer reads it without the lock.
        pub fn front(self: *Self) *T {
            return &self.values[self.front_index];
        }

        /// Publishes the back.
        pub fn swap(self: *Self, io: std.Io) void {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            self.front_index = ~self.front_index;
        }

        /// Locks the front until `unlock`.
        pub fn lock(self: *Self, io: std.Io) *const T {
            self.mutex.lockUncancelable(io);
            return &self.values[self.front_index];
        }

        pub fn unlock(self: *Self, io: std.Io) void {
            self.mutex.unlock(io);
        }

        /// A copy of the front.
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
