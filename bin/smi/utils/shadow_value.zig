const std = @import("std");

pub fn ShadowValue(comptime T: type) type {
    return struct {
        const Self = @This();

        value: T,
        mutex: std.Io.Mutex = .init,

        pub fn get(self: *Self, io: std.Io) T {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            return self.value;
        }

        pub fn set(self: *Self, io: std.Io, new: T) void {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            self.value = new;
        }
    };
}
