const std = @import("std");

/// Two stable slots with independent arenas. Readers borrow the front under the lock.
pub fn DoubleBuffer(comptime T: type) type {
    return struct {
        const Self = @This();
        const Slot = struct { value: T, arena: std.heap.ArenaAllocator };

        values: [2]Slot,
        front_index: u1 = 0,
        mutex: std.Io.Mutex = .init,

        pub fn deinit(self: *Self) void {
            for (&self.values) |*slot| slot.arena.deinit();
        }

        /// The side the writer fills. Only the writer calls it.
        pub fn back(self: *Self) *Slot {
            return &self.values[~self.front_index];
        }

        /// The writer holds the lock so it can commit other data alongside the swap.
        pub fn publishLocked(self: *Self) void {
            self.front_index = ~self.front_index;
        }

        /// Locks the front until `unlock`.
        pub fn lock(self: *Self, io: std.Io) *const Slot {
            self.mutex.lockUncancelable(io);
            return &self.values[self.front_index];
        }

        pub fn unlock(self: *Self, io: std.Io) void {
            self.mutex.unlock(io);
        }
    };
}
