const std = @import("std");
const stdx = @import("stdx");
const executor = @import("executor.zig");

pub fn Channel(comptime T: type, comptime capacity: usize) type {
    const Storage = stdx.queue.ArrayQueue(T, capacity);

    return struct {
        const Self = @This();

        q: Storage = .{},
        closed: bool = false,
        len: usize = capacity,
        space_notif: executor.Condition,
        value_notif: executor.Condition,

        pub fn init(exec: *executor.Executor) Self {
            return initWithLen(exec, capacity);
        }

        pub fn initWithLen(exec: *executor.Executor, len: usize) Self {
            return .{
                .len = len,
                .space_notif = executor.Condition.init(exec),
                .value_notif = executor.Condition.init(exec),
            };
        }

        pub fn close(self: *Self) void {
            self.closed = true;
            self.value_notif.signal();
        }

        pub fn try_send(self: *Self, val: T) bool {
            stdx.debug.assert(self.closed == false, "cannot send on closed Channel", .{});
            if (self.q.len() >= self.len) {
                return false;
            }
            self.q.push(val) catch stdx.debug.panic("tried to send on full Channel. This shouldn't happen.", .{});
            self.value_notif.signal();
            return true;
        }

        pub fn send(self: *Self, val: T) void {
            stdx.debug.assert(self.closed == false, "cannot send on closed Channel", .{});
            while (self.q.len() >= self.len) {
                self.space_notif.wait();
            }
            self.q.push(val) catch stdx.debug.panic("tried to send on full Channel. This shouldn't happen.", .{});
            self.value_notif.signal();
        }

        pub fn recv(self: *Self) ?T {
            while (self.closed == false or self.q.len() > 0) {
                if (self.q.pop()) |val| {
                    self.space_notif.signal();
                    return val;
                }
                self.value_notif.wait();
            }
            return null;
        }

        pub fn try_recv(self: *Self) ?T {
            if (self.closed == true) {
                return null;
            }
            if (self.q.pop()) |val| {
                self.space_notif.signal();
                return val;
            }
            return null;
        }
    };
}
