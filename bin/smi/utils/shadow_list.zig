const std = @import("std");

pub fn ShadowList(comptime T: type) type {
    return struct {
        const Self = @This();

        list: std.ArrayList(T) = .empty,
        mutex: std.Io.Mutex = .init,

        pub fn init() Self {
            return .{};
        }

        /// Creates a shadow that swaps into this list.
        pub fn shadow(self: *Self) Shadow {
            return .init(self);
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.list.deinit(allocator);
        }

        pub const Shadow = struct {
            list: std.ArrayList(T) = .empty,
            primary: *Self,

            pub fn init(primary: *Self) Shadow {
                return .{ .primary = primary };
            }

            pub fn deinit(self_: *Shadow, allocator: std.mem.Allocator) void {
                self_.list.deinit(allocator);
            }

            pub fn clearRetainingCapacity(self_: *Shadow) void {
                self_.list.clearRetainingCapacity();
            }

            pub fn append(self_: *Shadow, allocator: std.mem.Allocator, item: T) !void {
                try self_.list.append(allocator, item);
            }

            /// Swap this shadow's contents into the primary list under the primary's mutex.
            pub fn swap(self_: *Shadow, io: std.Io) void {
                self_.primary.mutex.lockUncancelable(io);
                defer self_.primary.mutex.unlock(io);
                const tmp = self_.primary.list;
                self_.primary.list = self_.list;
                self_.list = tmp;
            }
        };
    };
}
