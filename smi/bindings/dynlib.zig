const std = @import("std");
const builtin = @import("builtin");

pub fn Lib(comptime c: type) type {
    return struct {
        inner: std.DynLib = undefined,
        loaded: bool = false,

        const Self = @This();

        pub fn Fn(comptime name: [:0]const u8) type {
            return *const @TypeOf(@field(c, name));
        }

        pub fn sym(self: *const Self, comptime name: [:0]const u8) ?Fn(name) {
            if (!self.loaded) return null;
            return @constCast(&self.inner).lookup(Fn(name), name);
        }

        pub fn open(self: *Self, path: [:0]const u8) bool {
            self.inner = openDynLib(path) orelse return false;
            self.loaded = true;
            return true;
        }

        fn openDynLib(path: [:0]const u8) ?std.DynLib {
            return switch (builtin.os.tag) {
                .linux, .macos => .{
                    .inner = .{
                        .handle = std.c.dlopen(path, switch (builtin.os.tag) {
                            .linux => .{ .LAZY = true, .GLOBAL = true, .NODELETE = true },
                            .macos => .{ .LAZY = true, .LOCAL = true },
                            else => unreachable,
                        }) orelse return null,
                    },
                },
                else => std.DynLib.open(path) catch return null,
            };
        }
    };
}
