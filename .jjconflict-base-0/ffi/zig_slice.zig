const std = @import("std");
const c = @import("c");

pub const ZigSlice = struct {
    pub fn from(slice: anytype) c.zig_slice {
        return .{
            .ptr = @ptrCast(@constCast(slice.ptr)),
            .len = slice.len,
        };
    }

    pub fn to(comptime T: type, slice: c.zig_slice) []T {
        return @as([*c]T, @ptrCast(@alignCast(slice.ptr)))[0..slice.len];
    }
};
