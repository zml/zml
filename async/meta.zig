const std = @import("std");

pub fn FnSignature(comptime func: anytype, comptime argsT: ?type) type {
    return struct {
        pub const FuncT = if (@TypeOf(func) == type) func else @TypeOf(func);
        pub const ArgsT = blk: {
            if (@typeInfo(FuncT).Fn.params.len == 0) {
                break :blk @TypeOf(.{});
            }
            break :blk argsT orelse std.meta.ArgsTuple(FuncT);
        };
        pub const ReturnT = @TypeOf(@call(.auto, func, @as(ArgsT, undefined)));
        pub const ReturnPayloadT = blk: {
            break :blk switch (@typeInfo(ReturnT)) {
                .ErrorUnion => |u| u.payload,
                else => ReturnT,
            };
        };
        pub const ReturnErrorSet: ?type = blk: {
            break :blk switch (@typeInfo(ReturnT)) {
                .ErrorUnion => |u| u.error_set,
                else => null,
            };
        };
    };
}
