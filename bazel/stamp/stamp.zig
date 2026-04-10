const std = @import("std");

pub fn stamp(comptime name: []const u8) []const u8 {
    return std.mem.span(@extern([*c]const u8, .{ .name = name }));
}
