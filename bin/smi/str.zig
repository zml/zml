const std = @import("std");

/// Convert a null-terminated fixed buffer (e.g. [256]u8, [32]u8) to a slice.
pub fn slice(buf: anytype) []const u8 {
    return std.mem.span(@as([*:0]const u8, @ptrCast(buf)));
}

/// Compare a null-terminated [256]u8 to a plain slice.
pub fn eql(raw: [256]u8, expected: []const u8) bool {
    return std.mem.eql(u8, slice(&raw), expected);
}
