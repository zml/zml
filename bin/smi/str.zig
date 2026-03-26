const std = @import("std");

/// Convert a null-terminated fixed buffer (e.g. [256]u8, [32]u8) to a slice.
pub fn slice(buf: anytype) []const u8 {
    return std.mem.span(@as([*:0]const u8, @ptrCast(buf)));
}

/// Convert an optional null-terminated [256]u8 to a slice; returns "N/A" if null.
pub fn optSlice(buf: *const ?[256]u8) []const u8 {
    if (buf.*) |*b| return slice(b);
    return "N/A";
}

/// Compare a null-terminated [256]u8 to a plain slice.
pub fn eql(raw: [256]u8, expected: []const u8) bool {
    return std.mem.eql(u8, slice(&raw), expected);
}

/// Copy a slice into a null-terminated [256]u8 buffer.
pub fn fromSlice(s: []const u8) [256]u8 {
    var buf: [256]u8 = .{0} ** 256;
    const len = @min(s.len, 255);
    @memcpy(buf[0..len], s[0..len]);
    return buf;
}
