const std = @import("std");

pub fn calcUtf32LeLen(utf8: []const u8) !usize {
    return (try std.unicode.utf8CountCodepoints(utf8)) * @sizeOf(u32);
}

pub fn utf8ToUtf32Le(utf32le: []u32, utf8: []const u8) !usize {
    var view: std.unicode.Utf8View = .initUnchecked(utf8);
    var it = view.iterator();
    var idx: usize = 0;
    while (it.nextCodepoint()) |cp| : (idx += 1) {
        if (idx >= utf32le.len) {
            return error.BufferTooSmall;
        }
        utf32le[idx] = @intCast(cp);
    }
    return idx;
}

pub fn utf8ToUtf32LeZ(utf32le: []u32, utf8: []const u8) !usize {
    const n = try utf8ToUtf32Le(utf32le, utf8);
    utf32le[n] = 0;
    return n + 1;
}
