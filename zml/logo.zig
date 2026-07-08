const std = @import("std");

pub const Colors = struct {
    pub const reset = "\x1b[0m";
    pub const cyan = "\x1b[38;2;0;255;255m";
    pub const sky = "\x1b[38;2;0;221;255m";
    pub const blue = "\x1b[38;2;135;143;255m";
    pub const violet = "\x1b[38;2;215;90;219m";
    pub const pink = "\x1b[38;2;242;48;174m";
    pub const shine_cyan = "\x1b[1;38;2;140;255;255m";
    pub const shine_sky = "\x1b[1;38;2;128;244;255m";
    pub const shine_blue = "\x1b[1;38;2;205;210;255m";
    pub const shine_violet = "\x1b[1;38;2;255;170;255m";
    pub const shine_pink = "\x1b[1;38;2;255;150;220m";
};

pub const Row = struct {
    color: []const u8,
    shine: []const u8,
    text: []const u8,
};

pub fn writeRow(writer: *std.Io.Writer, row: Row) !void {
    var view: std.unicode.Utf8View = .initUnchecked(row.text);
    var iter = view.iterator();
    var is_shiny = false;

    try writer.writeAll(row.color);
    while (iter.nextCodepointSlice()) |codepoint| {
        const should_shine = isShineGlyph(codepoint);
        if (should_shine != is_shiny) {
            try writer.writeAll(if (should_shine) row.shine else row.color);
            is_shiny = should_shine;
        }
        try writer.writeAll(codepoint);
    }
    try writer.writeAll(Colors.reset);
}

pub fn isShineGlyph(codepoint: []const u8) bool {
    return std.mem.eql(u8, codepoint, "╔") or
        std.mem.eql(u8, codepoint, "╗") or
        std.mem.eql(u8, codepoint, "╚") or
        std.mem.eql(u8, codepoint, "╝") or
        std.mem.eql(u8, codepoint, "═") or
        std.mem.eql(u8, codepoint, "║") or
        std.mem.eql(u8, codepoint, ".") or
        std.mem.eql(u8, codepoint, "a") or
        std.mem.eql(u8, codepoint, "i");
}
