const std = @import("std");

pub const Color = struct {
    escape: []const u8,
};

pub const Colors = enum {
    reset,
    cyan,
    sky,
    blue,
    violet,
    pink,
    shine_cyan,
    shine_sky,
    shine_blue,
    shine_violet,
    shine_pink,

    pub fn get(self: Colors, support: ColorSupport) Color {
        return .{
            .escape = switch (support) {
                .true_color => self.trueColor(),
                .xterm_256 => self.xterm256(),
                else => "",
            },
        };
    }

    fn trueColor(self: Colors) []const u8 {
        return switch (self) {
            .reset => "\x1b[0m",
            .cyan => "\x1b[38;2;0;255;255m",
            .sky => "\x1b[38;2;0;221;255m",
            .blue => "\x1b[38;2;135;143;255m",
            .violet => "\x1b[38;2;215;90;219m",
            .pink => "\x1b[38;2;242;48;174m",
            .shine_cyan => "\x1b[1;38;2;140;255;255m",
            .shine_sky => "\x1b[1;38;2;128;244;255m",
            .shine_blue => "\x1b[1;38;2;205;210;255m",
            .shine_violet => "\x1b[1;38;2;255;170;255m",
            .shine_pink => "\x1b[1;38;2;255;150;220m",
        };
    }

    fn xterm256(self: Colors) []const u8 {
        return switch (self) {
            .reset => "\x1b[0m",
            .cyan => "\x1b[38;5;51m",
            .sky => "\x1b[38;5;45m",
            .blue => "\x1b[38;5;105m",
            .violet => "\x1b[38;5;170m",
            .pink => "\x1b[38;5;205m",
            .shine_cyan => "\x1b[1;38;5;123m",
            .shine_sky => "\x1b[1;38;5;123m",
            .shine_blue => "\x1b[1;38;5;189m",
            .shine_violet => "\x1b[1;38;5;219m",
            .shine_pink => "\x1b[1;38;5;212m",
        };
    }
};

pub const Row = struct {
    color: Colors,
    shine: Colors,
    text: []const u8,
};

pub const WriteOptions = struct {
    color_support: ColorSupport = .true_color,
};

pub const ColorSupport = enum {
    no_color,
    ansi,
    xterm_256,
    true_color,
};

pub fn detectColorSupport(io: std.Io, env_: ?*const std.process.Environ.Map) ColorSupport {
    var no_color: bool = false;
    var clicolor_force: bool = false;
    if (env_) |env| {
        no_color = if (env.get("NO_COLOR")) |v| v.len != 0 else false;
        clicolor_force = if (env.get("CLICOLOR_FORCE")) |v| v.len != 0 else false;
    }

    const mode = std.Io.Terminal.Mode.detect(io, std.Io.File.stdout(), no_color, clicolor_force) catch return .no_color;
    if (mode != .escape_codes) return .no_color;

    if (env_) |env| {
        if (env.get("COLORTERM")) |value| {
            if (std.ascii.eqlIgnoreCase(value, "truecolor") or
                std.ascii.eqlIgnoreCase(value, "24bit"))
            {
                return .true_color;
            }
        }

        if (env.get("TERM")) |value| {
            if (std.mem.endsWith(u8, value, "-direct")) return .true_color;
            if (std.mem.indexOf(u8, value, "truecolor") != null) return .true_color;

            if (std.mem.indexOf(u8, value, "256color") != null) return .xterm_256;
        }
    }

    return .ansi;
}

pub fn writeRow(writer: *std.Io.Writer, row: Row, options: WriteOptions) !void {
    const color = row.color.get(options.color_support).escape;
    const shine = row.shine.get(options.color_support).escape;
    const reset = Colors.reset.get(options.color_support).escape;

    var view: std.unicode.Utf8View = .initUnchecked(row.text);
    var iter = view.iterator();
    var is_shiny = false;

    try writer.writeAll(color);
    while (iter.nextCodepointSlice()) |codepoint| {
        const should_shine = isShineGlyph(codepoint);
        if (should_shine != is_shiny) {
            try writer.writeAll(if (should_shine) shine else color);
            is_shiny = should_shine;
        }
        try writer.writeAll(codepoint);
    }
    try writer.writeAll(reset);
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
