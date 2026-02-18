const std = @import("std");

// ── ANSI escape sequences & Unicode block characters ────────

pub const esc = struct {
    // Cursor control
    pub const hide_cursor = "\x1b[?25l";
    pub const show_cursor = "\x1b[?25h";
    pub const save_cursor = "\x1b7";
    pub const restore_cursor = "\x1b8";
    pub const cursor_home = "\x1b[H";

    // Screen control
    pub const clear_screen = "\x1b[2J";
    pub const clear_to_end = "\x1b[J";
    pub const reset_scroll_region = "\x1b[r";

    // Text style
    pub const reset = "\x1b[0m";
    pub const bold = "\x1b[1m";
    pub const dim_gray = "\x1b[0;90m";
    pub const bright_white = "\x1b[97m";

    // Unicode box drawing / block elements
    pub const horizontal_line = "\xe2\x94\x80"; // ─
    pub const upper_half_block = "\xe2\x96\x80"; // ▀
    pub const lower_half_block = "\xe2\x96\x84"; // ▄

    // Parameterized format strings (for use with fmt)
    pub const fg_rgb_fmt = "\x1b[38;2;{d};{d};{d}m";
    pub const bold_fg_rgb_fmt = "\x1b[1;38;2;{d};{d};{d}m";
    pub const fg_bg_rgb_fmt = "\x1b[38;2;{d};{d};{d};48;2;{d};{d};{d}m";
    pub const set_scroll_region_fmt = "\x1b[{d};999r";
    pub const move_cursor_fmt = "\x1b[{d};{d}H";
};

// ── Public types ─────────────────────────────────────────────

pub const Color = struct { r: u8, g: u8, b: u8 };

pub const Gradient = struct {
    start: Color = .{ .r = 140, .g = 20, .b = 8 },
    mid: Color = .{ .r = 230, .g = 120, .b = 5 },
    end: Color = .{ .r = 255, .g = 220, .b = 55 },
};

pub const Config = struct {
    num_bars: u16 = 60,
    half_height: u16 = 22,
    sensitivity: f32 = 12.0,
    attack: f32 = 0.6,
    decay: f32 = 0.12,
    gradient: Gradient = .{},
    padding_left: u16 = 4,
    title: []const u8,
    show_title: bool = true,
};

pub const FrameResult = struct {
    level: f32,
};

// ── State ────────────────────────────────────────────────────

pub const State = struct {
    config: Config,
    writer: *std.Io.Writer,
    smoothed: f32,
    frame_count: u32,
    buf: [BUF_SIZE]u8,
    pos: usize,

    gradient_lut: [MAX_HALF_HEIGHT]Color,

    const BUF_SIZE = 131072;
    const MAX_HALF_HEIGHT = 256;
    const Half = enum { upper, lower };

    pub fn init(config: Config, writer: *std.Io.Writer) State {
        var state = State{
            .config = config,
            .writer = writer,
            .smoothed = 0,
            .frame_count = 0,
            .gradient_lut = undefined,
            .buf = undefined,
            .pos = 0,
        };

        state.buildGradientLut();

        writer.writeAll(esc.hide_cursor ++ esc.clear_screen) catch {};

        return state;
    }

    pub fn deinit(self: *State) void {
        self.writer.writeAll(esc.reset_scroll_region ++ esc.show_cursor) catch {};
    }

    pub fn render(self: *State, rms: f32) FrameResult {
        // Smoothing
        const attack_rate: f32 = if (rms > self.smoothed) self.config.attack else self.config.decay;
        self.smoothed = self.smoothed * (1.0 - attack_rate) + rms * attack_rate;
        const level = @min(self.smoothed * self.config.sensitivity, 1.0);

        // Render
        self.pos = 0;
        self.put(esc.cursor_home);

        if (self.config.show_title) {
            self.put("\n");
            self.fmt(esc.bold_fg_rgb_fmt, .{ 230, 120, 5 });
            self.putPadding();
            self.put(self.config.title);
            self.put(" " ++ esc.dim_gray ++ esc.horizontal_line ++ " live transcription" ++ esc.reset ++ "\n\n");
        }

        const num_bars = self.config.num_bars;
        const half_height = self.config.half_height;
        const half_term = half_height / 2;

        // Compute bar heights
        var heights: [512]f32 = undefined;
        const center: f32 = @as(f32, @floatFromInt(num_bars)) / 2.0;
        const t: f32 = @as(f32, @floatFromInt(self.frame_count)) * 0.12;

        for (0..num_bars) |i| {
            const fi: f32 = @as(f32, @floatFromInt(i));
            const dist = (fi - center) / center;
            const envelope = @exp(-dist * dist * 3.5);

            const w1 = 0.28 * @sin(fi * 0.8 + t);
            const w2 = 0.18 * @sin(fi * 1.5 - t * 1.6);
            const w3 = 0.12 * @sin(fi * 2.4 + t * 0.7);
            const w4 = 0.08 * @sin(fi * 3.7 - t * 2.1);
            const wobble = w1 + w2 + w3 + w4;

            const h = level * envelope * (1.0 + wobble) * @as(f32, @floatFromInt(half_height));
            heights[i] = @max(@min(h, @as(f32, @floatFromInt(half_height))), 0.0);
        }

        self.renderHalf(.upper, half_term, half_height, num_bars, heights[0..num_bars]);
        self.renderHalf(.lower, half_term, half_height, num_bars, heights[0..num_bars]);

        self.frame_count +%= 1;

        return .{ .level = level };
    }

    /// Render a horizontal separator line below the wave visualization.
    pub fn renderSeparator(self: *State) void {
        self.put("\n");
        self.fmt(esc.fg_rgb_fmt, .{ 60, 60, 60 });
        self.putPadding();
        for (0..self.config.num_bars) |_| {
            self.put(esc.horizontal_line);
        }
        self.put(esc.reset ++ "\n\n");
    }

    // ── Private helpers ──────────────────────────────────────

    fn buildGradientLut(self: *State) void {
        const hh = self.config.half_height;
        const g = self.config.gradient;

        for (0..hh) |i| {
            const t: f32 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(hh - 1));
            if (t < 0.5) {
                const s = t * 2.0;
                self.gradient_lut[i] = .{
                    .r = lerpU8(@floatFromInt(g.start.r), @floatFromInt(g.mid.r), s),
                    .g = lerpU8(@floatFromInt(g.start.g), @floatFromInt(g.mid.g), s),
                    .b = lerpU8(@floatFromInt(g.start.b), @floatFromInt(g.mid.b), s),
                };
            } else {
                const s = (t - 0.5) * 2.0;
                self.gradient_lut[i] = .{
                    .r = lerpU8(@floatFromInt(g.mid.r), @floatFromInt(g.end.r), s),
                    .g = lerpU8(@floatFromInt(g.mid.g), @floatFromInt(g.end.g), s),
                    .b = lerpU8(@floatFromInt(g.mid.b), @floatFromInt(g.end.b), s),
                };
            }
        }
    }

    fn renderHalf(self: *State, half: Half, half_term: u16, half_height: u16, num_bars: u16, heights: []const f32) void {
        for (0..half_term) |tr| {
            self.putPadding();
            const tr16 = @as(u16, @intCast(tr));
            const upper_log, const lower_log = switch (half) {
                .upper => .{ half_height - 1 - 2 * tr16, half_height - 2 - 2 * tr16 },
                .lower => .{ 2 * tr16 + 1, 2 * tr16 + 2 },
            };
            for (0..num_bars) |col| {
                const h = heights[col];
                const upper_on = h > @as(f32, @floatFromInt(upper_log));
                const lower_on = h > @as(f32, @floatFromInt(lower_log));

                if (upper_on and lower_on) {
                    self.cellBoth(self.gradient_lut[upper_log], self.gradient_lut[lower_log]);
                } else if (half == .upper and lower_on) {
                    self.cellLower(self.gradient_lut[lower_log]);
                } else if (half == .lower and upper_on) {
                    self.cellUpper(self.gradient_lut[upper_log]);
                } else {
                    self.put(" ");
                }
            }
            self.put("\n");
        }
    }

    // ── Public buffer helpers (for caller to append after render) ──

    pub fn putPadding(self: *State) void {
        for (0..self.config.padding_left) |_| {
            self.put(" ");
        }
    }

    pub fn put(self: *State, s: []const u8) void {
        if (self.pos + s.len > BUF_SIZE) return;
        @memcpy(self.buf[self.pos..][0..s.len], s);
        self.pos += s.len;
    }

    pub fn fmt(self: *State, comptime f: []const u8, args: anytype) void {
        const result = std.fmt.bufPrint(self.buf[self.pos..], f, args) catch return;
        self.pos += result.len;
    }

    pub fn flush(self: *State) void {
        self.writer.writeAll(self.buf[0..self.pos]) catch {};
        self.pos = 0;
    }

    fn cellBoth(self: *State, upper: Color, lower: Color) void {
        self.fmt(esc.fg_bg_rgb_fmt, .{ upper.r, upper.g, upper.b, lower.r, lower.g, lower.b });
        self.put(esc.upper_half_block ++ esc.reset);
    }

    fn cellUpper(self: *State, col: Color) void {
        self.fmt(esc.fg_rgb_fmt, .{ col.r, col.g, col.b });
        self.put(esc.upper_half_block ++ esc.reset);
    }

    fn cellLower(self: *State, col: Color) void {
        self.fmt(esc.fg_rgb_fmt, .{ col.r, col.g, col.b });
        self.put(esc.lower_half_block ++ esc.reset);
    }
};

// ── Free functions ───────────────────────────────────────────

fn lerpU8(a: f32, b: f32, t: f32) u8 {
    return @intFromFloat(@max(0.0, @min(255.0, a + (b - a) * t)));
}

/// Compute RMS (root mean square) of audio samples for level metering.
pub fn computeRms(samples: []const f32) f32 {
    if (samples.len == 0) return 0;
    var sum: f32 = 0;
    for (samples) |s| {
        sum += s * s;
    }
    return @sqrt(sum / @as(f32, @floatFromInt(samples.len)));
}
