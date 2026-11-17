const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const theme = @import("../theme.zig");

const chart_bg: vaxis.Cell.Color = .{ .rgb = .{ 30, 30, 40 } };

/// Braille dot bit positions: [row_in_cell (0-3)][col_in_cell (0-1)]
/// Each braille character is a 2x4 dot grid encoded as U+2800 + pattern byte.
const dot_bits = [4][2]u8{
    .{ 0x01, 0x08 }, // row 0
    .{ 0x02, 0x10 }, // row 1
    .{ 0x04, 0x20 }, // row 2
    .{ 0x40, 0x80 }, // row 3
};

const glyphs = glyphs_blk: {
    var result: [256][3]u8 = undefined;
    for (&result, 0..) |*glyph, pattern| {
        const byte: u8 = @intCast(pattern);
        glyph.* = .{ 0xE2, 0xA0 + (byte >> 6), 0x80 + (byte & 0x3F) };
    }
    break :glyphs_blk result;
};

/// Render 0–100 samples into the top of a surface. Null samples leave gaps.
pub fn draw(surface: vxfw.Surface, column_start: u16, height: u16, values: []const ?u8) void {
    const width = surface.size.width - column_start;
    const dot_rows = @as(usize, height) * 4;
    const dot_cols = @as(usize, width) * 2;

    for (0..width) |cell_col| {
        var max_val: u8 = 0;
        var thresholds: [2]usize = @splat(dot_rows);
        for (&thresholds, 0..) |*threshold, offset| {
            const dot_x = cell_col * 2 + offset;
            if (dot_x + values.len < dot_cols) continue;
            const value = values[dot_x + values.len - dot_cols] orelse continue;
            max_val = @max(max_val, value);
            threshold.* = dot_rows - @as(usize, value) * dot_rows / 100;
        }
        const color = theme.colorForPercent(max_val);

        for (0..height) |cell_row| {
            var pattern: u8 = 0;
            for (dot_bits, 0..) |bits, dot_row| {
                const row = cell_row * 4 + dot_row;
                if (row >= thresholds[0]) pattern |= bits[0];
                if (row >= thresholds[1]) pattern |= bits[1];
            }

            surface.writeCell(column_start + @as(u16, @intCast(cell_col)), @intCast(cell_row), .{
                .char = .{ .grapheme = &glyphs[pattern] },
                .style = .{ .fg = color, .bg = chart_bg },
            });
        }
    }
}
