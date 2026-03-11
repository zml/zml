const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const theme = @import("../../theme.zig");

const BrailleChart = @This();

values: []const u8, // 0-100
height: u16 = 2,

pub const chart_bg: vaxis.Cell.Color = .{ .rgb = .{ 30, 30, 40 } };

/// Braille dot bit positions: [row_in_cell (0-3)][col_in_cell (0-1)]
/// Each braille character is a 2x4 dot grid encoded as U+2800 + pattern byte.
pub const dot_bits = [4][2]u8{
    .{ 0x01, 0x08 }, // row 0
    .{ 0x02, 0x10 }, // row 1
    .{ 0x04, 0x20 }, // row 2
    .{ 0x40, 0x80 }, // row 3
};

pub fn widget(self: *const BrailleChart) vxfw.Widget {
    return .{
        .userdata = @constCast(self),
        .drawFn = typeErasedDrawFn,
    };
}

fn typeErasedDrawFn(ptr: *anyopaque, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const self: *const BrailleChart = @ptrCast(@alignCast(ptr));
    return self.draw(ctx);
}

pub fn draw(self: *const BrailleChart, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const w = ctx.max.width orelse 40;
    var surface = try vxfw.Surface.init(ctx.arena, self.widget(), .{ .width = w, .height = self.height });
    try self.renderTo(ctx.arena, &surface, 0, 0, w);
    return surface;
}

/// Render braille data into an existing surface at the given offset.
pub fn renderTo(self: *const BrailleChart, arena: std.mem.Allocator, surface: *vxfw.Surface, col_offset: u16, row_offset: u16, w: u16) std.mem.Allocator.Error!void {
    const h = self.height;
    const dot_rows: u32 = @as(u32, h) * 4;
    const dot_cols: u32 = @as(u32, w) * 2;
    const data_len: u32 = @intCast(self.values.len);

    for (0..w) |cell_col| {
        var max_val: u8 = 0;
        for (0..2) |dx| {
            const dot_x: u32 = @as(u32, @intCast(cell_col)) * 2 + @as(u32, @intCast(dx));
            if (dot_x + data_len >= dot_cols) {
                max_val = @max(max_val, self.values[@intCast(dot_x + data_len - dot_cols)]);
            }
        }
        const color = theme.colorForPercent(max_val);

        for (0..h) |cell_row| {
            var pattern: u8 = 0;
            for (0..2) |dx| {
                const dot_x: u32 = @as(u32, @intCast(cell_col)) * 2 + @as(u32, @intCast(dx));
                if (dot_x + data_len >= dot_cols) {
                    const val = self.values[@intCast(dot_x + data_len - dot_cols)];
                    const fill: u32 = @as(u32, val) * dot_rows / 100;
                    const threshold: u32 = dot_rows - fill;
                    for (0..4) |dr| {
                        const abs_row: u32 = @as(u32, @intCast(cell_row)) * 4 + @as(u32, @intCast(dr));
                        if (abs_row >= threshold) pattern |= dot_bits[dr][dx];
                    }
                }
            }

            const buf = try arena.alloc(u8, 3);
            buf[0] = 0xE2;
            buf[1] = 0xA0 + (pattern >> 6);
            buf[2] = 0x80 + (pattern & 0x3F);

            surface.writeCell(col_offset + @as(u16, @intCast(cell_col)), row_offset + @as(u16, @intCast(cell_row)), .{
                .char = .{ .grapheme = buf, .width = 1 },
                .style = .{ .fg = color, .bg = chart_bg },
            });
        }
    }
}
