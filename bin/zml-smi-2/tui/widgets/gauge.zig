const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const theme = @import("../theme.zig");
const ui = @import("../lib/ui.zig");
const compose = @import("../lib/compose.zig");
const braille = @import("braille_chart.zig");

const Gauge = @This();

value: ?u8, // 0-100; null displays N/A
label: ?[]const u8 = null,
suffix: ?[]const u8 = null,
/// A history replaces the bar with a braille chart using the same text layout.
history: ?[]const ?u8 = null,
chart_height: u16 = 3,
/// Minimum label space to reserve (for aligning gauges with different label widths).
label_reserve: u16 = 0,
/// Minimum suffix space to reserve (for aligning gauges with different suffix widths).
suffix_reserve: u16 = 0,

pub const sep_w: u16 = 3; // "   " before suffix

pub fn draw(self: *const Gauge, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const width = @max(ctx.min.width, ctx.max.width orelse 30);
    const height = if (self.history != null) self.chart_height else 1;
    const text_row: i17 = @intCast(height / 2);
    const label_width: u16 = @intCast(@max(self.label_reserve, if (self.label) |label| ctx.stringWidth(label) + 1 else 0));
    const suffix_width: u16 = @intCast(ctx.stringWidth(self.suffix orelse ""));
    const suffix_area = @max(self.suffix_reserve, if (self.suffix != null) suffix_width + sep_w else 0);
    const percentage = if (self.value) |value|
        try std.fmt.allocPrint(ctx.arena, "{d:>3}%", .{value})
    else
        " N/A";
    const percentage_width: u16 = @intCast(ctx.stringWidth(percentage));
    // A bar has two brackets; a chart has one separator before its percentage.
    const framing: u16 = if (self.history != null) 1 else 2;
    const body_width = width -| (label_width + percentage_width + suffix_area + framing);
    const body_column: u16 = if (self.label != null) label_width else 0;

    var builder = compose.surfaceBuilder(ctx.arena);
    if (self.label) |label| {
        try builder.add(text_row, 0, try ui.drawRichLine(ctx, &.{
            .{ .text = label, .style = theme.label_style },
        }, label_width));
    }
    if (self.history) |values| {
        if (body_width > 0) {
            const size: vxfw.Size = .{ .width = body_width, .height = height };
            const chart = try vxfw.Surface.init(ctx.arena, ui.widget(self), size);
            braille.draw(chart, 0, height, values);
            try builder.add(0, body_column, chart);
        }
    } else {
        try builder.add(0, body_column, try self.drawBar(ctx, body_width));
    }

    const percentage_column = body_column + body_width + if (self.history == null) @as(u16, 2) else 0;
    try builder.add(text_row, percentage_column, try ui.drawRichLine(ctx, &.{
        .{ .text = if (self.history != null) " " else "", .style = .{} },
        .{ .text = percentage, .style = .{
            .bold = true,
            .fg = if (self.value) |value| theme.colorForPercent(value) else theme.dim,
        } },
    }, percentage_width + if (self.history != null) @as(u16, 1) else 0));
    if (self.suffix) |suffix| {
        // Charts anchor the suffix to the edge, including on a narrow terminal.
        const column = if (self.history != null)
            width -| suffix_width
        else
            percentage_column + percentage_width + suffix_area -| suffix_width;
        try builder.add(text_row, column, try ui.drawRichLine(ctx, &.{
            .{ .text = suffix, .style = theme.dim_style },
        }, suffix_width));
    }
    return builder.finish(.{ .width = width, .height = height }, ui.widget(self));
}

fn drawBar(self: *const Gauge, ctx: vxfw.DrawContext, width: u16) !vxfw.Surface {
    const filled: u16 = @intCast(@as(u32, width) * @min(self.value orelse 0, 100) / 100);
    const surface = try vxfw.Surface.init(ctx.arena, ui.widget(self), .{ .width = width + 2, .height = 1 });
    surface.buffer[0] = .{ .char = .{ .grapheme = "[" }, .style = theme.dim_style };
    surface.buffer[width + 1] = .{ .char = .{ .grapheme = "]" }, .style = theme.dim_style };
    @memset(surface.buffer[filled + 1 .. width + 1], .{
        .char = .{ .grapheme = "\u{2591}" },
        .style = .{ .fg = theme.gauge_empty },
    });
    // Each filled block keeps its position's gradient color.
    for (surface.buffer[1 .. filled + 1], 0..) |*cell, index| {
        const percentage: u8 = @intCast(index * 100 / @max(width -| 1, 1));
        cell.* = .{
            .char = .{ .grapheme = "\u{2588}" },
            .style = .{ .fg = theme.colorForPercent(percentage) },
        };
    }
    return surface;
}
