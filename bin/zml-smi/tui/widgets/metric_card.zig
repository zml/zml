const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const theme = @import("../theme.zig");
const ui = @import("../lib/ui.zig");
const compose = @import("../lib/compose.zig");
const Gauge = @import("gauge.zig");
const TitledBorder = @import("titled_border.zig");

const MetricCard = @This();

title: []const u8,
title_image: ?vaxis.Image = null,
title_right: ?[]const u8 = null,
cell_size: vxfw.Size = .{},
/// Lines of "label: value" pairs to display.
lines: []const LineEntry = &.{},
/// Gauges with bars or braille history.
gauges: []const Gauge = &.{},
/// Whether this card is highlighted (e.g. hovered).
highlighted: bool = false,

pub const LineEntry = struct {
    label: []const u8,
    value: []const u8,
    style: vaxis.Cell.Style = theme.value_style,
};

pub fn draw(self: *const MetricCard, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const bstyle: vaxis.Cell.Style = if (self.highlighted)
        .{ .fg = theme.accent, .bold = true }
    else
        theme.border_style;
    const tb: TitledBorder = .{
        .child = ui.drawWidget(self, drawContent),
        .title = self.title,
        .title_image = self.title_image,
        .value_label = self.title_right,
        .cell_size = self.cell_size,
        .border_style = bstyle,
    };
    return tb.draw(ctx);
}

fn drawContent(self: *const MetricCard, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const inner_w = ctx.max.width orelse 40;

    var sb = compose.surfaceBuilder(ctx.arena);
    var row: i17 = 0;

    var label_width: usize = 0;
    var suffix_width: usize = 0;
    for (self.gauges) |gauge| {
        if (gauge.label) |label| label_width = @max(label_width, ctx.stringWidth(label));
        if (gauge.suffix) |suffix| suffix_width = @max(suffix_width, ctx.stringWidth(suffix));
    }
    for (self.gauges) |entry| {
        if (row > 0) row += 1;
        var gauge = entry;
        gauge.label_reserve = @intCast(label_width + 1);
        gauge.suffix_reserve = @intCast(suffix_width + Gauge.sep_w);
        const surface = try gauge.draw(ui.fixedWidth(ctx, inner_w));
        try sb.add(row, 0, surface);
        row += @intCast(surface.size.height);
    }

    // Text lines
    for (self.lines) |line| {
        const text_surf = try ui.drawRichLine(ctx, &.{
            .{ .text = line.label, .style = theme.label_style },
            .{ .text = line.value, .style = line.style },
        }, inner_w);
        try sb.add(row, 0, text_surf);
        row += 1;
    }

    const content_h: u16 = @intCast(@max(row, 1));
    return sb.finish(.{ .width = inner_w, .height = content_h }, ui.drawWidget(self, drawContent));
}
