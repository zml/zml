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
/// Optional gauge bars.
gauges: []const GaugeEntry = &.{},
/// Optional braille mini-charts.
charts: []const ChartEntry = &.{},
/// Whether this card is highlighted (e.g. hovered).
highlighted: bool = false,

pub const LineEntry = struct {
    label: []const u8,
    value: []const u8,
    style: vaxis.Cell.Style = theme.value_style,
};

pub const GaugeEntry = struct {
    label: []const u8,
    value: ?u8,
    suffix: ?[]const u8 = null,
};

pub const ChartEntry = struct {
    label: []const u8,
    value: ?u8,
    suffix: ?[]const u8 = null,
    data: []const ?u8, // history values 0-100; null leaves a gap
    chart_height: u16 = 3,
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

fn maxFieldWidth(ctx: vxfw.DrawContext, entries: anytype, comptime field: []const u8) u16 {
    var max_w: u16 = 0;
    for (entries) |entry| {
        const raw = @field(entry, field);
        const value: []const u8 = if (@typeInfo(@TypeOf(raw)) == .optional) raw orelse continue else raw;
        max_w = @max(max_w, @as(u16, @intCast(ctx.stringWidth(value))));
    }
    return max_w;
}

fn drawContent(self: *const MetricCard, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const inner_w = ctx.max.width orelse 40;

    var sb = compose.surfaceBuilder(ctx.arena);
    var row: i17 = 0;

    // Both presentations share alignment, percentage, suffix, and row spacing.
    inline for (.{ self.charts, self.gauges }) |entries| {
        const label_width = maxFieldWidth(ctx, entries, "label") + 1;
        const suffix_width = maxFieldWidth(ctx, entries, "suffix") + Gauge.sep_w;
        for (entries) |entry| {
            if (row > 0) row += 1;
            const has_history = @hasField(@TypeOf(entry), "data");
            const gauge: Gauge = .{
                .value = entry.value,
                .label = entry.label,
                .suffix = entry.suffix,
                .history = if (has_history) entry.data else null,
                .chart_height = if (has_history) entry.chart_height else 1,
                .label_reserve = label_width,
                .suffix_reserve = suffix_width,
            };
            const surface = try gauge.draw(ui.fixedWidth(ctx, inner_w));
            try sb.add(row, 0, surface);
            row += @intCast(surface.size.height);
        }
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
