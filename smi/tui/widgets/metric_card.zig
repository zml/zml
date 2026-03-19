const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const theme = @import("../theme.zig");
const ui = @import("../lib/ui.zig");
const compose = @import("../lib/compose.zig");
const Gauge = @import("gauge.zig");
const BrailleChart = @import("braille_chart.zig");
const TitledBorder = @import("titled_border.zig");

const MetricCard = @This();

title: []const u8,
title_image: ?vaxis.Image = null,
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
    value: u8,
    suffix: ?[]const u8 = null,
};

pub const ChartEntry = struct {
    label: []const u8,
    value: u8,
    suffix: ?[]const u8 = null,
    data: []const u8, // history values 0-100
    chart_height: u16 = 3,
};

/// Returns the total height this card will occupy when drawn.
pub fn totalHeight(self: *const MetricCard) u16 {
    return self.contentHeight() + 4; // Border(2) + Padding(2)
}

fn contentHeight(self: *const MetricCard) u16 {
    var content_h: u16 = 0;

    // Charts: chart_height rows each, 1-row gap between
    for (self.charts, 0..) |chart, ci| {
        if (ci > 0) content_h += 1;
        content_h += chart.chart_height;
    }

    // Gauges: 1 row each, 1-row gap between
    const gauge_h: u16 = if (self.gauges.len > 0)
        @intCast(self.gauges.len * 2 - 1)
    else
        0;
    if (gauge_h > 0 and content_h > 0) content_h += 1;
    content_h += gauge_h;

    content_h += @intCast(self.lines.len);
    if (content_h == 0) content_h = 1;
    return content_h;
}

pub fn draw(self: *const MetricCard, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const bstyle: vaxis.Cell.Style = if (self.highlighted)
        .{ .fg = theme.accent, .bold = true }
    else
        theme.border_style;
    const tb: TitledBorder = .{
        .child = ui.drawWidget(self, drawContent),
        .title = self.title,
        .title_image = self.title_image,
        .cell_size = self.cell_size,
        .border_style = bstyle,
    };
    const total_h = self.totalHeight();
    return tb.draw(ctx.withConstraints(ctx.min, .{
        .width = ctx.max.width,
        .height = if (ctx.max.height) |h| h else total_h,
    }));
}

fn maxSuffixWidth(ctx: vxfw.DrawContext, suffixes: anytype) u16 {
    var max_w: u16 = 0;
    for (suffixes) |entry| {
        if (entry.suffix) |s| {
            const sw: u16 = @intCast(ctx.stringWidth(s));
            max_w = @max(max_w, sw + Gauge.sep_w);
        }
    }
    return max_w;
}

fn drawContent(self: *const MetricCard, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const inner_w = ctx.max.width orelse 40;
    const content_h = self.contentHeight();

    var sb = compose.surfaceBuilder(ctx.arena);
    var row: i17 = 0;

    // Charts (gauge-like layout: label [braille] pct% suffix, braille spans multiple rows)
    {
        const max_chart_suffix_w = maxSuffixWidth(ctx, self.charts);

        for (self.charts, 0..) |chart, ci| {
            if (ci > 0) row += 1;

            const ch_h = chart.chart_height;

            // Layout widths (mirrors gauge.zig)
            var label_w: u16 = @intCast(ctx.stringWidth(chart.label));
            label_w += 1; // trailing space

            const pct_str = try std.fmt.allocPrint(ctx.arena, "{d:>3}%", .{chart.value});
            const pct_w: u16 = @intCast(ctx.stringWidth(pct_str));

            const actual_suffix_area: u16 = if (chart.suffix) |s| @as(u16, @intCast(ctx.stringWidth(s))) + Gauge.sep_w else 0;
            const effective_suffix_area = @max(actual_suffix_area, max_chart_suffix_w);

            const reserved = label_w + pct_w + 1 + effective_suffix_area;
            const braille_w: u16 = if (reserved < inner_w) inner_w - reserved else 0;
            const text_row: i17 = row + @as(i17, @intCast(ch_h / 2));

            // Label (vertically centered, col 0)
            const label_surf = try ui.drawRichLine(ctx, &.{
                .{ .text = chart.label, .style = theme.label_style },
            }, label_w);
            try sb.add(text_row, 0, label_surf);

            // Braille chart (all rows, after label)
            if (braille_w > 0) {
                const braille: BrailleChart = .{ .values = chart.data, .height = ch_h };
                const braille_surf = try braille.draw(ui.fixedSize(ctx, braille_w, ch_h));
                try sb.add(row, @intCast(label_w), braille_surf);
            }

            // Pct + suffix (vertically centered, after braille)
            const right_col: i17 = @intCast(label_w + braille_w);
            const pct_surf = try ui.drawRichLine(ctx, &.{
                .{ .text = " ", .style = .{} },
                .{ .text = pct_str, .style = .{ .bold = true, .fg = theme.colorForPercent(chart.value) } },
            }, pct_w + 1);
            try sb.add(text_row, right_col, pct_surf);

            if (chart.suffix) |suffix| {
                const suffix_text_w: u16 = @intCast(ctx.stringWidth(suffix));
                const suffix_col: i17 = @intCast(inner_w -| suffix_text_w);
                const suffix_surf = try ui.drawRichLine(ctx, &.{
                    .{ .text = suffix, .style = theme.dim_style },
                }, suffix_text_w);
                try sb.add(text_row, suffix_col, suffix_surf);
            }

            row += @intCast(ch_h);
        }
    }

    // Gap between charts and gauges
    if (self.charts.len > 0 and self.gauges.len > 0) row += 1;

    // Gauges (with 1-row gap between consecutive gauges)
    const max_suffix_w = maxSuffixWidth(ctx, self.gauges);
    for (self.gauges, 0..) |ge, gi| {
        if (row >= content_h) break;
        const gauge: Gauge = .{
            .value = ge.value,
            .label = ge.label,
            .suffix = ge.suffix,
            .suffix_reserve = max_suffix_w,
        };
        const gauge_surf = try gauge.draw(ui.fixedSize(ctx, inner_w, 1));
        try sb.add(row, 0, gauge_surf);
        row += 1;
        if (gi + 1 < self.gauges.len) row += 1; // gap between gauges
    }

    // Text lines
    for (self.lines) |line| {
        if (row >= content_h) break;
        const text_surf = try ui.drawRichLine(ctx, &.{
            .{ .text = line.label, .style = theme.label_style },
            .{ .text = line.value, .style = line.style },
        }, inner_w);
        try sb.add(row, 0, text_surf);
        row += 1;
    }

    return sb.finish(.{ .width = inner_w, .height = content_h }, ui.drawWidget(self, drawContent));
}
