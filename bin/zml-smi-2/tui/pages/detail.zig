const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../data.zig");
const theme = @import("../theme.zig");
const ui = @import("../lib/ui.zig");
const Chart = @import("../widgets/chart.zig");
const ColumnLayout = @import("../widgets/column_layout.zig");
const ProcessTable = @import("../widgets/process_table.zig");
const presentation = @import("../presentation.zig");

pub fn draw(
    ctx: vxfw.DrawContext,
    state: *const data.SystemState,
    index: u16,
    process_table: *ProcessTable,
) std.mem.Allocator.Error!vxfw.Surface {
    const device = state.devices[index];
    const profile = device.profile;
    const reading = &state.frame.samples[index];
    const heading: vxfw.Text = .{
        .text = device.heading,
        .style = theme.header_style,
        .softwrap = false,
        .overflow = .clip,
    };

    const metrics = [_]data.Metric{ .util, .memory, .temperature, .power };
    const chart_count: usize = if (profile.environment == .hidden) 2 else metrics.len;
    var charts: [metrics.len]Chart = undefined;
    var chart_widgets: [metrics.len]vxfw.Widget = undefined;
    for (metrics[0..chart_count], charts[0..chart_count], chart_widgets[0..chart_count]) |kind, *chart, *widget| {
        chart.* = try presentation.historyChart(ctx.arena, state, index, kind);
        widget.* = ui.widget(chart);
    }
    const chart_layout: ColumnLayout = .{
        .children = chart_widgets[0..chart_count],
        .min_child_width = if (chart_count == 4) 60 else 40,
        .gap = 1,
    };

    const extra = try presentation.cards(ctx.arena, profile.sections, reading, state.frame.host);
    const card_widgets = try ctx.arena.alloc(vxfw.Widget, extra.len);
    for (extra, card_widgets) |*card, *widget| widget.* = ui.widget(card);
    const card_layout: ColumnLayout = .{ .children = card_widgets, .min_child_width = 30 };
    const processes = ui.widget(process_table);
    const sections = [_]vxfw.Widget{
        heading.widget(),
        ui.widget(&chart_layout),
        if (extra.len == 0) processes else ui.widget(&card_layout),
        processes,
    };
    const count: usize = if (extra.len == 0) 3 else 4;
    const layout: ColumnLayout = .{ .children = sections[0..count], .gap = 1 };
    const padded: vxfw.Padding = .{
        .child = ui.widget(&layout),
        .padding = .{ .top = 1, .left = 2, .right = 2 },
    };
    return padded.draw(ui.fixedWidth(ctx, ctx.max.width orelse 80));
}
