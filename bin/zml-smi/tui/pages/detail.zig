const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../data.zig");
const ui = @import("../lib/ui.zig");
const Chart = @import("../widgets/chart.zig");
const ColumnLayout = @import("../widgets/column_layout.zig");
const ProcessTable = @import("../widgets/process_table.zig");
const common = @import("detail/detail_common.zig");
const presentation = @import("../presentation.zig");

const Detail = @This();

state: *const data.SystemState,
device_id: u16,
process_table: *ProcessTable = undefined,

pub fn draw(self: *const Detail, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const index = self.device_id;
    const view = try presentation.detailView(ctx.arena, self.state.frame.?, index);

    const metrics = [_]data.Metric{ .util, .memory, .temperature, .power };
    const chart_count: usize = if (view.environment_charts) 4 else 2;
    var charts: [metrics.len]Chart = undefined;
    var chart_widgets: [metrics.len]vxfw.Widget = undefined;
    for (metrics[0..chart_count], charts[0..chart_count], chart_widgets[0..chart_count]) |kind, *chart, *widget| {
        chart.* = try common.historyChart(ctx, self.state, index, view.reading, kind, .{
            .title = if (kind == .util) view.util_title else null,
            .info_line = view.labels.get(kind),
        });
        widget.* = ui.widget(chart);
    }
    const chart_layout: ColumnLayout = .{
        .children = chart_widgets[0..chart_count],
        .min_child_width = if (chart_count == 4) 60 else 40,
        .gap = 1,
    };

    const extra = view.cards;
    const card_widgets = try ctx.arena.alloc(vxfw.Widget, extra.len);
    for (extra, card_widgets) |*card, *widget| widget.* = ui.widget(card);
    const card_layout: ColumnLayout = .{ .children = card_widgets, .min_child_width = 30 };
    var sections = [_]vxfw.Widget{
        view.heading.widget(), ui.widget(&chart_layout), ui.widget(&card_layout), ui.widget(self.process_table),
    };
    if (extra.len == 0) sections[2] = sections[3];
    const count: usize = if (extra.len == 0) 3 else 4;
    return common.pageFrame(ctx, .{ .children = sections[0..count], .gap = 1 }, ctx.max.width orelse 80, ui.widget(self));
}
