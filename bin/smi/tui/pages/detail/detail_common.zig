const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../../data.zig");
const theme = @import("../../theme.zig");
const utils = @import("../../lib/utils.zig");
const ui = @import("../../lib/ui.zig");
const compose = @import("../../lib/compose.zig");
const Chart = @import("../../widgets/chart.zig");
const ColumnLayout = @import("../../widgets/column_layout.zig");


pub fn headerText(arena: std.mem.Allocator, id: u8, name: []const u8) std.mem.Allocator.Error!vxfw.Text {
    return .{
        .text = try std.fmt.allocPrint(arena, "Device {d}: {s}", .{ id, name }),
        .style = theme.header_style,
        .softwrap = false,
        .overflow = .clip,
    };
}

/// Creates a Chart pre-filled with history data and standard defaults.
pub fn historyChart(
    ctx: vxfw.DrawContext,
    state: *const data.SystemState,
    id: u8,
    history: anytype,
    opts: struct {
        title: []const u8,
        value_label: []const u8,
        info_line: ?[]const u8 = null,
        y_min: u32 = 0,
        y_max: u32 = 100,
        y_unit: []const u8 = "%",
    },
) std.mem.Allocator.Error!Chart {
    return .{
        .title = opts.title,
        .data = try utils.normalizeRange(ctx.arena, try history[id].sliceLast(ctx.arena, data.history_len), opts.y_min, opts.y_max),
        .value_label = opts.value_label,
        .info_line = opts.info_line,
        .y_min = opts.y_min,
        .y_max = opts.y_max,
        .y_unit = opts.y_unit,
        .chart_height = 8,
        .tui_refresh_rate = state.tui_refresh_rate,
    };
}

/// Draws a page-level layout with standard margin (top=1, left/right=2) inside a full-width surface.
pub fn pageFrame(ctx: vxfw.DrawContext, layout: ColumnLayout, w: u16, parent_widget: vxfw.Widget) std.mem.Allocator.Error!vxfw.Surface {
    const padded = try compose.pad(ctx.arena, ui.widget(&layout), .{ .top = 1, .left = 2, .right = 2 });
    var surf = try padded.draw(ui.fixedWidth(ctx, w));
    surf.widget = parent_widget;
    return surf;
}
