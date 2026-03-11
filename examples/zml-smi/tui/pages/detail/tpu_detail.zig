const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../../data.zig");
const utils = @import("../../utils.zig");
const Chart = @import("../../widgets/components/chart.zig");
const ColumnLayout = @import("../../widgets/components/column_layout.zig");
const ProcessTable = @import("../../widgets/process_table.zig");
const common = @import("detail_common.zig");

pub fn draw(
    state: *const data.SystemState,
    process_table: *ProcessTable,
    ctx: vxfw.DrawContext,
    w: u16,
    id: u8,
    tp: *data.TpuInfo,
    parent_widget: vxfw.Widget,
) std.mem.Allocator.Error!vxfw.Surface {
    const content_w: u16 = w -| 4;

    // ── Header ──────────────────────────────────────────────
    const header = try common.headerText(ctx.arena, id, utils.strSlice(&tp.name));

    // ── Utilization + Memory charts ─────────────────────────
    const gpu_chart: Chart = .{
        .title = "Duty Cycle",
        .data = try utils.normalizeRange(ctx.arena, try state.history.util[id].sliceLast(ctx.arena, data.history_len), 0, 100),
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}%", .{tp.util_percent orelse 0}),
        .y_min = 0,
        .y_max = 100,
        .y_unit = "%",
        .chart_height = 8,
        .sample_interval_ms = state.sample_interval_ms,
    };

    const mem_used = tp.mem_used_bytes orelse 0;
    const mem_total = tp.mem_total_bytes orelse 0;
    const mem_chart: Chart = .{
        .title = "Memory",
        .data = try utils.normalizeRange(ctx.arena, try state.history.mem_util[id].sliceLast(ctx.arena, data.history_len), 0, 100),
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}%", .{if (mem_total > 0) mem_used * 100 / mem_total else 0}),
        .info_line = try std.fmt.allocPrint(ctx.arena, "HBM {d} / {d} MB", .{
            utils.bytesToMb(tp.mem_used_bytes), utils.bytesToMb(tp.mem_total_bytes),
        }),
        .y_min = 0,
        .y_max = 100,
        .y_unit = "%",
        .chart_height = 8,
        .sample_interval_ms = state.sample_interval_ms,
    };

    const charts_flow: ColumnLayout = .{
        .children = &.{ gpu_chart.widget(), mem_chart.widget() },
        .min_child_width = 40,
        .gap = 1,
    };

    // ── Compose vertical layout ─────────────────────────────
    const layout: ColumnLayout = .{
        .children = &.{
            header.widget(),
            charts_flow.widget(),
            process_table.widget(),
        },
        .gap = 1,
    };

    const layout_surf = try layout.widget().draw(ctx.withConstraints(
        .{ .width = content_w },
        .{ .width = content_w, .height = null },
    ));

    const children = try ctx.arena.alloc(vxfw.SubSurface, 1);
    children[0] = .{ .origin = .{ .row = 1, .col = 2 }, .surface = layout_surf };
    return .{
        .size = .{ .width = w, .height = layout_surf.size.height + 1 },
        .widget = parent_widget,
        .buffer = &.{},
        .children = children,
    };
}
