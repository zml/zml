const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../../data.zig");
const str = @import("../../../str.zig");
const utils = @import("../../lib/utils.zig");
const ui = @import("../../lib/ui.zig");
const ColumnLayout = @import("../../widgets/column_layout.zig");
const ProcessTable = @import("../../widgets/process_table.zig");
const common = @import("detail_common.zig");

pub fn draw(
    state: *const data.SystemState,
    process_table: *ProcessTable,
    ctx: vxfw.DrawContext,
    w: u16,
    id: u8,
    tp: data.TpuInfo,
    parent_widget: vxfw.Widget,
) std.mem.Allocator.Error!vxfw.Surface {
    // ── Header ──────────────────────────────────────────────
    const header = try common.headerText(ctx.arena, id, tp.name orelse "Unknown");

    // ── Utilization + Memory charts ─────────────────────────
    const util_chart = try common.historyChart(ctx, state, id, state.history.util, .{
        .title = "Duty Cycle",
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}%", .{tp.util_percent orelse 0}),
    });

    const mem_used = tp.mem_used_bytes orelse 0;
    const mem_total = tp.mem_total_bytes orelse 0;
    const mem_chart = try common.historyChart(ctx, state, id, state.history.mem_util, .{
        .title = "Memory",
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}%", .{if (mem_total > 0) mem_used * 100 / mem_total else 0}),
        .info_line = try std.fmt.allocPrint(ctx.arena, "HBM {d} / {d} MB", .{
            utils.bytesToMb(tp.mem_used_bytes), utils.bytesToMb(tp.mem_total_bytes),
        }),
    });

    const charts_flow: ColumnLayout = .{
        .children = &.{ ui.widget(&util_chart), ui.widget(&mem_chart) },
        .min_child_width = 40,
        .gap = 1,
    };

    // ── Compose vertical layout ─────────────────────────────
    return common.pageFrame(ctx, .{
        .children = &.{
            header.widget(),
            ui.widget(&charts_flow),
            ui.widget(process_table),
        },
        .gap = 1,
    }, w, parent_widget);
}
