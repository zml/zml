const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../../data.zig");
const utils = @import("../../utils.zig");
const Chart = @import("../../widgets/components/chart.zig");
const ColumnLayout = @import("../../widgets/components/column_layout.zig");
const MetricCard = @import("../../widgets/metric_card.zig");
const ProcessTable = @import("../../widgets/process_table.zig");
const common = @import("detail_common.zig");

pub fn draw(
    state: *const data.SystemState,
    process_table: *ProcessTable,
    ctx: vxfw.DrawContext,
    w: u16,
    id: u8,
    nc: *data.NeuronInfo,
    parent_widget: vxfw.Widget,
) std.mem.Allocator.Error!vxfw.Surface {
    const content_w: u16 = w -| 4;

    // ── Header ──────────────────────────────────────────────
    const header = try common.headerText(ctx.arena, id, utils.strSlice(&nc.name));

    // ── Utilization + Memory charts ─────────────────────────
    const util_chart: Chart = .{
        .title = "Core Utilization",
        .data = try utils.normalizeRange(ctx.arena, try state.history.util[id].sliceLast(ctx.arena, data.history_len), 0, 100),
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}%", .{nc.util_percent orelse 0}),
        .y_min = 0,
        .y_max = 100,
        .y_unit = "%",
        .chart_height = 8,
        .tui_refresh_rate = state.tui_refresh_rate,
    };

    const mem_used = nc.mem_used_bytes orelse 0;
    const mem_total = nc.mem_total_bytes orelse 0;
    const mem_chart: Chart = .{
        .title = "Memory",
        .data = try utils.normalizeRange(ctx.arena, try state.history.mem_util[id].sliceLast(ctx.arena, data.history_len), 0, 100),
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}%", .{if (mem_total > 0) mem_used * 100 / mem_total else 0}),
        .info_line = try std.fmt.allocPrint(ctx.arena, "HBM {d} / {d} MB", .{
            utils.bytesToMb(nc.mem_used_bytes), utils.bytesToMb(nc.mem_total_bytes),
        }),
        .y_min = 0,
        .y_max = 100,
        .y_unit = "%",
        .chart_height = 8,
        .tui_refresh_rate = state.tui_refresh_rate,
    };

    const charts_flow: ColumnLayout = .{
        .children = &.{ util_chart.widget(), mem_chart.widget() },
        .min_child_width = 40,
        .gap = 1,
    };

    // ── HBM Breakdown ───────────────────────────────────────
    const fmt_mb = struct {
        fn f(arena: std.mem.Allocator, val: ?u64) std.mem.Allocator.Error![]const u8 {
            return std.fmt.allocPrint(arena, " {d} MB", .{utils.bytesToMb(val)});
        }
    }.f;
    const hbm_card: MetricCard = .{
        .title = "HBM Breakdown",
        .lines = &.{
            .{ .label = "Tensors       ", .value = try fmt_mb(ctx.arena, nc.nc_tensors) },
            .{ .label = "Constants     ", .value = try fmt_mb(ctx.arena, nc.nc_constants) },
            .{ .label = "Model Code    ", .value = try fmt_mb(ctx.arena, nc.nc_model_code) },
            .{ .label = "Shared Scratch", .value = try fmt_mb(ctx.arena, nc.nc_shared_scratchpad) },
            .{ .label = "Nonshared     ", .value = try fmt_mb(ctx.arena, nc.nc_nonshared_scratchpad) },
            .{ .label = "Runtime       ", .value = try fmt_mb(ctx.arena, nc.nc_runtime) },
            .{ .label = "Driver        ", .value = try fmt_mb(ctx.arena, nc.nc_driver) },
            .{ .label = "DMA Rings     ", .value = try fmt_mb(ctx.arena, nc.nc_dma_rings) },
            .{ .label = "Collectives   ", .value = try fmt_mb(ctx.arena, nc.nc_collectives) },
            .{ .label = "Notifications ", .value = try fmt_mb(ctx.arena, nc.nc_notifications) },
            .{ .label = "Uncategorized ", .value = try fmt_mb(ctx.arena, nc.nc_uncategorized) },
        },
    };

    // ── Compose vertical layout ─────────────────────────────
    return common.pageFrame(ctx, .{
        .children = &.{
            header.widget(),
            charts_flow.widget(),
            hbm_card.widget(),
            process_table.widget(),
        },
        .gap = 1,
    }, w, content_w, parent_widget);
}
