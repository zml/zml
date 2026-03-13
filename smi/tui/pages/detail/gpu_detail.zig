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
    gpu: *data.GpuInfo,
    parent_widget: vxfw.Widget,
) std.mem.Allocator.Error!vxfw.Surface {
    const content_w: u16 = w -| 4;

    // ── Header ──────────────────────────────────────────────
    const header = try common.headerText(ctx.arena, id, utils.strSlice(&gpu.name));

    // ── GPU Utilization + Memory charts ─────────────────────
    const gpu_chart: Chart = .{
        .title = "GPU Utilization",
        .data = try utils.normalizeRange(ctx.arena, try state.history.util[id].sliceLast(ctx.arena, data.history_len), 0, 100),
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}%", .{gpu.util_percent orelse 0}),
        .info_line = try std.fmt.allocPrint(ctx.arena, "Enc {d}%  Dec {d}%", .{
            gpu.encoder_util_percent orelse 0, gpu.decoder_util_percent orelse 0,
        }),
        .y_min = 0,
        .y_max = 100,
        .y_unit = "%",
        .chart_height = 8,
        .tui_refresh_rate = state.tui_refresh_rate,
    };
    const mem_chart: Chart = .{
        .title = "Memory",
        .data = try utils.normalizeRange(ctx.arena, try state.history.mem_util[id].sliceLast(ctx.arena, data.history_len), 0, 100),
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}%", .{if (gpu.mem_total_bytes) |total| if (total > 0) (gpu.mem_used_bytes orelse 0) * 100 / total else 0 else 0}),
        .info_line = if (gpu.mem_bus_width) |bw|
            try std.fmt.allocPrint(ctx.arena, "VRAM {d} / {d} MB ({d}-bit)", .{
                utils.bytesToMb(gpu.mem_used_bytes), utils.bytesToMb(gpu.mem_total_bytes), bw,
            })
        else
            try std.fmt.allocPrint(ctx.arena, "VRAM {d} / {d} MB", .{
                utils.bytesToMb(gpu.mem_used_bytes), utils.bytesToMb(gpu.mem_total_bytes),
            }),
        .y_min = 0,
        .y_max = 100,
        .y_unit = "%",
        .chart_height = 8,
        .tui_refresh_rate = state.tui_refresh_rate,
    };

    // ── Temperature + Power charts ──────────────────────────
    const temp_chart: Chart = .{
        .title = "Temperature",
        .data = try utils.normalizeRange(ctx.arena, try state.history.temp[id].sliceLast(ctx.arena, data.history_len), 20, 100),
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}\u{00b0}C", .{gpu.temperature orelse 0}),
        .info_line = try std.fmt.allocPrint(ctx.arena, "Fan {d}%", .{gpu.fan_speed_percent orelse 0}),
        .y_min = 20,
        .y_max = 100,
        .y_unit = "\u{00b0}C",
        .chart_height = 8,
        .tui_refresh_rate = state.tui_refresh_rate,
    };

    const power_limit_mw = gpu.power_limit_mw orelse 0;
    const power_limit_w = power_limit_mw / 1000;
    const power_raw = try state.history.power[id].sliceLast(ctx.arena, data.history_len);
    const power_watts = try ctx.arena.alloc(u64, power_raw.len);
    for (power_raw, 0..) |p, j| power_watts[j] = p / 1000;
    const y_max = if (power_limit_w > 0) power_limit_w else 1;
    const power_chart: Chart = .{
        .title = "Power",
        .data = try utils.normalizeRange(ctx.arena, power_watts, 0, y_max),
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}W", .{(gpu.power_mw orelse 0) / 1000}),
        .info_line = if (power_limit_w > 0)
            try std.fmt.allocPrint(ctx.arena, "Limit {d}W", .{power_limit_w})
        else
            "N/A",
        .y_min = 0,
        .y_max = @intCast(@min(y_max, std.math.maxInt(u32))),
        .y_unit = "W",
        .chart_height = 8,
        .tui_refresh_rate = state.tui_refresh_rate,
    };

    const charts_flow: ColumnLayout = .{
        .children = &.{ gpu_chart.widget(), mem_chart.widget(), temp_chart.widget(), power_chart.widget() },
        .min_child_width = 60,
        .gap = 1,
    };

    // ── Clocks + PCIe ───────────────────────────────────────
    const clocks_card: MetricCard = .{
        .title = "Clocks",
        .lines = &.{
            .{ .label = "Graphics ", .value = try std.fmt.allocPrint(ctx.arena, " {d} / {d} MHz", .{
                gpu.clock_graphics_mhz orelse 0, gpu.clock_graphics_max_mhz orelse 0,
            }) },
            .{ .label = "SM       ", .value = try std.fmt.allocPrint(ctx.arena, " {d} MHz", .{gpu.clock_sm_mhz orelse 0}) },
            .{ .label = "Memory   ", .value = try std.fmt.allocPrint(ctx.arena, " {d} / {d} MHz", .{
                gpu.clock_mem_mhz orelse 0, gpu.clock_mem_max_mhz orelse 0,
            }) },
        },
    };
    const pcie_card: MetricCard = .{
        .title = "PCIe",
        .lines = &.{
            .{ .label = "Link     ", .value = try std.fmt.allocPrint(ctx.arena, " Gen{d} x{d}", .{
                gpu.pcie_link_gen orelse 0, gpu.pcie_link_width orelse 0,
            }) },
            .{ .label = "TX       ", .value = try utils.formatBandwidth(ctx.arena, gpu.pcie_tx_kbps orelse 0) },
            .{ .label = "RX       ", .value = try utils.formatBandwidth(ctx.arena, gpu.pcie_rx_kbps orelse 0) },
        },
    };

    const cards_flow: ColumnLayout = .{
        .children = &.{ clocks_card.widget(), pcie_card.widget() },
        .min_child_width = 30,
    };

    // ── Compose vertical layout ─────────────────────────────
    const layout: ColumnLayout = .{
        .children = &.{
            header.widget(),
            charts_flow.widget(),
            cards_flow.widget(),
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
