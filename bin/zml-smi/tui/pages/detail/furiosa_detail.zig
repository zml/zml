const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const smi = @import("zml-smi");
const data = @import("../../data.zig");
const utils = @import("../../lib/utils.zig");
const ui = @import("../../lib/ui.zig");
const Chart = @import("../../widgets/chart.zig");
const ColumnLayout = @import("../../widgets/column_layout.zig");
const MetricCard = @import("../../widgets/metric_card.zig");
const ProcessTable = @import("../../widgets/process_table.zig");
const common = @import("detail_common.zig");

pub fn draw(
    state: *const data.SystemState,
    process_table: *ProcessTable,
    ctx: vxfw.DrawContext,
    w: u16,
    id: u16,
    device: smi.Device,
    chip: smi.Sample.Furiosa,
    parent_widget: vxfw.Widget,
) std.mem.Allocator.Error!vxfw.Surface {
    const header = try common.headerText(ctx.arena, id, device);

    const util_chart = try common.historyChart(ctx, state, id, state.history.util, .{
        .title = "NPU Utilization",
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}%", .{chip.util_percent orelse 0}),
    });

    const mem_chart = try common.historyChart(ctx, state, id, state.history.mem_util, .{
        .title = "Memory",
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}%", .{utils.percent(chip.mem_used_bytes, chip.mem_total_bytes)}),
        .info_line = try std.fmt.allocPrint(ctx.arena, "DRAM {d} / {d} MB", .{ utils.bytesToMb(chip.mem_used_bytes), utils.bytesToMb(chip.mem_total_bytes) }),
    });

    const temp_chart = try common.historyChart(ctx, state, id, state.history.temp, .{
        .title = "Temperature",
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}\u{00b0}C", .{chip.temperature orelse 0}),
        .info_line = try std.fmt.allocPrint(ctx.arena, "Ambient {d}\u{00b0}C", .{chip.ambient_temperature orelse 0}),
        .y_min = 20,
        .y_max = 100,
        .y_unit = "\u{00b0}C",
    });

    const power_limit_w = (chip.power_limit_mw orelse 0) / 1000;
    const y_max = if (power_limit_w > 0) power_limit_w else 1;
    const power_chart: Chart = .{
        .title = "Power",
        .data = try utils.normalizeRange(ctx.arena, &state.history.power[id], 0, y_max * 1000),
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}W", .{(chip.power_mw orelse 0) / 1000}),
        .info_line = if (power_limit_w > 0) try std.fmt.allocPrint(ctx.arena, "Limit {d}W", .{power_limit_w}) else "N/A",
        .y_min = 0,
        .y_max = @intCast(@min(y_max, std.math.maxInt(u32))),
        .y_unit = "W",
        .chart_height = 8,
        .tui_refresh_rate = state.tui_refresh_rate,
    };

    const charts_flow: ColumnLayout = .{
        .children = &.{ ui.widget(&util_chart), ui.widget(&mem_chart), ui.widget(&temp_chart), ui.widget(&power_chart) },
        .min_child_width = 60,
        .gap = 1,
    };

    const state_card: MetricCard = .{
        .title = "State",
        .lines = &.{
            .{ .label = "Liveness ", .value = if (chip.liveness) |value| try std.fmt.allocPrint(ctx.arena, " {t}", .{value}) else " N/A" },
            .{ .label = "Governor ", .value = if (chip.governor) |value| try std.fmt.allocPrint(ctx.arena, " {t}", .{value}) else " N/A" },
            .{ .label = "Throttle ", .value = try flags(ctx.arena, smi.Sample.Furiosa.ThrottleReason, chip.throttle_reasons) },
        },
    };

    var hw_lines: std.ArrayList(MetricCard.LineEntry) = .empty;

    try hw_lines.append(ctx.arena, .{
        .label = "Core     ",
        .value = if (chip.clock_core_mhz) |mhz| try std.fmt.allocPrint(ctx.arena, " {d} MHz", .{mhz}) else " N/A",
    });

    try hw_lines.append(ctx.arena, .{
        .label = "Link     ",
        .value = if (chip.pcie_link_gen != null or chip.pcie_link_width != null)
            try std.fmt.allocPrint(ctx.arena, " Gen{d} x{d}", .{ chip.pcie_link_gen orelse 0, chip.pcie_link_width orelse 0 })
        else
            " N/A",
    });

    if (chip.pcie_bandwidth_mbps) |bw| {
        try hw_lines.append(ctx.arena, .{
            .label = "BW       ",
            .value = try std.fmt.allocPrint(ctx.arena, " {d} Mb/s", .{bw}),
        });
    }

    const hardware_card: MetricCard = .{ .title = "Clocks & PCIe", .lines = hw_lines.items };

    const cards_flow: ColumnLayout = .{
        .children = &.{ ui.widget(&state_card), ui.widget(&hardware_card) },
        .min_child_width = 30,
    };

    return common.pageFrame(ctx, .{
        .children = &.{
            header.widget(),
            ui.widget(&charts_flow),
            ui.widget(&cards_flow),
            ui.widget(process_table),
        },
        .gap = 1,
    }, w, parent_widget);
}

fn flags(arena: std.mem.Allocator, comptime E: type, value: ?u64) std.mem.Allocator.Error![]const u8 {
    const mask = value orelse return " N/A";
    if (mask == 0) return " none";

    var text: std.Io.Writer.Allocating = .init(arena);

    for (0..64) |bit| {
        if (mask >> @intCast(bit) & 1 == 0) continue;

        const separator = if (text.written().len == 0) " " else ", ";
        if (std.enums.fromInt(E, bit)) |flag| {
            text.writer.print("{s}{t}", .{ separator, flag }) catch return error.OutOfMemory;
        } else {
            text.writer.print("{s}bit {d}", .{ separator, bit }) catch return error.OutOfMemory;
        }
    }

    return text.written();
}
