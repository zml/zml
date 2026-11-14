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
    gpu: smi.Sample.Metal,
    parent_widget: vxfw.Widget,
) std.mem.Allocator.Error!vxfw.Surface {
    const header = try common.headerText(ctx.arena, id, device);

    const util_chart = try common.historyChart(ctx, state, id, state.history.util, .{
        .title = "GPU Utilization",
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}%", .{gpu.util_percent orelse 0}),
    });

    const mem_chart = try common.historyChart(ctx, state, id, state.history.mem_util, .{
        .title = "Memory",
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}%", .{utils.percent(gpu.mem_used_bytes, gpu.mem_total_bytes)}),
        .info_line = try std.fmt.allocPrint(ctx.arena, "In use {d} / {d} MB unified", .{
            utils.bytesToMb(gpu.mem_used_bytes), utils.bytesToMb(gpu.mem_total_bytes),
        }),
    });

    const temp_chart = try common.historyChart(ctx, state, id, state.history.temp, .{
        .title = "Temperature",
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}\u{00b0}C", .{gpu.temperature orelse 0}),
        .info_line = if (gpu.fan_speed_percent) |fan| try std.fmt.allocPrint(ctx.arena, "Fan {d}%", .{fan}) else "Fan N/A",
        .y_min = 20,
        .y_max = 100,
        .y_unit = "\u{00b0}C",
    });

    const power_mw = &state.history.power[id];
    var highest: u64 = 0;
    for (0..power_mw.len) |i| highest = @max(highest, power_mw.at(i));
    const y_max: u32 = std.math.lossyCast(u32, @max(2, (highest +| 999) / 1000));
    const power_chart: Chart = .{
        .title = "Power",
        .data = try utils.normalizeRange(ctx.arena, power_mw, 0, @as(u64, y_max) * 1000),
        .value_label = try watts(ctx.arena, gpu.power_mw),
        .info_line = try std.fmt.allocPrint(ctx.arena, "ANE {s}", .{try watts(ctx.arena, gpu.ane_power_mw)}),
        .y_min = 0,
        .y_max = y_max,
        .y_unit = "W",
        .chart_height = 8,
        .tui_refresh_rate = state.tui_refresh_rate,
    };

    const charts_flow: ColumnLayout = .{
        .children = &.{ ui.widget(&util_chart), ui.widget(&mem_chart), ui.widget(&temp_chart), ui.widget(&power_chart) },
        .min_child_width = 60,
        .gap = 1,
    };

    const clocks_card: MetricCard = .{ .title = "Clocks", .lines = &.{
        .{ .label = "Graphics ", .value = if (gpu.clock_graphics_mhz == null and gpu.clock_graphics_max_mhz == null)
            " N/A"
        else
            try std.fmt.allocPrint(ctx.arena, " {d} / {d} MHz", .{ gpu.clock_graphics_mhz orelse 0, gpu.clock_graphics_max_mhz orelse 0 }) },
    } };

    const power_card: MetricCard = .{ .title = "Power", .lines = &.{
        .{ .label = "GPU     ", .value = try std.fmt.allocPrint(ctx.arena, " {s}", .{try watts(ctx.arena, gpu.power_mw)}) },
        .{ .label = "ANE     ", .value = try std.fmt.allocPrint(ctx.arena, " {s}", .{try watts(ctx.arena, gpu.ane_power_mw)}) },
        .{ .label = "CPU     ", .value = try std.fmt.allocPrint(ctx.arena, " {s}", .{try watts(ctx.arena, gpu.cpu_power_mw)}) },
        .{ .label = "System  ", .value = try std.fmt.allocPrint(ctx.arena, " {s}", .{try watts(ctx.arena, gpu.system_power_mw)}) },
    } };

    const cards_flow: ColumnLayout = .{
        .children = &.{ ui.widget(&clocks_card), ui.widget(&power_card) },
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

fn watts(arena: std.mem.Allocator, milliwatts: ?u64) std.mem.Allocator.Error![]const u8 {
    const mw = milliwatts orelse return "N/A";

    return std.fmt.allocPrint(arena, "{d}.{d}W", .{ mw / 1000, mw % 1000 / 100 });
}
