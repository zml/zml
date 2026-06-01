const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
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
    tt: data.TenstorrentInfo,
    parent_widget: vxfw.Widget,
) std.mem.Allocator.Error!vxfw.Surface {
    // ── Header ──────────────────────────────────────────────
    const name_suffix: []const u8 = if (tt.driver_version) |v|
        try std.fmt.allocPrint(ctx.arena, " (fw {s})", .{v})
    else
        "";
    const header = try common.headerText(ctx.arena, id, try std.fmt.allocPrint(ctx.arena, "{s}{s}", .{ tt.name orelse "Unknown", name_suffix }));

    // ── Temperature + Power charts (the only live, bounded metrics) ─────────
    const temp_chart = try common.historyChart(ctx, state, id, state.history.temp, .{
        .title = "Temperature",
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}\u{00b0}C", .{tt.temperature orelse 0}),
        .info_line = if (tt.temperature_max) |m| try std.fmt.allocPrint(ctx.arena, "Limit {d}\u{00b0}C", .{m}) else null,
        .y_min = 20,
        .y_max = 100,
        .y_unit = "\u{00b0}C",
    });

    const power_limit_w = (tt.power_limit_mw orelse 0) / 1000;
    const power_raw = try state.history.power[id].sliceLast(ctx.arena, data.history_len);
    const power_watts = try ctx.arena.alloc(u64, power_raw.len);
    for (power_raw, 0..) |p, j| power_watts[j] = p / 1000;
    const y_max = if (power_limit_w > 0) power_limit_w else 1;
    const power_chart: Chart = .{
        .title = "Power",
        .data = try utils.normalizeRange(ctx.arena, power_watts, 0, y_max),
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}W", .{(tt.power_mw orelse 0) / 1000}),
        .info_line = if (power_limit_w > 0) try std.fmt.allocPrint(ctx.arena, "Limit {d}W", .{power_limit_w}) else "N/A",
        .y_min = 0,
        .y_max = @intCast(@min(y_max, std.math.maxInt(u32))),
        .y_unit = "W",
        .chart_height = 8,
        .tui_refresh_rate = state.tui_refresh_rate,
    };

    const charts_flow: ColumnLayout = .{
        .children = &.{ ui.widget(&temp_chart), ui.widget(&power_chart) },
        .min_child_width = 60,
        .gap = 1,
    };

    // ── Clocks ──────────────────────────────────────────────
    var clock_lines: std.ArrayList(MetricCard.LineEntry) = .empty;
    try clock_lines.append(ctx.arena, .{ .label = "AI   ", .value = try fmtMhz(ctx.arena, tt.clock_ai_mhz) });
    try clock_lines.append(ctx.arena, .{ .label = "ARC  ", .value = try fmtMhz(ctx.arena, tt.clock_arc_mhz) });
    try clock_lines.append(ctx.arena, .{ .label = "AXI  ", .value = try fmtMhz(ctx.arena, tt.clock_axi_mhz) });
    if (tt.clock_mem_mhz != null) {
        try clock_lines.append(ctx.arena, .{ .label = "Mem  ", .value = try fmtMhz(ctx.arena, tt.clock_mem_mhz) });
    }
    const clocks_card: MetricCard = .{ .title = "Clocks", .lines = clock_lines.items };

    // ── Sensors (voltage / current / extra temps) ───────────
    var sensor_lines: std.ArrayList(MetricCard.LineEntry) = .empty;
    try sensor_lines.append(ctx.arena, .{ .label = "Vcore  ", .value = try fmtMilli(ctx.arena, tt.voltage_mv, "V") });
    try sensor_lines.append(ctx.arena, .{ .label = "Current", .value = try fmtMilli(ctx.arena, tt.current_ma, "A") });
    if (tt.board_temperature != null) {
        try sensor_lines.append(ctx.arena, .{ .label = "Board T", .value = try fmtTemp(ctx.arena, tt.board_temperature) });
    }
    if (tt.dram_temperature != null) {
        try sensor_lines.append(ctx.arena, .{ .label = "DRAM T ", .value = try fmtTemp(ctx.arena, tt.dram_temperature) });
    }
    if (tt.fan_rpm) |rpm| {
        try sensor_lines.append(ctx.arena, .{ .label = "Fan    ", .value = try std.fmt.allocPrint(ctx.arena, " {d} RPM", .{rpm}) });
    }
    const sensors_card: MetricCard = .{ .title = "Sensors", .lines = sensor_lines.items };

    // ── PCIe ────────────────────────────────────────────────
    var pcie_lines: std.ArrayList(MetricCard.LineEntry) = .empty;
    try pcie_lines.append(ctx.arena, .{ .label = "Link ", .value = try fmtPcieLink(ctx.arena, tt.pcie_link_gen, tt.pcie_link_width) });
    if (tt.pcie_bandwidth_mbps) |bw| {
        try pcie_lines.append(ctx.arena, .{ .label = "BW   ", .value = try std.fmt.allocPrint(ctx.arena, " {d} Mb/s", .{bw}) });
    }
    const pcie_card: MetricCard = .{ .title = "PCIe", .lines = pcie_lines.items };

    // ── Board / firmware ────────────────────────────────────
    var board_lines: std.ArrayList(MetricCard.LineEntry) = .empty;
    if (tt.mem_total_bytes) |total| {
        try board_lines.append(ctx.arena, .{ .label = "GDDR6   ", .value = try std.fmt.allocPrint(ctx.arena, " {d} GB", .{total / (1024 * 1024 * 1024)}) });
    }
    try board_lines.append(ctx.arena, .{ .label = "Serial  ", .value = try fmtStr(ctx.arena, tt.board_serial) });
    try board_lines.append(ctx.arena, .{ .label = "ASIC ID ", .value = try fmtStr(ctx.arena, tt.asic_id) });
    try board_lines.append(ctx.arena, .{ .label = "Driver  ", .value = try fmtStr(ctx.arena, tt.driver_version) });
    try board_lines.append(ctx.arena, .{ .label = "FW bndl ", .value = try fmtStr(ctx.arena, tt.fw_bundle_version) });
    try board_lines.append(ctx.arena, .{ .label = "CM fw   ", .value = try fmtStr(ctx.arena, tt.cm_fw_version) });
    try board_lines.append(ctx.arena, .{ .label = "ETH fw  ", .value = try fmtStr(ctx.arena, tt.eth_fw_version) });
    try board_lines.append(ctx.arena, .{ .label = "DM app  ", .value = try fmtStr(ctx.arena, tt.dm_app_version) });
    if (tt.heartbeat) |hb| {
        try board_lines.append(ctx.arena, .{ .label = "Heartbeat", .value = try std.fmt.allocPrint(ctx.arena, " {d}", .{hb}) });
    }
    const board_card: MetricCard = .{ .title = "Board", .lines = board_lines.items };

    const cards_flow: ColumnLayout = .{
        .children = &.{ ui.widget(&clocks_card), ui.widget(&sensors_card), ui.widget(&pcie_card), ui.widget(&board_card) },
        .min_child_width = 30,
        .col_gap = 2,
        .gap = 1,
    };

    // ── Compose vertical layout ─────────────────────────────
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

fn fmtMhz(arena: std.mem.Allocator, v: ?u64) std.mem.Allocator.Error![]const u8 {
    if (v) |mhz| return std.fmt.allocPrint(arena, " {d} MHz", .{mhz});
    return " N/A";
}

fn fmtMilli(arena: std.mem.Allocator, v: ?u64, unit: []const u8) std.mem.Allocator.Error![]const u8 {
    if (v) |milli| return std.fmt.allocPrint(arena, " {d}.{d:0>3} {s}", .{ milli / 1000, milli % 1000, unit });
    return " N/A";
}

fn fmtTemp(arena: std.mem.Allocator, v: ?u64) std.mem.Allocator.Error![]const u8 {
    if (v) |t| return std.fmt.allocPrint(arena, " {d}\u{00b0}C", .{t});
    return " N/A";
}

fn fmtPcieLink(arena: std.mem.Allocator, gen: ?u64, width: ?u64) std.mem.Allocator.Error![]const u8 {
    if (gen == null and width == null) return " N/A";
    return std.fmt.allocPrint(arena, " Gen{d} x{d}", .{ gen orelse 0, width orelse 0 });
}

fn fmtStr(arena: std.mem.Allocator, v: ?[]const u8) std.mem.Allocator.Error![]const u8 {
    if (v) |s| return std.fmt.allocPrint(arena, " {s}", .{s});
    return " N/A";
}
