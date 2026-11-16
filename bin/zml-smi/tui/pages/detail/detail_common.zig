const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const smi = @import("zml-smi");
const data = @import("../../data.zig");
const theme = @import("../../theme.zig");
const utils = @import("../../lib/utils.zig");
const ui = @import("../../lib/ui.zig");
const compose = @import("../../lib/compose.zig");
const Chart = @import("../../widgets/chart.zig");
const ColumnLayout = @import("../../widgets/column_layout.zig");

/// "Device 0: NVIDIA GeForce RTX 5090 (615.71.09, CUDA 13.4)", with every detail.
pub fn headerText(arena: std.mem.Allocator, id: u16, device: smi.Device) std.mem.Allocator.Error!vxfw.Text {
    return .{
        .text = try std.fmt.allocPrint(arena, "Device {d}: {s}{s}", .{ id, device.name orelse "Unknown", try utils.versions(arena, device, .all) }),
        .style = theme.header_style,
        .softwrap = false,
        .overflow = .clip,
    };
}

/// Creates a Chart pre-filled with history data and standard defaults.
pub fn historyChart(
    ctx: vxfw.DrawContext,
    state: *const data.SystemState,
    id: u16,
    kind: data.Metric,
    opts: struct {
        title: ?[]const u8 = null,
        info_line: ?[]const u8 = null,
    },
) std.mem.Allocator.Error!Chart {
    const reading = state.sample(id).common();
    if (kind == .power) return powerChart(ctx, state, id, reading, opts.info_line);
    const history = try state.history(ctx.arena, id, kind);
    const temperature = kind == .temperature;
    const minimum: u32 = if (temperature) 20 else 0;
    return .{
        .title = opts.title orelse switch (kind) {
            .util => "Utilization",
            .memory => "Memory",
            .temperature => "Temperature",
            .power => unreachable,
        },
        .data = try utils.normalizeRange(ctx.arena, history, minimum, 100),
        .value_label = switch (kind) {
            .util => try metric(ctx.arena, reading.util_percent, "%"),
            .memory => try memoryPercent(ctx.arena, reading),
            .temperature => try metric(ctx.arena, reading.temperature, "\u{00b0}C"),
            .power => unreachable,
        },
        .info_line = opts.info_line,
        .y_min = minimum,
        .y_max = 100,
        .y_unit = if (temperature) "\u{00b0}C" else "%",
        .chart_height = 8,
        .sample_interval_ms = state.historyInterval(),
    };
}

pub fn metric(
    arena: std.mem.Allocator,
    value: ?u64,
    suffix: []const u8,
) std.mem.Allocator.Error![]const u8 {
    const number = value orelse return "N/A";
    return std.fmt.allocPrint(arena, "{d}{s}", .{ number, suffix });
}

pub fn memoryPercent(
    arena: std.mem.Allocator,
    reading: smi.Common,
) std.mem.Allocator.Error![]const u8 {
    const used = reading.mem_used_bytes orelse return "N/A";
    const total = reading.mem_total_bytes orelse return "N/A";
    if (total == 0) return "N/A";
    return std.fmt.allocPrint(arena, "{d}%", .{@as(u128, used) * 100 / total});
}

pub fn memoryInfo(
    arena: std.mem.Allocator,
    reading: smi.Common,
    prefix: []const u8,
    suffix: []const u8,
) std.mem.Allocator.Error![]const u8 {
    const used = if (reading.mem_used_bytes) |bytes|
        try std.fmt.allocPrint(arena, "{d}", .{utils.bytesToMb(bytes)})
    else
        "N/A";
    const total = if (reading.mem_total_bytes) |bytes|
        try std.fmt.allocPrint(arena, "{d}", .{utils.bytesToMb(bytes)})
    else
        "N/A";
    return std.fmt.allocPrint(arena, "{s}{s} / {s} MB{s}", .{ prefix, used, total, suffix });
}

pub fn watts(arena: std.mem.Allocator, milliwatts: ?u64) std.mem.Allocator.Error![]const u8 {
    const value = milliwatts orelse return "N/A";
    return std.fmt.allocPrint(arena, "{d}.{d}W", .{ value / 1000, value % 1000 / 100 });
}

/// Keeps missing samples as gaps and scales raw milliwatts without rounding them to watts.
pub fn powerChart(
    ctx: vxfw.DrawContext,
    state: *const data.SystemState,
    index: u16,
    reading: smi.Common,
    info_line: ?[]const u8,
) std.mem.Allocator.Error!Chart {
    const history = try state.history(ctx.arena, index, .power);
    var highest = reading.power_mw orelse 0;
    for (history) |value| {
        if (value) |milliwatts| highest = @max(highest, milliwatts);
    }
    const limit = reading.power_limit_mw orelse 0;
    const ceiling = if (limit > 0) limit else highest;
    const whole_watts = ceiling / 1000 + @intFromBool(ceiling % 1000 != 0);
    const maximum: u32 = std.math.lossyCast(u32, @max(2, whole_watts));
    return .{
        .title = "Power",
        .data = try utils.normalizeRange(ctx.arena, history, 0, @as(u64, maximum) * 1000),
        .value_label = try watts(ctx.arena, reading.power_mw),
        .info_line = info_line orelse if (limit > 0)
            try std.fmt.allocPrint(ctx.arena, "Limit {s}", .{try watts(ctx.arena, limit)})
        else
            "Limit N/A",
        .y_min = 0,
        .y_max = maximum,
        .y_unit = "W",
        .chart_height = 8,
        .sample_interval_ms = state.historyInterval(),
    };
}

/// Draws a page-level layout with standard margin (top=1, left/right=2) inside a full-width surface.
pub fn pageFrame(ctx: vxfw.DrawContext, layout: ColumnLayout, w: u16, parent_widget: vxfw.Widget) std.mem.Allocator.Error!vxfw.Surface {
    const padded = try compose.pad(ctx.arena, ui.widget(&layout), .{ .top = 1, .left = 2, .right = 2 });
    var surf = try padded.draw(ui.fixedWidth(ctx, w));
    surf.widget = parent_widget;
    return surf;
}
