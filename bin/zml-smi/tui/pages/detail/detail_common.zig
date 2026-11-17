const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const model = @import("zml-smi/model");
const data = @import("../../data.zig");
const utils = @import("../../lib/utils.zig");
const ui = @import("../../lib/ui.zig");
const compose = @import("../../lib/compose.zig");
const Chart = @import("../../widgets/chart.zig");
const ColumnLayout = @import("../../widgets/column_layout.zig");

/// Creates a Chart pre-filled with history data and standard defaults.
pub fn historyChart(
    ctx: vxfw.DrawContext,
    state: *const data.SystemState,
    id: u16,
    reading: model.Common,
    kind: data.Metric,
    opts: struct {
        title: ?[]const u8 = null,
        info_line: ?[]const u8 = null,
    },
) std.mem.Allocator.Error!Chart {
    const history = try state.history(ctx.arena, id, kind);
    const minimum: u32 = if (kind == .temperature) 20 else 0;
    const limit = reading.power_limit_mw orelse 0;
    const maximum: u32 = if (kind == .power) maximum_blk: {
        var ceiling = limit;
        if (limit == 0) {
            ceiling = reading.power_mw orelse 0;
            for (history) |value| ceiling = @max(ceiling, value orelse 0);
        }
        const whole_watts = ceiling / 1000 + @intFromBool(ceiling % 1000 != 0);
        break :maximum_blk std.math.lossyCast(u32, @max(2, whole_watts));
    } else 100;
    // Normalize power in milliwatts while labeling its axis in watts.
    const scale: u64 = if (kind == .power) 1000 else 1;
    const unit: []const u8 = switch (kind) {
        .util, .memory => "%",
        .temperature => "\u{00b0}C",
        .power => "W",
    };
    return .{
        .title = opts.title orelse switch (kind) {
            .util => "Utilization",
            .memory => "Memory",
            .temperature => "Temperature",
            .power => "Power",
        },
        .data = try utils.normalizeRange(ctx.arena, history, minimum, @as(u64, maximum) * scale),
        .value_label = switch (kind) {
            .util => try metric(ctx.arena, reading.util_percent, "%"),
            .memory => try memoryPercent(ctx.arena, reading),
            .temperature => try metric(ctx.arena, reading.temperature, unit),
            .power => try watts(ctx.arena, reading.power_mw),
        },
        .info_line = opts.info_line orelse if (kind != .power)
            null
        else if (limit > 0)
            try std.fmt.allocPrint(ctx.arena, "Limit {s}", .{try watts(ctx.arena, limit)})
        else
            "Limit N/A",
        .y_min = minimum,
        .y_max = maximum,
        .y_unit = unit,
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
    reading: model.Common,
) std.mem.Allocator.Error![]const u8 {
    const used = reading.mem_used_bytes orelse return "N/A";
    const total = reading.mem_total_bytes orelse return "N/A";
    if (total == 0) return "N/A";
    return std.fmt.allocPrint(arena, "{d}%", .{@as(u128, used) * 100 / total});
}

pub fn memoryInfo(
    arena: std.mem.Allocator,
    reading: model.Common,
    prefix: []const u8,
    suffix: []const u8,
) std.mem.Allocator.Error![]const u8 {
    return std.fmt.allocPrint(arena, "{s}{s} / {s} MB{s}", .{
        prefix, try memoryLabel(arena, reading.mem_used_bytes), try memoryLabel(arena, reading.mem_total_bytes), suffix,
    });
}

pub fn memoryLabel(arena: std.mem.Allocator, memory: ?u64) std.mem.Allocator.Error![]const u8 {
    return metric(arena, if (memory) |bytes| utils.bytesToMb(bytes) else null, "");
}

pub fn watts(arena: std.mem.Allocator, milliwatts: ?u64) std.mem.Allocator.Error![]const u8 {
    const value = milliwatts orelse return "N/A";
    return std.fmt.allocPrint(arena, "{d}.{d}W", .{ value / 1000, value % 1000 / 100 });
}

/// Draws a page-level layout with standard margin (top=1, left/right=2) inside a full-width surface.
pub fn pageFrame(ctx: vxfw.DrawContext, layout: ColumnLayout, w: u16, parent_widget: vxfw.Widget) std.mem.Allocator.Error!vxfw.Surface {
    const padded = try compose.pad(ctx.arena, ui.widget(&layout), .{ .top = 1, .left = 2, .right = 2 });
    var surf = try padded.draw(ui.fixedWidth(ctx, w));
    surf.widget = parent_widget;
    return surf;
}
