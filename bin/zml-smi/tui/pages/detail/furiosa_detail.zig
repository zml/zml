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
    chip: smi.Furiosa,
    parent_widget: vxfw.Widget,
) std.mem.Allocator.Error!vxfw.Surface {
    const header = try common.headerText(ctx.arena, id, device);

    const util_chart = try common.historyChart(ctx, state, id, state.history.util, .{
        .title = "NPU Utilization",
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}%", .{chip.util_percent orelse 0}),
    });

    const mem_used = chip.mem_used_bytes orelse 0;
    const mem_total = chip.mem_total_bytes orelse 0;
    const mem_chart = try common.historyChart(ctx, state, id, state.history.mem_util, .{
        .title = "Memory",
        .value_label = try std.fmt.allocPrint(ctx.arena, "{d}%", .{if (mem_total > 0) mem_used * 100 / mem_total else 0}),
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
    const power_raw = state.history.power[id].items;
    const power_watts = try ctx.arena.alloc(u64, power_raw.len);
    for (power_raw, power_watts) |mw, *watts| watts.* = mw / 1000;
    const y_max = if (power_limit_w > 0) power_limit_w else 1;
    const power_chart: Chart = .{
        .title = "Power",
        .data = try utils.normalizeRange(ctx.arena, power_watts, 0, y_max),
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
            .{ .label = "Liveness ", .value = try code(ctx.arena, smi.Furiosa.Liveness, chip.liveness) },
            .{ .label = "Governor ", .value = try code(ctx.arena, smi.Furiosa.Governor, chip.governor) },
            .{ .label = "Throttle ", .value = try flags(ctx.arena, smi.Furiosa.ThrottleReason, chip.throttle_reasons) },
        },
    };

    return common.pageFrame(ctx, .{
        .children = &.{
            header.widget(),
            ui.widget(&charts_flow),
            ui.widget(&state_card),
            ui.widget(process_table),
        },
        .gap = 1,
    }, w, parent_widget);
}

/// The name of a code, or its number when it has none.
fn code(arena: std.mem.Allocator, comptime E: type, value: ?u64) std.mem.Allocator.Error![]const u8 {
    const number = value orelse return " N/A";
    if (std.enums.fromInt(E, number)) |known| return std.fmt.allocPrint(arena, " {t}", .{known});
    return std.fmt.allocPrint(arena, " {d}", .{number});
}

/// The names of the bits set in a bitmask, or their positions when they have none.
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
