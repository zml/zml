const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../data.zig");
const ui = @import("../lib/ui.zig");
const Chart = @import("../widgets/chart.zig");
const ColumnLayout = @import("../widgets/column_layout.zig");
const ProcessTable = @import("../widgets/process_table.zig");
const common = @import("detail/detail_common.zig");
const cards = @import("detail/cards.zig");

const Detail = @This();

state: *const data.SystemState,
device_id: u16,
process_table: *ProcessTable = undefined,

pub fn draw(self: *const Detail, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const index = self.device_id;
    const sample = self.state.sample(index);
    const reading = sample.common();
    const header = try common.headerText(ctx.arena, index, self.state.device(index));
    const title: []const u8 = switch (sample) {
        .cuda, .rocm, .oneapi, .metal => "GPU Utilization",
        .furiosa => "NPU Utilization",
        .neuron => "Core Utilization",
        .tpu => "Duty Cycle",
    };
    const util_info = switch (sample) {
        .cuda, .rocm, .oneapi => |gpu| if (gpu.encoder_util_percent != null or
            gpu.decoder_util_percent != null)
            try std.fmt.allocPrint(ctx.arena, "Enc {s}  Dec {s}", .{
                try common.metric(ctx.arena, gpu.encoder_util_percent, "%"),
                try common.metric(ctx.arena, gpu.decoder_util_percent, "%"),
            })
        else
            null,
        else => null,
    };
    const memory_prefix: []const u8 = switch (sample) {
        .cuda, .rocm, .oneapi => "VRAM ",
        .metal => "In use ",
        .furiosa => "DRAM ",
        .neuron, .tpu => "HBM ",
    };
    const memory_suffix: []const u8 = switch (sample) {
        .cuda, .rocm, .oneapi => |gpu| if (gpu.mem_bus_width) |width|
            try std.fmt.allocPrint(ctx.arena, " ({d}-bit)", .{width})
        else
            "",
        .metal => " unified",
        else => "",
    };

    const chart_count: usize = if (sample == .neuron or sample == .tpu) 2 else 4;
    var charts: [4]Chart = undefined;
    charts[0] = try common.historyChart(ctx, self.state, index, reading, .util, .{
        .title = title,
        .info_line = util_info,
    });
    charts[1] = try common.historyChart(ctx, self.state, index, reading, .memory, .{
        .info_line = try common.memoryInfo(ctx.arena, reading, memory_prefix, memory_suffix),
    });
    if (chart_count == 4) {
        const temperature_info = switch (sample) {
            .furiosa => |chip| try std.fmt.allocPrint(ctx.arena, "Ambient {s}", .{
                try common.metric(ctx.arena, chip.ambient_temperature, "\u{00b0}C"),
            }),
            else => try std.fmt.allocPrint(ctx.arena, "Fan {s}", .{
                try common.metric(ctx.arena, reading.fan_speed_percent, "%"),
            }),
        };
        charts[2] = try common.historyChart(ctx, self.state, index, reading, .temperature, .{
            .info_line = temperature_info,
        });
        const power_info = if (sample == .metal)
            try std.fmt.allocPrint(ctx.arena, "ANE {s}", .{
                try common.watts(ctx.arena, sample.metal.ane_power_mw),
            })
        else
            null;
        charts[3] = try common.historyChart(ctx, self.state, index, reading, .power, .{ .info_line = power_info });
    }
    var chart_widgets: [4]vxfw.Widget = undefined;
    for (charts[0..chart_count], 0..) |*chart, position| chart_widgets[position] = ui.widget(chart);
    const chart_layout: ColumnLayout = .{
        .children = chart_widgets[0..chart_count],
        .min_child_width = if (chart_count == 4) 60 else 40,
        .gap = 1,
    };

    const extra = try cards.forSample(ctx.arena, sample, self.state.host());
    const card_widgets = try ctx.arena.alloc(vxfw.Widget, extra.len);
    for (extra, card_widgets) |*card, *widget| widget.* = ui.widget(card);
    const card_layout: ColumnLayout = .{ .children = card_widgets, .min_child_width = 30 };
    var sections = [_]vxfw.Widget{
        header.widget(), ui.widget(&chart_layout), ui.widget(&card_layout), ui.widget(self.process_table),
    };
    if (extra.len == 0) sections[2] = sections[3];
    const count: usize = if (extra.len == 0) 3 else 4;
    return common.pageFrame(ctx, .{ .children = sections[0..count], .gap = 1 }, ctx.max.width orelse 80, ui.widget(self));
}
