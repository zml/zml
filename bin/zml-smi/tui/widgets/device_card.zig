const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const smi = @import("zml-smi");
const data = @import("../data.zig");
const theme = @import("../theme.zig");
const utils = @import("../lib/utils.zig");
const ui = @import("../lib/ui.zig");
const image_cache = @import("../image_cache.zig");
const Gauge = @import("gauge.zig");
const MetricCard = @import("metric_card.zig");

const DeviceCard = @This();

device_id: u16 = 0,
highlighted: bool = false,

state: *const data.SystemState = undefined,
use_braille: bool = false,
viewing_device: *?u16 = undefined,

pub fn handleEvent(self: *DeviceCard, ctx: *vxfw.EventContext, event: vxfw.Event) anyerror!void {
    switch (event) {
        .mouse => |mouse| {
            if (mouse.type == .press and mouse.button == .left) {
                self.viewing_device.* = self.device_id;
                return ctx.consumeAndRedraw();
            }
        },
        .mouse_enter => {
            self.highlighted = true;
            try ctx.setMouseShape(.pointer);
            return ctx.consumeAndRedraw();
        },
        .mouse_leave => {
            self.highlighted = false;
            try ctx.setMouseShape(.default);
            ctx.redraw = true;
        },
        else => {},
    }
}

/// Platform-specific names for the common utilization and memory readings.
const Usage = struct {
    label: []const u8,
    util_label: []const u8,
};

pub fn draw(self: *DeviceCard, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const i = self.device_id;
    const device = self.state.device(i);
    const common = self.state.sample(i).common();
    const usage: Usage = switch (device.target) {
        .neuron => .{
            .label = "NC",
            .util_label = "Core",
        },
        .tpu => .{
            .label = "TPU",
            .util_label = "Duty",
        },
        .furiosa => .{
            .label = "NPU",
            .util_label = "NPU",
        },
        .cuda, .rocm, .oneapi, .metal => .{
            .label = "GPU",
            .util_label = "GPU",
        },
    };
    const suffix = switch (device.target) {
        .neuron, .tpu => "",
        .metal => try metalSuffix(ctx.arena, common),
        else => try gpuSuffix(ctx.arena, common),
    };
    const util_percent: ?u8 = if (common.util_percent) |value|
        @as(u8, @intCast(@min(value, 100)))
    else
        null;
    const mem_pct: ?u8 = if (data.memoryPercent(common)) |value|
        @as(u8, @intCast(value))
    else
        null;

    const card_title = try std.fmt.allocPrint(ctx.arena, "{s} {d}: {s}{s}", .{
        usage.label, i, device.name orelse "Unknown", try utils.versions(ctx.arena, device, .in_title),
    });

    const mem_total_str = try memoryLabel(ctx.arena, common.mem_total_bytes);
    const mem_used_str = try memoryLabel(ctx.arena, common.mem_used_bytes);

    const mem_pad = try utils.repeatStr(ctx.arena, " ", @intCast(mem_total_str.len -| mem_used_str.len));
    const mem_suffix = try std.fmt.allocPrint(ctx.arena, "{s}{s} / {s} MB", .{
        mem_pad, mem_used_str, mem_total_str,
    });

    var gauges = [2]Gauge{
        .{ .label = usage.util_label, .value = util_percent, .suffix = suffix },
        .{ .label = "MEM", .value = mem_pct, .suffix = mem_suffix },
    };
    if (self.use_braille) {
        for (&gauges, [_]data.Metric{ .util, .memory }) |*gauge, metric| {
            gauge.history = try utils.normalizeRange(
                ctx.arena,
                try self.state.history(ctx.arena, i, metric),
                0,
                100,
            );
        }
    }

    const card: MetricCard = .{
        .title = card_title,
        .title_image = image_cache.global.get(device.target),
        .title_right = if (device.remote) "remote" else null,
        .cell_size = ctx.cell_size,
        .gauges = &gauges,
        .highlighted = self.highlighted,
    };

    var card_surf = try card.draw(ctx);
    card_surf.widget = ui.widget(self);
    return card_surf;
}

/// "41°C |  10 W", the power padded to the width of the limit.
fn gpuSuffix(arena: std.mem.Allocator, common: smi.Common) std.mem.Allocator.Error![]const u8 {
    const temperature = try temperatureLabel(arena, common.temperature);
    const power_str = if (common.power_mw) |power|
        try std.fmt.allocPrint(arena, "{d}", .{power / 1000})
    else
        "N/A";
    const power_limit_str = if (common.power_limit_mw) |limit|
        try std.fmt.allocPrint(arena, "{d}", .{limit / 1000})
    else
        power_str;
    const pad = try utils.repeatStr(arena, " ", @intCast(power_limit_str.len -| power_str.len));
    return std.fmt.allocPrint(arena, "{s} | {s}{s} W", .{ temperature, pad, power_str });
}

fn metalSuffix(arena: std.mem.Allocator, common: smi.Common) std.mem.Allocator.Error![]const u8 {
    const temperature = try temperatureLabel(arena, common.temperature);
    const power = if (common.power_mw) |milliwatts|
        try std.fmt.allocPrint(arena, "{d}.{d} W", .{ milliwatts / 1000, milliwatts % 1000 / 100 })
    else
        "N/A";
    return std.fmt.allocPrint(arena, "{s} | {s}", .{ temperature, power });
}

fn temperatureLabel(arena: std.mem.Allocator, temperature: ?u64) std.mem.Allocator.Error![]const u8 {
    return if (temperature) |value|
        std.fmt.allocPrint(arena, "{d}\u{00b0}C", .{value})
    else
        "N/A";
}

fn memoryLabel(arena: std.mem.Allocator, memory: ?u64) std.mem.Allocator.Error![]const u8 {
    return if (memory) |value|
        std.fmt.allocPrint(arena, "{d}", .{value / (1024 * 1024)})
    else
        "N/A";
}
