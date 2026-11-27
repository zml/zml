const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const smi = @import("zml-smi");
const data = @import("../data.zig");
const theme = @import("../theme.zig");
const utils = @import("../lib/utils.zig");
const ui = @import("../lib/ui.zig");
const image_cache = @import("../image_cache.zig");
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

const Usage = struct {
    label: []const u8,
    util_label: []const u8,
    util_percent: ?u64,
    mem_used: ?u64,
    mem_total: ?u64,
    suffix: []const u8 = "",
};

pub fn draw(self: *DeviceCard, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const i = self.device_id;
    const device = self.state.device(i);
    const usage: Usage = switch (self.state.sample(i)) {
        .cuda, .rocm, .oneapi => |gpu| .{
            .label = "GPU",
            .util_label = "GPU",
            .util_percent = gpu.util_percent,
            .mem_used = gpu.mem_used_bytes,
            .mem_total = gpu.mem_total_bytes,
            .suffix = try gpuSuffix(ctx.arena, gpu),
        },
        .neuron => |core| .{
            .label = "NC",
            .util_label = "Core",
            .util_percent = core.util_percent,
            .mem_used = core.mem_used_bytes,
            .mem_total = core.mem_total_bytes,
        },
        .tpu => |chip| .{
            .label = "TPU",
            .util_label = "Duty",
            .util_percent = chip.util_percent,
            .mem_used = chip.mem_used_bytes,
            .mem_total = chip.mem_total_bytes,
        },
    };
    const util_percent: u8 = @intCast(@min(usage.util_percent orelse 0, 100));
    const mem_used = usage.mem_used orelse 0;
    const mem_total = usage.mem_total orelse 0;
    const mem_pct: u8 = utils.percent(usage.mem_used, usage.mem_total);

    const card_title = try std.fmt.allocPrint(ctx.arena, "{s} {d}: {s}{s}", .{
        usage.label, i, device.name orelse "Unknown", try utils.versions(ctx.arena, device, .in_title),
    });

    const mem_used_mb = utils.bytesToMb(mem_used);
    const mem_total_mb = utils.bytesToMb(mem_total);
    const mem_total_str = try std.fmt.allocPrint(ctx.arena, "{d}", .{mem_total_mb});
    const mem_used_str = try std.fmt.allocPrint(ctx.arena, "{d}", .{mem_used_mb});

    const mem_pad = try utils.repeatStr(ctx.arena, " ", @intCast(mem_total_str.len -| mem_used_str.len));
    const mem_suffix = try std.fmt.allocPrint(ctx.arena, "{s}{s} / {s} MB", .{
        mem_pad, mem_used_str, mem_total_str,
    });

    var card: MetricCard = .{
        .title = card_title,
        .title_image = image_cache.global.get(device.target),
        .title_right = if (device.remote) "remote" else null,
        .cell_size = ctx.cell_size,
        .highlighted = self.highlighted,
    };

    const util_chart_data = try utils.normalizeRange(ctx.arena, &self.state.history.util[i], 0, 100);
    const mem_chart_data = try utils.normalizeRange(ctx.arena, &self.state.history.mem_util[i], 0, 100);

    const chart_entries = [2]MetricCard.ChartEntry{
        .{ .label = usage.util_label, .value = util_percent, .suffix = usage.suffix, .data = util_chart_data },
        .{ .label = "MEM", .value = mem_pct, .suffix = mem_suffix, .data = mem_chart_data },
    };
    const gauge_entries = [2]MetricCard.GaugeEntry{
        .{ .label = usage.util_label, .value = util_percent, .suffix = usage.suffix },
        .{ .label = "MEM", .value = mem_pct, .suffix = mem_suffix },
    };

    if (self.use_braille) {
        card.charts = &chart_entries;
    } else {
        card.gauges = &gauge_entries;
    }

    var card_surf = try card.draw(ctx);
    card_surf.widget = ui.widget(self);
    return card_surf;
}

fn gpuSuffix(arena: std.mem.Allocator, gpu: smi.Sample.Gpu) std.mem.Allocator.Error![]const u8 {
    const power_str = try std.fmt.allocPrint(arena, "{d}", .{(gpu.power_mw orelse 0) / 1000});
    const power_limit_str = try std.fmt.allocPrint(arena, "{d}", .{(gpu.power_limit_mw orelse 0) / 1000});
    const pad = try utils.repeatStr(arena, " ", @intCast(power_limit_str.len -| power_str.len));
    return std.fmt.allocPrint(arena, "{d}\u{00b0}C | {s}{s} W", .{ gpu.temperature orelse 0, pad, power_str });
}
