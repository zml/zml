const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../data.zig");
const theme = @import("../theme.zig");
const utils = @import("../lib/utils.zig");
const ui = @import("../lib/ui.zig");
const image_cache = @import("../image_cache.zig");
const MetricCard = @import("metric_card.zig");

const DeviceCard = @This();

device_id: u8 = 0,
highlighted: bool = false,

state: *const data.SystemState = undefined,
use_braille: bool = false,
viewing_device: *?u8 = undefined,

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

pub fn draw(self: *DeviceCard, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const i = self.device_id;
    const dev = self.state.devices[i];
    const target: data.Target = std.meta.activeTag(dev.*);
    const cf = switch (dev.*) {
        inline else => |*sv| blk: {
            const f = sv.front().*;
            break :blk .{
                .name = f.name orelse "Unknown",
                .util_percent = @as(u8, @intCast(@min(f.util_percent orelse 0, 100))),
                .mem_used = f.mem_used_bytes orelse 0,
                .mem_total = f.mem_total_bytes orelse 0,
            };
        },
    };
    const mem_pct: u8 = if (cf.mem_total > 0) @intCast(@min(cf.mem_used * 100 / cf.mem_total, 100)) else 0;

    const card_title = try std.fmt.allocPrint(ctx.arena, "{s} {d}: {s}", .{ target.deviceLabel(), i, cf.name });
    const util_label = target.utilLabel();

    const dev_suffix = switch (dev.*) {
        .cuda, .rocm => |*sv| blk: {
            const gpu = sv.front().*;
            const power = (gpu.power_mw orelse 0) / 1000;
            const power_limit = (gpu.power_limit_mw orelse 0) / 1000;

            const power_limit_str = try std.fmt.allocPrint(ctx.arena, "{d}", .{power_limit});
            const power_str = try std.fmt.allocPrint(ctx.arena, "{d}", .{power});

            const pad = try utils.repeatStr(ctx.arena, " ", @intCast(power_limit_str.len -| power_str.len));
            break :blk try std.fmt.allocPrint(ctx.arena, "{d}\u{00b0}C | {s}{s} W", .{
                gpu.temperature orelse 0, pad, power_str,
            });
        },
        else => "",
    };

    const mem_used_mb = utils.bytesToMb(cf.mem_used);
    const mem_total_mb = utils.bytesToMb(cf.mem_total);
    const mem_total_str = try std.fmt.allocPrint(ctx.arena, "{d}", .{mem_total_mb});
    const mem_used_str = try std.fmt.allocPrint(ctx.arena, "{d}", .{mem_used_mb});

    const mem_pad = try utils.repeatStr(ctx.arena, " ", @intCast(mem_total_str.len -| mem_used_str.len));
    const mem_suffix = try std.fmt.allocPrint(ctx.arena, "{s}{s} / {s} MB", .{
        mem_pad, mem_used_str, mem_total_str,
    });

    var card: MetricCard = .{
        .title = card_title,
        .title_image = image_cache.global.get(switch (dev.*) {
            .cuda => "gpu_cuda",
            .rocm => "gpu_rocm",
            .neuron => "gpu_neuron",
            .tpu => "gpu_tpu",
        }),
        .cell_size = ctx.cell_size,
        .highlighted = self.highlighted,
    };

    const util_hist = try self.state.history.util[i].sliceLast(ctx.arena, data.history_len);
    const mem_hist = try self.state.history.mem_util[i].sliceLast(ctx.arena, data.history_len);
    const util_chart_data = try utils.normalizeRange(ctx.arena, util_hist, 0, 100);
    const mem_chart_data = try utils.normalizeRange(ctx.arena, mem_hist, 0, 100);

    const chart_entries = [2]MetricCard.ChartEntry{
        .{ .label = util_label, .value = cf.util_percent, .suffix = dev_suffix, .data = util_chart_data },
        .{ .label = "MEM", .value = mem_pct, .suffix = mem_suffix, .data = mem_chart_data },
    };
    const gauge_entries = [2]MetricCard.GaugeEntry{
        .{ .label = util_label, .value = cf.util_percent, .suffix = dev_suffix },
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
