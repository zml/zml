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

pub fn draw(self: *DeviceCard, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const i = self.device_id;
    const dev = self.state.devices[i];
    const target: data.Target = std.meta.activeTag(dev.*);
    // Every backend's info struct carries a name; util/mem are read per-branch
    // below, where they exist (Tenstorrent has neither).
    const name = switch (dev.*) {
        inline else => |*sv| sv.front().name orelse "Unknown",
    };

    const device_label, const util_label = switch (target) {
        .cuda, .rocm, .oneapi => .{ "GPU", "GPU" },
        .neuron => .{ "NC", "Core" },
        .tpu => .{ "TPU", "Duty" },
        .tenstorrent => .{ "TT", "TEMP" },
    };
    const driver_ver: []const u8 = switch (dev.*) {
        .cuda => |*sv| blk: {
            const gpu = sv.front();
            const drv = gpu.driver_version orelse "";
            const cuda = gpu.cuda_driver_version orelse "";
            break :blk if (drv.len > 0 and cuda.len > 0)
                try std.fmt.allocPrint(ctx.arena, " ({s}, CUDA {s})", .{ drv, cuda })
            else if (drv.len > 0)
                try std.fmt.allocPrint(ctx.arena, " ({s})", .{drv})
            else if (cuda.len > 0)
                try std.fmt.allocPrint(ctx.arena, " (CUDA {s})", .{cuda})
            else
                @as([]const u8, "");
        },
        .rocm, .oneapi => |*sv| if (sv.front().driver_version) |v| try std.fmt.allocPrint(ctx.arena, " ({s})", .{v}) else "",
        .neuron => |*sv| if (sv.front().driver_version) |v| try std.fmt.allocPrint(ctx.arena, " ({s})", .{v}) else "",
        .tenstorrent => |*sv| if (sv.front().driver_version) |v| try std.fmt.allocPrint(ctx.arena, " ({s})", .{v}) else "",
        else => "",
    };
    const card_title = try std.fmt.allocPrint(ctx.arena, "{s} {d}: {s}{s}", .{ device_label, i, name, driver_ver });

    const dev_suffix = switch (dev.*) {
        .cuda, .rocm, .oneapi => |*sv| blk: {
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

    var card: MetricCard = .{
        .title = card_title,
        .title_image = image_cache.global.get(switch (dev.*) {
            .cuda => "gpu_cuda",
            .rocm => "gpu_rocm",
            .oneapi => "gpu_oneapi",
            .neuron => "gpu_neuron",
            .tpu => "gpu_tpu",
            .tenstorrent => "gpu_tenstorrent",
        }),
        .title_right = if (self.state.devices[i].isRemote()) "remote" else null,
        .cell_size = ctx.cell_size,
        .highlighted = self.highlighted,
    };

    // Tenstorrent has no util/mem-usage telemetry, so its card shows the two
    // gauges that are real and bounded: temperature and power (vs their limits).
    var chart_entries: [2]MetricCard.ChartEntry = undefined;
    var gauge_entries: [2]MetricCard.GaugeEntry = undefined;

    switch (dev.*) {
        .tenstorrent => |*sv| {
            const tt = sv.front().*;
            const temp = tt.temperature orelse 0;
            const temp_max = tt.temperature_max orelse 100;
            const temp_pct: u8 = @intCast(@min(if (temp_max > 0) temp * 100 / temp_max else 0, 100));
            const power_w = (tt.power_mw orelse 0) / 1000;
            const power_lim_w = (tt.power_limit_mw orelse 0) / 1000;
            const power_pct: u8 = @intCast(@min(if (power_lim_w > 0) power_w * 100 / power_lim_w else 0, 100));
            const temp_suffix = try std.fmt.allocPrint(ctx.arena, "{d} / {d} \u{00b0}C", .{ temp, temp_max });
            const power_suffix = try std.fmt.allocPrint(ctx.arena, "{d} / {d} W", .{ power_w, power_lim_w });

            const temp_hist = try self.state.history.temp[i].sliceLast(ctx.arena, data.history_len);
            const power_raw = try self.state.history.power[i].sliceLast(ctx.arena, data.history_len);
            const power_watts = try ctx.arena.alloc(u64, power_raw.len);
            for (power_raw, 0..) |p, j| power_watts[j] = p / 1000;
            const temp_data = try utils.normalizeRange(ctx.arena, temp_hist, 20, 100);
            const power_data = try utils.normalizeRange(ctx.arena, power_watts, 0, if (power_lim_w > 0) power_lim_w else 1);

            chart_entries = .{
                .{ .label = "TEMP", .value = temp_pct, .suffix = temp_suffix, .data = temp_data },
                .{ .label = "PWR ", .value = power_pct, .suffix = power_suffix, .data = power_data },
            };
            gauge_entries = .{
                .{ .label = "TEMP", .value = temp_pct, .suffix = temp_suffix },
                .{ .label = "PWR ", .value = power_pct, .suffix = power_suffix },
            };
        },
        inline else => |*sv| {
            const f = sv.front().*;
            const util_percent: u8 = @intCast(@min(f.util_percent orelse 0, 100));
            const mem_used = f.mem_used_bytes orelse 0;
            const mem_total = f.mem_total_bytes orelse 0;
            const mem_pct: u8 = if (mem_total > 0) @intCast(@min(mem_used * 100 / mem_total, 100)) else 0;

            const mem_total_str = try std.fmt.allocPrint(ctx.arena, "{d}", .{utils.bytesToMb(mem_total)});
            const mem_used_str = try std.fmt.allocPrint(ctx.arena, "{d}", .{utils.bytesToMb(mem_used)});
            const mem_pad = try utils.repeatStr(ctx.arena, " ", @intCast(mem_total_str.len -| mem_used_str.len));
            const mem_suffix = try std.fmt.allocPrint(ctx.arena, "{s}{s} / {s} MB", .{ mem_pad, mem_used_str, mem_total_str });

            const util_hist = try self.state.history.util[i].sliceLast(ctx.arena, data.history_len);
            const mem_hist = try self.state.history.mem_util[i].sliceLast(ctx.arena, data.history_len);
            const util_chart_data = try utils.normalizeRange(ctx.arena, util_hist, 0, 100);
            const mem_chart_data = try utils.normalizeRange(ctx.arena, mem_hist, 0, 100);

            chart_entries = .{
                .{ .label = util_label, .value = util_percent, .suffix = dev_suffix, .data = util_chart_data },
                .{ .label = "MEM", .value = mem_pct, .suffix = mem_suffix, .data = mem_chart_data },
            };
            gauge_entries = .{
                .{ .label = util_label, .value = util_percent, .suffix = dev_suffix },
                .{ .label = "MEM", .value = mem_pct, .suffix = mem_suffix },
            };
        },
    }

    if (self.use_braille) {
        card.charts = &chart_entries;
    } else {
        card.gauges = &gauge_entries;
    }

    var card_surf = try card.draw(ctx);
    card_surf.widget = ui.widget(self);
    return card_surf;
}
