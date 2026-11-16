const std = @import("std");

const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;

const data = @import("../data.zig");
const compose = @import("../lib/compose.zig");
const ui = @import("../lib/ui.zig");
const theme = @import("../theme.zig");
const stamp = @import("bazel/stamp");

const InfoLines = @This();

state: *const data.SystemState,

const labels = [_][]const u8{
    "zml-smi ",
    "Hostname ",
    "Kernel ",
    "CPU ",
    "Cores ",
    "Memory ",
    "Uptime ",
    "Load ",
    "Devices ",
};

pub const entry_count: u16 = labels.len;

pub fn draw(self: *const InfoLines, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const w = ctx.max.width orelse 80;
    const host = self.state.host();

    const uptime_str = try formatUptime(ctx.arena, host.uptime_seconds orelse 0);
    const cores_str = try std.fmt.allocPrint(ctx.arena, "{d}", .{host.cpu_cores orelse 0});
    const load_str = try std.fmt.allocPrint(ctx.arena, "{d:.2} / {d:.2} / {d:.2}", .{
        host.load_1 orelse 0, host.load_5 orelse 0, host.load_15 orelse 0,
    });
    const total_kib = host.mem_total_kib orelse 0;
    const avail_kib = host.mem_available_kib orelse 0;
    const used_gb = (total_kib -| avail_kib) / (1024 * 1024);
    const total_gb = total_kib / (1024 * 1024);
    const mem_str = try std.fmt.allocPrint(ctx.arena, "{d} / {d} GB", .{ used_gb, total_gb });
    const devices_str = try std.fmt.allocPrint(ctx.arena, "{d}", .{self.state.deviceCount()});
    const values = [entry_count][]const u8{
        stamp.stamp("STABLE_ZML_SMI_VERSION"),
        host.hostname orelse "Unknown",
        host.kernel orelse "Unknown",
        host.cpu_name orelse "Unknown",
        cores_str,
        mem_str,
        uptime_str,
        load_str,
        devices_str,
    };
    var sb = compose.surfaceBuilder(ctx.arena);
    for (labels, values, 0..) |label, value, index| {
        if (w == 0) continue;
        const pad = try ctx.arena.alloc(u8, w -| (label.len + value.len));
        if (pad.len > 0) {
            @memset(pad, '-');
            pad[pad.len - 1] = ' ';
        }
        const text_surf = try ui.drawRichLine(ctx, &.{
            .{ .text = label, .style = theme.header_style },
            .{ .text = pad, .style = theme.dim_style },
            .{ .text = value, .style = theme.value_style },
        }, w);
        try sb.add(@intCast(index), 0, text_surf);
    }
    return sb.finish(.{ .width = w, .height = entry_count }, ui.widget(self));
}

fn formatUptime(arena: std.mem.Allocator, seconds: u64) std.mem.Allocator.Error![]const u8 {
    const days = seconds / 86400;
    const hours = (seconds % 86400) / 3600;
    const mins = (seconds % 3600) / 60;
    if (days > 0) {
        return std.fmt.allocPrint(arena, "{d}d {d}h {d}m", .{ days, hours, mins });
    } else if (hours > 0) {
        return std.fmt.allocPrint(arena, "{d}h {d}m", .{ hours, mins });
    } else {
        return std.fmt.allocPrint(arena, "{d}m", .{mins});
    }
}
