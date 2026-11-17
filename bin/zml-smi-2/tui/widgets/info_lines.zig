const std = @import("std");

const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;

const data = @import("../data.zig");
const compose = @import("../lib/compose.zig");
const ui = @import("../lib/ui.zig");
const theme = @import("../theme.zig");

const InfoLines = @This();

state: *const data.SystemState,

pub const entry_count: u16 = 9;

pub fn draw(self: *const InfoLines, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const w = ctx.max.width orelse 80;
    const host = self.state.frame.host;
    const load = try std.fmt.allocPrint(ctx.arena, "{d:.2} / {d:.2} / {d:.2}", .{
        host.load_1 orelse 0, host.load_5 orelse 0, host.load_15 orelse 0,
    });
    const total = host.mem_total_kib orelse 0;
    const used = total -| (host.mem_available_kib orelse 0);
    const memory = try std.fmt.allocPrint(ctx.arena, "{d} / {d} GB", .{ used >> 20, total >> 20 });
    const rows = [entry_count]struct { []const u8, []const u8 }{
        .{ "Monitor ", "zml-smi-2" },
        .{ "Hostname ", host.hostname orelse "Unknown" },
        .{ "Kernel ", host.kernel orelse "Unknown" },
        .{ "CPU ", host.cpu_name orelse "Unknown" },
        .{ "Cores ", try std.fmt.allocPrint(ctx.arena, "{d}", .{host.cpu_cores orelse 0}) },
        .{ "Memory ", memory },
        .{ "Uptime ", try formatUptime(ctx.arena, host.uptime_seconds orelse 0) },
        .{ "Load ", load },
        .{ "Devices ", try std.fmt.allocPrint(ctx.arena, "{d}", .{self.state.devices.len}) },
    };
    var sb = compose.surfaceBuilder(ctx.arena);
    for (rows, 0..) |row, index| {
        if (w == 0) continue;
        const label, const value = row;
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
    if (days > 0) return std.fmt.allocPrint(arena, "{d}d {d}h {d}m", .{ days, hours, mins });
    if (hours > 0) return std.fmt.allocPrint(arena, "{d}h {d}m", .{ hours, mins });
    return std.fmt.allocPrint(arena, "{d}m", .{mins});
}
