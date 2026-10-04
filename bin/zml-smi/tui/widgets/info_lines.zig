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

const Entry = struct {
    label: []const u8,
    value: []const u8,
};

pub fn draw(self: *const InfoLines, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const w = ctx.max.width orelse 80;
    var sb = compose.surfaceBuilder(ctx.arena);
    for (self.state.hosts, 0..) |summary, host_idx| {
        const host = summary.host.front().*;

        const hostname_str = try std.fmt.allocPrint(ctx.arena, "{s}", .{host.hostname orelse "Unknown"});
        const kernel_str = try std.fmt.allocPrint(ctx.arena, "{s}", .{host.kernel orelse "Unknown"});
        const cpu_str = try std.fmt.allocPrint(ctx.arena, "{s}", .{host.cpu_name orelse "Unknown"});
        const uptime_str = if (host.uptime_seconds) |seconds| try formatUptime(ctx.arena, seconds) else "Unknown";
        const cores_str = if (host.cpu_cores) |cores| try std.fmt.allocPrint(ctx.arena, "{d}", .{cores}) else "Unknown";
        const load_str = if (host.load_1 == null or host.load_5 == null or host.load_15 == null) "Unknown" else try std.fmt.allocPrint(ctx.arena, "{d:.2} / {d:.2} / {d:.2}", .{
            host.load_1 orelse 0, host.load_5 orelse 0, host.load_15 orelse 0,
        });
        const total_kib = host.mem_total_kib orelse 0;
        const avail_kib = host.mem_available_kib orelse 0;
        const used_gb = (total_kib -| avail_kib) / (1024 * 1024);
        const total_gb = total_kib / (1024 * 1024);
        const mem_str = if (host.mem_total_kib == null or host.mem_available_kib == null) "Unknown" else try std.fmt.allocPrint(ctx.arena, "{d} / {d} GB", .{ used_gb, total_gb });
        const devices_str = try std.fmt.allocPrint(ctx.arena, "{d}", .{summary.device_count});
        const values = [entry_count][]const u8{
            stamp.stamp("STABLE_ZML_SMI_VERSION"),
            hostname_str,
            kernel_str,
            cpu_str,
            cores_str,
            mem_str,
            uptime_str,
            load_str,
            devices_str,
        };
        var entries: [entry_count]Entry = undefined;
        for (&entries, labels, values) |*e, label, value| {
            e.* = .{ .label = label, .value = value };
        }

        for (entries, 0..) |entry, i| {
            if (w == 0) {
                continue;
            }

            const total_len = entry.label.len + entry.value.len;
            const segments = if (total_len < w) blk: {
                const pad_len = w - total_len;
                const pad = try ctx.arena.alloc(u8, pad_len);
                @memset(pad, '-');
                pad[pad_len - 1] = ' ';
                break :blk try ctx.arena.dupe(vaxis.Cell.Segment, &.{
                    .{ .text = entry.label, .style = theme.header_style },
                    .{ .text = pad, .style = theme.dim_style },
                    .{ .text = entry.value, .style = theme.value_style },
                });
            } else try ctx.arena.dupe(vaxis.Cell.Segment, &.{
                .{ .text = entry.label, .style = theme.header_style },
                .{ .text = entry.value, .style = theme.value_style },
            });

            const text_surf = try ui.drawRichLine(ctx, segments, w);

            try sb.add(@intCast(host_idx * (entry_count + 1) + i), 0, text_surf);
        }
    }
    const total_h = self.height();
    return sb.finish(.{ .width = w, .height = total_h }, ui.widget(self));
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

pub fn height(self: *const InfoLines) u16 {
    return @intCast(self.state.hosts.len * (entry_count + 1) -| 1);
}
