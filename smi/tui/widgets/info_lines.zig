const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("../data.zig");
const theme = @import("../theme.zig");
const utils = @import("../utils.zig");

const RichText = vxfw.RichText;

const InfoLines = @This();

state: *const data.SystemState,

pub fn widget(self: *const InfoLines) vxfw.Widget {
    return .{
        .userdata = @constCast(self),
        .drawFn = typeErasedDrawFn,
    };
}

fn typeErasedDrawFn(ptr: *anyopaque, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const self: *const InfoLines = @ptrCast(@alignCast(ptr));
    return self.draw(ctx);
}

const Entry = struct {
    label: []const u8,
    value: []const u8,
};

pub fn draw(self: *const InfoLines, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const w = ctx.max.width orelse 80;
    const host = self.state.host;

    const uptime_str = try utils.formatUptime(ctx.arena, host.uptime_seconds orelse 0);
    const cores_str = try std.fmt.allocPrint(ctx.arena, "{d}", .{host.cpu_cores orelse 0});
    const load = utils.parseLoadAvg(host.load_avg);
    const load_str = try std.fmt.allocPrint(ctx.arena, "{d:.2} / {d:.2} / {d:.2}", .{
        load[0], load[1], load[2],
    });
    const total_kib = host.mem_total_kib orelse 0;
    const avail_kib = host.mem_available_kib orelse 0;
    const used_gb = (total_kib -| avail_kib) / (1024 * 1024);
    const total_gb = total_kib / (1024 * 1024);
    const mem_str = try std.fmt.allocPrint(ctx.arena, "{d} / {d} GB", .{ used_gb, total_gb });
    const devices_str = try std.fmt.allocPrint(ctx.arena, "{d}", .{self.state.deviceCount()});
    const entries = [_]Entry{
        .{ .label = "zml-smi ", .value = "v0.1" },
        .{ .label = "Hostname ", .value = utils.strSlice(&host.hostname) },
        .{ .label = "Kernel ", .value = utils.strSlice(&host.kernel) },
        .{ .label = "CPU ", .value = utils.strSlice(&host.cpu_name) },
        .{ .label = "Cores ", .value = cores_str },
        .{ .label = "Memory ", .value = mem_str },
        .{ .label = "Uptime ", .value = uptime_str },
        .{ .label = "Load ", .value = load_str },
        .{ .label = "Devices ", .value = devices_str },
    };

    var children: std.ArrayList(vxfw.SubSurface) = .empty;

    for (entries, 0..) |entry, i| {
        if (w == 0) continue;

        const label_len = entry.label.len;
        const value_len = entry.value.len;
        const total_len = label_len + value_len;
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

        const rich: RichText = .{
            .text = segments,
            .softwrap = false,
            .overflow = .clip,
        };
        const text_surf = try rich.draw(ctx.withConstraints(
            .{},
            .{ .width = w, .height = 1 },
        ));

        try children.append(ctx.arena, .{
            .origin = .{ .row = @intCast(i), .col = 0 },
            .surface = text_surf,
        });
    }

    const total_h: u16 = @intCast(entries.len);
    return .{
        .size = .{ .width = w, .height = total_h },
        .widget = self.widget(),
        .buffer = &.{},
        .children = children.items,
    };
}
