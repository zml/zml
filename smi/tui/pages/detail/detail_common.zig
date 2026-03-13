const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const theme = @import("../../theme.zig");

pub fn headerText(arena: std.mem.Allocator, id: u8, name: []const u8) std.mem.Allocator.Error!vxfw.Text {
    return .{
        .text = try std.fmt.allocPrint(arena, "Device {d}: {s}", .{ id, name }),
        .style = theme.header_style,
        .softwrap = false,
        .overflow = .clip,
    };
}
