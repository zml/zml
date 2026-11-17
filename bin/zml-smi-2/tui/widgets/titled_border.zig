const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const theme = @import("../theme.zig");
const ui = @import("../lib/ui.zig");
const compose = @import("../lib/compose.zig");
const Image = @import("../lib/image.zig");

const TitledBorder = @This();

child: vxfw.Widget,
title: []const u8 = "",
title_image: ?vaxis.Image = null,
value_label: ?[]const u8 = null,
border_style: vaxis.Cell.Style = theme.border_style,

pub fn draw(self: *const TitledBorder, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const pad: vxfw.Padding = .{
        .child = self.child,
        .padding = .{ .left = 1, .right = 1, .top = 1, .bottom = 1 },
    };
    const border: vxfw.Border = .{ .child = pad.widget(), .style = self.border_style };
    var surface = try border.draw(ctx);
    if (self.title_image == null and self.title.len == 0 and self.value_label == null) return surface;

    var overlays = compose.surfaceBuilder(ctx.arena);
    try overlays.children.appendSlice(ctx.arena, surface.children);
    var title_column: u16 = 2;
    if (self.title_image) |image| {
        const icon: Image = .{ .image = image, .rows = 1 };
        const icon_surface = try icon.draw(ctx);
        try overlays.addZ(0, 2, icon_surface, 1);
        title_column += icon_surface.size.width;
    }

    // Both captions cover the border with padded text; only their anchors differ.
    const title: ?[]const u8 = if (self.title.len > 0) self.title else null;
    for ([_]?[]const u8{ title, self.value_label }, 0..) |caption, index| {
        const value = caption orelse continue;
        const text = try std.fmt.allocPrint(ctx.arena, " {s} ", .{value});
        const right = surface.size.width -| 1;
        const width: u16 = if (index == 1)
            @intCast(@min(ctx.stringWidth(text), right -| 2))
        else
            surface.size.width -| (title_column + 1);
        const label: vxfw.Text = .{
            .text = text,
            .style = if (index == 1) theme.value_style else theme.title_style,
            .softwrap = false,
            .overflow = .clip,
        };
        const column = if (index == 1) right -| width else title_column;
        try overlays.addZ(0, column, try label.draw(ui.maxSize(ctx, width, 1)), 1);
    }

    surface.children = overlays.children.items;
    return surface;
}
