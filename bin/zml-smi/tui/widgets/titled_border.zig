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
title_style: vaxis.Cell.Style = theme.title_style,
title_image: ?vaxis.Image = null,
cell_size: vxfw.Size = .{},
value_label: ?[]const u8 = null,
value_label_style: vaxis.Cell.Style = theme.value_style,
border_style: vaxis.Cell.Style = theme.border_style,
padding: vxfw.Padding.PadValues = .{ .left = 1, .right = 1, .top = 1, .bottom = 1 },

pub fn draw(self: *const TitledBorder, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const pad: vxfw.Padding = .{ .child = self.child, .padding = self.padding };
    const border: vxfw.Border = .{ .child = pad.widget(), .style = self.border_style };
    var surface = try border.draw(ctx);
    if (self.title_image == null and self.title.len == 0 and self.value_label == null) return surface;

    var overlays = compose.surfaceBuilder(ctx.arena);
    try overlays.children.appendSlice(ctx.arena, surface.children);
    var title_column: u16 = 2;
    if (self.title_image) |image| {
        const icon: Image = .{ .image = image, .rows = 1 };
        try overlays.addZ(0, 2, try icon.draw(ctx), 1);
        title_column += ui.imageColumns(image, 1, self.cell_size);
    }

    // Both captions cover the border with padded text; only their anchors differ.
    const Caption = struct { text: ?[]const u8, style: vaxis.Cell.Style, right: bool };
    const captions = [_]Caption{
        .{
            .text = if (self.title.len > 0) self.title else null,
            .style = self.title_style,
            .right = false,
        },
        .{ .text = self.value_label, .style = self.value_label_style, .right = true },
    };
    for (captions) |caption| {
        const value = caption.text orelse continue;
        const text = try std.fmt.allocPrint(ctx.arena, " {s} ", .{value});
        const right = surface.size.width -| 1;
        const width: u16 = if (caption.right)
            @intCast(@min(ctx.stringWidth(text), right -| 2))
        else
            surface.size.width -| (title_column + 1);
        const label: vxfw.Text = .{
            .text = text,
            .style = caption.style,
            .softwrap = false,
            .overflow = .clip,
        };
        try overlays.addZ(
            0,
            @intCast(if (caption.right) right -| width else title_column),
            try label.draw(ui.maxSize(ctx, width, 1)),
            1,
        );
    }

    surface.children = overlays.children.items;
    return surface;
}
