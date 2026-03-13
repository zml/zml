const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const theme = @import("../../theme.zig");
const ColumnLayout = @import("../../widgets/components/column_layout.zig");

pub fn headerText(arena: std.mem.Allocator, id: u8, name: []const u8) std.mem.Allocator.Error!vxfw.Text {
    return .{
        .text = try std.fmt.allocPrint(arena, "Device {d}: {s}", .{ id, name }),
        .style = theme.header_style,
        .softwrap = false,
        .overflow = .clip,
    };
}

/// Draws a page-level layout with standard margin (row=1, col=2) inside a full-width surface.
pub fn pageFrame(ctx: vxfw.DrawContext, layout: ColumnLayout, w: u16, content_w: u16, parent_widget: vxfw.Widget) std.mem.Allocator.Error!vxfw.Surface {
    const layout_surf = try layout.widget().draw(ctx.withConstraints(
        .{ .width = content_w },
        .{ .width = content_w, .height = null },
    ));

    const children = try ctx.arena.alloc(vxfw.SubSurface, 1);
    children[0] = .{ .origin = .{ .row = 1, .col = 2 }, .surface = layout_surf };
    return .{
        .size = .{ .width = w, .height = layout_surf.size.height + 1 },
        .widget = parent_widget,
        .buffer = &.{},
        .children = children,
    };
}
