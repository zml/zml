const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const theme = @import("../theme.zig");
const ui = @import("../lib/ui.zig");
const Image = @import("../lib/image.zig");

const Text = vxfw.Text;

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
    const pad: vxfw.Padding = .{
        .child = self.child,
        .padding = self.padding,
    };
    const border: vxfw.Border = .{
        .child = pad.widget(),
        .style = self.border_style,
    };
    var surf = try border.draw(ctx);

    // Count overlay children needed
    var overlay_count: usize = 0;
    var title_col: u16 = 2;
    var has_image = false;

    if (self.title_image) |img| {
        const icon = ui.imageCellSize(img, 1, self.cell_size);
        has_image = true;
        title_col += icon.cols;
        overlay_count += 1;
    }
    if (self.title.len > 0) {
        overlay_count += 1;
    }
    if (self.value_label != null) {
        overlay_count += 1;
    }

    if (overlay_count == 0) {
        return surf;
    }

    const new_children = try ctx.arena.alloc(vxfw.SubSurface, surf.children.len + overlay_count);
    @memcpy(new_children[0..surf.children.len], surf.children);
    var idx = surf.children.len;

    // Image icon overlay
    if (has_image) {
        const img = self.title_image.?;
        const img_widget: Image = .{ .image = img, .rows = 1 };
        const img_surf = try img_widget.draw(ctx);
        new_children[idx] = .{
            .origin = .{ .row = 0, .col = 2 },
            .surface = img_surf,
            .z_index = 1,
        };
        idx += 1;
    }

    // Title text overlay (top-left)
    if (self.title.len > 0) {
        const title_text = try std.fmt.allocPrint(ctx.arena, " {s} ", .{self.title});
        const title_widget: Text = .{
            .text = title_text,
            .style = self.title_style,
            .softwrap = false,
            .overflow = .clip,
        };
        const max_title_w = surf.size.width -| (title_col + 1);
        const title_surf = try title_widget.draw(ui.maxSize(ctx, max_title_w, 1));
        new_children[idx] = .{
            .origin = .{ .row = 0, .col = @intCast(title_col) },
            .surface = title_surf,
            .z_index = 1,
        };
        idx += 1;
    }

    // Value label overlay (top-right)
    if (self.value_label) |vlabel| {
        const label_text = try std.fmt.allocPrint(ctx.arena, " {s} ", .{vlabel});
        const label_widget: Text = .{
            .text = label_text,
            .style = self.value_label_style,
            .softwrap = false,
            .overflow = .clip,
        };
        const right = surf.size.width -| 1;
        const lw: u16 = @intCast(@min(ctx.stringWidth(label_text), right -| 2));
        const label_surf = try label_widget.draw(ui.maxSize(ctx, lw, 1));
        new_children[idx] = .{
            .origin = .{ .row = 0, .col = @intCast(right -| lw) },
            .surface = label_surf,
            .z_index = 1,
        };
    }

    surf.children = new_children;
    return surf;
}
