const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;

const ui = @import("ui.zig");

const Image = @This();

image: vaxis.Image,
/// Desired height in cell rows. Width is computed to maintain aspect ratio.
rows: u16,

pub fn draw(self: *const Image, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const widget = ui.widget(self);
    if (self.rows == 0 or self.image.width == 0 or self.image.height == 0) {
        return .empty(widget);
    }
    const cols = ui.imageColumns(self.image, self.rows, ctx.cell_size);
    const surface = try vxfw.Surface.init(ctx.arena, widget, .{ .width = cols, .height = self.rows });
    surface.writeCell(0, 0, .{
        .image = .{
            .img_id = self.image.id,
            // Some terminals fall back to native pixel size when only rows are set.
            .options = .{ .size = .{ .rows = self.rows, .cols = cols } },
        },
    });
    return surface;
}
