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

    // Preserve the image's aspect ratio by rounding to the nearest terminal column.
    const cell_h: u32 = if (ctx.cell_size.height > 0) ctx.cell_size.height else 20;
    const cell_w: u32 = if (ctx.cell_size.width > 0) ctx.cell_size.width else 10;
    const height_px = @as(u32, self.rows) * cell_h;
    const scale = @as(f64, @floatFromInt(height_px)) / @as(f64, @floatFromInt(self.image.height));
    const width_px: u32 = @intFromFloat(@as(f64, @floatFromInt(self.image.width)) * scale);
    const cols: u16 = @intCast(@max(1, (width_px + cell_w / 2) / cell_w));

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
