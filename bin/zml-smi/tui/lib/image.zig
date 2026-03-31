const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;

const ui = @import("ui.zig");
const Allocator = std.mem.Allocator;

const Image = @This();

/// The loaded image to display.
image: vaxis.Image,
/// Desired height in cell rows. Width is computed to maintain aspect ratio.
/// If null, uses max constraint height (or 1 if unbounded).
rows: ?u16 = null,

pub fn draw(self: *const Image, ctx: vxfw.DrawContext) Allocator.Error!vxfw.Surface {
    const target_rows = self.rows orelse ctx.max.height orelse 1;
    if (target_rows == 0 or self.image.width == 0 or self.image.height == 0) {
        return .{
            .size = .{ .width = 0, .height = 0 },
            .widget = ui.widget(self),
            .buffer = &.{},
            .children = &.{},
        };
    }

    // Compute cell columns from image aspect ratio and terminal cell pixel size
    const cell_h: u32 = if (ctx.cell_size.height > 0) ctx.cell_size.height else 20;
    const cell_w: u32 = if (ctx.cell_size.width > 0) ctx.cell_size.width else 10;
    const height_px = @as(u32, target_rows) * cell_h;
    const scale_f = @as(f64, @floatFromInt(height_px)) / @as(f64, @floatFromInt(self.image.height));
    const width_px: u32 = @intFromFloat(@as(f64, @floatFromInt(self.image.width)) * scale_f);
    const cols: u16 = @intCast((width_px + cell_w - 1) / cell_w);

    const size: vxfw.Size = .{ .width = cols, .height = target_rows };
    var surf = try vxfw.Surface.init(ctx.arena, ui.widget(self), size);
    surf.writeCell(0, 0, .{
        .image = .{
            .img_id = self.image.id,
            .options = .{
                .size = .{ .rows = target_rows },
            },
        },
    });
    return surf;
}

test Image {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    vxfw.DrawContext.init(.unicode);

    const img: Image = .{
        .image = .{ .id = 1, .width = 100, .height = 50 },
        .rows = 5,
    };
    const w = ui.widget(&img);

    const draw_ctx: vxfw.DrawContext = .{
        .arena = arena.allocator(),
        .min = .{},
        .max = .{ .width = 40, .height = 10 },
        .cell_size = .{ .width = 10, .height = 20 },
    };

    const surface = try w.draw(draw_ctx);
    try std.testing.expectEqual(@as(u16, 5), surface.size.height);
    // 5 rows * 20px = 100px height. scale = 100/50 = 2.0. width = 100*2 = 200px. cols = 200/10 = 20
    try std.testing.expectEqual(@as(u16, 20), surface.size.width);
    // Image cell should be at (0, 0)
    const cell = surface.buffer[0];
    try std.testing.expect(cell.image != null);
    try std.testing.expectEqual(@as(u32, 1), cell.image.?.img_id);
}

test "refAllDecls" {
    std.testing.refAllDecls(@This());
}
