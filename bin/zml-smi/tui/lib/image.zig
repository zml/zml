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
    const empty: vxfw.Surface = .{
        .size = .{ .width = 0, .height = 0 },
        .widget = ui.widget(self),
        .buffer = &.{},
        .children = &.{},
    };

    var target_rows = self.rows orelse ctx.max.height orelse 1;
    if (ctx.max.height) |max_h| target_rows = @min(target_rows, max_h);
    if (target_rows == 0 or ctx.max.width == 0 or self.image.width == 0 or self.image.height == 0) {
        return empty;
    }

    var cols = ui.imageCellSize(self.image, target_rows, ctx.cell_size).cols;
    if (cols == 0) return empty;

    if (ctx.max.width) |max_w| {
        if (cols > max_w) {
            target_rows = @intCast(@as(u32, target_rows) * max_w / cols);
            if (target_rows == 0) return empty;

            cols = @min(ui.imageCellSize(self.image, target_rows, ctx.cell_size).cols, max_w);
            if (cols == 0) return empty;
        }
    }

    if (@as(u32, cols) * target_rows > std.math.maxInt(u16)) return empty;

    const size: vxfw.Size = .{ .width = cols, .height = target_rows };
    var surf = try vxfw.Surface.init(ctx.arena, ui.widget(self), size);
    surf.writeCell(0, 0, .{
        .image = .{
            .img_id = self.image.id,
            .options = .{
                // Emit both rows and cols so that target bounds are unambiguous.
                // Rows alone lets some terminals fall back to native pixel size.
                .size = .{ .rows = target_rows, .cols = cols },
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
    // Both rows and cols must be emitted so terminals don't fall back to native pixel size.
    const size = cell.image.?.options.size.?;
    try std.testing.expectEqual(@as(?u16, 5), size.rows);
    try std.testing.expectEqual(@as(?u16, 20), size.cols);
}

test "Image rounds its width to the nearest column" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    vxfw.DrawContext.init(.unicode);

    const img: Image = .{ .image = .{ .id = 1, .width = 64, .height = 64 }, .rows = 1 };
    const surface = try img.draw(.{
        .arena = arena.allocator(),
        .min = .{},
        .max = .{ .width = 40, .height = 10 },
        .cell_size = .{ .width = 8, .height = 17 },
    });
    try std.testing.expectEqual(@as(u16, 2), surface.size.width);
}

test "Image clamps to ctx.max bounds and handles zero cell dimensions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    vxfw.DrawContext.init(.unicode);

    const img: Image = .{ .image = .{ .id = 1, .width = 1024, .height = 1008 }, .rows = 8 };

    const narrow = try img.draw(.{
        .arena = arena.allocator(),
        .min = .{},
        .max = .{ .width = 10, .height = 8 },
        .cell_size = .{ .width = 0, .height = 48 },
    });
    try std.testing.expect(narrow.size.width <= 10);
    try std.testing.expect(narrow.size.height <= 8);

    const zero_w = try img.draw(.{
        .arena = arena.allocator(),
        .min = .{},
        .max = .{ .width = 0, .height = 8 },
        .cell_size = .{ .width = 10, .height = 20 },
    });
    try std.testing.expectEqual(@as(u16, 0), zero_w.size.width);
    try std.testing.expectEqual(@as(u16, 0), zero_w.size.height);
}

test "refAllDecls" {
    std.testing.refAllDecls(@This());
}
