const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const ui = @import("ui.zig");
const Image = @import("image.zig");

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

    // One 17px row scales a square icon to 17px: closer to 2 columns of 8px than to 3.
    const img: Image = .{ .image = .{ .id = 1, .width = 64, .height = 64 }, .rows = 1 };
    const surface = try img.draw(.{
        .arena = arena.allocator(),
        .min = .{},
        .max = .{ .width = 40, .height = 10 },
        .cell_size = .{ .width = 8, .height = 17 },
    });
    try std.testing.expectEqual(@as(u16, 2), surface.size.width);
}

test "refAllDecls" {
    std.testing.refAllDecls(Image);
}
