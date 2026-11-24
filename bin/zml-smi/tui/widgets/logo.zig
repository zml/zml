const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const ui = @import("../lib/ui.zig");
const compose = @import("../lib/compose.zig");
const Image = @import("../lib/image.zig");
const zml_logo = @import("zml/logo");

const Logo = @This();

style: vaxis.Cell.Style = .{},
image: ?vaxis.Image = null,

pub const logo_width: u16 = 28;
pub const logo_height: u16 = @intCast(zml_logo.zml_art_blocks[0].rows.len);

pub fn draw(self: *const Logo, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    if (self.image) |img| {
        const img_w: Image = .{ .image = img, .rows = logo_height };
        return img_w.draw(ctx);
    }

    const w = if (ctx.max.width) |mw| @min(logo_width, mw) else logo_width;
    const h = if (ctx.max.height) |mh| @min(logo_height, mh) else logo_height;
    if (w == 0 or h == 0) {
        return .{
            .size = .{ .width = 0, .height = 0 },
            .widget = ui.widget(self),
            .buffer = &.{},
            .children = &.{},
        };
    }

    var sb = compose.surfaceBuilder(ctx.arena);
    for (0..h, 0..) |logo_row, row| {
        try sb.add(@intCast(row), 0, try self.drawLogoRow(ctx, logo_row, w));
    }

    return sb.finish(.{ .width = w, .height = h }, ui.widget(self));
}

fn drawLogoRow(self: *const Logo, ctx: vxfw.DrawContext, row_index: usize, max_w: u16) std.mem.Allocator.Error!vxfw.Surface {
    var segments: std.ArrayList(vaxis.Cell.Segment) = .empty;
    for (zml_logo.zml_art_blocks[0..]) |block| {
        const row = block.rows[row_index];
        if (row.text.len == 0) continue;

        var view: std.unicode.Utf8View = .initUnchecked(row.text);
        var iter = view.iterator();
        while (iter.nextCodepointSlice()) |codepoint| {
            const shiny = zml_logo.isShineGlyph(codepoint);
            try segments.append(ctx.arena, .{
                .text = codepoint,
                .style = .{
                    .fg = switch (if (shiny) row.shine else row.color) {
                        .reset => .default,
                        .cyan => .{ .rgb = .{ 0, 255, 255 } },
                        .sky => .{ .rgb = .{ 0, 221, 255 } },
                        .blue => .{ .rgb = .{ 135, 143, 255 } },
                        .violet => .{ .rgb = .{ 215, 90, 219 } },
                        .pink => .{ .rgb = .{ 242, 48, 174 } },
                        .shine_cyan => .{ .rgb = .{ 140, 255, 255 } },
                        .shine_sky => .{ .rgb = .{ 128, 244, 255 } },
                        .shine_blue => .{ .rgb = .{ 205, 210, 255 } },
                        .shine_violet => .{ .rgb = .{ 255, 170, 255 } },
                        .shine_pink => .{ .rgb = .{ 255, 150, 220 } },
                    },
                    .bg = self.style.bg,
                    .bold = shiny,
                },
            });
        }
    }
    return ui.drawRichLine(ctx, segments.items, max_w);
}

test "Logo clamps to narrow and zero max widths" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    vxfw.DrawContext.init(.unicode);

    const logo: Logo = .{};
    const narrow = try logo.draw(.{
        .arena = arena.allocator(),
        .min = .{},
        .max = .{ .width = 10, .height = null },
        .cell_size = .{ .width = 10, .height = 20 },
    });
    try std.testing.expectEqual(@as(u16, 10), narrow.size.width);

    const empty = try logo.draw(.{
        .arena = arena.allocator(),
        .min = .{},
        .max = .{ .width = 0, .height = null },
        .cell_size = .{ .width = 10, .height = 20 },
    });
    try std.testing.expectEqual(@as(u16, 0), empty.size.width);
    try std.testing.expectEqual(@as(u16, 0), empty.size.height);
}
