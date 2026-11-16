const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const ui = @import("../lib/ui.zig");
const Image = @import("../lib/image.zig");
const zml_logo = @import("zml/logo");

const Logo = @This();

image: ?vaxis.Image = null,

pub const logo_width: u16 = 28;
pub const logo_height: u16 = @intCast(zml_logo.zml_art_blocks[0].rows.len);

pub fn draw(self: *const Logo, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    if (self.image) |img| {
        const img_w: Image = .{ .image = img, .rows = logo_height };
        return img_w.draw(ctx);
    }

    const surface = try vxfw.Surface.init(ctx.arena, ui.widget(self), .{
        .width = logo_width,
        .height = logo_height,
    });
    for (0..logo_height) |row_index| {
        var col: u16 = 0;
        for (zml_logo.zml_art_blocks) |block| {
            const row = block.rows[row_index];
            var glyphs = (std.unicode.Utf8View.initUnchecked(row.text)).iterator();
            while (glyphs.nextCodepointSlice()) |glyph| {
                const shiny = zml_logo.isShineGlyph(glyph);
                const width: u8 = @intCast(ctx.stringWidth(glyph));
                surface.writeCell(col, @intCast(row_index), .{
                    .char = .{ .grapheme = glyph, .width = width },
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
                        .bold = shiny,
                    },
                });
                col += width;
            }
        }
    }
    return surface;
}
