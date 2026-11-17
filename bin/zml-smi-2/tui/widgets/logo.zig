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

// The shared logo encodes RGB as the final three true-color SGR parameters.
const palette = palette_blk: {
    @setEvalBranchQuota(10_000);
    var colors: std.EnumArray(zml_logo.Color, vaxis.Color) = .initFill(.default);
    for (std.enums.values(zml_logo.Color)) |color| {
        if (color == .reset) continue;
        const sgr = color.escape(.true_color);
        var channels = std.mem.splitBackwardsScalar(u8, sgr[0 .. sgr.len - 1], ';');
        var rgb: [3]u8 = undefined;
        for (0..3) |index| {
            rgb[2 - index] = std.fmt.parseInt(u8, channels.next().?, 10) catch
                @compileError("invalid ZML logo true-color sequence");
        }
        colors.set(color, .{ .rgb = rgb });
    }
    break :palette_blk colors;
};

pub fn draw(self: *const Logo, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    if (self.image) |image| return (Image{ .image = image, .rows = logo_height }).draw(ctx);

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
                    .style = .{ .fg = palette.get(if (shiny) row.shine else row.color), .bold = shiny },
                });
                col += width;
            }
        }
    }
    return surface;
}
