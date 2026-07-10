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
pub const image_height: u16 = logo_height;

pub fn draw(self: *const Logo, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    if (self.image) |img| {
        const img_w: Image = .{ .image = img, .rows = image_height };
        return img_w.draw(ctx);
    }

    var sb = compose.surfaceBuilder(ctx.arena);
    for (0..logo_height, 0..) |logo_row, row| {
        try sb.add(@intCast(row), 0, try self.drawLogoRow(ctx, logo_row));
    }
    return sb.finish(.{ .width = logo_width, .height = logo_height }, ui.widget(self));
}

fn drawLogoRow(self: *const Logo, ctx: vxfw.DrawContext, row_index: usize) std.mem.Allocator.Error!vxfw.Surface {
    const last_block = blk: {
        var index = zml_logo.zml_art_blocks.len;
        while (index > 0) {
            index -= 1;
            if (std.mem.trimEnd(u8, zml_logo.zml_art_blocks[index].rows[row_index].text, " \t\r\n").len != 0) break :blk index;
        }
        return vxfw.Surface.init(ctx.arena, ui.widget(self), .{ .width = logo_width, .height = 1 });
    };

    var segments: std.ArrayList(vaxis.Cell.Segment) = .empty;
    for (zml_logo.zml_art_blocks[0 .. last_block + 1], 0..) |block, block_index| {
        const is_last_block = block_index == last_block;
        const row = block.rows[row_index];
        const text = if (is_last_block) std.mem.trimEnd(u8, row.text, " \t\r\n") else row.text;
        if (text.len == 0) continue;

        var view: std.unicode.Utf8View = .initUnchecked(text);
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
    return ui.drawRichLine(ctx, segments.items, logo_width);
}
