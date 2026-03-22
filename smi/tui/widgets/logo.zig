const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const theme = @import("../theme.zig");
const ui = @import("../lib/ui.zig");
const compose = @import("../lib/compose.zig");
const Image = @import("../lib/image.zig");

const Logo = @This();

style: vaxis.Cell.Style = theme.header_style,
image: ?vaxis.Image = null,
compact: bool = false,

const ascii_lines = [_][]const u8{
    "                           ",
    "███████╗███╗   ███╗██╗     ",
    "╚══███╔╝████╗ ████║██║     ",
    "  ███╔╝ ██╔████╔██║██║     ",
    " ███╔╝  ██║╚██╔╝██║██║     ",
    "███████╗██║ ╚═╝ ██║███████╗",
    "╚══════╝╚═╝     ╚═╝╚══════╝",
};

pub const logo_width: u16 = 28;
pub const logo_height: u16 = ascii_lines.len;
pub const compact_height: u16 = ascii_lines.len - 1;
pub const image_height: u16 = 14;
pub const compact_image_height: u16 = 8;

pub fn draw(self: *const Logo, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    if (self.image) |img| {
        const h = if (self.compact) compact_image_height else image_height;
        const img_w: Image = .{ .image = img, .rows = h };
        return img_w.draw(ctx);
    }

    const lines: []const []const u8 = if (self.compact) ascii_lines[1..] else &ascii_lines;
    const h: u16 = if (self.compact) compact_height else logo_height;
    const logo_color: vaxis.Cell.Color = .{ .rgb = .{ 140, 180, 255 } };

    var sb = compose.surfaceBuilder(ctx.arena);
    for (lines, 0..) |line, row| {
        const text: vxfw.Text = .{
            .text = line,
            .style = .{ .bold = self.style.bold, .fg = logo_color },
            .softwrap = false,
        };
        try sb.add(@intCast(row), 0, try text.draw(ui.maxSize(ctx, logo_width, 1)));
    }
    return sb.finish(.{ .width = logo_width, .height = h }, ui.widget(self));
}
