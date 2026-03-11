const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const theme = @import("../theme.zig");

const Text = vxfw.Text;

const Logo = @This();

style: vaxis.Cell.Style = theme.header_style,
image: ?vaxis.Image = null,
compact: bool = false,

const logo_lines = [_][]const u8{
    "                           ",
    "                           ",
    "                           ",
    "███████╗███╗   ███╗██╗     ",
    "╚══███╔╝████╗ ████║██║     ",
    "  ███╔╝ ██╔████╔██║██║     ",
    " ███╔╝  ██║╚██╔╝██║██║     ",
    "███████╗██║ ╚═╝ ██║███████╗",
    "╚══════╝╚═╝     ╚═╝╚══════╝",
};

const compact_lines = [_][]const u8{
    "███████╗███╗   ███╗██╗     ",
    "╚══███╔╝████╗ ████║██║     ",
    "  ███╔╝ ██╔████╔██║██║     ",
    " ███╔╝  ██║╚██╔╝██║██║     ",
    "███████╗██║ ╚═╝ ██║███████╗",
    "╚══════╝╚═╝     ╚═╝╚══════╝",
};

pub const logo_width: u16 = 28;
pub const logo_height: u16 = 9;
pub const compact_height: u16 = 6;

pub fn widget(self: *const Logo) vxfw.Widget {
    return .{
        .userdata = @constCast(self),
        .drawFn = typeErasedDrawFn,
    };
}

fn typeErasedDrawFn(ptr: *anyopaque, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const self: *const Logo = @ptrCast(@alignCast(ptr));
    return self.draw(ctx);
}

pub fn draw(self: *const Logo, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    if (self.image) |img| {
        const h = if (self.compact) compact_image_height else image_height;
        const img_widget: vxfw.Image = .{ .image = img, .rows = h };
        return img_widget.draw(ctx);
    }

    return self.drawAscii(ctx);
}

pub const image_height: u16 = 14;
pub const compact_image_height: u16 = 8;

fn drawAscii(self: *const Logo, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const lines: []const []const u8 = if (self.compact) &compact_lines else &logo_lines;
    const h: u16 = if (self.compact) compact_height else logo_height;

    // Gradient colors for each row
    const all_colors = [logo_height]vaxis.Cell.Color{
        .{ .rgb = .{ 120, 170, 255 } },
        .{ .rgb = .{ 140, 180, 255 } },
        .{ .rgb = .{ 160, 190, 255 } },
        .{ .rgb = .{ 180, 200, 255 } },
        .{ .rgb = .{ 140, 180, 255 } },
        .{ .rgb = .{ 140, 180, 255 } },
        .{ .rgb = .{ 140, 180, 255 } },
        .{ .rgb = .{ 140, 180, 255 } },
        .{ .rgb = .{ 140, 180, 255 } },
    };
    const color_offset: usize = if (self.compact) logo_height - compact_height else 0;

    var children = try std.ArrayList(vxfw.SubSurface).initCapacity(ctx.arena, h);

    for (lines, 0..) |line, row| {
        const text_widget: Text = .{
            .text = line,
            .style = .{ .bold = self.style.bold, .fg = all_colors[row + color_offset] },
            .softwrap = false,
        };
        const text_surf = try text_widget.draw(ctx.withConstraints(
            .{},
            .{ .width = logo_width, .height = 1 },
        ));
        children.appendAssumeCapacity(.{
            .origin = .{ .row = @intCast(row), .col = 0 },
            .surface = text_surf,
        });
    }

    return .{
        .size = .{ .width = logo_width, .height = h },
        .widget = self.widget(),
        .buffer = &.{},
        .children = children.items,
    };
}
