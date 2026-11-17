const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const theme = @import("../theme.zig");
const ui = @import("../lib/ui.zig");

const StatusLine = @This();

viewing_device: ?u16,
use_braille: bool,

const Binding = struct {
    key: []const u8,
    desc: []const u8,
};

pub fn draw(self: *const StatusLine, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const w = ctx.max.width orelse 80;

    const bg: vaxis.Cell.Color = .{ .rgb = .{ 30, 30, 45 } };
    const key_style: vaxis.Style = .{
        .bold = true,
        .fg = .{ .rgb = .{ 30, 30, 40 } },
        .bg = theme.accent,
    };

    const bindings: []const Binding = if (self.viewing_device != null) &.{
        .{ .key = "Esc", .desc = "back" },
        .{ .key = "q", .desc = "quit" },
    } else &.{
        .{ .key = "v", .desc = if (self.use_braille) "gauges" else "charts" },
        .{ .key = "click", .desc = "device details" },
        .{ .key = "q", .desc = "quit" },
    };

    var segments: [1 + 3 * 5]vaxis.Cell.Segment = undefined;
    const space: vaxis.Cell.Segment = .{ .text = " ", .style = .{ .bg = bg } };
    segments[0] = space;
    for (bindings, 0..) |binding, index| {
        segments[1 + index * 5 ..][0..5].* = .{
            space,
            .{ .text = binding.key, .style = key_style },
            space,
            .{ .text = binding.desc, .style = .{ .fg = theme.text_secondary, .bg = bg } },
            .{ .text = "  \u{2502}", .style = .{ .fg = theme.dim, .bg = bg } },
        };
    }

    const rich: vxfw.RichText = .{
        .text = segments[0 .. bindings.len * 5], // Include leading space, skip final separator.
        .base_style = .{ .bg = bg },
        .softwrap = false,
        .overflow = .clip,
    };
    return rich.draw(ui.fixedSize(ctx, w, 1));
}
