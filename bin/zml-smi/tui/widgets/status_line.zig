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

    const key_style: vaxis.Style = .{ .bold = true, .fg = .{ .rgb = .{ 30, 30, 40 } }, .bg = theme.accent };
    const desc_style: vaxis.Style = .{ .fg = theme.text_secondary };
    const sep_style: vaxis.Style = .{ .fg = theme.dim };
    const bg_style: vaxis.Style = .{ .bg = .{ .rgb = .{ 30, 30, 45 } } };

    const chart_label = if (self.use_braille) "gauges" else "charts";

    const bindings: []const Binding = if (self.viewing_device != null) &.{
        .{ .key = "Esc", .desc = "back" },
        .{ .key = "q", .desc = "quit" },
    } else &.{
        .{ .key = "v", .desc = chart_label },
        .{ .key = "click", .desc = "device details" },
        .{ .key = "q", .desc = "quit" },
    };

    // Build segments: " key desc | key desc | ... "
    var segments: std.ArrayList(vaxis.Cell.Segment) = .empty;

    // Leading space
    try segments.append(ctx.arena, .{ .text = " ", .style = bg_style });

    for (bindings, 0..) |b, i| {
        try segments.append(ctx.arena, .{ .text = " ", .style = bg_style });
        try segments.append(ctx.arena, .{ .text = b.key, .style = key_style });
        try segments.append(ctx.arena, .{ .text = " ", .style = bg_style });
        try segments.append(ctx.arena, .{ .text = b.desc, .style = desc_style });
        if (i + 1 < bindings.len) {
            try segments.append(ctx.arena, .{ .text = "  \u{2502}", .style = sep_style });
        }
    }

    const rich: vxfw.RichText = .{
        .text = segments.items,
        .softwrap = false,
        .overflow = .clip,
    };
    const surf = try rich.draw(ui.fixedSize(ctx, w, 1));

    // Fill background across full width
    for (surf.buffer) |*cell| {
        if (cell.style.bg == .default) {
            cell.style.bg = bg_style.bg;
        }
    }

    return surf;
}
