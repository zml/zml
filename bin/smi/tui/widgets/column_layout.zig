const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const ui = @import("../lib/ui.zig");
const compose = @import("../lib/compose.zig");

const ColumnLayout = @This();

children: []const vxfw.Widget,
gap: u16 = 0,
col_gap: u16 = 0,
/// When set, children wrap into a grid with this minimum column width.
/// When null (default), behaves as a simple vertical stack.
min_child_width: ?u16 = null,

pub fn draw(self: *const ColumnLayout, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const n = self.children.len;
    if (n == 0) {
        return vxfw.Surface.init(ctx.arena, ui.widget(self), ctx.min);
    }

    const available_w = ctx.max.width orelse 80;
    const cols: u16 = if (self.min_child_width) |mcw|
        @intCast(@max(1, @min(n, available_w / @max(mcw, 1))))
    else
        1;

    var sb = compose.surfaceBuilder(ctx.arena);
    var row_y: u16 = 0;
    var max_width: u16 = 0;
    var i: usize = 0;

    while (i < n) {
        if (i > 0) {
            row_y += self.gap;
        }

        const row_end = @min(i + cols, n);
        var row_h: u16 = 0;

        for (i..row_end) |j| {
            const col_idx: u16 = @intCast(j - i);

            const col_x, const w = if (cols > 1) blk: {
                const total_col_gaps = @as(u16, cols - 1) * self.col_gap;
                const usable_w = available_w -| total_col_gaps;
                const child_w = usable_w / cols;
                const x = col_idx * (child_w + self.col_gap);
                break :blk .{ x, if (col_idx == cols - 1) available_w - x else child_w };
            } else .{ 0, available_w };

            const surf = try self.children[j].draw(ui.fixedWidth(ctx, w));
            try sb.add(row_y, col_x, surf);
            max_width = @max(max_width, col_x + surf.size.width);
            row_h = @max(row_h, surf.size.height);
        }

        row_y += row_h;
        i = row_end;
    }

    const result_w = if (ctx.max.width) |mw| @min(@max(max_width, ctx.min.width), mw) else @max(max_width, ctx.min.width);
    return sb.finish(.{ .width = result_w, .height = row_y }, ui.widget(self));
}
