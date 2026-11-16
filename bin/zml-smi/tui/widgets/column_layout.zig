const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const ui = @import("../lib/ui.zig");

const ColumnLayout = @This();

children: []const vxfw.Widget,
gap: u16 = 0,
col_gap: u16 = 0,
/// When set, children wrap into a grid with this minimum column width.
/// When null (default), behaves as a simple vertical stack.
min_child_width: ?u16 = null,

pub fn draw(self: *const ColumnLayout, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    if (self.children.len == 0) {
        return vxfw.Surface.init(ctx.arena, ui.widget(self), ctx.min);
    }

    const available_w = ctx.max.width orelse 80;
    const cols: u16 = if (self.min_child_width) |mcw|
        @intCast(@max(1, @min(self.children.len, available_w / @max(mcw, 1))))
    else
        1;

    const column_width = (available_w -| (cols - 1) * self.col_gap) / cols;
    const children = try ctx.arena.alloc(vxfw.SubSurface, self.children.len);
    var row_y: u16 = 0;
    var max_width: u16 = 0;
    var row_height: u16 = 0;
    for (self.children, children, 0..) |widget, *child, index| {
        const column: u16 = @intCast(index % cols);
        if (column == 0 and index != 0) {
            row_y += row_height + self.gap;
            row_height = 0;
        }
        const column_x = column * (column_width +| self.col_gap);
        const width = if (column == cols - 1) available_w - column_x else column_width;
        const surface = try widget.draw(ui.fixedWidth(ctx, width));
        child.* = .{ .origin = .{ .row = row_y, .col = column_x }, .surface = surface, .z_index = 0 };
        max_width = @max(max_width, column_x + surface.size.width);
        row_height = @max(row_height, surface.size.height);
    }

    return .{
        .size = .{
            .width = @min(@max(max_width, ctx.min.width), ctx.max.width orelse std.math.maxInt(u16)),
            .height = row_y + row_height,
        },
        .widget = ui.widget(self),
        .buffer = &.{},
        .children = children,
    };
}
