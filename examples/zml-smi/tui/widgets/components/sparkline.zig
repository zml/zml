const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const theme = @import("../../theme.zig");
const utils = @import("../../utils.zig");

const RichText = vxfw.RichText;

const Sparkline = @This();

/// Normalized values (0-100).
values: []const u8,
style: vaxis.Cell.Style = .{},
/// If true, each bar is colored based on its value.
gradient: bool = true,

pub fn widget(self: *const Sparkline) vxfw.Widget {
    return .{
        .userdata = @constCast(self),
        .drawFn = typeErasedDrawFn,
    };
}

fn typeErasedDrawFn(ptr: *anyopaque, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const self: *const Sparkline = @ptrCast(@alignCast(ptr));
    return self.draw(ctx);
}

pub fn draw(self: *const Sparkline, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const max_w = ctx.max.width orelse 40;
    const width: u16 = @max(1, @min(max_w, @as(u16, @intCast(self.values.len))));
    const height: u16 = @max(1, ctx.min.height);

    // Draw from the right (newest) to the left
    const data_len = self.values.len;
    const display_count = @min(data_len, width);
    const data_start = data_len - display_count;

    if (height == 1) {
        return self.drawSingleRow(ctx, display_count, data_start, width);
    } else {
        return self.drawMultiRow(ctx, display_count, data_start, width, height);
    }
}

fn drawSingleRow(self: *const Sparkline, ctx: vxfw.DrawContext, display_count: usize, data_start: usize, width: u16) std.mem.Allocator.Error!vxfw.Surface {
    var spans = try std.ArrayList(vaxis.Cell.Segment).initCapacity(ctx.arena, display_count);

    for (0..display_count) |i| {
        const val = self.values[data_start + i];
        const clamped = @min(val, 100);
        const block_idx = @as(usize, clamped) * 8 / 100;

        var style = self.style;
        if (self.gradient) {
            style.fg = theme.colorForPercent(clamped);
        }

        spans.appendAssumeCapacity(.{ .text = utils.blocks[block_idx], .style = style });
    }

    const rich: RichText = .{
        .text = spans.items,
        .softwrap = false,
        .overflow = .clip,
        .width_basis = .longest_line,
    };
    return rich.draw(ctx.withConstraints(
        .{ .width = 1, .height = 1 },
        .{ .width = width, .height = 1 },
    ));
}

fn drawMultiRow(self: *const Sparkline, ctx: vxfw.DrawContext, display_count: usize, data_start: usize, width: u16, height: u16) std.mem.Allocator.Error!vxfw.Surface {
    // Pre-compute per-column fill data
    const Column = struct { full_rows: u16, partial: usize, style: vaxis.Cell.Style };
    var columns = try std.ArrayList(Column).initCapacity(ctx.arena, display_count);
    for (0..display_count) |i| {
        const val = self.values[data_start + i];
        const clamped = @min(val, 100);
        const filled_eighths = @as(u32, clamped) * @as(u32, height) * 8 / 100;
        var style = self.style;
        if (self.gradient) style.fg = theme.colorForPercent(clamped);
        columns.appendAssumeCapacity(.{
            .full_rows = @intCast(filled_eighths / 8),
            .partial = filled_eighths % 8,
            .style = style,
        });
    }

    // Build per-row RichText SubSurfaces
    var children = try std.ArrayList(vxfw.SubSurface).initCapacity(ctx.arena, height);

    var row: u16 = 0;
    while (row < height) : (row += 1) {
        var spans = try std.ArrayList(vaxis.Cell.Segment).initCapacity(ctx.arena, display_count);
        for (columns.items) |col_data| {
            const from_bottom = height - 1 - row;
            if (from_bottom < col_data.full_rows) {
                spans.appendAssumeCapacity(.{ .text = utils.blocks[8], .style = col_data.style });
            } else if (from_bottom == col_data.full_rows and col_data.partial > 0) {
                spans.appendAssumeCapacity(.{ .text = utils.blocks[col_data.partial], .style = col_data.style });
            } else {
                spans.appendAssumeCapacity(.{ .text = " ", .style = .{} });
            }
        }
        const rich: RichText = .{
            .text = spans.items,
            .softwrap = false,
            .overflow = .clip,
        };
        const row_surf = try rich.draw(ctx.withConstraints(
            .{ .width = width },
            .{ .width = width, .height = 1 },
        ));
        children.appendAssumeCapacity(.{
            .origin = .{ .row = @intCast(row), .col = 0 },
            .surface = row_surf,
        });
    }

    return .{
        .size = .{ .width = width, .height = height },
        .widget = self.widget(),
        .buffer = &.{},
        .children = children.items,
    };
}
