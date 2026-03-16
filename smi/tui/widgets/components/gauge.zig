const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const theme = @import("../../theme.zig");
const utils = @import("../../utils.zig");

const RichText = vxfw.RichText;

const Gauge = @This();

value: u8, // 0-100
label: ?[]const u8 = null,
suffix: ?[]const u8 = null,
/// Minimum suffix space to reserve (for aligning gauges with different suffix widths).
suffix_reserve: u16 = 0,

pub const sep_w: u16 = 3; // "  " before suffix

pub fn widget(self: *const Gauge) vxfw.Widget {
    return .{
        .userdata = @constCast(self),
        .drawFn = typeErasedDrawFn,
    };
}

fn typeErasedDrawFn(ptr: *anyopaque, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const self: *const Gauge = @ptrCast(@alignCast(ptr));
    return self.draw(ctx);
}

pub fn draw(self: *const Gauge, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const max_w = ctx.max.width orelse 30;
    const width: u16 = @max(ctx.min.width, max_w);

    // Label width
    var label_w: u16 = 0;
    if (self.label) |label| {
        label_w = @intCast(ctx.stringWidth(label));
        label_w += 1; // space
    }

    // Percentage text: " XX%"
    const pct_str = try std.fmt.allocPrint(ctx.arena, "{d:>3}%", .{self.value});
    const pct_width: u16 = @intCast(ctx.stringWidth(pct_str));

    // Suffix area (text + separator); use reserve for bar calculation, actual for rendering
    const actual_suffix_area: u16 = if (self.suffix) |s| @as(u16, @intCast(ctx.stringWidth(s))) + sep_w else 0;
    const effective_suffix_area = @max(actual_suffix_area, self.suffix_reserve);

    // Reserved: label + "[" + "]" + pct + suffix_area
    const reserved = label_w + pct_width + 2 + effective_suffix_area; // 2 for [ ]
    if (reserved >= width) {
        // Not enough space for a bar — just show label
        var spans: std.ArrayList(vaxis.Cell.Segment) = .empty;
        if (self.label) |label| {
            try spans.append(ctx.arena, .{ .text = label, .style = theme.label_style });
        }
        const rich: RichText = .{
            .text = spans.items,
            .softwrap = false,
            .overflow = .clip,
            .width_basis = .parent,
        };
        return rich.draw(ctx.withConstraints(
            .{ .width = width, .height = 1 },
            .{ .width = width, .height = 1 },
        ));
    }

    const bar_width = width - reserved;
    const filled: u16 = @intCast(@as(u32, bar_width) * @as(u32, @min(self.value, 100)) / 100);
    const empty_count = bar_width - filled;

    // Pre-allocate spans: label(2) + bracket(1) + filled + empty(1) + bracket(1) + space(1) + pct(1) + sep(1) + suffix(1)
    var spans: std.ArrayList(vaxis.Cell.Segment) = .empty;
    try spans.ensureTotalCapacity(ctx.arena, filled + 12);

    // Label
    if (self.label) |label| {
        spans.appendAssumeCapacity(.{ .text = label, .style = theme.label_style });
        spans.appendAssumeCapacity(.{ .text = " ", .style = theme.label_style });
    }

    // Opening bracket
    spans.appendAssumeCapacity(.{ .text = "[", .style = theme.dim_style });

    // Filled blocks — per-block gradient color
    for (0..filled) |i| {
        const pos_pct: u8 = if (bar_width > 1)
            @intCast(@as(u32, @intCast(i)) * 100 / (@as(u32, bar_width) - 1))
        else
            0;
        spans.appendAssumeCapacity(.{
            .text = "\u{2588}",
            .style = .{ .fg = theme.colorForPercent(pos_pct) },
        });
    }

    // Empty blocks
    if (empty_count > 0) {
        const empty_str = try utils.repeatStr(ctx.arena, "\u{2591}", empty_count);
        spans.appendAssumeCapacity(.{ .text = empty_str, .style = .{ .fg = theme.gauge_empty } });
    }

    // Closing bracket (no space — pct% collides with bar)
    spans.appendAssumeCapacity(.{ .text = "]", .style = theme.dim_style });

    // Percentage text
    spans.appendAssumeCapacity(.{ .text = pct_str, .style = .{ .bold = true, .fg = theme.colorForPercent(self.value) } });

    // Suffix (right-aligned)
    if (self.suffix) |suffix| {
        const suffix_text_w: u16 = @intCast(ctx.stringWidth(suffix));
        const pad = effective_suffix_area -| suffix_text_w;
        if (pad > 0) {
            const pad_str = try utils.repeatStr(ctx.arena, " ", pad);
            spans.appendAssumeCapacity(.{ .text = pad_str, .style = theme.dim_style });
        }
        spans.appendAssumeCapacity(.{ .text = suffix, .style = theme.dim_style });
    }

    const rich: RichText = .{
        .text = spans.items,
        .softwrap = false,
        .overflow = .clip,
        .width_basis = .parent,
    };
    return rich.draw(ctx.withConstraints(
        .{ .width = width, .height = 1 },
        .{ .width = width, .height = 1 },
    ));
}
