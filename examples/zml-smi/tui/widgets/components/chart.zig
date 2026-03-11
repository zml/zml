const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const theme = @import("../../theme.zig");
const TitledBorder = @import("titled_border.zig");
const BrailleChart = @import("braille_chart.zig");

const Chart = @This();

// Layout constants for rows below the chart data area
const x_axis_h: u16 = 1; // x-axis line
const time_labels_h: u16 = 1; // time labels below x-axis
const info_line_h: u16 = 1; // info text below time labels
const border_h: u16 = 3; // TitledBorder overhead (border + padding)

title: []const u8,
data: []const u8, // 0-100 normalized
value_label: ?[]const u8 = null,
info_line: ?[]const u8 = null,
y_min: u32 = 0,
y_max: u32 = 100,
y_unit: []const u8 = "%",
chart_height: u16 = 6,
tui_refresh_rate: u16,

pub fn widget(self: *const Chart) vxfw.Widget {
    return .{
        .userdata = @constCast(self),
        .drawFn = typeErasedDrawFn,
    };
}

fn typeErasedDrawFn(ptr: *anyopaque, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const self: *const Chart = @ptrCast(@alignCast(ptr));
    const content_h: u16 = self.chart_height + x_axis_h + time_labels_h + info_line_h;
    const total_h: u16 = content_h + border_h;
    const tb: TitledBorder = .{
        .child = .{ .userdata = @constCast(self), .drawFn = typeErasedContentFn },
        .title = self.title,
        .value_label = self.value_label,
    };
    return tb.draw(ctx.withConstraints(ctx.min, .{
        .width = ctx.max.width,
        .height = if (ctx.max.height) |h| h else total_h,
    }));
}

fn typeErasedContentFn(ptr: *anyopaque, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const self: *const Chart = @ptrCast(@alignCast(ptr));
    return self.drawContent(ctx);
}

fn drawContent(self: *const Chart, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const max_w = ctx.max.width orelse 40;
    const content_h: u16 = self.chart_height + x_axis_h + time_labels_h + info_line_h;
    const content_widget: vxfw.Widget = .{ .userdata = @constCast(self), .drawFn = typeErasedContentFn };

    // ── Y-axis label widths ──────────────────────────────────
    const y_labels = [3]struct { row: u16, val: u32 }{
        .{ .row = 0, .val = self.y_max },
        .{ .row = self.chart_height / 2, .val = (self.y_min + self.y_max) / 2 },
        .{ .row = self.chart_height, .val = self.y_min },
    };
    var y_strs: [3][]const u8 = undefined;
    var y_label_w: u16 = 0;
    for (&y_strs, y_labels) |*s, lbl| {
        s.* = try std.fmt.allocPrint(ctx.arena, "{d}{s}", .{ lbl.val, self.y_unit });
        y_label_w = @max(y_label_w, @as(u16, @intCast(ctx.stringWidth(s.*))));
    }

    // ── Column layout ────────────────────────────────────────
    const chart_start: u16 = y_label_w + 1;
    const chart_w: u16 = if (max_w > chart_start) max_w - chart_start else 0;

    var surface = try vxfw.Surface.init(ctx.arena, content_widget, .{ .width = max_w, .height = content_h });
    if (chart_w == 0) return surface;

    // ── Y-axis labels + axis line ────────────────────────────
    for (y_strs, y_labels) |text, lbl| {
        const w: u16 = @intCast(ctx.stringWidth(text));
        writeStr(&surface, y_label_w - w, lbl.row, text, theme.dim_style);
    }
    for (0..self.chart_height + 1) |r| {
        const row: u16 = @intCast(r);
        const ch: []const u8 = if (row == self.chart_height)
            "\u{2514}"
        else
            "\u{2502}";
        surface.writeCell(y_label_w, row, .{
            .char = .{ .grapheme = ch, .width = 1 },
            .style = theme.dim_style,
        });
    }

    // ── Braille data area ────────────────────────────────────
    const braille: BrailleChart = .{ .values = self.data, .height = self.chart_height };
    try braille.renderTo(ctx.arena, &surface, chart_start, 0, chart_w);

    // ── X-axis line ──────────────────────────────────────────
    const tick_mid: u16 = chart_w / 2;
    const tick_right: u16 = chart_w -| 1;
    for (0..chart_w) |c| {
        const col: u16 = @intCast(c);
        const is_tick = (col == 0 or col == tick_mid or col == tick_right);
        surface.writeCell(chart_start + col, self.chart_height, .{
            .char = .{ .grapheme = if (is_tick) "\u{252C}" else "\u{2500}", .width = 1 },
            .style = theme.dim_style,
        });
    }

    // ── X-axis time labels ───────────────────────────────────
    const total_ms: u32 = @as(u32, chart_w) * 2 * @as(u32, self.tui_refresh_rate);
    const total_s: u16 = @intCast(std.math.divCeil(u32, total_ms, 1000) catch unreachable);
    const mid_s: u16 = @intCast(std.math.divCeil(u32, total_ms / 2, 1000) catch unreachable);
    const label_row = self.chart_height + x_axis_h;
    const left_str = try fmtTime(ctx.arena, total_s);
    const mid_str = try fmtTime(ctx.arena, mid_s);

    writeStr(&surface, chart_start, label_row, left_str, theme.dim_style);

    const mid_w: u16 = @intCast(ctx.stringWidth(mid_str));
    writeStr(&surface, chart_start + tick_mid -| (mid_w / 2), label_row, mid_str, theme.dim_style);

    const right_str: []const u8 = "now";
    const right_w: u16 = @intCast(ctx.stringWidth(right_str));
    writeStr(&surface, chart_start + tick_right + 1 -| right_w, label_row, right_str, theme.dim_style);

    // ── Info line ────────────────────────────────────────────
    if (self.info_line) |info| {
        writeStr(&surface, chart_start, self.chart_height + x_axis_h + time_labels_h, info, theme.dim_style);
    }

    return surface;
}

fn writeStr(surface: *vxfw.Surface, col: u16, row: u16, text: []const u8, style: vaxis.Cell.Style) void {
    var c = col;
    var i: usize = 0;
    while (i < text.len) {
        const len = std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
        const end = @min(i + len, text.len);
        surface.writeCell(c, row, .{
            .char = .{ .grapheme = text[i..end], .width = 1 },
            .style = style,
        });
        c += 1;
        i = end;
    }
}

fn fmtTime(arena: std.mem.Allocator, seconds: u16) std.mem.Allocator.Error![]const u8 {
    if (seconds >= 120) {
        return std.fmt.allocPrint(arena, "-{d}m", .{seconds / 60});
    } else {
        return std.fmt.allocPrint(arena, "-{d}s", .{seconds});
    }
}
