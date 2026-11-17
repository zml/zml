const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const theme = @import("../theme.zig");
const ui = @import("../lib/ui.zig");
const TitledBorder = @import("titled_border.zig");
const braille = @import("braille_chart.zig");

const Chart = @This();

const chart_height: u16 = 8;
const time_row: u16 = chart_height + 1;
const info_row: u16 = time_row + 1;

title: []const u8,
data: []const ?u8, // 0-100 normalized; null leaves a gap
value_label: ?[]const u8 = null,
info_line: ?[]const u8 = null,
y_min: u32 = 0,
y_max: u32 = 100,
y_unit: []const u8 = "%",
sample_interval_ms: u32,

pub fn draw(self: *const Chart, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const tb: TitledBorder = .{
        .child = ui.drawWidget(self, drawContent),
        .title = self.title,
        .value_label = self.value_label,
    };
    return tb.draw(ctx);
}

fn drawContent(self: *const Chart, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const max_w = ctx.max.width orelse 40;

    // ── Y-axis label widths ──────────────────────────────────
    var y_labels: [3]struct { text: []const u8, width: u16 } = undefined;
    const y_values = [3]u32{ self.y_max, (self.y_min + self.y_max) / 2, self.y_min };
    var y_label_w: u16 = 0;
    for (&y_labels, y_values) |*label, value| {
        const text = try std.fmt.allocPrint(ctx.arena, "{d}{s}", .{ value, self.y_unit });
        label.* = .{ .text = text, .width = @intCast(ctx.stringWidth(text)) };
        y_label_w = @max(y_label_w, label.width);
    }

    // ── Column layout ────────────────────────────────────────
    const chart_start: u16 = y_label_w + 1;
    const chart_w = max_w -| chart_start;

    const surface = try vxfw.Surface.init(ctx.arena, ui.drawWidget(self, drawContent), .{
        .width = max_w,
        .height = info_row + 1,
    });
    if (chart_w == 0) return surface;

    // ── Y-axis labels + axis line ────────────────────────────
    for (y_labels, [3]u16{ 0, chart_height / 2, chart_height }) |label, row| {
        writeLabel(surface, y_label_w - label.width, row, label.text);
    }
    for (0..chart_height + 1) |row| {
        const glyph = if (row == chart_height) "\u{2514}" else "\u{2502}";
        writeGlyph(surface, y_label_w, @intCast(row), glyph);
    }

    // ── Braille data area ────────────────────────────────────
    braille.draw(surface, chart_start, chart_height, self.data);

    // ── X-axis line ──────────────────────────────────────────
    const tick_mid: u16 = chart_w / 2;
    const tick_right = chart_w - 1;
    for (0..chart_w) |col| {
        const is_tick = col == 0 or col == tick_mid or col == tick_right;
        const glyph = if (is_tick) "\u{252C}" else "\u{2500}";
        writeGlyph(surface, chart_start + @as(u16, @intCast(col)), chart_height, glyph);
    }

    // ── X-axis time labels ───────────────────────────────────
    const total_ms = (@as(u64, chart_w) * 2 - 1) * self.sample_interval_ms;
    const oldest = try fmtTime(ctx.arena, @divCeil(total_ms, 1000));
    const middle = try fmtTime(ctx.arena, @divCeil(total_ms / 2, 1000));
    const now = "now";
    // Time labels are ASCII, so their byte lengths are also their column widths.
    const half_width: u16 = @intCast(middle.len / 2);
    writeLabel(surface, chart_start, time_row, oldest);
    writeLabel(surface, chart_start + tick_mid -| half_width, time_row, middle);
    writeLabel(surface, max_w -| @as(u16, now.len), time_row, now);

    // ── Info line ────────────────────────────────────────────
    if (self.info_line) |info| writeLabel(surface, chart_start, info_row, info);

    return surface;
}

fn writeGlyph(surface: vxfw.Surface, col: u16, row: u16, glyph: []const u8) void {
    surface.writeCell(col, row, .{
        .char = .{ .grapheme = glyph, .width = 1 },
        .style = theme.dim_style,
    });
}

fn writeLabel(surface: vxfw.Surface, start_col: u16, row: u16, text: []const u8) void {
    var col = start_col;
    var index: usize = 0;
    while (index < text.len and col < surface.size.width) : (col += 1) {
        const len = std.unicode.utf8ByteSequenceLength(text[index]) catch 1;
        const end = @min(index + len, text.len);
        writeGlyph(surface, col, row, text[index..end]);
        index = end;
    }
}

fn fmtTime(arena: std.mem.Allocator, seconds: u64) std.mem.Allocator.Error![]const u8 {
    if (seconds >= 120) return std.fmt.allocPrint(arena, "-{d}m", .{seconds / 60});
    return std.fmt.allocPrint(arena, "-{d}s", .{seconds});
}
