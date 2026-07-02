const std = @import("std");

const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;

const ui = @import("../lib/ui.zig");
const theme = @import("../theme.zig");

const BarChart = @This();

// Define the bucket data structure for bar chart
pub const BucketData = struct {
    upper_bound: i64,
    percentage: u8,
};

// Bar chart data structure for histograms
buckets: []const BucketData,
bar_height: u16 = 5,
show_values: bool = true,
show_bounds: bool = true,
label: ?[]const u8 = null,

pub fn draw(self: *const BarChart, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const max_w = ctx.max.width orelse 40;
    const bar_width: u16 = 4; // Fixed width for each bar
    const bar_spacing: u16 = 1; // Space between bars
    const chars_per_bar: u16 = bar_width + bar_spacing;
    const num_bars: u16 = @min((max_w / chars_per_bar), @as(u16, @intCast(self.buckets.len))); // Each bar needs ~5 chars minimum
    const used_width: u16 = @min(max_w, chars_per_bar * num_bars);

    // Always allocate space for label for simplicity
    const total_height: u16 = self.bar_height + 3; // +3 for axis, values, and label
    var surface = try vxfw.Surface.init(ctx.arena, ui.widget(self), .{ .width = used_width, .height = total_height });
    try self.renderTo(ctx.arena, ctx, &surface, 0, 0, used_width);

    // Draw label if present
    if (self.label) |label_text| {
        writeStr(&surface, 0, total_height - 1, label_text, theme.dim_style);
    }

    return surface;
}

/// Write a string to the surface
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

/// Format a number with K/M/G/T/P suffixes for powers of 1024
fn formatBound(arena: std.mem.Allocator, value: i64) std.mem.Allocator.Error![]const u8 {
    const units = [_][]const u8{
        "", "K", "M", "G", "T", "P", "H",
    };

    var i: u8 = 0;
    var current: i64 = value;

    while (current >= 1000 and i < units.len) : (i += 1) {
        current = @divFloor(current, 1000);
    }

    const result = try std.fmt.allocPrint(arena, "{d}{s}", .{ current, units[i] });
    std.debug.assert(result.len <= 5);
    return result;
}

/// Render bar chart data into an existing surface at the given offset.
pub fn renderTo(self: *const BarChart, arena: std.mem.Allocator, ctx: vxfw.DrawContext, surface: *vxfw.Surface, col_offset: u16, row_offset: u16, w: u16) std.mem.Allocator.Error!void {
    const bar_width: u16 = 4; // Fixed width for each bar
    const bar_spacing: u16 = 1; // Space between bars
    const chars_per_bar: u16 = bar_width + bar_spacing;
    const num_bars: u16 = @min(@as(u16, @intCast(w / chars_per_bar)), @as(u16, @intCast(self.buckets.len)));
    const chart_height = self.bar_height;
    var max_percentage: u8 = 0;
    if (self.buckets.len > 0) {
        for (self.buckets) |bucket| {
            max_percentage = @max(max_percentage, bucket.percentage);
        }
    }

    // Draw x-axis line
    for (0..w) |col| {
        surface.writeCell(col_offset + @as(u16, @intCast(col)), row_offset + chart_height, .{
            .char = .{ .grapheme = "-", .width = 1 },
            .style = theme.dim_style,
        });
    }

    // Draw bars
    for (0..num_bars) |i| {
        const bucket_idx = @as(usize, @intCast((@as(u64, @intCast(i)) * @as(u64, @intCast(self.buckets.len))) % @as(u64, @intCast(num_bars))));
        const bucket = self.buckets[bucket_idx];
        const bar_col = col_offset + @as(u16, @intCast(i * chars_per_bar));

        if (bucket.percentage == 0) continue;

        // Skip display of the last bucket (typically has arbitrary large upper bound)
        const is_last_bucket = bucket_idx == self.buckets.len - 1;

        // Calculate filled height and empty height
        const filled_height = if (max_percentage > 0) @as(u16, @intCast(bucket.percentage * @as(u32, chart_height) / @as(u32, max_percentage))) else 0;
        const empty_height = chart_height - filled_height;

        // Draw empty part (if any)
        for (0..empty_height) |row| {
            const row_pos = row_offset + @as(u16, @intCast(row));
            for (0..bar_width) |bar_col_idx| {
                surface.writeCell(bar_col + @as(u16, @intCast(bar_col_idx)), row_pos, .{
                    .char = .{ .grapheme = " ", .width = 1 },
                    .style = theme.dim_style,
                });
            }
        }

        // Draw filled part (from bottom up)
        const color = theme.colorForPercent(bucket.percentage);
        for (0..filled_height) |row| {
            const row_pos = row_offset + @as(u16, @intCast(chart_height - 1 - row));
            for (0..bar_width) |bar_col_idx| {
                surface.writeCell(bar_col + @as(u16, @intCast(bar_col_idx)), row_pos, .{
                    .char = .{ .grapheme = "█", .width = 1 },
                    .style = .{ .fg = color, .bg = .default },
                });
            }
        }

        // Show value and bound labels (stacked)
        if (self.show_values) {
            const value_str = try std.fmt.allocPrint(arena, "{d}%", .{bucket.percentage});
            const value_w = @as(u16, @intCast(ctx.stringWidth(value_str)));
            const value_pos = bar_col + @as(u16, @intCast(bar_width / 2 - value_w / 2));
            const value_row: u16 = row_offset + chart_height + 1;
            for (0..value_w) |col_idx| {
                const ch_str = value_str[col_idx .. col_idx + 1];
                surface.writeCell(value_pos + @as(u16, @intCast(col_idx)), value_row, .{
                    .char = .{ .grapheme = ch_str, .width = 1 },
                    .style = theme.dim_style,
                });
            }
        }

        if (self.show_bounds and !is_last_bucket) {
            const bound_str = try formatBound(arena, bucket.upper_bound);
            const bound_w = @as(u16, @intCast(ctx.stringWidth(bound_str)));
            const bound_pos = if (bound_w < bar_width)
                bar_col + @as(u16, @intCast(bar_width / 2 - bound_w / 2))
            else
                bar_col;
            const bound_row: u16 = row_offset + chart_height + 2;

            for (0..bound_w) |col_idx| {
                const ch_str = bound_str[col_idx .. col_idx + 1];
                surface.writeCell(bound_pos + @as(u16, @intCast(col_idx)), bound_row, .{
                    .char = .{ .grapheme = ch_str, .width = 1 },
                    .style = theme.dim_style,
                });
            }
        }
    }
}
