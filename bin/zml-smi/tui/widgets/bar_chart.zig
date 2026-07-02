const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const theme = @import("../theme.zig");
const ui = @import("../lib/ui.zig");

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

pub fn draw(self: *const BarChart, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const max_w = ctx.max.width orelse 40;
    const bar_width: u16 = 4; // Fixed width for each bar
    const bar_spacing: u16 = 1; // Space between bars
    const chars_per_bar: u16 = bar_width + bar_spacing;
    const num_bars: u16 = @min((max_w / chars_per_bar), @as(u16, @intCast(self.buckets.len))); // Each bar needs ~5 chars minimum
    const used_width: u16 = @min(max_w, chars_per_bar * num_bars);

    var surface = try vxfw.Surface.init(ctx.arena, ui.widget(self), .{ .width = used_width, .height = self.bar_height + 2 }); // +2 for axis and labels
    try self.renderTo(ctx.arena, ctx, &surface, 0, 0, used_width);
    return surface;
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

        // Draw filled part
        const color = theme.colorForPercent(bucket.percentage);
        for (filled_height..chart_height) |row| {
            const row_pos = row_offset + @as(u16, @intCast(row));
            for (0..bar_width) |bar_col_idx| {
                surface.writeCell(bar_col + @as(u16, @intCast(bar_col_idx)), row_pos, .{
                    .char = .{ .grapheme = "█", .width = 1 },
                    .style = .{ .fg = color, .bg = .default },
                });
            }
        }

        // Show value labels if enabled
        if (self.show_values) {
            const value_str = try std.fmt.allocPrint(arena, "{d}%", .{bucket.percentage});
            const value_w = @as(u16, @intCast(ctx.stringWidth(value_str)));
            const value_pos = bar_col + @as(u16, @intCast(bar_width / 2 - value_w / 2));
            for (0..value_w) |col_idx| {
                const ch_str = value_str[col_idx .. col_idx + 1];
                surface.writeCell(value_pos + @as(u16, @intCast(col_idx)), row_offset + chart_height + 1, .{
                    .char = .{ .grapheme = ch_str, .width = 1 },
                    .style = theme.dim_style,
                });
            }
        }
    }
}
