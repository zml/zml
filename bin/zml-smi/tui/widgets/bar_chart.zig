const std = @import("std");

const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;

const ui = @import("../lib/ui.zig");
const theme = @import("../theme.zig");

const BarChart = @This();

// Define the bucket data structure for bar chart
pub const BucketData = struct {
    upper_bound: f64,
    percentage: u8,
};

// Bar chart data structure for histograms
buckets: []const BucketData,
bar_height: u16 = 5,
show_values: bool = true,
show_bounds: bool = true,
label: ?[]const u8 = null,

pub fn height(self: *const BarChart) u16 {
    var total_height: u16 = self.bar_height + 1;
    if (self.label) |_| total_height += 1;
    if (self.show_values) total_height += 1;
    if (self.show_bounds) total_height += 1;
    return total_height + 1;
}

pub fn draw(self: *const BarChart, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
    const max_w = ctx.max.width orelse 40;
    const bar_width: u16 = 4; // Fixed width for each bar
    const bar_spacing: u16 = 1; // Space between bars
    const chars_per_bar: u16 = bar_width + bar_spacing;
    const num_bars: u16 = @min((max_w / chars_per_bar), @as(u16, @intCast(self.buckets.len))); // Each bar needs ~5 chars minimum
    const used_width: u16 = @min(max_w, chars_per_bar * num_bars);

    var surface = try vxfw.Surface.init(ctx.arena, ui.widget(self), .{ .width = used_width, .height = self.height() });

    // Draw label if present
    // if (self.label) |label_text| {
    //     writeStr(&surface, 0, 0, label_text, theme.label_style);
    // }

    try self.renderTo(ctx.arena, ctx, &surface);
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

fn writeAscii(surface: *vxfw.Surface, col: u16, row: u16, text: []const u8, style: vaxis.Cell.Style) void {
    for (0..text.len) |col_idx_usize| {
        const col_idx: u16 = @intCast(col_idx_usize);
        surface.writeCell(col + col_idx, row, .{
            .char = .{ .grapheme = text[col_idx .. col_idx + 1], .width = 1 },
            .style = style,
        });
    }
}

/// Format a number with K/M/G/T/P suffixes for powers of 1024
fn formatBound(arena: std.mem.Allocator, value: f64) std.mem.Allocator.Error![]const u8 {
    const units = [_][]const u8{
        "", "K", "M", "G", "T", "P", "H",
    };

    var i: u8 = 0;
    var current: f64 = value;

    while (current >= 1000 and i < units.len) : (i += 1) {
        current = @divFloor(current, 1000);
    }

    var result: []const u8 = try std.fmt.allocPrint(arena, "{d:.3}", .{current});
    result = std.mem.trim(u8, result, "0");
    result = std.mem.trimEnd(u8, result, ".");
    if (units[i].len > 0) {
        result = result[0..@min(4, result.len)];
        result = try std.fmt.allocPrint(arena, "{s}{s}", .{ std.mem.trim(u8, result, "0"), units[i] });
    }
    return result;
}

/// Render bar chart data into an existing surface at the given offset.
pub fn renderTo(self: *const BarChart, arena: std.mem.Allocator, ctx: vxfw.DrawContext, surface: *vxfw.Surface) std.mem.Allocator.Error!void {
    const size = surface.size;
    const empty_bar = "    ";
    const bar_width: u16 = empty_bar.len; // Fixed width for each bar
    const bar_spacing: u16 = 1; // Space between bars
    // const chars_per_bar: u16 = bar_width + bar_spacing;

    const bar_height: u16 = self.bar_height;
    const bar_area: u16 = bar_width * bar_height;
    var max_percentage: u16 = 0;
    if (self.buckets.len > 0) {
        for (self.buckets) |bucket| {
            max_percentage = @max(max_percentage, bucket.percentage);
        }
    }

    const row_label: u16 = 0;
    const row_top_of_bar = if (self.label) |_| row_label + 1 else row_label;
    const row_values = row_top_of_bar + bar_height + 1;
    const row_bounds = if (self.show_values) row_values + 1 else row_values;
    const row_border = if (self.show_bounds) row_bounds + 1 else row_bounds;
    std.debug.assert(size.height == row_border + 1);
    // Draw title
    if (self.label) |label_text| {
        writeStr(surface, 0, row_label, label_text, theme.label_style);
    }

    // Draw x-axis line
    for (0..size.width) |col| {
        surface.writeCell(@intCast(col), row_border, .{
            .char = .{ .grapheme = "-", .width = 1 },
            .style = theme.border_style,
        });
    }

    // Clean the entire bar chart area from previous renders to prevent scrolling artifacts
    // Clear bar area, value area, and bounds area
    const max_row = if (self.show_values and !self.show_bounds) row_values else row_bounds;
    for (0..size.width) |col| {
        for (row_top_of_bar..max_row) |row| {
            writeAscii(surface, @as(u16, @intCast(col)), @as(u16, @intCast(row)), empty_bar, theme.dim_style);
        }
    }

    // Draw bars
    var bar_col: u16 = 0;
    var bucket_idx: u16 = 0;

    while (bucket_idx < self.buckets.len and bar_col < size.width) : (bucket_idx += 1) {
        const bucket = self.buckets[bucket_idx];
        if (bucket.percentage == 0) continue;
        defer bar_col += bar_width + bar_spacing;

        // Calculate filled height and remainder
        const bucket_area: u16 = bucket.percentage * bar_area / max_percentage;
        const full_rows = bucket_area / bar_width;
        const remainder: u8 = @intCast(bucket_area % bar_width);
        const empty_height = bar_height - (if (remainder == 0) full_rows else full_rows + 1);

        const color = theme.colorForPercent(bucket.percentage);
        var row_idx: u16 = row_top_of_bar + empty_height;
        // Draw the middle partial row (split into 4 columns based on remainder)
        // The partial row sits at the boundary between empty rows (top) and full rows (bottom)
        if (remainder > 0) {
            surface.writeCell(bar_col, row_idx, .{
                .char = .{ .grapheme = "████"[0 .. remainder * "█".len], .width = remainder },
                .style = .{ .fg = color, .bg = .default },
            });
            row_idx += 1;
        }

        // Draw fully filled rows (█) from the row after partial row to the bottom
        for (row_idx..row_idx + full_rows) |row_idx_usize| {
            surface.writeCell(bar_col, @intCast(row_idx_usize), .{
                .char = .{ .grapheme = "████", .width = 4 },
                .style = .{ .fg = color, .bg = .default },
            });
        }

        // Show value and bound labels (stacked)
        if (self.show_values) {
            const value_str = try std.fmt.allocPrint(arena, "{d}%", .{bucket.percentage});
            // defer arena.free(value_str);
            const value_w: u16 = @intCast(ctx.stringWidth(value_str));
            const value_pos = bar_col + @as(u16, @intCast(bar_width / 2 - value_w / 2));
            writeStr(surface, value_pos, row_values, value_str, theme.value_style);
        }

        // Skip display of the last bucket (typically has arbitrary large upper bound)
        const is_last_bucket = bucket_idx == self.buckets.len - 1;
        if (self.show_bounds and !is_last_bucket) {
            const bound_str = try formatBound(arena, bucket.upper_bound);
            const bound_w: u16 = @intCast(ctx.stringWidth(bound_str));
            const bound_pos = if (bound_w < bar_width)
                bar_col + @as(u16, @intCast(bar_width / 2 - bound_w / 2))
            else
                bar_col;
            writeStr(surface, bound_pos, row_bounds, bound_str, theme.value_style);
        }
    }
}
