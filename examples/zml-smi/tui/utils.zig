const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;

pub const ImageCellSize = struct {
    cols: u16,
    rows: u16,
};

/// Compute the cell dimensions for an image scaled to a target row count,
/// maintaining aspect ratio based on the terminal's cell pixel size.
pub fn imageCellSize(img: vaxis.Image, target_rows: u16, cell_size: vxfw.Size) ImageCellSize {
    const cell_h: u32 = if (cell_size.height > 0) cell_size.height else 20;
    const cell_w: u32 = if (cell_size.width > 0) cell_size.width else 10;
    const height_px = @as(u32, target_rows) * cell_h;
    const scale_f = @as(f64, @floatFromInt(height_px)) / @as(f64, @floatFromInt(img.height));
    const width_px: u32 = @intFromFloat(@as(f64, @floatFromInt(img.width)) * scale_f);
    const img_cols: u16 = @intCast((width_px + cell_w - 1) / cell_w);
    return .{ .cols = img_cols, .rows = target_rows };
}

pub fn formatUptime(arena: std.mem.Allocator, seconds: u64) std.mem.Allocator.Error![]const u8 {
    const days = seconds / 86400;
    const hours = (seconds % 86400) / 3600;
    const mins = (seconds % 3600) / 60;
    if (days > 0) {
        return std.fmt.allocPrint(arena, "{d}d {d}h {d}m", .{ days, hours, mins });
    } else if (hours > 0) {
        return std.fmt.allocPrint(arena, "{d}h {d}m", .{ hours, mins });
    } else {
        return std.fmt.allocPrint(arena, "{d}m", .{mins});
    }
}

pub fn repeatStr(arena: std.mem.Allocator, char: []const u8, count: u16) std.mem.Allocator.Error![]const u8 {
    const buf = try arena.alloc(u8, char.len * count);
    for (0..count) |i| {
        @memcpy(buf[i * char.len ..][0..char.len], char);
    }
    return buf;
}

/// Map values from [min_val, max_val] to [0, 100], clamped.
pub fn normalizeRange(arena: std.mem.Allocator, raw: []const u64, min_val: u64, max_val: u64) std.mem.Allocator.Error![]u8 {
    const result = try arena.alloc(u8, raw.len);
    const range = max_val -| min_val;
    for (raw, 0..) |v, i| {
        result[i] = if (v <= min_val)
            0
        else if (v >= max_val)
            100
        else if (range > 0)
            @intCast((v - min_val) * 100 / range)
        else
            0;
    }
    return result;
}

pub fn strSlice(buf: *const ?[256]u8) []const u8 {
    if (buf.*) |*b| return std.mem.sliceTo(@as([*:0]const u8, @ptrCast(b)), 0);
    return "N/A";
}

pub fn bytesToMb(val: ?u64) u64 {
    return if (val) |v| v / (1024 * 1024) else 0;
}

pub fn parseLoadAvg(raw: ?[256]u8) [3]f32 {
    const str = if (raw) |*b| std.mem.sliceTo(@as([*:0]const u8, @ptrCast(b)), 0) else return .{ 0, 0, 0 };
    var result: [3]f32 = .{ 0, 0, 0 };
    var it = std.mem.splitScalar(u8, str, ' ');
    for (&result) |*r| {
        const token = it.next() orelse break;
        r.* = std.fmt.parseFloat(f32, token) catch 0;
    }
    return result;
}

pub fn formatBandwidth(arena: std.mem.Allocator, kbps: u64) std.mem.Allocator.Error![]const u8 {
    if (kbps >= 1_000_000) {
        const gbps = @as(f64, @floatFromInt(kbps)) / 1_000_000.0;
        return std.fmt.allocPrint(arena, " {d:.1} GB/s", .{gbps});
    } else if (kbps >= 1_000) {
        const mbps = @as(f64, @floatFromInt(kbps)) / 1_000.0;
        return std.fmt.allocPrint(arena, " {d:.1} MB/s", .{mbps});
    } else {
        return std.fmt.allocPrint(arena, " {d} KB/s", .{kbps});
    }
}
