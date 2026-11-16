const std = @import("std");
const smi = @import("zml-smi");

pub fn bytesToMb(val: ?u64) u64 {
    return if (val) |v| v / (1024 * 1024) else 0;
}

pub fn repeatStr(arena: std.mem.Allocator, char: []const u8, count: u16) std.mem.Allocator.Error![]const u8 {
    const buf = try arena.alloc(u8, char.len * count);
    for (0..count) |i| {
        @memcpy(buf[i * char.len ..][0..char.len], char);
    }
    return buf;
}

/// Maps measurements to [0, 100], preserving missing observations as gaps.
pub fn normalizeRange(
    arena: std.mem.Allocator,
    raw: []const ?u64,
    min_val: u64,
    max_val: u64,
) std.mem.Allocator.Error![]?u8 {
    const result = try arena.alloc(?u8, raw.len);
    const range = max_val -| min_val;
    for (raw, result) |reading, *normalized| {
        const value = reading orelse {
            normalized.* = null;
            continue;
        };
        normalized.* = if (value <= min_val)
            0
        else if (value >= max_val)
            100
        else if (range > 0)
            @intCast(@as(u128, value - min_val) * 100 / range)
        else
            0;
    }
    return result;
}

/// " (driver, CUDA 13.4)": the driver version and the details, all of them or those shown in titles.
pub fn versions(arena: std.mem.Allocator, device: smi.Device, comptime details: enum { in_title, all }) std.mem.Allocator.Error![]const u8 {
    var text: std.Io.Writer.Allocating = .init(arena);
    const writer = &text.writer;
    if (device.driver_version) |driver| writer.writeAll(driver) catch return error.OutOfMemory;
    for (device.details) |detail| {
        if (details == .in_title and !detail.in_title) continue;
        if (text.written().len > 0) writer.writeAll(", ") catch return error.OutOfMemory;
        writer.print("{s} {s}", .{ detail.label, detail.value }) catch return error.OutOfMemory;
    }
    if (text.written().len == 0) return "";
    return std.fmt.allocPrint(arena, " ({s})", .{text.written()});
}
