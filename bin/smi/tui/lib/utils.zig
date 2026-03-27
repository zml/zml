const std = @import("std");

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

/// Maps values from [min_val, max_val] to [0, 100], clamped.
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
