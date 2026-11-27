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
pub fn normalizeRange(arena: std.mem.Allocator, raw: *const std.Deque(u64), min_val: u64, max_val: u64) std.mem.Allocator.Error![]u8 {
    const result = try arena.alloc(u8, raw.len);
    const range = max_val -| min_val;
    for (0..raw.len) |i| {
        const v = raw.at(i);
        result[i] = blk: {
            if (v <= min_val) break :blk 0;
            if (v >= max_val) break :blk 100;
            if (range > 0) break :blk @intCast(@as(u128, v - min_val) * 100 / range);

            break :blk 0;
        };
    }
    return result;
}
