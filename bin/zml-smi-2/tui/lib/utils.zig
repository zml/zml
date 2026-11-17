const std = @import("std");

pub fn mebibytes(bytes: ?u64) ?u64 {
    return if (bytes) |value| value / (1024 * 1024) else null;
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
        normalized.* = if (reading) |value| percent_blk: {
            if (value <= min_val) break :percent_blk 0;
            if (value >= max_val) break :percent_blk 100;
            break :percent_blk @intCast(@as(u128, value - min_val) * 100 / range);
        } else null;
    }
    return result;
}
