const std = @import("std");
const utils = @import("utils.zig");

const normalizeRange = utils.normalizeRange;

test "normalization preserves missing observations and safely scales large values" {
    const allocator = std.testing.allocator;
    const readings = [_]?u64{ null, 0, 25, 50, 100, 150 };
    const normalized = try normalizeRange(allocator, &readings, 0, 100);
    defer allocator.free(normalized);
    try std.testing.expectEqualSlices(?u8, &.{ null, 0, 25, 50, 100, 100 }, normalized);

    const maximum = std.math.maxInt(u64);
    const large = try normalizeRange(allocator, &.{ 0, maximum / 2, maximum }, 0, maximum);
    defer allocator.free(large);
    try std.testing.expectEqualSlices(?u8, &.{ 0, 49, 100 }, large);
}
