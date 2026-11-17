const std = @import("std");
const scale = @import("zml-smi-2/native").units.scale;

test scale {
    try std.testing.expectEqual(2, scale(1500, .micro, .milli));
    try std.testing.expectEqual(1, scale(1499, .micro, .milli));
    try std.testing.expectEqual(58, scale(58_000, .milli, .one));
    try std.testing.expectEqual(230_000, scale(230_000_000, .micro, .milli));
    try std.testing.expectEqual(3_000_000, scale(3, .one, .micro));
    try std.testing.expectEqual(std.math.maxInt(u64), scale(std.math.maxInt(u64) / 10, .giga, .nano));
}
