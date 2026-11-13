const std = @import("std");

pub const Prefix = enum(u3) { nano, micro, milli, one, kilo, mega, giga };

pub fn scale(value: u64, from: Prefix, to: Prefix) u64 {
    const from_step: u8 = @backingInt(from);
    const to_step: u8 = @backingInt(to);
    if (from_step >= to_step) {
        return std.math.mul(u64, value, factor(from_step - to_step)) catch std.math.maxInt(u64);
    }
    const divisor = factor(to_step - from_step);
    const rounded_up: u64 = @intFromBool(value % divisor * 2 >= divisor);
    return value / divisor + rounded_up;
}

fn factor(steps: u8) u64 {
    return std.math.powi(u64, 1000, steps) catch unreachable;
}

test scale {
    try std.testing.expectEqual(2, scale(1500, .micro, .milli));
    try std.testing.expectEqual(1, scale(1499, .micro, .milli));
    try std.testing.expectEqual(58, scale(58_000, .milli, .one));
    try std.testing.expectEqual(230_000, scale(230_000_000, .micro, .milli));
    try std.testing.expectEqual(3_000_000, scale(3, .one, .micro));
    try std.testing.expectEqual(std.math.maxInt(u64), scale(std.math.maxInt(u64) / 10, .giga, .nano));
}
