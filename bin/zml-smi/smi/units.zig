const std = @import("std");

/// Decimal prefixes, a factor of 1000 apart.
pub const Prefix = enum(u3) { nano, micro, milli, one, kilo, mega, giga };

/// Converts between prefixes of one unit, rounding to nearest: `scale(1500, .micro, .milli)` is 2.
/// A result that does not fit is the largest u64.
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
