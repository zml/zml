const std = @import("std");

/// Decimal prefixes, a factor of 1000 apart.
pub const Prefix = enum(u3) { nano, micro, milli, one, kilo, mega, giga };

/// Converts between prefixes of one unit, rounding to nearest: `scale(1500, .micro, .milli)` is 2.
/// A result that does not fit is the largest u64.
pub fn scale(value: u64, from: Prefix, to: Prefix) u64 {
    const steps = @as(i4, @backingInt(from)) - @as(i4, @backingInt(to));
    const factor = std.math.powi(u64, 1000, @abs(steps)) catch unreachable;
    if (steps >= 0) return value *| factor;
    return value / factor + @intFromBool(value % factor >= factor / 2);
}
