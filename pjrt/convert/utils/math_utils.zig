const std = @import("std");

pub fn nanoToPico(n: u128) u128 {
    return n * 1000;
}

pub fn picoToMicro(p: anytype) f64 {
    return @as(f64, @floatFromInt(p)) / 1E6;
}
