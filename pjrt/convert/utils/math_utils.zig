const std = @import("std");

pub fn nanoToPico(n: u64) u64 {
    return n * 1000;
}

pub fn picoToMicro(p: u64) f64 {
    return @as(f64, @floatFromInt(p)) / 1E6;
}
