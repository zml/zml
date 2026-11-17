//! Shared low-level helpers for device and host readers.
const std = @import("std");

pub const sysfs = @import("sysfs.zig");
pub const hwmon = @import("hwmon.zig");
pub const pci = @import("pci.zig");
pub const fdinfo = @import("fdinfo.zig");
pub const proc = @import("proc.zig");
pub const dynlib = @import("dynlib.zig");
pub const units = @import("units.zig");

const counter = @import("counter.zig");
pub const Counter = counter.Counter;
pub const Counters = counter.Counters;
pub const Growth = counter.Growth;

/// A fixed-size macOS sysctl value.
pub fn sysctl(comptime T: type, name: [*:0]const u8) ?T {
    var value: T = std.mem.zeroes(T);
    var len: usize = @sizeOf(T);
    if (std.c.sysctlbyname(name, &value, &len, null, 0) != 0) return null;
    return value;
}

pub const testing = if (@import("builtin").is_test) struct {
    pub const counters = counter;
    pub const sandbox = @import("sandbox.zig");
} else struct {};
