const std = @import("std");
const sysfs = @import("sysfs.zig");

pub fn find(device: sysfs.Dir, name: ?[]const u8) !sysfs.Dir {
    const hwmon = try device.openDir("hwmon");
    defer hwmon.close();

    var it = hwmon.iterate();
    while (try it.next(device.io)) |entry| {
        if (!std.mem.startsWith(u8, entry.name, "hwmon")) continue;
        const dir = hwmon.openDir(entry.name) catch continue;
        const wanted = name orelse return dir;
        var buffer: [64]u8 = undefined;
        if (std.mem.eql(u8, dir.read("name", &buffer) catch "", wanted)) return dir;
        dir.close();
    }

    return error.NotFound;
}

pub fn numbered(comptime pattern: []const u8) [15][]const u8 {
    var paths: [15][]const u8 = undefined;

    for (&paths, 1..) |*path, i| {
        path.* = std.fmt.comptimePrint(pattern, .{i});
    }

    return paths;
}

pub fn firstReadable(dir: sysfs.Dir, paths: []const []const u8) ?[]const u8 {
    for (paths) |path| {
        _ = dir.int(u64, path) catch continue;
        return path;
    }

    return null;
}

pub fn labeled(arena: std.mem.Allocator, dir: sysfs.Dir, sensor: []const u8, suffix: []const u8, label: []const u8) ?[]const u8 {
    var it = dir.iterate();
    var label_buf: [64]u8 = undefined;
    var path_buf: [64]u8 = undefined;

    while (it.next(dir.io) catch null) |entry| {
        const rest = std.mem.cutPrefix(u8, entry.name, sensor) orelse continue;
        const index = std.mem.cutSuffix(u8, rest, "_label") orelse continue;
        if (!std.ascii.eqlIgnoreCase(dir.read(entry.name, &label_buf) catch continue, label)) continue;

        const file = std.fmt.bufPrint(&path_buf, "{s}{s}_{s}", .{ sensor, index, suffix }) catch continue;
        _ = dir.int(u64, file) catch continue;
        return arena.dupe(u8, file) catch null;
    }

    return null;
}

/// Converts an 8-bit PWM duty cycle (0..255) to a 0..100 percentage, rounded to the nearest integer.
pub fn pwmPercent(pwm: u64) u64 {
    const max: u64 = std.math.maxInt(u8);
    return (@as(u64, @min(pwm, max)) * 100 + max / 2) / max;
}

test pwmPercent {
    try std.testing.expectEqual(0, pwmPercent(0));
    try std.testing.expectEqual(50, pwmPercent(128));
    try std.testing.expectEqual(100, pwmPercent(255));
    try std.testing.expectEqual(100, pwmPercent(300));
}
