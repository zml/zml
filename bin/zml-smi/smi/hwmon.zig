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
    for (&paths, 1..) |*path, i| path.* = std.fmt.comptimePrint(pattern, .{i});
    return paths;
}

pub fn firstReadable(dir: sysfs.Dir, paths: []const []const u8) ?[]const u8 {
    for (paths) |path| {
        _ = dir.int(u64, path) catch continue;
        return path;
    }
    return null;
}

pub fn labeled(dir: sysfs.Dir, comptime sensor: []const u8, comptime suffix: []const u8, label: []const u8) ?[]const u8 {
    const labels = comptime numbered(sensor ++ "{d}_label");
    const files = comptime numbered(sensor ++ "{d}_" ++ suffix);
    var buffer: [64]u8 = undefined;
    for (labels, files) |label_path, file| {
        if (!std.ascii.eqlIgnoreCase(dir.read(label_path, &buffer) catch continue, label)) continue;
        _ = dir.int(u64, file) catch continue;
        return file;
    }
    return null;
}

pub fn pwmPercent(pwm: u64) u64 {
    return (@as(u64, @min(pwm, 255)) * 100 + 127) / 255;
}

test pwmPercent {
    try std.testing.expectEqual(0, pwmPercent(0));
    try std.testing.expectEqual(50, pwmPercent(128));
    try std.testing.expectEqual(100, pwmPercent(255));
    try std.testing.expectEqual(100, pwmPercent(300));
}
