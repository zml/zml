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

pub const Candidate = struct {
    prefix: []const u8,
    label: ?[]const u8 = null,
    suffix: []const u8 = "input",
};

pub fn resolve(arena: std.mem.Allocator, dir: sysfs.Dir, candidates: []const Candidate) ?[]const u8 {
    var buffer: [64]u8 = undefined;
    for (candidates) |candidate| {
        const name = fileName(dir, candidate, &buffer) orelse continue;
        _ = dir.int(u64, name) catch continue;
        return arena.dupe(u8, name) catch null;
    }
    return null;
}

fn fileName(dir: sysfs.Dir, candidate: Candidate, buffer: []u8) ?[]const u8 {
    const label = candidate.label orelse
        return std.fmt.bufPrint(buffer, "{s}_{s}", .{ candidate.prefix, candidate.suffix }) catch null;

    for (1..16) |index| {
        var label_name: [64]u8 = undefined;
        var text: [64]u8 = undefined;
        const path = std.fmt.bufPrint(&label_name, "{s}{d}_label", .{ candidate.prefix, index }) catch return null;
        if (!std.ascii.eqlIgnoreCase(dir.read(path, &text) catch continue, label)) continue;
        return std.fmt.bufPrint(buffer, "{s}{d}_{s}", .{ candidate.prefix, index, candidate.suffix }) catch null;
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
