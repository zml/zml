const std = @import("std");
const sysfs = @import("sysfs.zig");

/// The hwmon directory of a device: the one called `name`, or the first when `name` is null.
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

/// A file that may hold a value: `temp2_input`, or with a label, the `temp<N>_input`
/// whose `temp<N>_label` is the label.
pub const Candidate = struct {
    prefix: []const u8,
    label: ?[]const u8 = null,
    suffix: []const u8 = "input",
};

/// The name of the first candidate that reads as a number, copied into `arena`.
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

/// A fan PWM duty, 0 to 255, as a percentage.
pub fn pwmPercent(pwm: u64) u64 {
    return (@as(u64, @min(pwm, 255)) * 100 + 127) / 255;
}

test resolve {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "hwmon/hwmon2");
    try tmp.dir.createDirPath(io, "hwmon/hwmon3");
    try tmp.dir.writeFile(io, .{ .sub_path = "hwmon/hwmon2/name", .data = "nvme\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "hwmon/hwmon3/name", .data = "xe\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "hwmon/hwmon3/temp2_label", .data = "PKG\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "hwmon/hwmon3/temp2_input", .data = "58000\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "hwmon/hwmon3/power1_max", .data = "230000000\n" });

    const device: sysfs.Dir = .{ .handle = tmp.dir, .io = io };
    try std.testing.expectError(error.NotFound, find(device, "i915"));
    const dir = try find(device, "xe");
    defer dir.close();

    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const temperature = resolve(arena.allocator(), dir, &.{ .{ .prefix = "temp", .label = "pkg" }, .{ .prefix = "temp1" } });
    try std.testing.expectEqualStrings("temp2_input", temperature.?);
    const power_limit = resolve(arena.allocator(), dir, &.{ .{ .prefix = "power1", .suffix = "cap" }, .{ .prefix = "power1", .suffix = "max" } });
    try std.testing.expectEqualStrings("power1_max", power_limit.?);
    try std.testing.expectEqual(null, resolve(arena.allocator(), dir, &.{.{ .prefix = "fan1" }}));
}

test pwmPercent {
    try std.testing.expectEqual(0, pwmPercent(0));
    try std.testing.expectEqual(50, pwmPercent(128));
    try std.testing.expectEqual(100, pwmPercent(255));
    try std.testing.expectEqual(100, pwmPercent(300));
}
