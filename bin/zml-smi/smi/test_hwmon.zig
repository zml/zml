const std = @import("std");
const hwmon = @import("hwmon.zig");
const sysfs = @import("sysfs.zig");

const resolve = hwmon.resolve;
const pwmPercent = hwmon.pwmPercent;

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
    try std.testing.expectError(error.NotFound, hwmon.find(device, "i915"));
    const dir = try hwmon.find(device, "xe");
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
