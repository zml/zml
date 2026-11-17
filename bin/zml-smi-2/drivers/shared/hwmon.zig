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
        var buffer: [64]u8 = undefined;
        if (name == null or std.mem.eql(u8, dir.read("name", &buffer) catch "", name.?)) return dir;
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

/// Names the sensor's path field and unit conversion in a receiver-only probe method.
pub fn Scalar(comptime Probe: type, comptime binding: anytype) fn (*const Probe) ?u64 {
    return struct {
        fn read(probe: *const Probe) ?u64 {
            const sensors = probe.hwmon orelse return null;
            const file = @field(sensors.*, @tagName(binding[0])) orelse return null;
            return sensors.dir.scaled(file, binding[1], binding[2]) catch null;
        }
    }.read;
}

fn fileName(dir: sysfs.Dir, candidate: Candidate, buffer: []u8) ?[]const u8 {
    const label = candidate.label orelse
        return std.fmt.bufPrint(buffer, "{s}_{s}", .{ candidate.prefix, candidate.suffix }) catch null;

    for (1..16) |index| {
        var text: [64]u8 = undefined;
        const path = std.fmt.bufPrint(buffer, "{s}{d}_label", .{ candidate.prefix, index }) catch
            return null;
        if (!std.ascii.eqlIgnoreCase(dir.read(path, &text) catch continue, label)) continue;
        return std.fmt.bufPrint(buffer, "{s}{d}_{s}", .{ candidate.prefix, index, candidate.suffix }) catch null;
    }
    return null;
}

/// A fan PWM duty, 0 to 255, as a percentage.
pub fn pwmPercent(pwm: u64) u64 {
    return (@as(u64, @min(pwm, 255)) * 100 + 127) / 255;
}
