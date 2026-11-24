const std = @import("std");
const sysfs = @import("sysfs.zig");

pub const Link = struct {
    gen: ?u64 = null,
    width: ?u64 = null,

    pub fn mbps(self: Link) ?u64 {
        return (laneMbps(self.gen orelse return null) orelse return null) * (self.width orelse return null);
    }
};

pub fn link(dir: sysfs.Dir, comptime which: enum { current, max }) Link {
    var buffer: [32]u8 = undefined;
    const speed = dir.read(@tagName(which) ++ "_link_speed", &buffer) catch null;
    return .{
        .gen = generation(speed orelse ""),
        .width = dir.int(u64, @tagName(which) ++ "_link_width") catch null,
    };
}

pub fn generation(speed: []const u8) ?u64 {
    const transfers = std.fmt.parseFloat(f32, std.mem.sliceTo(speed, ' ')) catch return null;
    if (!(transfers > 0 and transfers < 100)) return null;
    return switch (@as(u32, @intFromFloat(transfers * 10))) {
        25 => 1,
        50 => 2,
        80 => 3,
        160 => 4,
        320 => 5,
        640 => 6,
        else => null,
    };
}

pub fn laneMbps(gen: u64) ?u64 {
    return switch (gen) {
        1 => 2_500,
        2 => 5_000,
        3 => 8_000,
        4 => 16_000,
        5 => 32_000,
        6 => 64_000,
        else => null,
    };
}

test generation {
    try std.testing.expectEqual(1, generation("2.5 GT/s PCIe"));
    try std.testing.expectEqual(3, generation("8 GT/s"));
    try std.testing.expectEqual(4, generation("16.0 GT/s PCIe"));
    try std.testing.expectEqual(null, generation("Unknown"));
}

test Link {
    try std.testing.expectEqual(10_000, (Link{ .gen = 1, .width = 4 }).mbps());
    try std.testing.expectEqual(512_000, (Link{ .gen = 5, .width = 16 }).mbps());
    try std.testing.expectEqual(null, (Link{ .gen = 5 }).mbps());
}
