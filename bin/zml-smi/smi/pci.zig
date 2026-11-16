const std = @import("std");
const sysfs = @import("sysfs.zig");

/// The PCI devices of one vendor, from /sys/bus/pci/devices.
pub const Devices = struct {
    root: sysfs.Dir,
    it: std.Io.Dir.Iterator,
    vendor: u16,

    pub const Entry = struct {
        /// The device id.
        id: u16,
        /// The caller closes it.
        dir: sysfs.Dir,
    };

    pub fn open(io: std.Io, vendor: u16) !Devices {
        const root = try sysfs.Dir.open(io, "/sys/bus/pci/devices");
        return .{ .root = root, .it = root.iterate(), .vendor = vendor };
    }

    pub fn close(self: *Devices) void {
        self.root.close();
    }

    pub fn next(self: *Devices) ?Entry {
        while (self.it.next(self.root.io) catch null) |entry| {
            const dir = self.root.openDir(entry.name) catch continue;
            if ((dir.int(u16, "vendor") catch 0) == self.vendor) {
                if (dir.int(u16, "device")) |id| return .{ .id = id, .dir = dir } else |_| {}
            }
            dir.close();
        }
        return null;
    }
};

pub const Link = struct {
    gen: ?u64 = null,
    width: ?u64 = null,

    /// The line rate across every lane, in megabits per second.
    pub fn mbps(self: Link) ?u64 {
        return (laneMbps(self.gen orelse return null) orelse return null) * (self.width orelse return null);
    }
};

/// The link a device negotiated, or the widest it supports.
pub fn link(dir: sysfs.Dir, comptime which: enum { current, max }) Link {
    var buffer: [32]u8 = undefined;
    const speed = dir.read(@tagName(which) ++ "_link_speed", &buffer) catch null;
    return .{
        .gen = generation(speed orelse ""),
        .width = dir.int(u64, @tagName(which) ++ "_link_width") catch null,
    };
}

/// The device, or the bridge above it that carries its link: the one with the widest link,
/// climbing only while the vendor is the device's own.
pub fn upstream(device: sysfs.Dir) !sysfs.Dir {
    const max_depth = 8;
    const vendor = try device.int(u16, "vendor");
    const segments: [max_depth][3]u8 = @splat("../".*);
    const parents = std.mem.asBytes(&segments);
    var best_depth: usize = 0;
    var best_mbps = link(device, .max).mbps() orelse 0;

    for (1..max_depth + 1) |depth| {
        const dir = device.openDir(parents[0 .. 3 * depth - 1]) catch break;
        defer dir.close();
        if ((dir.int(u16, "vendor") catch break) != vendor) break;
        const mbps = link(dir, .max).mbps() orelse 0;
        if (mbps > best_mbps) {
            best_depth = depth;
            best_mbps = mbps;
        }
    }
    return device.openDir(if (best_depth == 0) "." else parents[0 .. 3 * best_depth - 1]);
}

const transfers_tenths = [_]u32{ 25, 50, 80, 160, 320, 640 };

/// The generation of a link speed such as "32.0 GT/s PCIe".
pub fn generation(speed: []const u8) ?u64 {
    const transfers = std.fmt.parseFloat(f32, std.mem.sliceTo(speed, ' ')) catch return null;
    if (!(transfers > 0 and transfers < 100)) return null;
    const tenths: u32 = @intFromFloat(transfers * 10);
    const index = std.mem.indexOfScalar(u32, &transfers_tenths, tenths) orelse return null;
    return index + 1;
}

/// The line rate of one lane, in megabits per second.
pub fn laneMbps(gen: u64) ?u64 {
    if (gen == 0 or gen > transfers_tenths.len) return null;
    return @as(u64, transfers_tenths[@intCast(gen - 1)]) * 100;
}
