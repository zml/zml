const std = @import("std");
const pci = @import("zml-smi/native").pci;
const sysfs = @import("zml-smi/native").sysfs;

const generation = pci.generation;
const Link = pci.Link;
const upstream = pci.upstream;
const Devices = pci.Devices;

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

test upstream {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "root/bridge/gpu");
    try tmp.dir.writeFile(io, .{ .sub_path = "root/bridge/vendor", .data = "0x8086\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "root/bridge/max_link_speed", .data = "32.0 GT/s PCIe\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "root/bridge/max_link_width", .data = "16\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "root/bridge/gpu/vendor", .data = "0x8086\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "root/bridge/gpu/max_link_speed", .data = "2.5 GT/s PCIe\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "root/bridge/gpu/max_link_width", .data = "1\n" });

    const root: sysfs.Dir = .{ .handle = tmp.dir, .io = io };
    const gpu = try root.openDir("root/bridge/gpu");
    defer gpu.close();
    const dir = try upstream(gpu);
    defer dir.close();
    try std.testing.expectEqual(512_000, pci.link(dir, .max).mbps());
}

test Devices {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "0000:03:00.0");
    try tmp.dir.createDirPath(io, "0000:04:00.0");
    try tmp.dir.writeFile(io, .{ .sub_path = "0000:03:00.0/vendor", .data = "0x8086\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "0000:03:00.0/device", .data = "0xe223\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "0000:04:00.0/vendor", .data = "0x10de\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "0000:04:00.0/device", .data = "0x2b85\n" });

    const root: sysfs.Dir = .{ .handle = tmp.dir, .io = io };
    var devices: Devices = .{ .root = root, .it = root.iterate(), .vendor = 0x8086 };
    const entry = devices.next().?;
    defer entry.dir.close();
    try std.testing.expectEqual(0xe223, entry.id);
    try std.testing.expectEqual(null, devices.next());
}
