const std = @import("std");
const smi = @import("zml-smi");
const OneApi = @import("zml-smi/platforms/oneapi");

const parseLspciName = OneApi.testing.parseLspciName;
const largestBar = OneApi.testing.largestBar;
const memoryBytes = OneApi.testing.memoryBytes;
const Fan = OneApi.testing.Fan;

test parseLspciName {
    try std.testing.expectEqualStrings(
        "Intel Corporation Battlemage G31 [Arc Pro B70]",
        parseLspciName("0000:84:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Arc Pro B70] [8086:e223] (rev 04)\n").?,
    );
    // A generic name is no better than the device id.
    try std.testing.expectEqual(null, parseLspciName("0000:84:00.0 Display controller [0380]: Intel Corporation Device e223 [8086:e223] (rev 04)\n"));
}

test largestBar {
    try std.testing.expectEqual(34_359_738_368, largestBar(
        \\0x0000048800000000 0x0000048fffffffff 0x000000000014220c
        \\0x00000000f6c00000 0x00000000f6dfffff 0x0000000000046200
        \\0x0000000000000000 0x0000000000000000 0x0000000000000000
    ));
    try std.testing.expectEqual(null, largestBar("0x00000000f6c00000 0x00000000f6dfffff 0x0000000000046200\n"));
}

test memoryBytes {
    const xe = smi.fdinfo.parse("drm-total-system:\t184 MiB\ndrm-total-vram0:\t4096 KiB\ndrm-resident-vram0:\t3072 KiB\n");
    try std.testing.expectEqual(3072 * 1024, memoryBytes(&xe));
    const system = smi.fdinfo.parse("drm-total-system0:\t2 MiB\n");
    try std.testing.expectEqual(2 * 1024 * 1024, memoryBytes(&system));
}

test Fan {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const dir: smi.sysfs.Dir = .{ .handle = tmp.dir, .io = io };

    // fan1 has no maximum, so the speed is fan2's.
    try tmp.dir.writeFile(io, .{ .sub_path = "fan1_input", .data = "900\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "fan2_input", .data = "1500\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "fan2_max", .data = "3000\n" });
    try std.testing.expectEqual(50, Fan.percent(Fan.find(arena.allocator(), dir).?, dir));

    // A PWM comes first, whatever its index.
    try tmp.dir.writeFile(io, .{ .sub_path = "pwm3", .data = "255\n" });
    try std.testing.expectEqual(100, Fan.percent(Fan.find(arena.allocator(), dir).?, dir));
}
