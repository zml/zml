const std = @import("std");
const sysfs = @import("zml-smi-2/native").sysfs;

const Dir = sysfs.Dir;
const fieldValue = sysfs.fieldValue;

test Dir {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "vendor", .data = "0x8086\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "temp1_input", .data = "58500\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "meminfo", .data = "MemTotal:       65137624 kB\nMemTotalX: 1 kB\nFlags: fpu vme de pse tsc msr pae mce cx8\nMemAvailable:   33697096 kB\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "cmdline", .data = "python\x00-m\x00llmd\x00" });

    const dir: Dir = .{ .handle = tmp.dir, .io = io };
    var buffer: [16]u8 = undefined;

    try std.testing.expectEqualStrings("0x8086", try dir.read("vendor", &buffer));
    try std.testing.expectEqual(0x8086, try dir.int(u16, "vendor"));
    try std.testing.expectEqual(59, try dir.scaled("temp1_input", .milli, .one));
    try std.testing.expectError(error.StreamTooLong, dir.read("vendor", buffer[0..4]));
    try std.testing.expectEqualStrings("python\x00-m", try dir.head("cmdline", buffer[0..9]));

    var line: [64]u8 = undefined;
    try std.testing.expectEqualStrings("33697096 kB", try dir.field("meminfo", "MemAvailable", &line));
    try std.testing.expectEqualStrings("65137624 kB", try dir.field("meminfo", "MemTotal", &line));
    try std.testing.expectError(error.NotFound, dir.field("meminfo", "Mem", &line));
    // A line longer than the buffer is skipped, not cut.
    try std.testing.expectEqualStrings("33697096 kB", try dir.field("meminfo", "MemAvailable", line[0..30]));
}

test fieldValue {
    try std.testing.expectEqualStrings("0000:03:00.0", fieldValue("PCI_SLOT_NAME=0000:03:00.0", "PCI_SLOT_NAME").?);
    try std.testing.expectEqualStrings("AMD Ryzen", fieldValue("model name\t: AMD Ryzen", "model name").?);
    try std.testing.expectEqual(null, fieldValue("model names: x", "model name"));
}
