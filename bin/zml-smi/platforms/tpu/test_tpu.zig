const std = @import("std");
const model = @import("zml-smi/model");
const source = @import("zml-smi/source");
const Tpu = @import("Tpu.zig");

const chipOf = Tpu.testing.chipOf;
const byId = Tpu.testing.byId;
const addHolder = Tpu.testing.addHolder;

test chipOf {
    try std.testing.expectEqualStrings("TPU v3", chipOf(0x0027, 0x004f).?.name);
    try std.testing.expectEqualStrings("TPU v5e", chipOf(0x0063, 0x0001).?.name);
    try std.testing.expect(chipOf(0x0027, null) == null);
    try std.testing.expect(chipOf(0x0001, null) == null);
}

test byId {
    const used = byId(c_longlong, &.{ 1, 0, -1, 64 }, &.{ 2048, -5, 7, 7 });
    try std.testing.expectEqual(2048, used[1]);
    try std.testing.expectEqual(null, used[0]);
    const duty = byId(f64, &.{ 0, 1 }, &.{ 12.5, std.math.nan(f64) });
    try std.testing.expectEqual(13, duty[0]);
    try std.testing.expectEqual(null, duty[1]);
}

test addHolder {
    var samples: [4]model.Sample = @splat(.{ .tpu = .{} });
    samples[2].tpu = .{ .util_percent = 40, .mem_used_bytes = 8192 };
    var reading: source.Reading = .{ .samples = &samples, .allocator = std.testing.allocator };
    defer reading.processes.deinit(std.testing.allocator);

    // Chip 1 of a two-device chip is devices 2 and 3, listed once however many descriptors.
    try addHolder(&reading, 2, 7, "1");
    try addHolder(&reading, 2, 7, "1");
    // The vfio container and a chip beyond the devices make no row.
    try addHolder(&reading, 2, 7, "vfio");
    try addHolder(&reading, 2, 7, "2");

    try std.testing.expectEqual(2, reading.processes.items.len);
    try std.testing.expectEqual(2, reading.processes.items[0].device_idx);
    try std.testing.expectEqual(40, reading.processes.items[0].dev_util_percent);
    try std.testing.expectEqual(8, reading.processes.items[0].dev_mem_kib);
    try std.testing.expectEqual(3, reading.processes.items[1].device_idx);
    try std.testing.expectEqual(null, reading.processes.items[1].dev_mem_kib);
}
