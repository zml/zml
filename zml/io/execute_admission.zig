//! The arithmetic behind `Loader.loadExecute` admission: what a submission
//! needs on each device, what the devices have left, and whether the next
//! submission fits beside the ones still pending. Everything is per device
//! (index = `platform.devices` index) in bytes; nothing here touches a
//! device, so the rules are tested without one.
const std = @import("std");

/// Allocator slack per device. BFC rounds every allocation up to 256 bytes;
/// a 64-pack, 16-source, 8-device load is about 1e5 shards, under 32 MiB of
/// rounding, and nothing else runs on the devices during a load.
pub const reserve_bytes: u64 = 64 << 20;

/// Bytes the loader may still take on one device: the limit (null when the
/// plugin reports none, as CPU does) minus what is in use, minus its own
/// submitted-but-unallocated bytes (workers allocate tensors lazily, so
/// `allocated` may briefly exceed `submitted`), minus the reserve.
pub fn room(limit: ?u64, in_use: u64, submitted: u64, allocated: u64, reserve: u64) ?u64 {
    return (limit orelse return null) -| in_use -| (submitted -| allocated) -| reserve;
}

/// Whether one submission's `inputs` (placements, counted here until they
/// are published) and `execution` (output placement plus the executable's
/// temporaries) fit beside the pending executions on every device.
pub fn admits(rooms: []const u64, pending: []const u64, inputs: []const u64, execution: []const u64) bool {
    for (rooms, pending, inputs, execution) |r, p, i, e| {
        if (p +| i +| e > r) return false;
    }
    return true;
}

test "room subtracts use, unlanded bytes and the reserve, saturating" {
    try std.testing.expectEqual(null, room(null, 0, 0, 0, 0));
    try std.testing.expectEqual(55, room(100, 30, 50, 40, 5).?);
    // More in use than the limit admits: no room, no underflow.
    try std.testing.expectEqual(0, room(100, 130, 0, 0, 0).?);
    // A worker allocated ahead of the front end's accounting: nothing unlanded.
    try std.testing.expectEqual(70, room(100, 30, 10, 40, 0).?);
    try std.testing.expectEqual(0, room(100, 90, 0, 0, 20).?);
}

test "admits needs room on every device" {
    try std.testing.expect(admits(&.{ 40, 40 }, &.{ 20, 20 }, &.{ 10, 10 }, &.{ 5, 5 }));
    // The second device is one byte short.
    try std.testing.expect(!admits(&.{ 40, 34 }, &.{ 20, 20 }, &.{ 10, 10 }, &.{ 5, 5 }));
    try std.testing.expect(admits(&.{}, &.{}, &.{}, &.{}));
}
