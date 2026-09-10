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

pub const DeviceStats = struct {
    /// Null when the plugin reports no allocator limit (CPU).
    bytes_limit: ?u64,
    bytes_in_use: u64,
};

/// What one `loadExecute` submission needs on each device.
pub const Cost = struct {
    /// Input placements. Once published they are part of the loader's
    /// submitted bytes and no longer counted here.
    inputs: []const u64,
    /// Output placement plus the executable's temporaries: taken when the
    /// submission executes and, for the output, kept.
    execution: []const u64,
};

pub const Decision = enum { admit, retire_oldest };

/// Bytes the loader may still take on one device: the limit minus what is
/// in use, minus its own submitted-but-unallocated bytes (workers allocate
/// tensors lazily, so `allocated` may briefly exceed `submitted`), minus the
/// reserve. Null without a limit.
pub fn room(stats: DeviceStats, submitted: u64, allocated: u64, reserve: u64) ?u64 {
    const limit = stats.bytes_limit orelse return null;
    return limit -| stats.bytes_in_use -| (submitted -| allocated) -| reserve;
}

/// Fills `out` for every device; false when a device has no limit.
pub fn roomPerDevice(out: []u64, stats: []const DeviceStats, submitted: []const u64, allocated: []const u64, reserve: u64) bool {
    for (out, stats, submitted, allocated) |*o, s, sub, alloc| {
        o.* = room(s, sub, alloc, reserve) orelse return false;
    }
    return true;
}

/// Whether `cost` fits beside the pending executions on every device.
pub fn admits(rooms: []const u64, pending: []const u64, cost: Cost) bool {
    for (rooms, pending, cost.inputs, cost.execution) |r, p, i, e| {
        if (p +| i +| e > r) return false;
    }
    return true;
}

/// Admit when the submission fits, or when nothing pending could be retired
/// to make room: the loader cannot do better than try.
pub fn decide(rooms: []const u64, pending: []const u64, cost: Cost, pending_executes: usize) Decision {
    if (pending_executes == 0) return .admit;
    return if (admits(rooms, pending, cost)) .admit else .retire_oldest;
}

pub fn addPending(pending: []u64, execution: []const u64) void {
    for (pending, execution) |*p, e| p.* +|= e;
}

pub fn subPending(pending: []u64, execution: []const u64) void {
    for (pending, execution) |*p, e| {
        std.debug.assert(p.* >= e);
        p.* -= e;
    }
}

test "room subtracts use, unlanded bytes and the reserve, saturating" {
    try std.testing.expectEqual(null, room(.{ .bytes_limit = null, .bytes_in_use = 0 }, 0, 0, 0));
    try std.testing.expectEqual(55, room(.{ .bytes_limit = 100, .bytes_in_use = 30 }, 50, 40, 5).?);
    // More in use than the limit admits: no room, no underflow.
    try std.testing.expectEqual(0, room(.{ .bytes_limit = 100, .bytes_in_use = 130 }, 0, 0, 0).?);
    // A worker allocated ahead of the front end's accounting: nothing unlanded.
    try std.testing.expectEqual(70, room(.{ .bytes_limit = 100, .bytes_in_use = 30 }, 10, 40, 0).?);
    try std.testing.expectEqual(0, room(.{ .bytes_limit = 100, .bytes_in_use = 90 }, 0, 0, 20).?);
}

test "roomPerDevice fails as soon as one device has no limit" {
    var out: [2]u64 = undefined;
    const known = [_]DeviceStats{ .{ .bytes_limit = 100, .bytes_in_use = 10 }, .{ .bytes_limit = 50, .bytes_in_use = 20 } };
    try std.testing.expect(roomPerDevice(&out, &known, &.{ 0, 0 }, &.{ 0, 0 }, 5));
    try std.testing.expectEqualSlices(u64, &.{ 85, 25 }, &out);
    const unknown = [_]DeviceStats{ .{ .bytes_limit = 100, .bytes_in_use = 10 }, .{ .bytes_limit = null, .bytes_in_use = 0 } };
    try std.testing.expect(!roomPerDevice(&out, &unknown, &.{ 0, 0 }, &.{ 0, 0 }, 5));
}

test "admits needs room on every device" {
    const cost: Cost = .{ .inputs = &.{ 10, 10 }, .execution = &.{ 5, 5 } };
    try std.testing.expect(admits(&.{ 40, 40 }, &.{ 20, 20 }, cost));
    // The second device is one byte short.
    try std.testing.expect(!admits(&.{ 40, 34 }, &.{ 20, 20 }, cost));
    try std.testing.expect(admits(&.{}, &.{}, .{ .inputs = &.{}, .execution = &.{} }));
}

test "pending accounting round-trips" {
    var pending = [_]u64{ 1, 2 };
    addPending(&pending, &.{ 10, 20 });
    try std.testing.expectEqualSlices(u64, &.{ 11, 22 }, &pending);
    subPending(&pending, &.{ 10, 20 });
    try std.testing.expectEqualSlices(u64, &.{ 1, 2 }, &pending);
}

test "decide retires until the submission fits and admits an oversized one alone" {
    const cost: Cost = .{ .inputs = &.{5}, .execution = &.{5} };
    var pending = [_]u64{30};
    var pending_executes: usize = 3;
    var decisions: [4]Decision = undefined;
    for (&decisions) |*decision| {
        decision.* = decide(&.{25}, &pending, cost, pending_executes);
        if (decision.* == .retire_oldest) {
            subPending(&pending, &.{10});
            pending_executes -= 1;
        }
    }
    try std.testing.expectEqualSlices(Decision, &.{ .retire_oldest, .retire_oldest, .admit, .admit }, &decisions);
    try std.testing.expectEqual(.admit, decide(&.{5}, &.{0}, cost, 0));
}
