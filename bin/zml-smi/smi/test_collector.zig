const std = @import("std");
const smi = @import("smi.zig");

const Collector = smi.Collector;
const Reading = smi.Reading;

const Fake = struct {
    devices: []const smi.Device,
    polls: u64 = 0,

    pub const two_readings = true;

    pub fn open(
        _: std.mem.Allocator,
        _: std.mem.Allocator,
        _: std.Io,
        devices: []const smi.Device,
    ) !Fake {
        return .{ .devices = devices };
    }

    pub fn poll(self: *Fake, _: std.mem.Allocator, reading: *Reading) !void {
        const samples = reading.samples;
        self.polls += 1;
        samples[0].cuda.util_percent = self.polls;
        if (samples.len > 1) samples[1] = .{ .tpu = .{ .util_percent = 100 } };
        try reading.addProcess(.{ .device_idx = @intCast(samples.len) });
        try reading.addProcess(.{
            .pid = 42,
            .device_idx = @intCast(samples.len - 1),
            .dev_util_percent = @intCast(self.polls),
            .remote = true,
            .username = try reading.allocator.dupe(u8, if (self.polls == 1) "alice" else "carol"),
        });
        if (self.polls % 2 == 0) return error.Unavailable;
        reading.host.cpu_power_mw = self.polls;
    }
};

test "source publication keeps readings, history, and owned process rows together" {
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var collector: Collector = .init(arena.allocator(), std.testing.allocator, io, .{ .history = true });
    defer collector.deinit();
    const devices: []const smi.Device = &.{ .{ .target = .cuda }, .{ .target = .cuda } };
    try collector.open(Fake, .{devices});
    try collector.open(Fake, .{devices[0..1]});
    try std.testing.expect(collector.two_readings);

    collector.round();
    const frame = try collector.capture(arena.allocator(), true);
    try std.testing.expectEqual(1, frame.samples[0].cuda.util_percent);
    try std.testing.expectEqual(null, frame.samples[1].cuda.util_percent);
    try std.testing.expectEqual(1, frame.samples[2].cuda.util_percent);
    try std.testing.expectEqual(1, frame.host.cpu_power_mw);
    try std.testing.expectEqual(2, frame.processes.len);
    try std.testing.expectEqual(2, frame.processes[1].device_idx);
    try std.testing.expectEqual(1, frame.histories[0].len);

    collector.round();
    const next = try collector.capture(arena.allocator(), true);
    try std.testing.expectEqual(2, next.samples[0].cuda.util_percent);
    try std.testing.expectEqual(null, next.host.cpu_power_mw);
    try std.testing.expectEqual(0, next.processes.len);
    try std.testing.expectEqual(2, next.histories[0].len);
    // Reuse the arena from the first publication. Its captured strings must survive.
    collector.round();
    try std.testing.expectEqualStrings("alice", frame.processes[1].username);
}

test "collector joins workers before deinit" {
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var collector: Collector = .init(arena.allocator(), std.testing.allocator, io, .{ .poll_interval_ms = 1 });
    defer collector.deinit();
    const devices: []const smi.Device = &.{.{ .target = .cuda }};
    try collector.open(Fake, .{devices});
    try collector.start();
    while (true) {
        const source = collector.sources.items[0];
        const util = source.readings.lock(io).value.samples[0].cuda.util_percent orelse 0;
        source.readings.unlock(io);
        if (util >= 3) break;
        try io.sleep(.fromMilliseconds(1), .awake);
    }
}
