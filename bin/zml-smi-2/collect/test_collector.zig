const std = @import("std");
const contract = @import("zml-smi-2/contract");
const source = @import("zml-smi-2/source");
const Collector = @import("collector.zig").Collector;

test {
    _ = @import("test_double_buffer.zig");
}

const Fake = struct {
    devices: []const contract.Device,
    polls: u64 = 0,
    closes: usize = 0,

    pub const two_readings = true;

    pub fn close(self: *Fake) void {
        self.closes += 1;
    }

    pub fn poll(self: *Fake, _: std.mem.Allocator, reading: *source.Reading) !void {
        const samples = reading.samples;
        self.polls += 1;
        samples[0].put(.util_percent, self.polls);
        if (self.polls == 1 and samples.len > 1) {
            samples[1].put(.util_percent, 100);
            reading.host.hostname = try reading.allocator.dupe(u8, "device host");
        }
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

const FakeHost = struct {
    pub fn read(
        _: *FakeHost,
        _: std.mem.Allocator,
        _: []contract.ProcessInfo,
        _: *contract.HostData,
    ) void {}
};

test "source publication keeps readings, history, and owned process rows together" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var host: FakeHost = .{};
    const host_reader: source.HostReader = .from(&host);
    var collector: Collector = .init(arena.allocator(), gpa, io, host_reader, .{ .hostname = "local" }, .{
        .history = true,
    });
    defer collector.deinit();
    const devices: []const contract.Device = &.{ .{ .target = .cuda }, .{ .target = .cuda } };
    var first: Fake = .{ .devices = devices };
    var second: Fake = .{ .devices = devices[0..1] };
    try collector.add(.from(&first));
    try collector.add(.from(&second));
    try std.testing.expect(collector.two_readings);

    collector.round();
    const frame = try collector.capture(arena.allocator(), true);
    try std.testing.expectEqual(3, collector.devices.items.len);
    try std.testing.expectEqual(3, frame.samples.len);
    try std.testing.expectEqual(1, frame.samples[0].get(.util_percent));
    try std.testing.expectEqual(100, frame.samples[1].get(.util_percent));
    try std.testing.expectEqual(1, frame.samples[2].get(.util_percent));
    try std.testing.expectEqual(1, frame.host.cpu_power_mw);
    try std.testing.expectEqualStrings("device host", frame.host.hostname.?);
    try std.testing.expectEqual(2, frame.processes.len);
    try std.testing.expectEqual(2, frame.processes[1].device_idx);
    try std.testing.expectEqual(1, frame.history.?.points[0].len);
    try std.testing.expectEqual(100, frame.history.?.points[1][0].values.get(.util_percent));

    collector.round();
    const next = try collector.capture(arena.allocator(), true);
    try std.testing.expectEqual(2, next.samples[0].get(.util_percent));
    try std.testing.expectEqual(null, next.samples[1].get(.util_percent));
    try std.testing.expectEqual(null, next.host.cpu_power_mw);
    try std.testing.expectEqualStrings("local", next.host.hostname.?);
    try std.testing.expectEqual(0, next.processes.len);
    try std.testing.expectEqual(2, next.history.?.points[0].len);
    // Reuse the backing slot. Missing metrics clear and captured strings survive.
    collector.round();
    const latest = try collector.capture(arena.allocator(), false);
    try std.testing.expectEqual(null, latest.samples[1].get(.util_percent));
    try std.testing.expectEqualStrings("device host", frame.host.hostname.?);
    try std.testing.expectEqualStrings("alice", frame.processes[1].username);
}

test "collector joins workers before deinit" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var host: FakeHost = .{};
    const host_reader: source.HostReader = .from(&host);
    var collector: Collector = .init(arena.allocator(), gpa, io, host_reader, .{}, .{
        .poll_interval_ms = 1,
    });
    defer collector.deinit();
    var backend: Fake = .{ .devices = &.{.{ .target = .cuda }} };
    try collector.add(.from(&backend));
    try collector.start();
    while (true) {
        const publication = collector.sources.items[0];
        const util = publication.readings.lock(io).value.samples[0].get(.util_percent) orelse 0;
        publication.readings.unlock(io);
        if (util >= 3) break;
        try io.sleep(.fromMilliseconds(1), .awake);
    }
}

test "source registration transfers ownership only on success" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var host: FakeHost = .{};
    var collector: Collector = .init(std.testing.failing_allocator, gpa, io, .from(&host), .{}, .{});
    defer collector.deinit();

    var empty: Fake = .{ .devices = &.{} };
    try collector.add(.from(&empty));
    try std.testing.expectEqual(1, empty.closes);

    var backend: Fake = .{ .devices = &.{.{ .target = .cuda }} };
    try std.testing.expectError(error.OutOfMemory, collector.add(.from(&backend)));
    try std.testing.expectEqual(0, backend.closes);
    try std.testing.expectEqual(0, collector.devices.items.len);
}
