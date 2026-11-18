const std = @import("std");
const contract = @import("zml-smi-2/contract");
const source = @import("zml-smi-2/source");
const Collector = @import("collector.zig").Collector(FakeHost);

test {
    _ = @import("test_double_buffer.zig");
}

const Fake = struct {
    devices: []const contract.Device,
    power: source.HostPower = .{},
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
        if (self.polls == 1 and samples.len > 1) samples[1].put(.util_percent, 100);
        try reading.addProcess(.{ .device_idx = @intCast(samples.len) });
        try reading.addProcess(.{
            .pid = 42,
            .device_idx = @intCast(samples.len - 1),
            .dev_util_percent = @intCast(self.polls),
            .remote = true,
            .username = try reading.allocator.dupe(u8, if (self.polls == 1) "alice" else "carol"),
        });
        if (self.polls % 2 == 0) return error.Unavailable;
        reading.host_power = self.power;
    }
};

const FakeHost = struct {
    power: source.HostPower = .{},

    pub fn read(
        self: *FakeHost,
        _: std.mem.Allocator,
        _: []contract.ProcessInfo,
        data: *contract.HostData,
    ) void {
        data.cpu_power_mw = self.power.cpu_power_mw;
        data.system_power_mw = self.power.system_power_mw;
    }
};

test "source publication keeps readings, history, and owned process rows together" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var host: FakeHost = .{ .power = .{ .cpu_power_mw = 90, .system_power_mw = 120 } };
    var collector: Collector = .init(arena.allocator(), gpa, io, &host, .{ .hostname = "local" }, .{
        .history = true,
    });
    defer collector.deinit();
    const devices: []const contract.Device = &.{ .{ .target = .cuda }, .{ .target = .cuda } };
    var first: Fake = .{
        .devices = devices,
        .power = .{ .cpu_power_mw = 10, .system_power_mw = 20 },
    };
    var second: Fake = .{ .devices = devices[0..1], .power = .{ .cpu_power_mw = 0 } };
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
    try std.testing.expectEqual(0, frame.host.cpu_power_mw);
    try std.testing.expectEqual(20, frame.host.system_power_mw);
    try std.testing.expectEqualStrings("local", frame.host.hostname.?);
    try std.testing.expectEqual(2, frame.processes.len);
    try std.testing.expectEqual(2, frame.processes[1].device_idx);
    try std.testing.expectEqual(1, frame.history.?.points[0].len);
    try std.testing.expectEqual(100, frame.history.?.points[1][0].values.get(.util_percent));

    collector.round();
    const next = try collector.capture(arena.allocator(), true);
    try std.testing.expectEqual(2, next.samples[0].get(.util_percent));
    try std.testing.expectEqual(null, next.samples[1].get(.util_percent));
    try std.testing.expectEqual(90, next.host.cpu_power_mw);
    try std.testing.expectEqual(120, next.host.system_power_mw);
    try std.testing.expectEqualStrings("local", next.host.hostname.?);
    try std.testing.expectEqual(0, next.processes.len);
    try std.testing.expectEqual(2, next.history.?.points[0].len);
    // Reuse the backing slot. Missing metrics clear and captured strings survive.
    collector.round();
    const latest = try collector.capture(arena.allocator(), false);
    try std.testing.expectEqual(null, latest.samples[1].get(.util_percent));
    try std.testing.expectEqual(0, frame.host.cpu_power_mw);
    try std.testing.expectEqual(20, frame.host.system_power_mw);
    try std.testing.expectEqualStrings("alice", frame.processes[1].username);
}

test "collector joins workers before deinit" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var host: FakeHost = .{};
    var collector: Collector = .init(arena.allocator(), gpa, io, &host, .{}, .{
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
    var collector: Collector = .init(std.testing.failing_allocator, gpa, io, &host, .{}, .{});
    defer collector.deinit();

    var empty: Fake = .{ .devices = &.{} };
    try collector.add(.from(&empty));
    try std.testing.expectEqual(1, empty.closes);

    var backend: Fake = .{ .devices = &.{.{ .target = .cuda }} };
    try std.testing.expectError(error.OutOfMemory, collector.add(.from(&backend)));
    try std.testing.expectEqual(0, backend.closes);
    try std.testing.expectEqual(0, collector.devices.items.len);
}
