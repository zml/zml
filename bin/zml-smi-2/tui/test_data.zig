const std = @import("std");
const contract = @import("zml-smi-2/contract");
const data = @import("data.zig");
const spec = @import("spec.zig");

const Metric = data.Metric;
const historyIndex = data.testing.historyIndex;

const fixture_devices: []const spec.DeviceView = &.{.{
    .title = "Fixture device",
    .heading = "Device 0: Fixture device",
    .remote = false,
    .profile = &.{
        .util_label = "UTIL",
        .util_title = "Utilization",
        .memory_prefix = "Memory",
    },
}};

test "TUI captures a frame and uses its device history metadata" {
    const Source = struct {
        captures: u64 = 0,

        const points: []const contract.HistoryPoint = &.{.{
            .at = .fromNanoseconds(0),
            .values = .initDefault(@as(?u64, null), .{ .util_percent = 21 }),
        }};
        const histories: []const []const contract.HistoryPoint = &.{points};

        fn capture(context: *anyopaque, allocator: std.mem.Allocator) std.mem.Allocator.Error!contract.Frame {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.captures += 1;
            var sample: contract.Metrics = .{};
            sample.put(.util_percent, self.captures);
            return .{
                .samples = try allocator.dupe(contract.Metrics, &.{sample}),
                .processes = &.{},
                .host = .{},
                .history = .{
                    .points = histories,
                    .captured_at = .fromNanoseconds(125 * std.time.ns_per_ms),
                    .interval_ms = 125,
                    .capacity = 3,
                },
            };
        }
    };
    var source: Source = .{};
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var state: data.SystemState = .{
        .provider = .{ .context = &source, .capture = Source.capture },
        .devices = fixture_devices,
        .tui_refresh_rate = 100,
    };
    try state.capture(arena.allocator());
    try std.testing.expectEqual(1, state.devices.len);
    try std.testing.expectEqualStrings("Fixture device", state.devices[0].title);
    try std.testing.expectEqual(@as(?u64, 1), state.frame.samples[0].get(.util_percent));
    try std.testing.expectEqual(125, state.frame.history.?.interval_ms);
    try std.testing.expectEqualSlices(?u64, &.{ null, 21, null }, try state.history(arena.allocator(), 0, .util));
}

test "history buckets follow collection timestamps" {
    const now: std.Io.Timestamp = .fromNanoseconds(2_000 * std.time.ns_per_ms);
    try std.testing.expectEqual(@as(?usize, 3), historyIndex(now, now, 500, 4));
    try std.testing.expectEqual(@as(?usize, 1), historyIndex(
        .fromNanoseconds(1_000 * std.time.ns_per_ms),
        now,
        500,
        4,
    ));
    try std.testing.expectEqual(null, historyIndex(.fromNanoseconds(0), now, 500, 4));
    try std.testing.expectEqual(null, historyIndex(now, now, 500, 0));
}

test "live and captured metrics use the same raw readings" {
    var sample: contract.Metrics = .{};
    sample.put(.util_percent, 0);
    sample.put(.mem_used_bytes, 3);
    sample.put(.mem_total_bytes, 2);
    sample.put(.power_mw, std.math.maxInt(u64));
    const history = contract.historyMetrics(&sample);
    const expected: std.EnumArray(Metric, ?u128) = .init(.{
        .util = 0,
        .memory = 150,
        .power = std.math.maxInt(u64),
        .temperature = null,
    });
    for (std.enums.values(Metric)) |kind| {
        try std.testing.expectEqual(expected.get(kind), kind.read(&sample));
        try std.testing.expectEqual(expected.get(kind), kind.read(&history));
    }
}

test "captured history keeps failed and absent observations as gaps" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const points = [_]contract.HistoryPoint{
        .{
            .values = .initDefault(@as(?u64, null), .{
                .util_percent = 0,
                .mem_used_bytes = 0,
                .mem_total_bytes = 100,
                .power_mw = 100_000,
            }),
            .at = .fromNanoseconds(500 * std.time.ns_per_ms),
        },
        .{
            .values = .initFill(null),
            .at = .fromNanoseconds(1_500 * std.time.ns_per_ms),
        },
    };
    const state: data.SystemState = .{
        .provider = undefined,
        .devices = fixture_devices,
        .frame = .{ .history = .{
            .points = &.{&points},
            .captured_at = .fromNanoseconds(1_500 * std.time.ns_per_ms),
            .interval_ms = 500,
            .capacity = 4,
        } },
        .tui_refresh_rate = 100,
    };
    for (std.enums.values(Metric)) |metric| {
        const values = try state.history(arena.allocator(), 0, metric);
        const value: ?u64 = switch (metric) {
            .util, .memory => 0,
            .temperature => null,
            .power => 100_000,
        };
        const expected: []const ?u64 = &.{ null, value, null, null };
        try std.testing.expectEqualSlices(?u64, expected, values);
    }
}
