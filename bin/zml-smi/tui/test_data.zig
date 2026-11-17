const std = @import("std");
const model = @import("zml-smi/model");
const data = @import("data.zig");

const Metric = data.Metric;
const historyIndex = data.testing.historyIndex;
const historySeries = data.testing.historySeries;
const memoryPercent = data.memoryPercent;

test "TUI captures a frame and uses its device history metadata" {
    const Source = struct {
        captures: u64 = 0,

        const devices: []const model.Device = &.{.{ .target = .tpu }};
        const points: []const model.HistoryPoint = &.{.{ .at = .fromNanoseconds(0), .common = .{ .util_percent = 21 } }};
        const histories: []const []const model.HistoryPoint = &.{points};

        fn capture(context: *anyopaque, allocator: std.mem.Allocator) std.mem.Allocator.Error!model.Frame {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.captures += 1;
            return .{
                .devices = devices,
                .samples = try allocator.dupe(model.Sample, &.{.{ .tpu = .{ .util_percent = self.captures } }}),
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
        .tui_refresh_rate = 100,
    };
    try state.capture(arena.allocator());
    try std.testing.expectEqual(1, state.deviceCount());
    try std.testing.expectEqual(.tpu, state.frame.?.devices[0].target);
    try std.testing.expectEqual(@as(?u64, 1), state.frame.?.samples[0].tpu.util_percent);
    try std.testing.expectEqual(125, state.historyInterval());
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

test "missing memory is distinct from zero utilization" {
    try std.testing.expectEqual(null, memoryPercent(.{ .mem_total_bytes = 100 }));
    try std.testing.expectEqual(@as(?u64, 0), memoryPercent(.{
        .mem_used_bytes = 0,
        .mem_total_bytes = 100,
    }));
    try std.testing.expectEqual(@as(?u64, 100), memoryPercent(.{
        .mem_used_bytes = std.math.maxInt(u64),
        .mem_total_bytes = 1,
    }));
}

test "captured history keeps failed and absent observations as gaps" {
    var frame: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer frame.deinit();
    const points = [_]model.HistoryPoint{
        .{
            .common = .{ .util_percent = 0, .mem_used_bytes = 0, .mem_total_bytes = 100 },
            .at = .fromNanoseconds(500 * std.time.ns_per_ms),
        },
        .{
            .common = .{},
            .at = .fromNanoseconds(1_500 * std.time.ns_per_ms),
        },
    };
    for ([_]Metric{ .util, .memory, .temperature }) |metric| {
        const values = try historySeries(
            frame.allocator(),
            &points,
            metric,
            .fromNanoseconds(1_500 * std.time.ns_per_ms),
            500,
            4,
        );
        const expected: []const ?u64 = if (metric == .temperature)
            &.{ null, null, null, null }
        else
            &.{ null, 0, null, null };
        try std.testing.expectEqualSlices(?u64, expected, values);
    }
}
