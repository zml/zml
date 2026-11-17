const std = @import("std");
const counter = @import("zml-smi/native").testing.counters;

const Counter = counter.Counter;
const Growth = counter.Growth;
const Counters = counter.Counters;

fn at(ms: i64) std.Io.Timestamp {
    return .fromNanoseconds(@as(i96, ms) * std.time.ns_per_ms);
}

test Counter {
    var reading: Counter = .{};
    try std.testing.expectEqual(null, counter.testing.step(&reading, .{ .total = 100, .at = at(0) }));

    const growth = counter.testing.step(&reading, .{ .total = 600, .at = at(500) }).?;
    try std.testing.expectEqual(500, growth.amount);
    try std.testing.expectEqual(500, growth.elapsed.toMilliseconds());

    // The total went backwards, then no time passed.
    try std.testing.expectEqual(null, counter.testing.step(&reading, .{ .total = 10, .at = at(1000) }));
    try std.testing.expectEqual(null, counter.testing.step(&reading, .{ .total = 20, .at = at(1000) }));
}

test Growth {
    const half_second: std.Io.Duration = .fromMilliseconds(500);
    // 10 mJ in 500 ms is 20 mW.
    try std.testing.expectEqual(20_000, (Growth{ .amount = 10_000, .elapsed = half_second }).perSecond());
    try std.testing.expectEqual(std.math.maxInt(u64), (Growth{ .amount = std.math.maxInt(u64), .elapsed = .fromNanoseconds(1) }).perSecond());

    try std.testing.expectEqual(50, (Growth{ .amount = 250_000_000, .elapsed = half_second }).percentOfTime());
    try std.testing.expectEqual(100, (Growth{ .amount = 2_000_000_000, .elapsed = half_second }).percentOfTime());

    const cycles: Growth = .{ .amount = 30, .elapsed = half_second };
    try std.testing.expectEqual(30, cycles.percentOf(.{ .amount = 100, .elapsed = half_second }));
    try std.testing.expectEqual(0, cycles.percentOf(.{ .amount = 0, .elapsed = half_second }));
}

test Counters {
    const io = std.testing.io;
    var counters: Counters(u32) = .init(std.testing.allocator);
    defer counters.deinit();

    try std.testing.expectEqual(null, try counters.advance(1, io, 10));
    try std.testing.expectEqual(null, try counters.advance(2, io, 10));
    counters.sweep();

    _ = try counters.advance(1, io, 20);
    counters.sweep();
    // Key 2 was not advanced since the last sweep, so it starts over.
    try std.testing.expect(counters.map.contains(1));
    try std.testing.expect(!counters.map.contains(2));
}
