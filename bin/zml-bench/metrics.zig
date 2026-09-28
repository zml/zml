const std = @import("std");

pub const Stats = struct {
    count: u64 = 0,
    meanMs: ?f64 = null,
    minMs: ?f64 = null,
    p50Ms: ?f64 = null,
    p95Ms: ?f64 = null,
    p99Ms: ?f64 = null,
    maxMs: ?f64 = null,

    pub fn fromSamples(samples: []f64) Stats {
        if (samples.len == 0) return .{};
        std.mem.sort(f64, samples, {}, std.sort.asc(f64));
        var sum: f64 = 0;
        for (samples) |sample| sum += sample;
        return .{
            .count = samples.len,
            .meanMs = sum / @as(f64, @floatFromInt(samples.len)),
            .minMs = samples[0],
            .p50Ms = percentile(samples, 0.5),
            .p95Ms = percentile(samples, 0.95),
            .p99Ms = percentile(samples, 0.99),
            .maxMs = samples[samples.len - 1],
        };
    }

    fn percentile(sorted: []const f64, p: f64) f64 {
        const position = @as(f64, @floatFromInt(sorted.len - 1)) * p;
        const lower: usize = @intFromFloat(position);
        const upper = @min(lower + 1, sorted.len - 1);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - @as(f64, @floatFromInt(lower)));
    }
};

/// Bounded timing storage: 16 logarithmic buckets per power of two in microseconds.
/// Percentiles use bucket upper bounds (within about 4.5% + 1us); mean/min/max are exact.
pub const Histogram = struct {
    buckets: [512]u64 = @splat(0),
    count: u64 = 0,
    sumMs: f64 = 0,
    minMs: f64 = std.math.inf(f64),
    maxMs: f64 = 0,

    pub fn add(self: *Histogram, ms: f64) void {
        std.debug.assert(ms >= 0 and std.math.isFinite(ms));
        const bucket: usize = @intFromFloat(@min(511, @floor(@log2(ms * 1000 + 1) * 16)));
        self.buckets[bucket] += 1;
        self.count += 1;
        self.sumMs += ms;
        self.minMs = @min(self.minMs, ms);
        self.maxMs = @max(self.maxMs, ms);
    }

    pub fn merge(self: *Histogram, other: *const Histogram) void {
        for (&self.buckets, other.buckets) |*dest, source| dest.* += source;
        self.count += other.count;
        self.sumMs += other.sumMs;
        self.minMs = @min(self.minMs, other.minMs);
        self.maxMs = @max(self.maxMs, other.maxMs);
    }

    pub fn stats(self: *const Histogram) Stats {
        if (self.count == 0) return .{};
        return .{
            .count = self.count,
            .meanMs = self.sumMs / @as(f64, @floatFromInt(self.count)),
            .minMs = self.minMs,
            .p50Ms = self.percentile(0.5),
            .p95Ms = self.percentile(0.95),
            .p99Ms = self.percentile(0.99),
            .maxMs = self.maxMs,
        };
    }

    fn percentile(self: *const Histogram, p: f64) f64 {
        const rank: u64 = @intFromFloat(@ceil(@as(f64, @floatFromInt(self.count)) * p));
        var count: u64 = 0;
        for (self.buckets, 0..) |bucket, i| {
            count += bucket;
            if (count >= rank) return if (i == self.buckets.len - 1) self.maxMs else @min(self.maxMs, (@exp2(@as(f64, @floatFromInt(i + 1)) / 16) - 1) / 1000);
        }
        unreachable;
    }
};

test "exact request percentiles and bounded interval histogram" {
    var samples = [_]f64{ 300, 100, 200 };
    const stats = Stats.fromSamples(&samples);
    try std.testing.expectEqual(@as(?f64, 200), stats.p50Ms);
    try std.testing.expectEqual(@as(?f64, 290), stats.p95Ms);
    var histogram: Histogram = .{};
    histogram.add(0);
    histogram.add(10);
    var other: Histogram = .{};
    other.add(20);
    other.add(30);
    histogram.merge(&other);
    const intervals = histogram.stats();
    try std.testing.expectEqual(@as(u64, 4), intervals.count);
    try std.testing.expectEqual(@as(?f64, 15), intervals.meanMs);
    try std.testing.expect(intervals.p50Ms.? >= 10 and intervals.p50Ms.? <= 10.451);
    try std.testing.expectEqual(@as(?f64, 30), intervals.p99Ms);
    try std.testing.expectEqual(null, (Histogram{}).stats().meanMs);
}
