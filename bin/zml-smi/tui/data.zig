const std = @import("std");
const smi = @import("zml-smi");

const implementation = @This();

pub const Metric = enum { util, memory, temperature, power };

pub const SystemState = struct {
    collector: *smi.Collector,
    snapshot: ?smi.Collector.Snapshot = null,
    /// Bounded by the collector's history capacity when drawing a chart.
    history_len: usize = 128,
    captured_at: std.Io.Timestamp = .fromNanoseconds(0),
    tui_refresh_rate: u16,

    /// Called once before drawing a frame. Widgets borrow only this capture.
    pub fn capture(self: *SystemState, frame: std.mem.Allocator) std.mem.Allocator.Error!void {
        self.snapshot = try self.collector.capture(frame, true);
        self.captured_at = .now(self.collector.io, .awake);
    }

    pub fn historyInterval(self: *const SystemState) u32 {
        const milliseconds = @max(
            @divTrunc(self.collector.interval.nanoseconds, std.time.ns_per_ms),
            1,
        );
        return std.math.cast(u32, milliseconds) orelse std.math.maxInt(u32);
    }

    /// Projects only a visible chart's metric into timestamp buckets.
    pub fn history(
        self: *const SystemState,
        frame: std.mem.Allocator,
        device_index: u16,
        metric: Metric,
    ) std.mem.Allocator.Error![]const ?u64 {
        const snapshot = self.snapshot orelse return &.{};
        return historySeries(
            frame,
            snapshot.histories[device_index],
            metric,
            self.captured_at,
            self.historyInterval(),
            @min(self.history_len, smi.Collector.history_capacity),
        );
    }

    pub fn deviceCount(self: *const SystemState) usize {
        return self.collector.devices.items.len;
    }

    pub fn device(self: *const SystemState, index: u16) smi.Device {
        return self.collector.devices.items[index];
    }

    pub fn sample(self: *const SystemState, index: u16) smi.Sample {
        return if (self.snapshot) |snapshot|
            snapshot.samples[index]
        else
            .init(self.device(index).target);
    }

    pub fn host(self: *const SystemState) smi.HostData {
        return if (self.snapshot) |snapshot| snapshot.host else .{};
    }

    pub fn processes(self: *const SystemState) []const smi.ProcessInfo {
        return if (self.snapshot) |snapshot| snapshot.processes else &.{};
    }
};

pub fn memoryPercent(common: smi.Common) ?u64 {
    const used = common.mem_used_bytes orelse return null;
    const total = common.mem_total_bytes orelse return null;
    if (total == 0) return null;
    return @intCast(@min(@as(u128, used) * 100 / total, 100));
}

fn historySeries(
    frame: std.mem.Allocator,
    points: []const smi.Collector.HistoryPoint,
    metric: Metric,
    now: std.Io.Timestamp,
    interval_ms: u32,
    count: usize,
) std.mem.Allocator.Error![]const ?u64 {
    const values = try frame.alloc(?u64, count);
    @memset(values, null);
    for (points) |point| {
        const index = historyIndex(point.at, now, interval_ms, count) orelse continue;
        values[index] = switch (metric) {
            .util => point.common.util_percent,
            .memory => memoryPercent(point.common),
            .temperature => point.common.temperature,
            .power => point.common.power_mw,
        };
    }
    return values;
}

fn historyIndex(
    completed: std.Io.Timestamp,
    now: std.Io.Timestamp,
    interval_ms: u32,
    count: usize,
) ?usize {
    if (count == 0 or interval_ms == 0) return null;
    const elapsed = @max(completed.durationTo(now).nanoseconds, 0);
    const age = @divTrunc(elapsed, @as(i96, interval_ms) * std.time.ns_per_ms);
    if (age >= count) return null;
    return count - 1 - @as(usize, @intCast(age));
}

pub const testing = if (@import("builtin").is_test) struct {
    pub const historyIndex = implementation.historyIndex;
    pub const historySeries = implementation.historySeries;
} else struct {};
