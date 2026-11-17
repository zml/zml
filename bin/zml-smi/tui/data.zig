const std = @import("std");
const model = @import("zml-smi/model");

const implementation = @This();

pub const Metric = enum { util, memory, temperature, power };

/// Captures readings into a frame allocator. Device indices stay stable across captures.
pub const FrameProvider = struct {
    context: *anyopaque,
    capture: *const fn (*anyopaque, std.mem.Allocator) std.mem.Allocator.Error!model.Frame,
};

pub const SystemState = struct {
    provider: FrameProvider,
    frame: ?model.Frame = null,
    /// Bounded by the available history capacity when drawing a chart.
    history_len: usize = 128,
    tui_refresh_rate: u16,

    /// Called once before drawing a frame. Widgets borrow only this capture.
    pub fn capture(self: *SystemState, allocator: std.mem.Allocator) std.mem.Allocator.Error!void {
        self.frame = try self.provider.capture(self.provider.context, allocator);
    }

    pub fn historyInterval(self: *const SystemState) u32 {
        const history_data = (self.frame orelse return 1).history orelse return 1;
        return history_data.interval_ms;
    }

    /// Projects only a visible chart's metric into timestamp buckets.
    pub fn history(
        self: *const SystemState,
        frame: std.mem.Allocator,
        device_index: u16,
        metric: Metric,
    ) std.mem.Allocator.Error![]const ?u64 {
        const history_data = (self.frame orelse return &.{}).history orelse return &.{};
        return historySeries(
            frame,
            history_data.points[device_index],
            metric,
            history_data.captured_at,
            history_data.interval_ms,
            @min(self.history_len, history_data.capacity),
        );
    }

    pub fn deviceCount(self: *const SystemState) usize {
        return if (self.frame) |frame| frame.devices.len else 0;
    }

    pub fn host(self: *const SystemState) model.HostData {
        return if (self.frame) |frame| frame.host else .{};
    }

    pub fn processes(self: *const SystemState) []const model.ProcessInfo {
        return if (self.frame) |frame| frame.processes else &.{};
    }
};

pub fn memoryPercent(common: model.Common) ?u64 {
    const used = common.mem_used_bytes orelse return null;
    const total = common.mem_total_bytes orelse return null;
    if (total == 0) return null;
    return @intCast(@min(@as(u128, used) * 100 / total, 100));
}

fn historySeries(
    frame: std.mem.Allocator,
    points: []const model.HistoryPoint,
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
