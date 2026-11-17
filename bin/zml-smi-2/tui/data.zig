const std = @import("std");
const contract = @import("zml-smi-2/contract");
const spec = @import("spec.zig");

const implementation = @This();

pub const Metric = enum {
    util,
    memory,
    temperature,
    power,

    /// Read a live sample or a compact history sample, preserving the raw memory ratio.
    pub fn read(self: Metric, readings: anytype) ?u128 {
        const scalar = switch (self) {
            .util => readings.get(.util_percent),
            .memory => return memoryRatio(readings),
            .temperature => readings.get(.temperature),
            .power => readings.get(.power_mw),
        } orelse return null;
        return scalar;
    }
};

/// Captures readings into a frame allocator. Device indices stay stable across captures.
pub const FrameProvider = struct {
    context: *anyopaque,
    capture: *const fn (*anyopaque, std.mem.Allocator) std.mem.Allocator.Error!contract.Frame,
};

pub const SystemState = struct {
    provider: FrameProvider,
    devices: []const spec.DeviceView,
    frame: contract.Frame = .{},
    /// The first asset is the overview header logo.
    logos: []const []const u8 = &.{},
    /// Bounded by the available history capacity when drawing a chart.
    history_len: usize = 128,
    tui_refresh_rate: u16,

    /// Called once before drawing a frame. Widgets borrow only this capture.
    pub fn capture(self: *SystemState, allocator: std.mem.Allocator) std.mem.Allocator.Error!void {
        self.frame = try self.provider.capture(self.provider.context, allocator);
    }

    /// Projects only a visible chart's metric into timestamp buckets.
    pub fn history(
        self: *const SystemState,
        arena: std.mem.Allocator,
        device_index: u16,
        metric: Metric,
    ) std.mem.Allocator.Error![]const ?u64 {
        const history_data = self.frame.history orelse return &.{};
        const count = @min(self.history_len, history_data.capacity);
        const values = try arena.alloc(?u64, count);
        @memset(values, null);
        for (history_data.points[device_index]) |point| {
            const index = historyIndex(
                point.at,
                history_data.captured_at,
                history_data.interval_ms,
                count,
            ) orelse continue;
            values[index] = if (metric.read(&point.values)) |value|
                @intCast(if (metric == .memory) @min(value, 100) else value)
            else
                null;
        }
        return values;
    }
};

fn memoryRatio(readings: anytype) ?u128 {
    const used = readings.get(.mem_used_bytes) orelse return null;
    const total = readings.get(.mem_total_bytes) orelse return null;
    if (total == 0) return null;
    return @as(u128, used) * 100 / total;
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
} else struct {};
