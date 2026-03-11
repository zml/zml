const std = @import("std");
const di = @import("../info/device_info.zig");
pub const DeviceInfo = di.DeviceInfo;
pub const GpuInfo = di.GpuInfo;
pub const NeuronInfo = di.NeuronInfo;
pub const TpuInfo = di.TpuInfo;
pub const Target = di.Target;
pub const Targets = @import("../platform.zig").Targets;
pub const HostInfo = @import("../info/host_info.zig").HostInfo;
pub const ProcessScanner = @import("../bindings/linux/process.zig").ProcessScanner;
pub const ProcessList = @import("../info/process_info.zig").ProcessList;

pub const history_len: usize = 500;

pub fn RingBuffer(comptime T: type, comptime capacity: usize) type {
    return struct {
        const Self = @This();

        buf: [capacity]T = [_]T{0} ** capacity,
        head: usize = 0,
        len: usize = 0,

        pub fn push(self: *Self, value: T) void {
            self.buf[self.head] = value;
            self.head = (self.head + 1) % capacity;
            if (self.len < capacity) self.len += 1;
        }

        pub fn latest(self: *const Self) T {
            if (self.len == 0) return 0;
            return self.buf[(self.head + capacity - 1) % capacity];
        }

        pub fn sliceLast(self: *const Self, arena: std.mem.Allocator, n: usize) std.mem.Allocator.Error![]const T {
            const count = @min(n, self.len);
            if (count == 0) return &.{};
            const out = try arena.alloc(T, count);
            const start = (self.head + capacity - count) % capacity;
            for (0..count) |i| {
                out[i] = self.buf[(start + i) % capacity];
            }
            return out;
        }
    };
}

pub const HistoryBuffers = struct {
    util: []RingBuffer(u64, history_len) = &.{},
    mem_util: []RingBuffer(u64, history_len) = &.{},
    temp: []RingBuffer(u64, history_len) = &.{},
    power: []RingBuffer(u64, history_len) = &.{},

    pub fn init(allocator: std.mem.Allocator, count: usize) !HistoryBuffers {
        var h: HistoryBuffers = .{
            .util = try allocator.alloc(RingBuffer(u64, history_len), count),
            .mem_util = try allocator.alloc(RingBuffer(u64, history_len), count),
            .temp = try allocator.alloc(RingBuffer(u64, history_len), count),
            .power = try allocator.alloc(RingBuffer(u64, history_len), count),
        };

        @memset(h.util, .{});
        @memset(h.mem_util, .{});
        @memset(h.temp, .{});
        @memset(h.power, .{});

        return h;
    }

    pub fn deinit(self: *HistoryBuffers, allocator: std.mem.Allocator) void {
        allocator.free(self.util);
        allocator.free(self.mem_util);
        allocator.free(self.temp);
        allocator.free(self.power);
    }
};

pub const SystemState = struct {
    devices: []*DeviceInfo = &.{},
    host: *HostInfo,
    history: HistoryBuffers = .{},
    targets: Targets = .{},
    process_scanner: ?*ProcessScanner = null,
    sample_interval_ms: u16,

    pub fn init(allocator: std.mem.Allocator, devices: []*DeviceInfo, host_info: *HostInfo, targets: Targets, sample_interval_ms: u16) !SystemState {
        return .{
            .devices = devices,
            .host = host_info,
            .history = try HistoryBuffers.init(allocator, devices.len),
            .targets = targets,
            .sample_interval_ms = sample_interval_ms,
        };
    }

    pub fn deinit(self: *SystemState, allocator: std.mem.Allocator) void {
        self.history.deinit(allocator);
    }

    pub fn deviceCount(self: *const SystemState) usize {
        return self.devices.len;
    }

    pub fn getProcesses(self: *const SystemState) ?*const ProcessList {
        if (self.process_scanner) |scanner| return scanner.getFront();
        return null;
    }

    pub fn recordHistory(self: *SystemState) void {
        for (0..self.deviceCount()) |i| {
            switch (self.devices[i].*) {
                .cuda, .rocm => |gpu| {
                    self.history.util[i].push(gpu.util_percent orelse 0);
                    const used = gpu.mem_used_bytes orelse 0;
                    const total = gpu.mem_total_bytes orelse 0;
                    self.history.mem_util[i].push(if (total > 0) used * 100 / total else 0);
                    self.history.temp[i].push(gpu.temperature orelse 0);
                    self.history.power[i].push(gpu.power_mw orelse 0);
                },
                inline else => |info| {
                    self.history.util[i].push(info.util_percent orelse 0);
                    const used = info.mem_used_bytes orelse 0;
                    const total = info.mem_total_bytes orelse 0;
                    self.history.mem_util[i].push(if (total > 0) used * 100 / total else 0);
                    self.history.temp[i].push(0);
                    self.history.power[i].push(0);
                },
            }
        }
    }
};
