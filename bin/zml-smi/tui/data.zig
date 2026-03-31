const std = @import("std");
const smi_info = @import("zml-smi/info");
const di = smi_info.device_info;
pub const DeviceInfo = di.DeviceInfo;
pub const GpuInfo = di.GpuInfo;
pub const NeuronInfo = di.NeuronInfo;
pub const TpuInfo = di.TpuInfo;
pub const Target = di.Target;
pub const Targets = @import("zml-smi/platform").Targets;
const hi = smi_info.host_info;
pub const HostInfo = hi.HostInfo;
pub const HostData = hi.HostData;
pub const pi = smi_info.process_info;
pub const ProcessDoubleBuffer = @import("zml-smi/utils").double_buffer.DoubleBuffer(std.ArrayList(pi.ProcessInfo));
pub const ProcessEnricher = @import("zml-smi/bindings/linux").process.ProcessEnricher;

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
            if (self.len < capacity) {
                self.len += 1;
            }
        }

        pub fn sliceLast(self: *const Self, arena: std.mem.Allocator, n: usize) std.mem.Allocator.Error![]const T {
            const count = @min(n, self.len);
            if (count == 0) {
                return &.{};
            }
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
    devices: []*DeviceInfo,
    host: *HostInfo,
    history: HistoryBuffers = .{},
    targets: Targets,
    process_lists: []*ProcessDoubleBuffer,
    enricher: *ProcessEnricher,
    allocator: std.mem.Allocator,
    io: std.Io,
    tui_refresh_rate: u16,

    pub const Config = struct {
        devices: []*DeviceInfo,
        host: *HostInfo,
        targets: Targets,
        tui_refresh_rate: u16,
        process_lists: []*ProcessDoubleBuffer,
        enricher: *ProcessEnricher,
        io: std.Io,
    };

    pub fn init(allocator: std.mem.Allocator, cfg: Config) !SystemState {
        return .{
            .devices = cfg.devices,
            .host = cfg.host,
            .history = try HistoryBuffers.init(allocator, cfg.devices.len),
            .targets = cfg.targets,
            .tui_refresh_rate = cfg.tui_refresh_rate,
            .process_lists = cfg.process_lists,
            .enricher = cfg.enricher,
            .allocator = allocator,
            .io = cfg.io,
        };
    }

    pub fn deinit(self: *SystemState, allocator: std.mem.Allocator) void {
        self.history.deinit(allocator);
    }

    pub fn deviceCount(self: *const SystemState) usize {
        return self.devices.len;
    }

    pub fn recordHistory(self: *SystemState) void {
        for (0..self.deviceCount()) |i| {
            switch (self.devices[i].*) {
                .cuda, .rocm => |*sv| {
                    const gpu = sv.front().*;
                    self.history.util[i].push(gpu.util_percent orelse 0);
                    const used = gpu.mem_used_bytes orelse 0;
                    const total = gpu.mem_total_bytes orelse 0;
                    self.history.mem_util[i].push(if (total > 0) used * 100 / total else 0);
                    self.history.temp[i].push(gpu.temperature orelse 0);
                    self.history.power[i].push(gpu.power_mw orelse 0);
                },
                inline else => |*sv| {
                    const info = sv.front().*;
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
