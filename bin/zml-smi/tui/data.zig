const std = @import("std");
const builtin = @import("builtin");
const smi_info = @import("zml-smi/info");
const di = smi_info.device_info;
pub const DeviceInfo = di.DeviceInfo;
pub const GpuInfo = di.GpuInfo;
pub const NeuronInfo = di.NeuronInfo;
pub const TpuInfo = di.TpuInfo;
pub const Target = di.Target;
pub const Targets = smi_info.Targets;
const hi = smi_info.host_info;
pub const HostInfo = hi.HostInfo;
pub const HostData = hi.HostData;
pub const pi = smi_info.process_info;
pub const ProcessDoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer(std.ArrayList(pi.ProcessInfo));
pub const ProcessEnricher = if (builtin.os.tag == .macos)
    @import("zml-smi/platforms/macos").process.ProcessEnricher
else
    @import("zml-smi/platforms/linux").process.ProcessEnricher;

pub const History = std.Deque(u64);

pub const HistoryBuffers = struct {
    util: []History = &.{},
    mem_util: []History = &.{},
    temp: []History = &.{},
    power: []History = &.{},

    pub fn init(allocator: std.mem.Allocator, count: usize) !HistoryBuffers {
        const h: HistoryBuffers = .{
            .util = try allocator.alloc(History, count),
            .mem_util = try allocator.alloc(History, count),
            .temp = try allocator.alloc(History, count),
            .power = try allocator.alloc(History, count),
        };

        @memset(h.util, .empty);
        @memset(h.mem_util, .empty);
        @memset(h.temp, .empty);
        @memset(h.power, .empty);

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
    /// Charts plot one sample per braille dot, two per cell: the history
    /// keeps enough of them to fill the widest terminal seen.
    history_len: usize = 0,
    targets: Targets,
    process_lists: []*ProcessDoubleBuffer,
    enricher: *ProcessEnricher,
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    io: std.Io,
    tui_refresh_rate: u16,

    pub const Config = struct {
        devices: []*DeviceInfo,
        host: *HostInfo,
        targets: Targets,
        tui_refresh_rate: u16,
        process_lists: []*ProcessDoubleBuffer,
        enricher: *ProcessEnricher,
        gpa: std.mem.Allocator,
        arena: std.mem.Allocator,
        io: std.Io,
    };

    pub fn init(cfg: Config) !SystemState {
        return .{
            .devices = cfg.devices,
            .host = cfg.host,
            .history = try HistoryBuffers.init(cfg.arena, cfg.devices.len),
            .targets = cfg.targets,
            .tui_refresh_rate = cfg.tui_refresh_rate,
            .process_lists = cfg.process_lists,
            .enricher = cfg.enricher,
            .gpa = cfg.gpa,
            .arena = cfg.arena,
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
                .cuda, .rocm, .oneapi => |*sv| {
                    const gpu = sv.front().*;
                    self.record(&self.history.util[i], gpu.util_percent orelse 0);
                    const used = gpu.mem_used_bytes orelse 0;
                    const total = gpu.mem_total_bytes orelse 0;
                    self.record(&self.history.mem_util[i], if (total > 0) used * 100 / total else 0);
                    self.record(&self.history.temp[i], gpu.temperature orelse 0);
                    self.record(&self.history.power[i], gpu.power_mw orelse 0);
                },
                inline else => |*sv| {
                    const info = sv.front().*;
                    self.record(&self.history.util[i], info.util_percent orelse 0);
                    const used = info.mem_used_bytes orelse 0;
                    const total = info.mem_total_bytes orelse 0;
                    self.record(&self.history.mem_util[i], if (total > 0) used * 100 / total else 0);
                    self.record(&self.history.temp[i], 0);
                    self.record(&self.history.power[i], 0);
                },
            }
        }
    }

    fn record(self: *SystemState, history: *History, value: u64) void {
        if (history.len > 0 and history.len >= self.history_len) {
            _ = history.popFront();
        }
        history.pushBack(self.arena, value) catch {};
    }
};
