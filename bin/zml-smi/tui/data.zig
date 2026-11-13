const std = @import("std");
const smi = @import("zml-smi");

pub const History = std.ArrayList(u64);

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
    collector: *smi.Collector,
    history: HistoryBuffers = .{},
    /// Charts plot one sample per braille dot, two per cell: the history
    /// keeps enough of them to fill the widest terminal seen.
    history_len: usize = 0,
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    io: std.Io,
    tui_refresh_rate: u16,

    pub const Config = struct {
        collector: *smi.Collector,
        tui_refresh_rate: u16,
        gpa: std.mem.Allocator,
        arena: std.mem.Allocator,
        io: std.Io,
    };

    pub fn init(cfg: Config) !SystemState {
        return .{
            .collector = cfg.collector,
            .history = try HistoryBuffers.init(cfg.arena, cfg.collector.devices.items.len),
            .tui_refresh_rate = cfg.tui_refresh_rate,
            .gpa = cfg.gpa,
            .arena = cfg.arena,
            .io = cfg.io,
        };
    }

    pub fn deinit(self: *SystemState, allocator: std.mem.Allocator) void {
        self.history.deinit(allocator);
    }

    pub fn deviceCount(self: *const SystemState) usize {
        return self.collector.devices.items.len;
    }

    pub fn device(self: *const SystemState, id: u16) smi.Device {
        return self.collector.devices.items[id];
    }

    pub fn sample(self: *const SystemState, id: u16) smi.Sample {
        return self.collector.sample(@fromBackingInt(id));
    }

    pub fn host(self: *const SystemState) smi.HostData {
        return self.collector.host.read(self.io);
    }

    pub fn recordHistory(self: *SystemState) void {
        for (0..self.deviceCount()) |i| {
            switch (self.sample(@intCast(i))) {
                .cuda, .rocm, .oneapi => |gpu| {
                    self.recordUsage(i, gpu.util_percent, gpu.mem_used_bytes, gpu.mem_total_bytes);
                    self.record(&self.history.temp[i], gpu.temperature orelse 0);
                    self.record(&self.history.power[i], gpu.power_mw orelse 0);
                },
                .neuron => |core| {
                    self.recordUsage(i, core.util_percent, core.mem_used_bytes, core.mem_total_bytes);
                    self.record(&self.history.temp[i], 0);
                    self.record(&self.history.power[i], 0);
                },
                .tpu => |chip| {
                    self.recordUsage(i, chip.util_percent, chip.mem_used_bytes, chip.mem_total_bytes);
                    self.record(&self.history.temp[i], 0);
                    self.record(&self.history.power[i], 0);
                },
            }
        }
    }

    fn recordUsage(self: *SystemState, i: usize, util: ?u64, used: ?u64, total: ?u64) void {
        self.record(&self.history.util[i], util orelse 0);
        const total_bytes = total orelse 0;
        self.record(&self.history.mem_util[i], if (total_bytes > 0) (used orelse 0) * 100 / total_bytes else 0);
    }

    fn record(self: *SystemState, history: *History, value: u64) void {
        if (history.items.len > 0 and history.items.len >= self.history_len) {
            _ = history.orderedRemove(0);
        }
        history.append(self.arena, value) catch {};
    }
};
