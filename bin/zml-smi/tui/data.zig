const std = @import("std");
const smi = @import("zml-smi");
const utils = @import("lib/utils.zig");

pub const History = std.Deque(u64);

pub const HistoryBuffers = struct {
    util: []History = &.{},
    mem_util: []History = &.{},
    temp: []History = &.{},
    power: []History = &.{},

    pub fn init(allocator: std.mem.Allocator, count: usize) !HistoryBuffers {
        const util = try allocator.alloc(History, count);
        errdefer allocator.free(util);

        const mem_util = try allocator.alloc(History, count);
        errdefer allocator.free(mem_util);

        const temp = try allocator.alloc(History, count);
        errdefer allocator.free(temp);

        const power = try allocator.alloc(History, count);

        const h: HistoryBuffers = .{
            .util = util,
            .mem_util = mem_util,
            .temp = temp,
            .power = power,
        };

        @memset(h.util, .empty);
        @memset(h.mem_util, .empty);
        @memset(h.temp, .empty);
        @memset(h.power, .empty);

        return h;
    }

    pub fn deinit(self: *HistoryBuffers, allocator: std.mem.Allocator) void {
        for (self.util) |*h| h.deinit(allocator);
        for (self.mem_util) |*h| h.deinit(allocator);
        for (self.temp) |*h| h.deinit(allocator);
        for (self.power) |*h| h.deinit(allocator);

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
            .history = try HistoryBuffers.init(cfg.gpa, cfg.collector.devices.items.len),
            .tui_refresh_rate = cfg.tui_refresh_rate,
            .gpa = cfg.gpa,
            .arena = cfg.arena,
            .io = cfg.io,
        };
    }

    pub fn deinit(self: *SystemState) void {
        self.history.deinit(self.gpa);
    }

    pub fn deviceCount(self: *const SystemState) usize {
        return self.collector.devices.items.len;
    }

    pub fn device(self: *const SystemState, id: u16) smi.Device {
        return self.collector.devices.items[id];
    }

    pub fn sample(self: *const SystemState, id: u16) smi.Sample {
        return self.collector.sample(id);
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
        self.record(&self.history.mem_util[i], utils.percent(used, total));
    }

    fn record(self: *SystemState, history: *History, value: u64) void {
        if (history.len > 0 and history.len >= self.history_len) {
            _ = history.popFront();
        }
        history.pushBack(self.gpa, value) catch {};
    }
};

test "HistoryBuffers deinit frees inner histories" {
    var history: HistoryBuffers = try .init(std.testing.allocator, 2);
    defer history.deinit(std.testing.allocator);

    try history.util[0].pushBack(std.testing.allocator, 42);
    try history.mem_util[1].pushBack(std.testing.allocator, 75);
    try history.temp[0].pushBack(std.testing.allocator, 65);
    try history.power[1].pushBack(std.testing.allocator, 150);
}
