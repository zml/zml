const std = @import("std");
const OneApi = @import("oneapi.zig");
const sysfs = @import("sysfs.zig");
const device_info = @import("zml-smi/info").device_info;
const GpuInfo = device_info.GpuInfo;
const DoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer;
const Collector = @import("zml-smi/collector").Collector;
const process = @import("process.zig");

pub fn start(collector: *Collector) !void {
    const oneapi = try collector.arena.create(OneApi);
    oneapi.* = try OneApi.init(collector.arena, collector.io);

    const dev_offset: u16 = @intCast(collector.device_infos.items.len);
    var warmups: std.ArrayList(Warmup) = .empty;
    var process_targets: std.ArrayList(process.Target) = .empty;

    for (0..oneapi.deviceCount()) |i| {
        const poll_arena = try collector.createPollArena();
        const dev = try collector.arena.create(Device);
        dev.* = Device.open(oneapi, poll_arena, i, @intCast(dev_offset + process_targets.items.len)) catch continue;
        const initial = dev.initial(collector.arena);
        const info = try collector.addDevice(.{ .oneapi = .{ .values = .{ initial, initial } } });

        try collector.spawnPoll(pollOnce, .{ poll_arena, collector.io, &info.oneapi, dev });
        if (collector.poll_only) {
            try warmups.append(collector.arena, .{ .arena = poll_arena, .db = &info.oneapi, .dev = dev });
        }
        try process_targets.append(collector.arena, dev.target());
    }

    if (process_targets.items.len == 0) return;

    const processes = try collector.createProcessList();
    try process.init(collector, processes, try process_targets.toOwnedSlice(collector.arena));

    if (collector.poll_only) {
        for (warmups.items) |warmup| {
            pollOnce(warmup.arena, collector.io, warmup.db, warmup.dev);
        }
    }
}

const Warmup = struct {
    arena: *std.heap.ArenaAllocator,
    db: *DoubleBuffer(GpuInfo),
    dev: *Device,
};

fn pollOnce(arena: *std.heap.ArenaAllocator, io: std.Io, db: *DoubleBuffer(GpuInfo), dev: *Device) void {
    _ = arena.reset(.retain_capacity);

    const targets = [1]process.Target{dev.target()};
    var usage = sysfs.collectDeviceUsage(arena.allocator(), io, &targets) catch std.AutoHashMapUnmanaged(u16, sysfs.DeviceSample){};
    defer usage.deinit(arena.allocator());

    const current = Snapshot{
        .activity = if (usage.get(dev.device_idx)) |sample| sample.engine else null,
        .energy = dev.energy() catch null,
        .mem_used_bytes = if (usage.get(dev.device_idx)) |sample| sample.mem_kib * 1024 else 0,
    };

    const back = db.back();
    back.* = db.front().*;

    inline for (metrics) |m| {
        @field(back, m.field) = m.query(dev.*) catch null;
    }
    back.mem_used_bytes = current.mem_used_bytes;
    back.util_percent = if (sysfs.processUtil(dev.activity_prev, current.activity)) |util| @intCast(util) else 0;
    back.power_mw = sysfs.powerMilliwatts(dev.energy_prev, current.energy);

    dev.saveSnapshot(current);
    db.swap();
}

const Device = struct {
    oneapi: *const OneApi,
    arena: *std.heap.ArenaAllocator,
    handle: OneApi.Handle,
    device_idx: u16,
    activity_prev: ?sysfs.EngineSample = null,
    energy_prev: ?sysfs.EnergySample = null,

    fn open(oneapi: *const OneApi, arena: *std.heap.ArenaAllocator, index: usize, device_idx: u16) !Device {
        return .{
            .oneapi = oneapi,
            .arena = arena,
            .handle = try oneapi.handleByIndex(index),
            .device_idx = device_idx,
        };
    }

    fn target(self: Device) OneApi.Target {
        return self.oneapi.target(self.handle, self.device_idx);
    }

    fn initial(self: Device, arena: std.mem.Allocator) GpuInfo {
        return .{
            .name = self.oneapi.name(arena, self.handle) catch null,
            .driver_version = self.oneapi.driverVersion(arena) catch null,
            .power_limit_mw = self.powerLimit() catch null,
            .clock_graphics_max_mhz = self.maxClockGraphics() catch null,
            .mem_total_bytes = self.memTotal() catch null,
            .pcie_link_gen = self.pcieLinkGen() catch null,
            .pcie_link_width = self.pcieLinkWidth() catch null,
            .pcie_bandwidth_mbps = self.pcieBandwidth() catch null,
        };
    }

    fn saveSnapshot(self: *Device, current: Snapshot) void {
        self.activity_prev = current.activity;
        self.energy_prev = current.energy;
    }

    pub fn temperature(self: Device) !u64 {
        return self.oneapi.temperature(self.arena.allocator(), self.handle);
    }

    pub fn energy(self: Device) !sysfs.EnergySample {
        return self.oneapi.energy(self.arena.allocator(), self.handle);
    }

    pub fn powerLimit(self: Device) !u64 {
        return self.oneapi.powerLimit(self.arena.allocator(), self.handle);
    }

    pub fn clockGraphics(self: Device) !u64 {
        return self.oneapi.clockGraphics(self.arena.allocator(), self.handle);
    }

    pub fn maxClockGraphics(self: Device) !u64 {
        return self.oneapi.maxClockGraphics(self.arena.allocator(), self.handle);
    }

    pub fn memTotal(self: Device) !u64 {
        return self.oneapi.memTotal(self.arena.allocator(), self.handle);
    }

    pub fn pcieLinkGen(self: Device) !u64 {
        return self.oneapi.pcieLinkGen(self.arena.allocator(), self.handle);
    }

    pub fn pcieLinkWidth(self: Device) !u64 {
        return self.oneapi.pcieLinkWidth(self.arena.allocator(), self.handle);
    }

    pub fn pcieBandwidth(self: Device) !u64 {
        return self.oneapi.pcieBandwidth(self.arena.allocator(), self.handle);
    }
};

const Snapshot = struct {
    activity: ?sysfs.EngineSample,
    energy: ?sysfs.EnergySample,
    mem_used_bytes: ?u64,
};

const metrics = .{
    .{ .field = "power_limit_mw", .query = Device.powerLimit },
    .{ .field = "temperature", .query = Device.temperature },
    .{ .field = "clock_graphics_mhz", .query = Device.clockGraphics },
    .{ .field = "clock_graphics_max_mhz", .query = Device.maxClockGraphics },
    .{ .field = "mem_total_bytes", .query = Device.memTotal },
    .{ .field = "pcie_bandwidth_mbps", .query = Device.pcieBandwidth },
    .{ .field = "pcie_link_gen", .query = Device.pcieLinkGen },
    .{ .field = "pcie_link_width", .query = Device.pcieLinkWidth },
};
