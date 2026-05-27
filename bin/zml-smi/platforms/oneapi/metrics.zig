const std = @import("std");
const OneApi = @import("oneapi.zig");
const device_info = @import("zml-smi/info").device_info;
const GpuInfo = device_info.GpuInfo;
const DoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer;
const Collector = @import("zml-smi/collector").Collector;
const poll_metrics = @import("zml-smi/info").poll_metrics;
const process = @import("process.zig");

pub fn start(collector: *Collector) !void {
    const oneapi = try collector.arena.create(OneApi);
    oneapi.* = try OneApi.init(collector.arena, collector.io);

    const count = oneapi.monitor.devices.len;
    const dev_offset: u16 = @intCast(collector.device_infos.items.len);

    for (0..count) |i| {
        const dev = Device.open(oneapi, i) catch continue;
        const initial: GpuInfo = .{
            .name = dev.name(collector.arena) catch null,
            .driver_version = dev.driverVersion(collector.arena) catch null,
            .power_limit_mw = dev.powerLimit() catch null,
            .clock_graphics_max_mhz = dev.maxClockGraphics() catch null,
            .mem_total_bytes = dev.memTotal() catch null,
            .pcie_link_gen = dev.pcieLinkGen() catch null,
            .pcie_link_width = dev.pcieLinkWidth() catch null,
            .pcie_bandwidth_mbps = dev.pcieBandwidth() catch null,
        };
        const info = try collector.addDevice(.{ .oneapi = .{ .values = .{ initial, initial } } });

        try collector.spawnPoll(pollOnce, .{ null, &info.oneapi, dev });
    }

    const processes = try collector.createProcessList();
    try process.init(collector, processes, oneapi, dev_offset);
}

const pollOnce = poll_metrics.poll(*DoubleBuffer(GpuInfo), Device, metrics);

const Device = struct {
    oneapi: *OneApi,
    handle: OneApi.Handle,

    fn open(oneapi: *OneApi, index: usize) !Device {
        return .{
            .oneapi = oneapi,
            .handle = try oneapi.handleByIndex(index),
        };
    }

    fn name(self: Device, arena: std.mem.Allocator) ![]const u8 {
        return self.oneapi.name(arena, self.handle);
    }

    fn driverVersion(self: Device, arena: std.mem.Allocator) ![]const u8 {
        return self.oneapi.driverVersion(arena);
    }

    pub fn powerUsage(self: Device) !u64 {
        return self.oneapi.powerUsage(self.handle);
    }

    pub fn temperature(self: Device) !u64 {
        return self.oneapi.temperature(self.handle);
    }

    pub fn gpuUtil(self: Device) !u64 {
        return self.oneapi.gpuUtil(self.handle);
    }

    pub fn powerLimit(self: Device) !u64 {
        return self.oneapi.powerLimit(self.handle);
    }

    pub fn clockGraphics(self: Device) !u64 {
        return self.oneapi.clockGraphics(self.handle);
    }

    pub fn maxClockGraphics(self: Device) !u64 {
        return self.oneapi.maxClockGraphics(self.handle);
    }

    pub fn memUsed(self: Device) !u64 {
        return self.oneapi.memUsed(self.handle);
    }

    pub fn memTotal(self: Device) !u64 {
        return self.oneapi.memTotal(self.handle);
    }

    pub fn pcieLinkGen(self: Device) !u64 {
        return self.oneapi.pcieLinkGen(self.handle);
    }

    pub fn pcieLinkWidth(self: Device) !u64 {
        return self.oneapi.pcieLinkWidth(self.handle);
    }

    pub fn pcieBandwidth(self: Device) !u64 {
        return self.oneapi.pcieBandwidth(self.handle);
    }
};

const metrics = .{
    .{ .field = "power_mw", .query = Device.powerUsage },
    .{ .field = "power_limit_mw", .query = Device.powerLimit },
    .{ .field = "temperature", .query = Device.temperature },
    .{ .field = "util_percent", .query = Device.gpuUtil },
    .{ .field = "clock_graphics_mhz", .query = Device.clockGraphics },
    .{ .field = "clock_graphics_max_mhz", .query = Device.maxClockGraphics },
    .{ .field = "mem_used_bytes", .query = Device.memUsed },
    .{ .field = "mem_total_bytes", .query = Device.memTotal },
    .{ .field = "pcie_bandwidth_mbps", .query = Device.pcieBandwidth },
    .{ .field = "pcie_link_gen", .query = Device.pcieLinkGen },
    .{ .field = "pcie_link_width", .query = Device.pcieLinkWidth },
};
