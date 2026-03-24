const std = @import("std");
const amdsmi = @import("amdsmi.zig");
const DeviceInfo = @import("../../device_info.zig").DeviceInfo;
const worker = @import("../../worker.zig");

pub fn init(io: std.Io, allocator: std.mem.Allocator, device_infos: *std.ArrayList(DeviceInfo), signal: *worker.Signal) !void {
    try amdsmi.init();
    const count = try amdsmi.getDeviceCount();

    for (0..count) |i| {
        const dev = Device.open(@intCast(i)) catch continue;

        const info = try device_infos.addOne(allocator);
        info.* = .{};

        inline for (metrics) |metric| {
            try worker.spawnWorker(io, info, metric.field, metric.query, dev, signal);
        }
    }
}

const Device = struct {
    handle: amdsmi.Handle,

    pub fn open(index: u32) !Device {
        return .{ .handle = try amdsmi.getHandleByIndex(index) };
    }

    pub fn getName(self: Device) ![256]u8 {
        var buf: [256]u8 = .{0} ** 256;
        _ = try amdsmi.getName(self.handle, &buf);
        return buf;
    }

    // Power
    pub fn getPowerUsage(self: Device) !u64 {
        const pw = try amdsmi.getPowerUsage(self.handle);
        if (pw == std.math.maxInt(u32)) return error.not_supported;
        return pw;
    }
    pub fn getPowerLimit(self: Device) !u64 {
        // Header says W, but empirically power_limit is in µW; convert to mW
        const limit = try amdsmi.getPowerLimit(self.handle);
        if (limit == std.math.maxInt(u32)) return error.not_supported;
        return @as(u64, limit) / 1000;
    }

    // Thermal
    pub fn getTemperature(self: Device) !u64 {
        const temp = try amdsmi.getTemperature(self.handle);
        return @intCast(temp);
    }
    pub fn getFanSpeed(self: Device) !u64 {
        const speed = try amdsmi.getFanSpeed(self.handle);
        return @intCast(@divTrunc(speed * 100, 255));
    }

    // Utilization
    pub fn getGpuUtil(self: Device) !u64 {
        return try amdsmi.getGpuUtil(self.handle);
    }
    pub fn getMemUtil(self: Device) !u64 {
        return try amdsmi.getMemUtil(self.handle);
    }
    pub fn getEncoderUtil(self: Device) !u64 {
        return try amdsmi.getMmUtil(self.handle);
    }
    pub fn getDecoderUtil(self: Device) !u64 {
        return try amdsmi.getMmUtil(self.handle);
    }

    // Clocks
    pub fn getClockGraphics(self: Device) !u64 {
        return try amdsmi.getClockGraphics(self.handle);
    }
    pub fn getClockSm(self: Device) !u64 {
        return try amdsmi.getClockGraphics(self.handle);
    }
    pub fn getClockMem(self: Device) !u64 {
        return try amdsmi.getClockMem(self.handle);
    }
    pub fn getMaxClockGraphics(self: Device) !u64 {
        return try amdsmi.getMaxClockGraphics(self.handle);
    }
    pub fn getMaxClockMem(self: Device) !u64 {
        return try amdsmi.getMaxClockMem(self.handle);
    }

    // Performance state
    pub fn getPState(self: Device) !u64 {
        return try amdsmi.getPerfLevel(self.handle);
    }

    // Memory
    pub fn getMemUsed(self: Device) !u64 {
        return amdsmi.getMemUsed(self.handle);
    }
    pub fn getMemFree(self: Device) !u64 {
        const total = try amdsmi.getMemTotal(self.handle);
        const used = try amdsmi.getMemUsed(self.handle);
        return total - used;
    }
    pub fn getMemTotal(self: Device) !u64 {
        return amdsmi.getMemTotal(self.handle);
    }

    // PCIe
    pub fn getPcieTx(self: Device) !u64 {
        const bw = try amdsmi.getPcieBandwidth(self.handle);
        if (bw == std.math.maxInt(u32)) return error.not_supported;
        return @as(u64, bw) * 125;
    }
    pub fn getPcieRx(self: Device) !u64 {
        const bw = try amdsmi.getPcieBandwidth(self.handle);
        if (bw == std.math.maxInt(u32)) return error.not_supported;
        return @as(u64, bw) * 125;
    }
    pub fn getPcieLinkGen(self: Device) !u64 {
        return try amdsmi.getPcieLinkGen(self.handle);
    }
    pub fn getPcieLinkWidth(self: Device) !u64 {
        return try amdsmi.getPcieWidth(self.handle);
    }
};

const metrics = .{
    .{ .field = "name", .query = Device.getName },
    .{ .field = "power_mw", .query = Device.getPowerUsage },
    .{ .field = "power_limit_mw", .query = Device.getPowerLimit },
    .{ .field = "temperature", .query = Device.getTemperature },
    .{ .field = "fan_speed_percent", .query = Device.getFanSpeed },
    .{ .field = "gpu_util_percent", .query = Device.getGpuUtil },
    .{ .field = "mem_util_percent", .query = Device.getMemUtil },
    .{ .field = "encoder_util_percent", .query = Device.getEncoderUtil },
    .{ .field = "decoder_util_percent", .query = Device.getDecoderUtil },
    .{ .field = "clock_graphics_mhz", .query = Device.getClockGraphics },
    .{ .field = "clock_sm_mhz", .query = Device.getClockSm },
    .{ .field = "clock_mem_mhz", .query = Device.getClockMem },
    .{ .field = "clock_graphics_max_mhz", .query = Device.getMaxClockGraphics },
    .{ .field = "clock_mem_max_mhz", .query = Device.getMaxClockMem },
    .{ .field = "pstate", .query = Device.getPState },
    .{ .field = "mem_used_bytes", .query = Device.getMemUsed },
    .{ .field = "mem_free_bytes", .query = Device.getMemFree },
    .{ .field = "mem_total_bytes", .query = Device.getMemTotal },
    .{ .field = "pcie_tx_kbps", .query = Device.getPcieTx },
    .{ .field = "pcie_rx_kbps", .query = Device.getPcieRx },
    .{ .field = "pcie_link_gen", .query = Device.getPcieLinkGen },
    .{ .field = "pcie_link_width", .query = Device.getPcieLinkWidth },
};
