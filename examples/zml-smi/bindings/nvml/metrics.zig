const std = @import("std");
const nvml = @import("nvml.zig");
const device_info = @import("../../info/device_info.zig");
const DeviceInfo = device_info.DeviceInfo;
const GpuInfo = device_info.GpuInfo;
const worker = @import("../../worker.zig");

pub fn init(io: std.Io, allocator: std.mem.Allocator, device_infos: *std.ArrayList(*DeviceInfo)) !void {
    try nvml.init();
    const count = try nvml.getDeviceCount();

    for (0..count) |i| {
        const dev = Device.open(@intCast(i)) catch continue;

        const info = try allocator.create(DeviceInfo);
        info.* = .{ .cuda = .{ .name = dev.getName() catch null } };
        try device_infos.append(allocator, info);

        inline for (metrics) |metric| {
            try worker.spawnWorker(io, &info.cuda, metric.field, metric.query, dev);
        }
    }
}

const Device = struct {
    handle: nvml.Handle,

    pub fn open(index: u32) !Device {
        return .{ .handle = try nvml.getHandleByIndex(index) };
    }

    fn getName(self: Device) ![256]u8 {
        var buf: [256]u8 = .{0} ** 256;
        _ = try nvml.getName(self.handle, &buf);
        return buf;
    }

    // Power
    pub fn getPowerUsage(self: Device) !u64 {
        return @intCast(try nvml.getPowerUsage(self.handle));
    }
    pub fn getPowerLimit(self: Device) !u64 {
        return @intCast(try nvml.getPowerLimit(self.handle));
    }
    pub fn getTotalEnergy(self: Device) !u64 {
        return nvml.getTotalEnergy(self.handle);
    }

    // Thermal
    pub fn getTemperature(self: Device) !u64 {
        return @intCast(try nvml.getTemperature(self.handle));
    }
    pub fn getFanSpeed(self: Device) !u64 {
        return @intCast(try nvml.getFanSpeed(self.handle));
    }

    // Utilization
    pub fn getGpuUtil(self: Device) !u64 {
        return @intCast(try nvml.getUtilizationGpu(self.handle));
    }
    pub fn getMemUtil(self: Device) !u64 {
        return @intCast(try nvml.getUtilizationMem(self.handle));
    }
    pub fn getEncoderUtil(self: Device) !u64 {
        return @intCast(try nvml.getEncoderUtil(self.handle));
    }
    pub fn getDecoderUtil(self: Device) !u64 {
        return @intCast(try nvml.getDecoderUtil(self.handle));
    }

    // Clocks
    pub fn getClockGraphics(self: Device) !u64 {
        return @intCast(try nvml.getClockGraphics(self.handle));
    }
    pub fn getClockSm(self: Device) !u64 {
        return @intCast(try nvml.getClockSm(self.handle));
    }
    pub fn getClockMem(self: Device) !u64 {
        return @intCast(try nvml.getClockMem(self.handle));
    }
    pub fn getMaxClockGraphics(self: Device) !u64 {
        return @intCast(try nvml.getMaxClockGraphics(self.handle));
    }
    pub fn getMaxClockMem(self: Device) !u64 {
        return @intCast(try nvml.getMaxClockMem(self.handle));
    }

    // Performance state
    pub fn getPState(self: Device) !u64 {
        return @intCast(try nvml.getPState(self.handle));
    }

    // Memory
    pub fn getMemUsed(self: Device) !u64 {
        return nvml.getMemUsed(self.handle);
    }
    pub fn getMemFree(self: Device) !u64 {
        return nvml.getMemFree(self.handle);
    }
    pub fn getMemTotal(self: Device) !u64 {
        return nvml.getMemTotal(self.handle);
    }
    pub fn getMemBusWidth(self: Device) !u64 {
        return @intCast(try nvml.getMemBusWidth(self.handle));
    }

    // PCIe
    pub fn getPcieTx(self: Device) !u64 {
        return @intCast(try nvml.getPcieTxKBps(self.handle));
    }
    pub fn getPcieRx(self: Device) !u64 {
        return @intCast(try nvml.getPcieRxKBps(self.handle));
    }
    pub fn getPcieLinkGen(self: Device) !u64 {
        return @intCast(try nvml.getPcieLinkGen(self.handle));
    }
    pub fn getPcieLinkWidth(self: Device) !u64 {
        return @intCast(try nvml.getPcieLinkWidth(self.handle));
    }
};

const metrics = .{
    .{ .field = "power_mw", .query = Device.getPowerUsage },
    .{ .field = "power_limit_mw", .query = Device.getPowerLimit },
    .{ .field = "total_energy_mj", .query = Device.getTotalEnergy },
    .{ .field = "temperature", .query = Device.getTemperature },
    .{ .field = "fan_speed_percent", .query = Device.getFanSpeed },
    .{ .field = "util_percent", .query = Device.getGpuUtil },
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
    .{ .field = "mem_bus_width", .query = Device.getMemBusWidth },
    .{ .field = "pcie_tx_kbps", .query = Device.getPcieTx },
    .{ .field = "pcie_rx_kbps", .query = Device.getPcieRx },
    .{ .field = "pcie_link_gen", .query = Device.getPcieLinkGen },
    .{ .field = "pcie_link_width", .query = Device.getPcieLinkWidth },
};
