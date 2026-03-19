const std = @import("std");
const Nvml = @import("nvml.zig");
const device_info = @import("../../info/device_info.zig");
const DeviceInfo = device_info.DeviceInfo;
const GpuInfo = device_info.GpuInfo;
const Collector = @import("../../collector.zig").Collector;
const Worker = @import("../../worker.zig").Worker;
const process = @import("process.zig");

pub const target: device_info.Target = .cuda;

pub fn start(collector: *Collector) !void {
    const nvml = try collector.arena.create(Nvml);
    nvml.* = try Nvml.init();
    const count = try nvml.getDeviceCount();

    for (0..count) |i| {
        const dev = Device.open(nvml, @intCast(i)) catch continue;
        const info = try collector.addDevice(.{ .cuda = .{ .name = dev.getName() catch null } });
        try collector.worker.spawn(collector.io, pollDevice, .{ collector.io, collector.worker, &info.cuda, dev });
    }

    const processes = try collector.createProcessList();
    try process.init(collector.worker, collector.io, collector.gpa, processes, nvml);
}

const pollDevice = Worker.pollMetrics(*GpuInfo, Device, metrics);

const Device = struct {
    nvml: *const Nvml,
    handle: Nvml.Handle,

    pub fn open(nvml: *const Nvml, index: u32) !Device {
        return .{ .nvml = nvml, .handle = try nvml.getHandleByIndex(index) };
    }

    fn getName(self: Device) ![256]u8 {
        var buf: [256]u8 = .{0} ** 256;
        _ = try self.nvml.getName(self.handle, &buf);
        return buf;
    }

    // Power
    pub fn getPowerUsage(self: Device) !u64 {
        return @intCast(try self.nvml.getPowerUsage(self.handle));
    }
    pub fn getPowerLimit(self: Device) !u64 {
        return @intCast(try self.nvml.getPowerLimit(self.handle));
    }
    pub fn getTotalEnergy(self: Device) !u64 {
        return self.nvml.getTotalEnergy(self.handle);
    }

    // Thermal
    pub fn getTemperature(self: Device) !u64 {
        return @intCast(try self.nvml.getTemperature(self.handle));
    }
    pub fn getFanSpeed(self: Device) !u64 {
        return @intCast(try self.nvml.getFanSpeed(self.handle));
    }

    // Utilization
    pub fn getGpuUtil(self: Device) !u64 {
        return @intCast(try self.nvml.getUtilizationGpu(self.handle));
    }
    pub fn getEncoderUtil(self: Device) !u64 {
        return @intCast(try self.nvml.getEncoderUtil(self.handle));
    }
    pub fn getDecoderUtil(self: Device) !u64 {
        return @intCast(try self.nvml.getDecoderUtil(self.handle));
    }

    // Clocks
    pub fn getClockGraphics(self: Device) !u64 {
        return @intCast(try self.nvml.getClockGraphics(self.handle));
    }
    pub fn getClockSm(self: Device) !u64 {
        return @intCast(try self.nvml.getClockSm(self.handle));
    }
    pub fn getClockMem(self: Device) !u64 {
        return @intCast(try self.nvml.getClockMem(self.handle));
    }
    pub fn getMaxClockGraphics(self: Device) !u64 {
        return @intCast(try self.nvml.getMaxClockGraphics(self.handle));
    }
    pub fn getMaxClockMem(self: Device) !u64 {
        return @intCast(try self.nvml.getMaxClockMem(self.handle));
    }

    // Memory
    pub fn getMemUsed(self: Device) !u64 {
        return self.nvml.getMemUsed(self.handle);
    }
    pub fn getMemTotal(self: Device) !u64 {
        return self.nvml.getMemTotal(self.handle);
    }
    pub fn getMemBusWidth(self: Device) !u64 {
        return @intCast(try self.nvml.getMemBusWidth(self.handle));
    }

    // PCIe
    pub fn getPcieTx(self: Device) !u64 {
        return @intCast(try self.nvml.getPcieTxKBps(self.handle));
    }
    pub fn getPcieRx(self: Device) !u64 {
        return @intCast(try self.nvml.getPcieRxKBps(self.handle));
    }
    pub fn getPcieLinkGen(self: Device) !u64 {
        return @intCast(try self.nvml.getPcieLinkGen(self.handle));
    }
    pub fn getPcieLinkWidth(self: Device) !u64 {
        return @intCast(try self.nvml.getPcieLinkWidth(self.handle));
    }
};

const metrics = .{
    .{ .field = "power_mw", .query = Device.getPowerUsage },
    .{ .field = "power_limit_mw", .query = Device.getPowerLimit },
    .{ .field = "total_energy_mj", .query = Device.getTotalEnergy },
    .{ .field = "temperature", .query = Device.getTemperature },
    .{ .field = "fan_speed_percent", .query = Device.getFanSpeed },
    .{ .field = "util_percent", .query = Device.getGpuUtil },
    .{ .field = "encoder_util_percent", .query = Device.getEncoderUtil },
    .{ .field = "decoder_util_percent", .query = Device.getDecoderUtil },
    .{ .field = "clock_graphics_mhz", .query = Device.getClockGraphics },
    .{ .field = "clock_sm_mhz", .query = Device.getClockSm },
    .{ .field = "clock_mem_mhz", .query = Device.getClockMem },
    .{ .field = "clock_graphics_max_mhz", .query = Device.getMaxClockGraphics },
    .{ .field = "clock_mem_max_mhz", .query = Device.getMaxClockMem },
    .{ .field = "mem_used_bytes", .query = Device.getMemUsed },
    .{ .field = "mem_total_bytes", .query = Device.getMemTotal },
    .{ .field = "mem_bus_width", .query = Device.getMemBusWidth },
    .{ .field = "pcie_tx_kbps", .query = Device.getPcieTx },
    .{ .field = "pcie_rx_kbps", .query = Device.getPcieRx },
    .{ .field = "pcie_link_gen", .query = Device.getPcieLinkGen },
    .{ .field = "pcie_link_width", .query = Device.getPcieLinkWidth },
};
