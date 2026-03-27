const std = @import("std");
const Nvml = @import("nvml.zig");
const device_info = @import("../../info/device_info.zig");
const DeviceInfo = device_info.DeviceInfo;
const GpuInfo = device_info.GpuInfo;
const DoubleBuffer = @import("../../utils/double_buffer.zig").DoubleBuffer;
const Collector = @import("../../collector.zig").Collector;
const Worker = @import("../../worker.zig").Worker;
const process = @import("process.zig");

pub const target: device_info.Target = .cuda;

pub fn start(collector: *Collector) !void {
    const nvml = try collector.arena.create(Nvml);
    nvml.* = try Nvml.init();
    const count = try nvml.deviceCount();
    const dev_offset: u8 = @intCast(collector.device_infos.items.len);

    for (0..count) |i| {
        const dev = Device.open(nvml, @intCast(i)) catch continue;
        const initial: GpuInfo = .{ .name = dev.name(collector.arena) catch null };
        const info = try collector.addDevice(.{ .cuda = .{ .values = .{ initial, initial } } });
        try collector.worker.spawn(collector.io, pollDevice, .{ collector.io, collector.worker, &info.cuda, dev });
    }

    const processes = try collector.createProcessList();
    try process.init(collector.worker, collector.io, collector.gpa, processes, nvml, dev_offset);
}

const pollDevice = Worker.pollMetrics(*DoubleBuffer(GpuInfo), Device, metrics);

const Device = struct {
    nvml: *const Nvml,
    handle: Nvml.Handle,

    pub fn open(nvml: *const Nvml, index: u32) !Device {
        return .{ .nvml = nvml, .handle = try nvml.handleByIndex(index) };
    }

    fn name(self: Device, arena: std.mem.Allocator) ![]const u8 {
        var buf: [256]u8 = .{0} ** 256;
        const slice = try self.nvml.name(self.handle, &buf);
        return try arena.dupe(u8, slice);
    }

    // Power
    pub fn powerUsage(self: Device) !u64 {
        return @intCast(try self.nvml.powerUsage(self.handle));
    }
    pub fn powerLimit(self: Device) !u64 {
        return @intCast(try self.nvml.powerLimit(self.handle));
    }
    pub fn totalEnergy(self: Device) !u64 {
        return self.nvml.totalEnergy(self.handle);
    }

    // Thermal
    pub fn temperature(self: Device) !u64 {
        return @intCast(try self.nvml.temperature(self.handle));
    }
    pub fn fanSpeed(self: Device) !u64 {
        return @intCast(try self.nvml.fanSpeed(self.handle));
    }

    // Utilization
    pub fn gpuUtil(self: Device) !u64 {
        return @intCast(try self.nvml.utilizationGpu(self.handle));
    }
    pub fn encoderUtil(self: Device) !u64 {
        return @intCast(try self.nvml.encoderUtil(self.handle));
    }
    pub fn decoderUtil(self: Device) !u64 {
        return @intCast(try self.nvml.decoderUtil(self.handle));
    }

    // Clocks
    pub fn clockGraphics(self: Device) !u64 {
        return @intCast(try self.nvml.clockGraphics(self.handle));
    }
    pub fn clockSm(self: Device) !u64 {
        return @intCast(try self.nvml.clockSm(self.handle));
    }
    pub fn clockMem(self: Device) !u64 {
        return @intCast(try self.nvml.clockMem(self.handle));
    }
    pub fn maxClockGraphics(self: Device) !u64 {
        return @intCast(try self.nvml.maxClockGraphics(self.handle));
    }
    pub fn maxClockMem(self: Device) !u64 {
        return @intCast(try self.nvml.maxClockMem(self.handle));
    }

    // Memory
    pub fn memUsed(self: Device) !u64 {
        return self.nvml.memUsed(self.handle);
    }
    pub fn memTotal(self: Device) !u64 {
        return self.nvml.memTotal(self.handle);
    }
    pub fn memBusWidth(self: Device) !u64 {
        return @intCast(try self.nvml.memBusWidth(self.handle));
    }

    // PCIe
    pub fn pcieTx(self: Device) !u64 {
        return @intCast(try self.nvml.pcieTxKBps(self.handle));
    }
    pub fn pcieRx(self: Device) !u64 {
        return @intCast(try self.nvml.pcieRxKBps(self.handle));
    }
    pub fn pcieLinkGen(self: Device) !u64 {
        return @intCast(try self.nvml.pcieLinkGen(self.handle));
    }
    pub fn pcieLinkWidth(self: Device) !u64 {
        return @intCast(try self.nvml.pcieLinkWidth(self.handle));
    }
};

const metrics = .{
    .{ .field = "power_mw", .query = Device.powerUsage },
    .{ .field = "power_limit_mw", .query = Device.powerLimit },
    .{ .field = "total_energy_mj", .query = Device.totalEnergy },
    .{ .field = "temperature", .query = Device.temperature },
    .{ .field = "fan_speed_percent", .query = Device.fanSpeed },
    .{ .field = "util_percent", .query = Device.gpuUtil },
    .{ .field = "encoder_util_percent", .query = Device.encoderUtil },
    .{ .field = "decoder_util_percent", .query = Device.decoderUtil },
    .{ .field = "clock_graphics_mhz", .query = Device.clockGraphics },
    .{ .field = "clock_sm_mhz", .query = Device.clockSm },
    .{ .field = "clock_mem_mhz", .query = Device.clockMem },
    .{ .field = "clock_graphics_max_mhz", .query = Device.maxClockGraphics },
    .{ .field = "clock_mem_max_mhz", .query = Device.maxClockMem },
    .{ .field = "mem_used_bytes", .query = Device.memUsed },
    .{ .field = "mem_total_bytes", .query = Device.memTotal },
    .{ .field = "mem_bus_width", .query = Device.memBusWidth },
    .{ .field = "pcie_tx_kbps", .query = Device.pcieTx },
    .{ .field = "pcie_rx_kbps", .query = Device.pcieRx },
    .{ .field = "pcie_link_gen", .query = Device.pcieLinkGen },
    .{ .field = "pcie_link_width", .query = Device.pcieLinkWidth },
};
