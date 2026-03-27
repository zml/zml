const std = @import("std");
const AmdSmi = @import("amdsmi.zig");
const device_info = @import("../../info/device_info.zig");
const DeviceInfo = device_info.DeviceInfo;
const GpuInfo = device_info.GpuInfo;
const DoubleBuffer = @import("../../utils/double_buffer.zig").DoubleBuffer;
const Collector = @import("../../collector.zig").Collector;
const Worker = @import("../../worker.zig").Worker;
const process = @import("process.zig");

pub const target: device_info.Target = .rocm;

pub fn start(collector: *Collector) !void {
    const amdsmi = try collector.arena.create(AmdSmi);
    amdsmi.* = try AmdSmi.init(collector.arena);
    const count = amdsmi.deviceCount();
    const dev_offset: u8 = @intCast(collector.device_infos.items.len);

    for (0..count) |i| {
        const dev = Device.open(amdsmi, @intCast(i)) catch continue;
        const initial: GpuInfo = .{ .name = dev.name(collector.arena) catch null };
        const info = try collector.addDevice(.{ .rocm = .{ .values = .{ initial, initial } } });
        try collector.worker.spawn(collector.io, pollDevice, .{ collector.io, collector.worker, &info.rocm, dev });
    }

    const processes = try collector.createProcessList();
    try process.init(collector.worker, collector.io, collector.gpa, processes, amdsmi, dev_offset);
}

const pollDevice = Worker.pollMetrics(*DoubleBuffer(GpuInfo), Device, metrics);

const Device = struct {
    amdsmi: *const AmdSmi,
    handle: AmdSmi.Handle,

    pub fn open(amdsmi: *const AmdSmi, index: u32) !Device {
        return .{ .amdsmi = amdsmi, .handle = try amdsmi.handleByIndex(index) };
    }

    fn name(self: Device, arena: std.mem.Allocator) ![]const u8 {
        var buf: [AmdSmi.name_buf_len]u8 = .{0} ** AmdSmi.name_buf_len;
        const slice = try self.amdsmi.name(self.handle, &buf);
        return try arena.dupe(u8, slice);
    }

    // Power
    pub fn powerUsage(self: Device) !u64 {
        const pw = try self.amdsmi.powerUsage(self.handle);
        return @as(u64, pw) * 1000;
    }
    pub fn powerLimit(self: Device) !u64 {
        // Header says W, but empirically power_limit is in µW; convert to mW
        const limit = try self.amdsmi.powerLimit(self.handle);
        if (limit == std.math.maxInt(u32)) {
            return error.not_supported;
        }
        return @as(u64, limit) / 1000;
    }

    // Thermal
    pub fn temperature(self: Device) !u64 {
        const temp = try self.amdsmi.temperature(self.handle);
        return @intCast(temp);
    }
    pub fn fanSpeed(self: Device) !u64 {
        const speed = try self.amdsmi.fanSpeed(self.handle);
        return @intCast(@divTrunc(speed * 100, 255));
    }

    // Utilization
    pub fn gpuUtil(self: Device) !u64 {
        return try self.amdsmi.gpuUtil(self.handle);
    }

    // Clocks
    pub fn clockGraphics(self: Device) !u64 {
        return try self.amdsmi.clockGraphics(self.handle);
    }
    pub fn clockSoc(self: Device) !u64 {
        return try self.amdsmi.clockSoc(self.handle);
    }
    pub fn clockMem(self: Device) !u64 {
        return try self.amdsmi.clockMem(self.handle);
    }
    pub fn maxClockGraphics(self: Device) !u64 {
        return try self.amdsmi.maxClockGraphics(self.handle);
    }
    pub fn maxClockMem(self: Device) !u64 {
        return try self.amdsmi.maxClockMem(self.handle);
    }

    // Memory
    pub fn memUsed(self: Device) !u64 {
        return self.amdsmi.memUsed(self.handle);
    }
    pub fn memTotal(self: Device) !u64 {
        return self.amdsmi.memTotal(self.handle);
    }

    // PCIe
    pub fn pcieBandwidth(self: Device) !u64 {
        const bw = try self.amdsmi.pcieBandwidth(self.handle);
        if (bw == std.math.maxInt(u32)) {
            return error.not_supported;
        }
        return bw;
    }
    pub fn pcieLinkGen(self: Device) !u64 {
        return try self.amdsmi.pcieLinkGen(self.handle);
    }
    pub fn pcieLinkWidth(self: Device) !u64 {
        return try self.amdsmi.pcieWidth(self.handle);
    }
};

const metrics = .{
    .{ .field = "power_mw", .query = Device.powerUsage },
    .{ .field = "power_limit_mw", .query = Device.powerLimit },
    .{ .field = "temperature", .query = Device.temperature },
    .{ .field = "fan_speed_percent", .query = Device.fanSpeed },
    .{ .field = "util_percent", .query = Device.gpuUtil },
    .{ .field = "clock_graphics_mhz", .query = Device.clockGraphics },
    .{ .field = "clock_soc_mhz", .query = Device.clockSoc },
    .{ .field = "clock_mem_mhz", .query = Device.clockMem },
    .{ .field = "clock_graphics_max_mhz", .query = Device.maxClockGraphics },
    .{ .field = "clock_mem_max_mhz", .query = Device.maxClockMem },
    .{ .field = "mem_used_bytes", .query = Device.memUsed },
    .{ .field = "mem_total_bytes", .query = Device.memTotal },
    .{ .field = "pcie_bandwidth_mbps", .query = Device.pcieBandwidth },
    .{ .field = "pcie_link_gen", .query = Device.pcieLinkGen },
    .{ .field = "pcie_link_width", .query = Device.pcieLinkWidth },
};
