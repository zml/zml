const std = @import("std");
const amdsmi = @import("amdsmi.zig");
const device_info = @import("../../info/device_info.zig");
const DeviceInfo = device_info.DeviceInfo;
const GpuInfo = device_info.GpuInfo;
const Worker = @import("../../worker.zig").Worker;
const pi = @import("../../info/process_info.zig");
const process = @import("process.zig");

pub const Backend = struct {
    processes: std.ArrayList(pi.ProcessInfo) = .{},

    pub fn start(self: *Backend, w: *Worker, io: std.Io, allocator: std.mem.Allocator, device_infos: *std.ArrayList(*DeviceInfo), proc_allocator: std.mem.Allocator) !void {
        try amdsmi.init();
        const count = try amdsmi.getDeviceCount();

        for (0..count) |i| {
            const dev = Device.open(@intCast(i)) catch continue;

            const info = try allocator.create(DeviceInfo);
            info.* = .{ .rocm = .{ .name = dev.getName() catch null } };
            try device_infos.append(allocator, info);

            inline for (metrics) |metric| {
                try w.spawnWorker(io, &info.rocm, metric.field, metric.query, dev);
            }
        }

        try process.init(w, io, proc_allocator, &self.processes);
    }

    pub fn deinit(self: *Backend, proc_allocator: std.mem.Allocator) void {
        self.processes.deinit(proc_allocator);
    }
};

const Device = struct {
    handle: amdsmi.Handle,

    pub fn open(index: u32) !Device {
        return .{ .handle = try amdsmi.getHandleByIndex(index) };
    }

    fn getName(self: Device) ![256]u8 {
        var buf: [256]u8 = .{0} ** 256;
        _ = try amdsmi.getName(self.handle, &buf);
        return buf;
    }

    // Power
    pub fn getPowerUsage(self: Device) !u64 {
        const pw = try amdsmi.getPowerUsage(self.handle);
        return @as(u64, pw) * 1000;
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

    // Memory
    pub fn getMemUsed(self: Device) !u64 {
        return amdsmi.getMemUsed(self.handle);
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
    .{ .field = "power_mw", .query = Device.getPowerUsage },
    .{ .field = "power_limit_mw", .query = Device.getPowerLimit },
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
    .{ .field = "pcie_tx_kbps", .query = Device.getPcieTx },
    .{ .field = "pcie_rx_kbps", .query = Device.getPcieRx },
    .{ .field = "pcie_link_gen", .query = Device.getPcieLinkGen },
    .{ .field = "pcie_link_width", .query = Device.getPcieLinkWidth },
};
