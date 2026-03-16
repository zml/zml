const std = @import("std");
const AmdSmi = @import("amdsmi.zig");
const device_info = @import("../../info/device_info.zig");
const DeviceInfo = device_info.DeviceInfo;
const GpuInfo = device_info.GpuInfo;
const Worker = @import("../../worker.zig").Worker;
const pi = @import("../../info/process_info.zig");
const ProcessShadowList = @import("../../shadow_list.zig").ShadowList(pi.ProcessInfo);
const process = @import("process.zig");

pub const Backend = struct {
    amdsmi: AmdSmi = undefined,
    processes: ProcessShadowList = .init(),

    pub fn start(self: *Backend, w: *Worker, io: std.Io, allocator: std.mem.Allocator, device_infos: *std.ArrayList(*DeviceInfo), proc_allocator: std.mem.Allocator) !void {
        self.amdsmi = try AmdSmi.init(allocator);
        const count = self.amdsmi.getDeviceCount();

        for (0..count) |i| {
            const dev = Device.open(&self.amdsmi, @intCast(i)) catch continue;

            const info = try allocator.create(DeviceInfo);
            info.* = .{ .rocm = .{ .name = dev.getName() catch null } };
            try device_infos.append(allocator, info);

            inline for (metrics) |metric| {
                try w.spawnWorker(io, &info.rocm, metric.field, metric.query, dev);
            }
        }

        try process.init(w, io, proc_allocator, &self.processes, &self.amdsmi);
    }

    pub fn deinit(self: *Backend, proc_allocator: std.mem.Allocator) void {
        self.processes.deinit(proc_allocator);
    }
};

const Device = struct {
    amdsmi: *const AmdSmi,
    handle: AmdSmi.Handle,

    pub fn open(amdsmi: *const AmdSmi, index: u32) !Device {
        return .{ .amdsmi = amdsmi, .handle = try amdsmi.getHandleByIndex(index) };
    }

    fn getName(self: Device) ![256]u8 {
        var buf: [256]u8 = .{0} ** 256;
        _ = try self.amdsmi.getName(self.handle, buf[0..AmdSmi.name_buf_len]);
        return buf;
    }

    // Power
    pub fn getPowerUsage(self: Device) !u64 {
        const pw = try self.amdsmi.getPowerUsage(self.handle);
        return @as(u64, pw) * 1000;
    }
    pub fn getPowerLimit(self: Device) !u64 {
        // Header says W, but empirically power_limit is in µW; convert to mW
        const limit = try self.amdsmi.getPowerLimit(self.handle);
        if (limit == std.math.maxInt(u32)) return error.not_supported;
        return @as(u64, limit) / 1000;
    }

    // Thermal
    pub fn getTemperature(self: Device) !u64 {
        const temp = try self.amdsmi.getTemperature(self.handle);
        return @intCast(temp);
    }
    pub fn getFanSpeed(self: Device) !u64 {
        const speed = try self.amdsmi.getFanSpeed(self.handle);
        return @intCast(@divTrunc(speed * 100, 255));
    }

    // Utilization
    pub fn getGpuUtil(self: Device) !u64 {
        return try self.amdsmi.getGpuUtil(self.handle);
    }

    // Clocks
    pub fn getClockGraphics(self: Device) !u64 {
        return try self.amdsmi.getClockGraphics(self.handle);
    }
    pub fn getClockSoc(self: Device) !u64 {
        return try self.amdsmi.getClockSoc(self.handle);
    }
    pub fn getClockMem(self: Device) !u64 {
        return try self.amdsmi.getClockMem(self.handle);
    }
    pub fn getMaxClockGraphics(self: Device) !u64 {
        return try self.amdsmi.getMaxClockGraphics(self.handle);
    }
    pub fn getMaxClockMem(self: Device) !u64 {
        return try self.amdsmi.getMaxClockMem(self.handle);
    }

    // Memory
    pub fn getMemUsed(self: Device) !u64 {
        return self.amdsmi.getMemUsed(self.handle);
    }
    pub fn getMemTotal(self: Device) !u64 {
        return self.amdsmi.getMemTotal(self.handle);
    }

    // PCIe
    pub fn getPcieBandwidth(self: Device) !u64 {
        const bw = try self.amdsmi.getPcieBandwidth(self.handle);
        if (bw == std.math.maxInt(u32)) return error.not_supported;
        return bw;
    }
    pub fn getPcieLinkGen(self: Device) !u64 {
        return try self.amdsmi.getPcieLinkGen(self.handle);
    }
    pub fn getPcieLinkWidth(self: Device) !u64 {
        return try self.amdsmi.getPcieWidth(self.handle);
    }
};

const metrics = .{
    .{ .field = "power_mw", .query = Device.getPowerUsage },
    .{ .field = "power_limit_mw", .query = Device.getPowerLimit },
    .{ .field = "temperature", .query = Device.getTemperature },
    .{ .field = "fan_speed_percent", .query = Device.getFanSpeed },
    .{ .field = "util_percent", .query = Device.getGpuUtil },
    .{ .field = "clock_graphics_mhz", .query = Device.getClockGraphics },
    .{ .field = "clock_soc_mhz", .query = Device.getClockSoc },
    .{ .field = "clock_mem_mhz", .query = Device.getClockMem },
    .{ .field = "clock_graphics_max_mhz", .query = Device.getMaxClockGraphics },
    .{ .field = "clock_mem_max_mhz", .query = Device.getMaxClockMem },
    .{ .field = "mem_used_bytes", .query = Device.getMemUsed },
    .{ .field = "mem_total_bytes", .query = Device.getMemTotal },
    .{ .field = "pcie_bandwidth_mbps", .query = Device.getPcieBandwidth },
    .{ .field = "pcie_link_gen", .query = Device.getPcieLinkGen },
    .{ .field = "pcie_link_width", .query = Device.getPcieLinkWidth },
};
