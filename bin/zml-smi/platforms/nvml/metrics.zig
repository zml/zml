const std = @import("std");
const Nvml = @import("nvml.zig");
const device_info = @import("zml-smi/info").device_info;
const GpuInfo = device_info.GpuInfo;
const poll_metrics = @import("zml-smi/info").poll_metrics;
const DoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer;
const Collector = @import("zml-smi/collector").Collector;
const process = @import("process.zig");
const sysfs = @import("zml-smi/sysfs");

const pcie_every: u8 = 4;

const Slot = struct {
    info: *DoubleBuffer(GpuInfo),
    dev: Device,
};

const Ctx = struct {
    gpa: std.mem.Allocator,
    nvml: *const Nvml,
    slots: []Slot,
    processes: *process.ProcessDoubleBuffer,
    last_seen_ts: []u64,
    /// Backs the /proc/meminfo reads of unified-memory GPUs; null when there are none.
    poll_arena: ?*std.heap.ArenaAllocator,
    dev_offset: u16,
    device_count: u32,
    tick: u8 = 0,
};

pub fn start(collector: *Collector) !void {
    const nvml = try collector.arena.create(Nvml);
    nvml.* = try Nvml.init();

    const count = try nvml.deviceCount();
    const dev_offset: u16 = @intCast(collector.device_infos.items.len);

    var poll_arena: ?*std.heap.ArenaAllocator = null;
    var slots: std.ArrayList(Slot) = try .initCapacity(collector.arena, count);
    for (0..count) |i| {
        var dev = Device.open(nvml, @intCast(i)) catch continue;
        // Unified-memory GPUs (e.g. GB10 / DGX Spark) have no dedicated memory and NVML
        // reports memory info as not supported: report the system memory they share.
        if (nvml.memTotal(dev.handle)) |_| {} else |err| if (err == error.error_not_supported) {
            if (poll_arena == null) poll_arena = try collector.createPollArena();
            dev.system_memory = .{ .io = collector.io, .arena = poll_arena.? };
        }
        const initial: GpuInfo = .{
            .name = dev.name(collector.arena) catch null,
            .driver_version = dev.driverVersion(collector.arena) catch null,
            .cuda_driver_version = dev.cudaDriverVersion(collector.arena) catch null,
        };
        const info = try collector.addDevice(.{ .cuda = .{ .values = .{ initial, initial } } });
        slots.appendAssumeCapacity(.{ .info = &info.cuda, .dev = dev });
    }

    const last_seen_ts = try collector.arena.alloc(u64, count);
    @memset(last_seen_ts, 0);

    const ctx = try collector.arena.create(Ctx);
    ctx.* = .{
        .gpa = collector.gpa,
        .nvml = nvml,
        .slots = slots.items,
        .processes = try collector.createProcessList(),
        .last_seen_ts = last_seen_ts,
        .poll_arena = poll_arena,
        .dev_offset = dev_offset,
        .device_count = count,
    };
    try collector.spawnPoll(pollOnce, .{ctx});
}

fn pollOnce(ctx: *Ctx) void {
    const pcie = ctx.tick % pcie_every == 0;
    ctx.tick +%= 1;
    if (ctx.poll_arena) |a| _ = a.reset(.retain_capacity);

    for (ctx.slots) |slot| {
        pollDevice(slot.info, slot.dev, pcie);
    }
    process.pollOnce(ctx.gpa, ctx.processes, ctx.nvml, ctx.dev_offset, ctx.device_count, ctx.last_seen_ts);
}

fn pollDevice(db: *DoubleBuffer(GpuInfo), dev: Device, pcie: bool) void {
    const back = db.back();
    back.* = db.front().*;
    poll_metrics.apply(metrics, back, dev);
    if (pcie) poll_metrics.apply(pcie_metrics, back, dev);
    db.swap();
}

const Device = struct {
    nvml: *const Nvml,
    handle: Nvml.Handle,
    /// set when the GPU uses system memory (NVML memory info not supported)
    system_memory: ?SystemMemory = null,

    pub fn open(nvml: *const Nvml, index: u32) !Device {
        return .{ .nvml = nvml, .handle = try nvml.handleByIndex(index) };
    }

    fn name(self: Device, arena: std.mem.Allocator) ![]const u8 {
        var buf: [Nvml.name_buf_len]u8 = undefined;
        const slice = try self.nvml.name(self.handle, &buf);

        return try arena.dupe(u8, slice);
    }

    fn driverVersion(self: Device, arena: std.mem.Allocator) ![]const u8 {
        var buf: [Nvml.driver_version_buf_size]u8 = undefined;
        const slice = try self.nvml.driverVersion(&buf);

        return try arena.dupe(u8, slice);
    }

    fn cudaDriverVersion(self: Device, arena: std.mem.Allocator) ![]const u8 {
        const v = try self.nvml.cudaDriverVersion();
        return try std.fmt.allocPrint(arena, "{}.{}", .{
            Nvml.cudaDriverVersionMajor(v),
            Nvml.cudaDriverVersionMinor(v),
        });
    }

    // Power
    pub fn powerUsage(self: Device) !u64 {
        return @intCast(try self.nvml.powerUsage(self.handle));
    }

    pub fn powerLimit(self: Device) !u64 {
        return @intCast(try self.nvml.powerLimit(self.handle));
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
        if (self.system_memory) |sm| {
            const total = try sm.memTotal();
            return total - @min(total, try sm.memAvailable());
        }
        return self.nvml.memUsed(self.handle);
    }

    pub fn memTotal(self: Device) !u64 {
        if (self.system_memory) |sm| return sm.memTotal();
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

    pub fn pcieBandwidth(self: Device) !u64 {
        return @intCast(try self.nvml.pcieSpeed(self.handle));
    }

    pub fn pcieLinkGen(self: Device) !u64 {
        return @intCast(try self.nvml.pcieLinkGen(self.handle));
    }

    pub fn pcieLinkWidth(self: Device) !u64 {
        return @intCast(try self.nvml.pcieLinkWidth(self.handle));
    }
};

/// System memory as seen by a unified-memory GPU, from /proc/meminfo
/// (used = MemTotal - MemAvailable, as NVIDIA recommends for DGX Spark).
const SystemMemory = struct {
    io: std.Io,
    arena: *std.heap.ArenaAllocator,

    fn memTotal(self: SystemMemory) !u64 {
        return 1024 * try sysfs.readFieldInt(self.arena.allocator(), self.io, "/proc/meminfo", "MemTotal");
    }

    fn memAvailable(self: SystemMemory) !u64 {
        return 1024 * try sysfs.readFieldInt(self.arena.allocator(), self.io, "/proc/meminfo", "MemAvailable");
    }
};

const metrics = .{
    .{ .field = "power_mw", .query = Device.powerUsage },
    .{ .field = "power_limit_mw", .query = Device.powerLimit },
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
    .{ .field = "pcie_bandwidth_mbps", .query = Device.pcieBandwidth },
    .{ .field = "pcie_link_gen", .query = Device.pcieLinkGen },
    .{ .field = "pcie_link_width", .query = Device.pcieLinkWidth },
};

const pcie_metrics = .{
    .{ .field = "pcie_tx_kbps", .query = Device.pcieTx },
    .{ .field = "pcie_rx_kbps", .query = Device.pcieRx },
};
