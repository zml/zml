const std = @import("std");
const c = @import("c");
const sysfs = @import("../../utils/sysfs.zig");
const device_info = @import("../../info/device_info.zig");
const DeviceInfo = device_info.DeviceInfo;
const NeuronInfo = device_info.NeuronInfo;
const DoubleBuffer = @import("../../utils/double_buffer.zig").DoubleBuffer;
const Collector = @import("../../collector.zig").Collector;
const Worker = @import("../../worker.zig").Worker;
const Nrt = @import("nrt.zig");
const process = @import("process.zig");

const base_path = "/sys/devices/virtual/neuron_device";

pub const target: device_info.Target = .neuron;

const State = struct {
    nrt: Nrt,
    handles: [c.MAX_NEURON_DEVICE_COUNT]*c.ndl_device_t = undefined,
    handle_count: usize = 0,
};

pub fn start(collector: *Collector) !void {
    const state = try collector.arena.create(State);
    state.* = .{ .nrt = try Nrt.init(collector.io) };

    var dev_index_buf: [c.MAX_NEURON_DEVICE_COUNT]c_int = undefined;
    const dev_indexes = state.nrt.availableDevices(&dev_index_buf);
    if (dev_indexes.len == 0) return;

    const total_nc = state.nrt.totalNcCount() catch return;
    const nc_per_device = total_nc / @as(u32, @intCast(dev_indexes.len));

    const GiB: u64 = 1024 * 1024 * 1024;
    var neuron_infos: std.ArrayList(*DeviceInfo) = .{};
    const dev_offset: u8 = @intCast(collector.device_infos.items.len);

    for (dev_indexes) |device_idx| {
        const dev_handle = state.nrt.openDevice(device_idx) catch continue;
        state.handles[state.handle_count] = dev_handle;
        state.handle_count += 1;
        const device_type = Nrt.deviceType(dev_handle);
        const mem_per_core: u64 = switch (device_type) {
            .inf1 => 2 * GiB, // inf1: 8 GiB / 4 cores
            .inf2_trn1 => 16 * GiB, // inf2/trn1: 32 GiB / 2 cores
            .trn2 => 12 * GiB, // trn2: 96 GiB / 8 cores
            .trn3 => 18 * GiB, // trn3: 144 GiB / 8 cores
            else => 0,
        };

        for (0..nc_per_device) |ci| {
            const dev: Device = .{
                .io = collector.io,
                .device_idx = @intCast(device_idx),
                .core_idx = @intCast(ci),
                .mem_per_core = mem_per_core,
            };

            const initial: NeuronInfo = .{ .name = dev.name(collector.arena) catch null };
            const info = try collector.addDevice(.{ .neuron = .{ .values = .{ initial, initial } } });
            try neuron_infos.append(collector.arena, info);

            try collector.worker.spawn(collector.io, pollDevice, .{ collector.io, collector.worker, &info.neuron, dev });
        }
    }

    if (neuron_infos.items.len > 0) {
        const processes = try collector.createProcessList();
        try process.init(collector.worker, collector.io, collector.gpa, processes, &state.nrt, state.handles[0..state.handle_count], nc_per_device, neuron_infos.items, dev_offset);
    }
}

const pollDevice = Worker.pollMetrics(*DoubleBuffer(NeuronInfo), Device, metrics);

const Device = struct {
    io: std.Io,
    device_idx: u32,
    core_idx: u32,
    mem_per_core: u64 = 0,

    fn devicePath(self: Device, buf: *[std.Io.Dir.max_path_bytes]u8, sub_path: []const u8) ![]const u8 {
        return std.fmt.bufPrint(buf, base_path ++ "/neuron{d}/{s}", .{ self.device_idx, sub_path });
    }

    fn corePath(self: Device, buf: *[std.Io.Dir.max_path_bytes]u8, sub_path: []const u8) ![]const u8 {
        return std.fmt.bufPrint(buf, base_path ++ "/neuron{d}/neuron_core{d}/{s}", .{ self.device_idx, self.core_idx, sub_path });
    }

    fn name(self: Device, arena: std.mem.Allocator) ![]const u8 {
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        var read_buf: [256]u8 = undefined;
        return try arena.dupe(u8, try sysfs.readString(self.io, try self.devicePath(&path_buf, "info/architecture/device_name"), &read_buf));
    }

    fn readCoreMem(self: Device, comptime subdir: []const u8) !u64 {
        var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        return sysfs.readInt(self.io, try self.corePath(&buf, "stats/memory_usage/device_mem/" ++ subdir ++ "/total"));
    }

    pub fn memTotal(self: Device) !u64 {
        if (self.mem_per_core == 0) return error.NrtUnavailable;
        return self.mem_per_core;
    }

    // Sysfs-based metrics
    pub fn memUsed(self: Device) !u64 {
        var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        return sysfs.readInt(self.io, try self.corePath(&buf, "stats/memory_usage/device_mem/total"));
    }

    pub fn tensors(self: Device) !u64 {
        return self.readCoreMem("tensors");
    }

    pub fn constants(self: Device) !u64 {
        return self.readCoreMem("constants");
    }

    pub fn modelCode(self: Device) !u64 {
        return self.readCoreMem("model_code");
    }

    pub fn sharedScratchpad(self: Device) !u64 {
        return self.readCoreMem("model_shared_scratchpad");
    }

    pub fn nonsharedScratchpad(self: Device) !u64 {
        return self.readCoreMem("nonshared_scratchpad");
    }

    pub fn runtime(self: Device) !u64 {
        return self.readCoreMem("runtime_memory");
    }

    pub fn driver(self: Device) !u64 {
        return self.readCoreMem("driver_memory");
    }

    pub fn dmaRings(self: Device) !u64 {
        return self.readCoreMem("dma_rings");
    }

    pub fn collectives(self: Device) !u64 {
        return self.readCoreMem("collectives");
    }

    pub fn notifications(self: Device) !u64 {
        return self.readCoreMem("notifications");
    }

    pub fn uncategorized(self: Device) !u64 {
        return self.readCoreMem("uncategorized");
    }
};

const metrics = .{
    .{ .field = "mem_total_bytes", .query = Device.memTotal },
    .{ .field = "mem_used_bytes", .query = Device.memUsed },
    .{ .field = "nc_tensors", .query = Device.tensors },
    .{ .field = "nc_constants", .query = Device.constants },
    .{ .field = "nc_model_code", .query = Device.modelCode },
    .{ .field = "nc_shared_scratchpad", .query = Device.sharedScratchpad },
    .{ .field = "nc_nonshared_scratchpad", .query = Device.nonsharedScratchpad },
    .{ .field = "nc_runtime", .query = Device.runtime },
    .{ .field = "nc_driver", .query = Device.driver },
    .{ .field = "nc_dma_rings", .query = Device.dmaRings },
    .{ .field = "nc_collectives", .query = Device.collectives },
    .{ .field = "nc_notifications", .query = Device.notifications },
    .{ .field = "nc_uncategorized", .query = Device.uncategorized },
};
