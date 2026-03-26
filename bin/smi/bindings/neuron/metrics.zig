const std = @import("std");
const c = @import("c");
const sysfs = @import("../../utils/sysfs.zig");
const device_info = @import("../../info/device_info.zig");
const DeviceInfo = device_info.DeviceInfo;
const NeuronInfo = device_info.NeuronInfo;
const ShadowValue = @import("../../utils/shadow_value.zig").ShadowValue;
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

    const total_nc = state.nrt.getTotalNcCount() catch return;
    const nc_per_device = total_nc / @as(u32, @intCast(dev_indexes.len));

    const GiB: u64 = 1024 * 1024 * 1024;
    var neuron_infos: std.ArrayList(*DeviceInfo) = .{};
    const dev_offset: u8 = @intCast(collector.device_infos.items.len);

    for (dev_indexes) |device_idx| {
        const dev_handle = state.nrt.openDevice(device_idx) catch continue;
        state.handles[state.handle_count] = dev_handle;
        state.handle_count += 1;
        const device_type = Nrt.getDeviceType(dev_handle);
        const mem_per_core: u64 = switch (device_type) {
            1 => 2 * GiB, // inf1: 8 GiB / 4 cores
            2 => 16 * GiB, // inf2/trn1: 32 GiB / 2 cores
            3 => 12 * GiB, // trn2: 96 GiB / 8 cores
            4 => 18 * GiB, // trn3: 144 GiB / 8 cores
            else => 0,
        };

        for (0..nc_per_device) |ci| {
            const info = try collector.addDevice(.{ .neuron = .{ .value = .{} } });
            try neuron_infos.append(collector.arena, info);

            const dev: Device = .{
                .io = collector.io,
                .device_idx = @intCast(device_idx),
                .core_idx = @intCast(ci),
                .mem_per_core = mem_per_core,
            };

            info.neuron.value.name = dev.getName() catch null;

            try collector.worker.spawn(collector.io, pollDevice, .{ collector.io, collector.worker, &info.neuron, dev });
        }
    }

    if (neuron_infos.items.len > 0) {
        const processes = try collector.createProcessList();
        try process.init(collector.worker, collector.io, collector.gpa, processes, &state.nrt, state.handles[0..state.handle_count], nc_per_device, neuron_infos.items, dev_offset);
    }
}

const pollDevice = Worker.pollMetrics(*ShadowValue(NeuronInfo), Device, metrics);

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

    fn getName(self: Device) ![256]u8 {
        var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        return sysfs.readString(self.io, try self.devicePath(&buf, "info/architecture/device_name"));
    }

    fn readCoreMem(self: Device, comptime subdir: []const u8) !u64 {
        var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        return sysfs.readInt(self.io, try self.corePath(&buf, "stats/memory_usage/device_mem/" ++ subdir ++ "/total"));
    }

    pub fn getMemTotal(self: Device) !u64 {
        if (self.mem_per_core == 0) return error.NrtUnavailable;
        return self.mem_per_core;
    }

    // Sysfs-based metrics
    pub fn getMemUsed(self: Device) !u64 {
        var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        return sysfs.readInt(self.io, try self.corePath(&buf, "stats/memory_usage/device_mem/total"));
    }

    pub fn getTensors(self: Device) !u64 {
        return self.readCoreMem("tensors");
    }

    pub fn getConstants(self: Device) !u64 {
        return self.readCoreMem("constants");
    }

    pub fn getModelCode(self: Device) !u64 {
        return self.readCoreMem("model_code");
    }

    pub fn getSharedScratchpad(self: Device) !u64 {
        return self.readCoreMem("model_shared_scratchpad");
    }

    pub fn getNonsharedScratchpad(self: Device) !u64 {
        return self.readCoreMem("nonshared_scratchpad");
    }

    pub fn getRuntime(self: Device) !u64 {
        return self.readCoreMem("runtime_memory");
    }

    pub fn getDriver(self: Device) !u64 {
        return self.readCoreMem("driver_memory");
    }

    pub fn getDmaRings(self: Device) !u64 {
        return self.readCoreMem("dma_rings");
    }

    pub fn getCollectives(self: Device) !u64 {
        return self.readCoreMem("collectives");
    }

    pub fn getNotifications(self: Device) !u64 {
        return self.readCoreMem("notifications");
    }

    pub fn getUncategorized(self: Device) !u64 {
        return self.readCoreMem("uncategorized");
    }
};

const metrics = .{
    .{ .field = "mem_total_bytes", .query = Device.getMemTotal },
    .{ .field = "mem_used_bytes", .query = Device.getMemUsed },
    .{ .field = "nc_tensors", .query = Device.getTensors },
    .{ .field = "nc_constants", .query = Device.getConstants },
    .{ .field = "nc_model_code", .query = Device.getModelCode },
    .{ .field = "nc_shared_scratchpad", .query = Device.getSharedScratchpad },
    .{ .field = "nc_nonshared_scratchpad", .query = Device.getNonsharedScratchpad },
    .{ .field = "nc_runtime", .query = Device.getRuntime },
    .{ .field = "nc_driver", .query = Device.getDriver },
    .{ .field = "nc_dma_rings", .query = Device.getDmaRings },
    .{ .field = "nc_collectives", .query = Device.getCollectives },
    .{ .field = "nc_notifications", .query = Device.getNotifications },
    .{ .field = "nc_uncategorized", .query = Device.getUncategorized },
};
