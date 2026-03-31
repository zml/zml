const std = @import("std");
const sysfs = @import("zml-smi/sysfs");
const DoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer;
const device_info = @import("zml-smi/info").device_info;
const DeviceInfo = device_info.DeviceInfo;
const NeuronInfo = device_info.NeuronInfo;
const Collector = @import("zml-smi/collector").Collector;
const poll_metrics = @import("zml-smi/info").poll_metrics;
const Nrt = @import("nrt.zig");
const process = @import("process.zig");

const base_path = "/sys/devices/virtual/neuron_device";

pub fn start(collector: *Collector) !void {
    const nrt = try collector.arena.create(Nrt);
    nrt.* = try Nrt.init(collector.arena, collector.io);

    if (nrt.handles.len == 0) {
        return;
    }

    const total_nc = nrt.totalNcCount() catch return;
    const nc_per_device = total_nc / @as(u32, @intCast(nrt.handles.len));

    const GiB: u64 = 1024 * 1024 * 1024;
    var neuron_infos: std.ArrayList(*DeviceInfo) = .{};
    const dev_offset: u8 = @intCast(collector.device_infos.items.len);

    for (nrt.handles, nrt.device_indexes) |dev_handle, device_idx| {
        const device_type = Nrt.deviceType(dev_handle);
        const mem_per_device: u64 = switch (device_type) {
            .inf1 => 8 * GiB,
            .inf2_trn1 => 32 * GiB,
            .trn2 => 96 * GiB,
            .trn3 => 144 * GiB,
            else => 0,
        };

        for (0..nc_per_device) |ci| {
            const poll_arena = try collector.createPollArena();

            const dev: Device = .{
                .io = collector.io,
                .arena = poll_arena,
                .device_idx = @intCast(device_idx),
                .core_idx = @intCast(ci),
                .mem_per_device = mem_per_device,
                .nc_per_device = nc_per_device,
            };

            const initial: NeuronInfo = .{ .name = dev.name(collector.arena) catch null };
            const info = try collector.addDevice(.{ .neuron = .{ .values = .{ initial, initial } } });
            try neuron_infos.append(collector.arena, info);
            try collector.spawnPoll(pollOnce, .{ poll_arena, &info.neuron, dev }, .{ .needs_warmup = true });
        }
    }

    if (neuron_infos.items.len > 0) {
        const processes = try collector.createProcessList();
        try process.init(collector, processes, nrt, nc_per_device, neuron_infos.items, dev_offset);
    }
}

const pollOnce = poll_metrics.poll(*DoubleBuffer(NeuronInfo), Device, metrics);

const Device = struct {
    io: std.Io,
    arena: *std.heap.ArenaAllocator,
    device_idx: u32,
    core_idx: u32,
    mem_per_device: u64 = 0,
    nc_per_device: u32 = 1,

    fn devicePath(self: Device, sub_path: []const u8) ![]const u8 {
        return std.fmt.allocPrint(self.arena.allocator(), base_path ++ "/neuron{d}/{s}", .{ self.device_idx, sub_path });
    }

    fn corePath(self: Device, sub_path: []const u8) ![]const u8 {
        return std.fmt.allocPrint(self.arena.allocator(), base_path ++ "/neuron{d}/neuron_core{d}/{s}", .{ self.device_idx, self.core_idx, sub_path });
    }

    fn name(self: Device, allocator: std.mem.Allocator) ![]const u8 {
        return sysfs.readString(allocator, self.io, try self.devicePath("info/architecture/device_name"));
    }

    fn readCoreMem(self: Device, comptime subdir: []const u8) !u64 {
        return sysfs.readInt(self.arena.allocator(), self.io, try self.corePath("stats/memory_usage/device_mem/" ++ subdir ++ "/total"));
    }

    pub fn memTotal(self: Device) !u64 {
        if (self.mem_per_device == 0) {
            return error.NrtUnavailable;
        }
        return self.mem_per_device;
    }

    // Sysfs-based metrics — RAM is shared across cores, so sum all cores on the device.
    pub fn memUsed(self: Device) !u64 {
        var total: u64 = 0;
        for (0..self.nc_per_device) |ci| {
            const path = try std.fmt.allocPrint(self.arena.allocator(), base_path ++ "/neuron{d}/neuron_core{d}/stats/memory_usage/device_mem/total", .{ self.device_idx, ci });
            total += try sysfs.readInt(self.arena.allocator(), self.io, path);
        }

        return total;
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
