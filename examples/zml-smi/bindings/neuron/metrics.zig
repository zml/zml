const std = @import("std");
const sysfs = @import("../../sysfs.zig");
const DeviceInfo = @import("../../device_info.zig").DeviceInfo;
const worker = @import("../../worker.zig");
const schema = @import("schema.zig");

const base_path = "/sys/devices/virtual/neuron_device";

const monitor_config =
    \\{"period":"1s","neuron_runtimes":[{"tag_filter":".*","metrics":[{"type":"neuroncore_counters"}]}]}
;

pub fn init(io: std.Io, allocator: std.mem.Allocator, device_infos: *std.ArrayList(DeviceInfo), signal: *worker.Signal) !void {
    var device_idx: u32 = 0;
    while (device_idx < 64) : (device_idx += 1) {
        var dev_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const dev_path = std.fmt.bufPrint(&dev_buf, base_path ++ "/neuron{d}/info/architecture/device_name", .{device_idx}) catch break;
        _ = sysfs.readString(io, dev_path) catch break;

        var core_idx: u32 = 0;
        while (core_idx < 64) : (core_idx += 1) {
            var core_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
            const core_path = std.fmt.bufPrint(&core_buf, base_path ++ "/neuron{d}/neuron_core{d}/stats/memory_usage/device_mem/tensors/total", .{ device_idx, core_idx }) catch break;
            _ = sysfs.readInt(io, core_path) catch break;

            const info = try device_infos.addOne(allocator);
            info.* = .{ .gpu_util_percent = 0 };

            const dev = Device{
                .io = io,
                .device_idx = device_idx,
                .core_idx = @intCast(core_idx),
            };

            info.name = dev.getName() catch null;

            inline for (metrics) |metric| {
                try worker.spawnWorker(io, info, metric.field, metric.query, dev, signal);
            }
        }
    }

    if (device_infos.items.len > 0) {
        startMonitor(io, allocator, device_infos, signal) catch {};
    }
}

const Device = struct {
    io: std.Io,
    device_idx: u32,
    core_idx: u32,

    fn devicePath(self: Device, buf: *[std.Io.Dir.max_path_bytes]u8, sub_path: []const u8) ![]const u8 {
        return std.fmt.bufPrint(buf, base_path ++ "/neuron{d}/{s}", .{ self.device_idx, sub_path });
    }

    fn corePath(self: Device, buf: *[std.Io.Dir.max_path_bytes]u8, sub_path: []const u8) ![]const u8 {
        return std.fmt.bufPrint(buf, base_path ++ "/neuron{d}/neuron_core{d}/{s}", .{ self.device_idx, self.core_idx, sub_path });
    }

    pub fn getName(self: Device) ![256]u8 {
        var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        return sysfs.readString(self.io, try self.devicePath(&buf, "info/architecture/device_name"));
    }

    pub fn getMemUsed(self: Device) !u64 {
        var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        return sysfs.readInt(self.io, try self.corePath(&buf, "stats/memory_usage/device_mem/total"));
    }
};

fn memQuery(comptime sysfs_dir: []const u8) type {
    return struct {
        pub fn get(dev: Device) !u64 {
            var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
            return sysfs.readInt(dev.io, try dev.corePath(&buf, "stats/memory_usage/device_mem/" ++ sysfs_dir ++ "/total"));
        }
    };
}

const metrics = .{
    .{ .field = "mem_used_bytes", .query = Device.getMemUsed },
    .{ .field = "nc_tensors", .query = memQuery("tensors").get },
    .{ .field = "nc_constants", .query = memQuery("constants").get },
    .{ .field = "nc_model_code", .query = memQuery("model_code").get },
    .{ .field = "nc_shared_scratchpad", .query = memQuery("model_shared_scratchpad").get },
    .{ .field = "nc_nonshared_scratchpad", .query = memQuery("nonshared_scratchpad").get },
    .{ .field = "nc_runtime", .query = memQuery("runtime_memory").get },
    .{ .field = "nc_driver", .query = memQuery("driver_memory").get },
    .{ .field = "nc_dma_rings", .query = memQuery("dma_rings").get },
    .{ .field = "nc_collectives", .query = memQuery("collectives").get },
    .{ .field = "nc_notifications", .query = memQuery("notifications").get },
    .{ .field = "nc_uncategorized", .query = memQuery("uncategorized").get },
};

// --- neuron-monitor subprocess for utilization + device memory capacity ---

fn startMonitor(io: std.Io, allocator: std.mem.Allocator, device_infos: *std.ArrayList(DeviceInfo), signal: *worker.Signal) !void {
    const config_path = "/tmp/zml-smi-neuron-monitor.conf";
    var config_file = try std.Io.Dir.createFileAbsolute(io, config_path, .{});
    try config_file.writeStreamingAll(io, monitor_config);
    config_file.close(io);

    var child = try std.process.spawn(io, .{
        .argv = &.{ "neuron-monitor", "-c", config_path },
        .stdin = .ignore,
        .stdout = .pipe,
        .stderr = .ignore,
    });

    const stdout_file = child.stdout orelse return error.NoPipe;
    _ = try io.concurrent(monitorLoop, .{ io, allocator, device_infos, signal, stdout_file });
}

fn monitorLoop(io: std.Io, allocator: std.mem.Allocator, device_infos: *std.ArrayList(DeviceInfo), signal: *worker.Signal, stdout_file: std.Io.File) void {
    var read_buf: [4096]u8 = undefined;
    var file_reader = stdout_file.reader(io, &read_buf);
    var line_buf: std.Io.Writer.Allocating = std.Io.Writer.Allocating.initCapacity(allocator, 4096) catch return;

    while (true) {
        _ = file_reader.interface.streamDelimiterEnding(&line_buf.writer, '\n') catch return;
        file_reader.interface.toss(1);
        updateFromMonitor(allocator, device_infos.items, line_buf.written());
        signal.notify(io);
        line_buf.clearRetainingCapacity();
    }
}

fn updateFromMonitor(allocator: std.mem.Allocator, infos: []DeviceInfo, json_data: []const u8) void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const parsed = std.json.parseFromSlice(
        schema.MonitorReport,
        arena.allocator(),
        json_data,
        .{ .ignore_unknown_fields = true },
    ) catch return;

    const report = parsed.value;

    const hw = report.neuron_hardware_info;
    if (hw.neuron_device_memory_size > 0) {
        const cores_per_device = @max(hw.neuroncore_per_device_count, 1);
        const mem_per_core = hw.neuron_device_memory_size / cores_per_device;
        for (infos) |*info| {
            info.mem_total_bytes = mem_per_core;
        }
    }

    for (report.neuron_runtime_data) |runtime| {
        const counters = runtime.report.neuroncore_counters orelse continue;
        var it = counters.neuroncores_in_use.map.iterator();
        while (it.next()) |entry| {
            const core_idx = std.fmt.parseInt(usize, entry.key_ptr.*, 10) catch continue;
            if (core_idx >= infos.len) continue;
            infos[core_idx].gpu_util_percent = @intFromFloat(@round(entry.value_ptr.neuroncore_utilization));
        }
    }
}
