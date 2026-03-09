const std = @import("std");
const nvml = @import("nvml.zig");
const pi = @import("../../info/process_info.zig");
const Worker = @import("../../worker.zig").Worker;

const max_devices: usize = 16;

pub fn init(w: *Worker, io: std.Io, allocator: std.mem.Allocator, list: *std.ArrayList(pi.ProcessInfo)) !void {
    try w.spawnCustomWorker(io, pollLoop, .{ io, w, allocator, list });
}

fn pollLoop(io: std.Io, w: *const Worker, allocator: std.mem.Allocator, list: *std.ArrayList(pi.ProcessInfo)) void {
    const interval: std.Io.Duration = .fromMilliseconds(w.poll_interval_ms);
    io.sleep(interval, .awake) catch {};

    var last_seen_ts: [max_devices]u64 = .{0} ** max_devices;
    var shadow: std.ArrayList(pi.ProcessInfo) = .{};
    defer shadow.deinit(allocator);

    while (w.isRunning()) {
        shadow.clearRetainingCapacity();

        const device_count = nvml.getDeviceCount() catch 0;
        for (0..@min(device_count, max_devices)) |dev_idx| {
            const handle = nvml.getHandleByIndex(@intCast(dev_idx)) catch continue;
            const idx: u8 = @intCast(dev_idx);

            var buf: [64]nvml.ProcessInfo_t = undefined;
            collectFromQuery(allocator, &shadow, idx, nvml.getComputeRunningProcesses(handle, &buf) catch &.{});
            collectFromQuery(allocator, &shadow, idx, nvml.getGraphicsRunningProcesses(handle, &buf) catch &.{});

            // Apply utilization samples
            var util_buf: [256]nvml.ProcessUtilSample_t = undefined;
            const last_ts = last_seen_ts[dev_idx];
            const utils = nvml.getProcessUtilization(handle, &util_buf, last_ts) catch continue;

            for (utils) |sample| {
                if (sample.smUtil > 100 or sample.timeStamp <= last_ts) continue;

                last_seen_ts[dev_idx] = @max(last_seen_ts[dev_idx], sample.timeStamp);

                for (shadow.items) |*entry| {
                    if (entry.pid == sample.pid and entry.device_idx == idx) {
                        entry.gpu_util_percent = @intCast(sample.smUtil);
                    }
                }
            }
        }

        const tmp = list.*;
        list.* = shadow;
        shadow = tmp;

        io.sleep(interval, .awake) catch {};
    }
}

fn collectFromQuery(allocator: std.mem.Allocator, list: *std.ArrayList(pi.ProcessInfo), dev_idx: u8, procs: []const nvml.ProcessInfo_t) void {
    for (procs) |gp| {
        list.append(allocator, .{
            .pid = gp.pid,
            .device_idx = dev_idx,
            .gpu_mem_kib = @intCast(gp.usedGpuMemory / 1024),
        }) catch return;
    }
}
