const std = @import("std");
const nvml = @import("nvml.zig");
const pi = @import("../../info/process_info.zig");
const Worker = @import("../../worker.zig").Worker;

pub fn init(w: *Worker, io: std.Io, allocator: std.mem.Allocator, list: *std.ArrayList(pi.ProcessInfo)) !void {
    try w.spawnCustomWorker(io, pollLoop, .{ io, w, allocator, list });
}

fn pollLoop(io: std.Io, w: *const Worker, allocator: std.mem.Allocator, list: *std.ArrayList(pi.ProcessInfo)) void {
    const interval: std.Io.Duration = .fromMilliseconds(w.poll_interval_ms);
    io.sleep(interval, .awake) catch {};

    const device_count = nvml.getDeviceCount() catch 0;
    const last_seen_ts = allocator.alloc(u64, device_count) catch return;
    defer allocator.free(last_seen_ts);
    @memset(last_seen_ts, 0);

    var shadow: std.ArrayList(pi.ProcessInfo) = .{};
    defer shadow.deinit(allocator);

    while (w.isRunning()) {
        shadow.clearRetainingCapacity();

        for (0..device_count) |dev_idx| {
            const handle = nvml.getHandleByIndex(@intCast(dev_idx)) catch continue;
            const idx: u8 = @intCast(dev_idx);

            const compute = nvml.getComputeRunningProcesses(allocator, handle) catch &.{};
            defer if (compute.len > 0) allocator.free(compute);
            collectFromQuery(allocator, &shadow, idx, compute);

            const graphics = nvml.getGraphicsRunningProcesses(allocator, handle) catch &.{};
            defer if (graphics.len > 0) allocator.free(graphics);
            collectFromQuery(allocator, &shadow, idx, graphics);

            // Apply utilization samples
            const last_ts = last_seen_ts[dev_idx];
            const utils = nvml.getProcessUtilization(allocator, handle, last_ts) catch continue;
            defer if (utils.len > 0) allocator.free(utils);

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
