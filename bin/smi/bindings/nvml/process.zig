const std = @import("std");
const Nvml = @import("nvml.zig");
const pi = @import("../../info/process_info.zig");
const ProcessShadowList = @import("../../utils/shadow_list.zig").ShadowList(pi.ProcessInfo);
const Worker = @import("../../worker.zig").Worker;

pub fn init(w: *Worker, io: std.Io, allocator: std.mem.Allocator, list: *ProcessShadowList, nvml: *const Nvml, dev_offset: u8) !void {
    try w.spawn(io, pollLoop, .{ io, w, allocator, list, nvml, dev_offset });
}

fn pollLoop(io: std.Io, w: *const Worker, allocator: std.mem.Allocator, list: *ProcessShadowList, nvml: *const Nvml, dev_offset: u8) void {
    const interval: std.Io.Duration = .fromMilliseconds(w.poll_interval_ms);
    io.sleep(interval, .awake) catch {};

    const device_count = nvml.deviceCount() catch 0;
    const last_seen_ts = allocator.alloc(u64, device_count) catch return;
    defer allocator.free(last_seen_ts);
    @memset(last_seen_ts, 0);

    var sl = list.shadow();
    defer sl.deinit(allocator);

    while (w.isRunning()) {
        const start: std.Io.Timestamp = .now(io, .awake);

        sl.clearRetainingCapacity();

        for (0..device_count) |dev_idx| {
            const handle = nvml.handleByIndex(@intCast(dev_idx)) catch continue;
            const idx: u8 = @intCast(dev_idx + dev_offset);

            const compute = nvml.computeRunningProcesses(allocator, handle) catch &.{};
            defer if (compute.len > 0) allocator.free(compute);
            collectFromQuery(allocator, &sl, idx, compute);

            const graphics = nvml.graphicsRunningProcesses(allocator, handle) catch &.{};
            defer if (graphics.len > 0) allocator.free(graphics);
            collectFromQuery(allocator, &sl, idx, graphics);

            // Apply utilization samples
            const last_ts = last_seen_ts[dev_idx];
            const utils = nvml.processUtilization(allocator, handle, last_ts) catch continue;
            defer if (utils.len > 0) allocator.free(utils);

            for (utils) |sample| {
                if (sample.smUtil > 100 or sample.timeStamp <= last_ts) continue;

                last_seen_ts[dev_idx] = @max(last_seen_ts[dev_idx], sample.timeStamp);

                for (sl.list.items) |*entry| {
                    if (entry.pid == sample.pid and entry.device_idx == idx) {
                        entry.dev_util_percent = @intCast(sample.smUtil);
                    }
                }
            }
        }

        sl.swap(io);

        const elapsed = start.untilNow(io, .awake);
        if (elapsed.nanoseconds < interval.nanoseconds) {
            io.sleep(.fromNanoseconds(interval.nanoseconds - elapsed.nanoseconds), .awake) catch {};
        }
    }
}

fn collectFromQuery(allocator: std.mem.Allocator, sl: *ProcessShadowList.Shadow, dev_idx: u8, procs: []const Nvml.ProcessInfo_t) void {
    for (procs) |gp| {
        sl.append(allocator, .{
            .pid = gp.pid,
            .device_idx = dev_idx,
            .dev_mem_kib = @intCast(gp.usedGpuMemory / 1024),
        }) catch return;
    }
}
