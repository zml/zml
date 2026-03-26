const std = @import("std");
const c = @import("c");
const Nrt = @import("nrt.zig");
const pi = @import("../../info/process_info.zig");
const DeviceInfo = @import("../../info/device_info.zig").DeviceInfo;
const ProcessShadowList = @import("../../utils/shadow_list.zig").ShadowList(pi.ProcessInfo);
const Worker = @import("../../worker.zig").Worker;

pub fn init(w: *Worker, io: std.Io, allocator: std.mem.Allocator, list: *ProcessShadowList, nrt: *const Nrt, handles: []const *c.ndl_device_t, nc_per_device: u32, device_infos: []*DeviceInfo, dev_offset: u8) !void {
    try w.spawn(io, pollLoop, .{ io, w, allocator, list, nrt, handles, nc_per_device, device_infos, dev_offset });
}

fn pollLoop(io: std.Io, w: *const Worker, allocator: std.mem.Allocator, list: *ProcessShadowList, nrt: *const Nrt, handles: []const *c.ndl_device_t, nc_per_device: u32, device_infos: []*DeviceInfo, dev_offset: u8) void {
    const interval: std.Io.Duration = .fromMilliseconds(w.poll_interval_ms);
    io.sleep(interval, .awake) catch {};

    if (handles.len == 0) return;

    var sl = list.shadow();
    defer sl.deinit(allocator);

    while (w.isRunning()) {
        const start: std.Io.Timestamp = .now(io, .awake);

        sl.clearRetainingCapacity();

        for (handles, 0..) |handle, dev_i| {
            const apps_result = nrt.getAllAppsInfo(handle) catch continue;
            defer if (apps_result.ptr) |ptr| std.c.free(@ptrCast(ptr));

            const apps = if (apps_result.ptr) |ptr| ptr[0..apps_result.count] else continue;

            var total_us_per_core: [c.MAX_NC_PER_DEVICE]u64 = .{0} ** c.MAX_NC_PER_DEVICE;

            for (apps) |*app| {
                if (app.pid <= 0) continue;

                const nds = nrt.ndsOpen(handle, app.pid) catch continue;
                defer nrt.ndsClose(nds);

                for (0..nc_per_device) |ci| {
                    const time_in_use = nrt.getNcCounter(nds, @intCast(ci), c.NDS_NC_COUNTER_TIME_IN_USE) catch continue;
                    total_us_per_core[ci] += time_in_use;

                    if (time_in_use == 0) continue;

                    const dev_idx: u32 = @intCast(dev_i * nc_per_device + ci);

                    var info: pi.ProcessInfo = .{
                        .pid = @bitCast(app.pid),
                        .device_idx = @intCast(dev_idx + dev_offset),
                        .dev_mem_kib = @intCast(app.device_mem_size / 1024),
                    };

                    if (dev_idx < device_infos.len) {
                        if (device_infos[dev_idx].neuron.get(io).util_percent) |util| {
                            info.dev_util_percent = @intCast(util);
                        }
                    }

                    if (info.dev_mem_kib != null and info.dev_mem_kib.? > 0)
                        sl.append(allocator, info) catch break;
                }
            }

            const now: std.Io.Timestamp = .now(io, .awake);

            for (0..nc_per_device) |ci| {
                const dev_idx = dev_i * nc_per_device + ci;
                if (dev_idx >= device_infos.len) break;

                var ni = device_infos[dev_idx].neuron.get(io);
                const prev_us = ni.prev_total_us;

                ni.prev_total_us = total_us_per_core[ci];

                if (ni.prev_timestamp) |prev| {
                    const elapsed = prev.untilNow(io, .awake);
                    const elapsed_us: u64 = @intCast(@divFloor(elapsed.nanoseconds, std.time.ns_per_us));
                    if (elapsed_us > 0) {
                        const delta_us = total_us_per_core[ci] -| prev_us;
                        ni.util_percent = @min(100, delta_us * 100 / elapsed_us);
                    }
                }

                ni.prev_timestamp = now;
                device_infos[dev_idx].neuron.set(io, ni);
            }
        }

        sl.swap(io);

        const elapsed = start.untilNow(io, .awake);
        if (elapsed.nanoseconds < interval.nanoseconds) {
            io.sleep(.fromNanoseconds(interval.nanoseconds - elapsed.nanoseconds), .awake) catch {};
        }
    }
}
