const std = @import("std");
const c = @import("c");
const Nrt = @import("nrt.zig");
const smi_info = @import("zml-smi/info");
const pi = smi_info.process_info;
const DeviceInfo = smi_info.device_info.DeviceInfo;
const ProcessDoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer(std.ArrayList(pi.ProcessInfo));
const Collector = @import("zml-smi/collector").Collector;

pub fn init(collector: *Collector, list: *ProcessDoubleBuffer, nrt: *const Nrt, nc_per_device: u32, device_infos: []*DeviceInfo, dev_offset: u8) !void {
    if (nrt.handles.len == 0) return;
    try collector.spawnPoll(pollOnce, .{ collector.io, collector.gpa, list, nrt, nc_per_device, device_infos, dev_offset });
}

fn pollOnce(io: std.Io, allocator: std.mem.Allocator, list: *ProcessDoubleBuffer, nrt: *const Nrt, nc_per_device: u32, device_infos: []*DeviceInfo, dev_offset: u8) void {
    const back = list.back();

    back.clearRetainingCapacity();

    for (nrt.handles, 0..) |handle, dev_i| {
        const apps_result = nrt.allAppsInfo(handle) catch continue;
        defer {
            if (apps_result.ptr) |ptr| {
                std.c.free(@ptrCast(ptr));
            }
        }

        const apps = if (apps_result.ptr) |ptr| ptr[0..apps_result.count] else continue;

        var total_us_per_core: [c.MAX_NC_PER_DEVICE]u64 = .{0} ** c.MAX_NC_PER_DEVICE;

        for (apps) |*app| {
            if (app.pid <= 0) {
                continue;
            }

            const nds = nrt.ndsOpen(handle, app.pid) catch continue;
            defer nrt.ndsClose(nds);

            for (0..nc_per_device) |ci| {
                const time_in_use = nrt.ncCounter(nds, @intCast(ci), c.NDS_NC_COUNTER_TIME_IN_USE) catch continue;
                total_us_per_core[ci] += time_in_use;

                if (time_in_use == 0) {
                    continue;
                }

                const dev_idx: u32 = @intCast(dev_i * nc_per_device + ci);

                var info: pi.ProcessInfo = .{
                    .pid = @bitCast(app.pid),
                    .device_idx = @intCast(dev_idx + dev_offset),
                    .dev_mem_kib = @intCast(app.device_mem_size / 1024),
                };

                if (dev_idx < device_infos.len) {
                    if (device_infos[dev_idx].neuron.front().util_percent) |util| {
                        info.dev_util_percent = @intCast(util);
                    }
                }

                if (info.dev_mem_kib != null and info.dev_mem_kib.? > 0)
                    back.append(allocator, info) catch break;
            }
        }

        const now: std.Io.Timestamp = .now(io, .awake);

        for (0..nc_per_device) |ci| {
            const dev_idx = dev_i * nc_per_device + ci;
            if (dev_idx >= device_infos.len) {
                break;
            }

            const ni_back = device_infos[dev_idx].neuron.back();
            ni_back.* = device_infos[dev_idx].neuron.front().*;
            const prev_us = ni_back.prev_total_us;

            ni_back.prev_total_us = total_us_per_core[ci];

            if (ni_back.prev_timestamp) |prev| {
                const elapsed = prev.untilNow(io, .awake);
                const elapsed_us: u64 = @intCast(@divFloor(elapsed.nanoseconds, std.time.ns_per_us));

                if (elapsed_us > 0) {
                    const delta_us = total_us_per_core[ci] -| prev_us;
                    ni_back.util_percent = @min(100, delta_us * 100 / elapsed_us);
                }
            }

            ni_back.prev_timestamp = now;
            device_infos[dev_idx].neuron.swap();
        }
    }

    list.swap();
}
