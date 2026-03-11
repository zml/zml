const nvml = @import("nvml.zig");
const pi = @import("../../info/process_info.zig");
const ProcessInfo = pi.ProcessInfo;

const max_devices: usize = 16;
var last_seen_ts: [max_devices]u64 = .{0} ** max_devices;

pub fn enrichProcesses(result: []ProcessInfo) void {
    const device_count = nvml.getDeviceCount() catch return;
    for (0..@min(device_count, max_devices)) |dev_idx| {
        const handle = nvml.getHandleByIndex(@intCast(dev_idx)) catch continue;
        const idx: u8 = @intCast(dev_idx);
        var buf: [64]nvml.ProcessInfo_t = undefined;
        enrichFromQuery(result, idx, nvml.getComputeRunningProcesses(handle, &buf) catch &.{});
        enrichFromQuery(result, idx, nvml.getGraphicsRunningProcesses(handle, &buf) catch &.{});

        for (result) |*proc| {
            if (proc.device_idx == idx) proc.gpu_util_percent = 0;
        }
        var util_buf: [256]nvml.ProcessUtilSample_t = undefined;
        const last_ts = last_seen_ts[dev_idx];
        const utils = nvml.getProcessUtilization(handle, &util_buf, last_ts) catch continue;
        for (utils) |sample| {
            if (sample.smUtil > 100 or sample.timeStamp <= last_ts) continue;
            last_seen_ts[dev_idx] = @max(last_seen_ts[dev_idx], sample.timeStamp);
            for (result) |*proc| {
                if (proc.pid == sample.pid) {
                    proc.gpu_util_percent = @intCast(sample.smUtil);
                }
            }
        }
    }
}

fn enrichFromQuery(result: []ProcessInfo, dev_idx: u8, procs: []const nvml.ProcessInfo_t) void {
    for (procs) |gp| {
        for (result) |*proc| {
            if (proc.pid == gp.pid) {
                proc.device_idx = dev_idx;
                proc.gpu_mem_kib = @intCast(gp.usedGpuMemory / 1024);
            }
        }
    }
}
