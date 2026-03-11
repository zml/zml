const std = @import("std");
const amdsmi = @import("amdsmi.zig");
const pi = @import("../../info/process_info.zig");
const ProcessInfo = pi.ProcessInfo;

pub fn enrichProcesses(result: []ProcessInfo) void {
    const device_count = amdsmi.getDeviceCount() catch return;

    for (0..device_count) |dev_idx| {
        const handle = amdsmi.getHandleByIndex(@intCast(dev_idx)) catch continue;
        const idx: u8 = @intCast(dev_idx);

        var buf: [64]amdsmi.ProcInfo = undefined;
        const procs = amdsmi.getProcessList(handle, &buf) catch |err| {
            std.log.err("amdsmi getProcessList dev {d}: {s}", .{ dev_idx, @errorName(err) });
            continue;
        };

        // std.log.err("amdsmi dev {d}: {d} procs", .{ dev_idx, procs.len });
        for (procs) |proc| {
            // std.log.err("  pid={d} vram={d} gfx_ns={any} mem={d}", .{ proc.pid, proc.memory_usage.vram_mem, proc.engine_usage, proc.mem });
            for (result) |*info| {
                if (info.pid == proc.pid) {
                    // TODO: the tested device doesnt report gpu usage per process (always 0), so we need to find a way to test it

                    info.device_idx = idx;
                    info.gpu_mem_kib = @intCast(proc.memory_usage.vram_mem / 1024);
                }
            }
        }
    }
}
