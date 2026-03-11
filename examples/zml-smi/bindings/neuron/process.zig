const std = @import("std");
const pi = @import("../../info/process_info.zig");
const ProcessInfo = pi.ProcessInfo;
const schema = @import("schema.zig");

const max_runtimes: usize = 32;

const RuntimeEntry = struct {
    pid: u32 = 0,
    device_idx: ?u8 = null,
    device_mem_bytes: u64 = 0,
    util_percent: ?u16 = null,
};

var cache: [2][max_runtimes]RuntimeEntry = .{.{RuntimeEntry{}} ** max_runtimes} ** 2;
var cache_counts: [2]u32 = .{ 0, 0 };
var current: std.atomic.Value(u8) = .init(0);

pub fn updateCache(runtimes: []const schema.RuntimeData) void {
    const back: u8 = 1 - current.load(.acquire);
    var n: u32 = 0;
    for (runtimes) |rt| {
        if (rt.pid == 0 or n >= max_runtimes) break;
        const mem_used = rt.report.memory_used orelse {
            cache[back][n] = .{ .pid = rt.pid };
            n += 1;
            continue;
        };
        const breakdown = mem_used.neuron_runtime_used_bytes.usage_breakdown orelse {
            cache[back][n] = .{ .pid = rt.pid, .device_mem_bytes = mem_used.neuron_runtime_used_bytes.neuron_device };
            n += 1;
            continue;
        };

        // Find the neuroncore with the most memory usage for this process
        var best_core: ?u8 = null;
        var best_mem: u64 = 0;
        var it = breakdown.neuroncore_memory_usage.map.iterator();
        while (it.next()) |entry| {
            const v = entry.value_ptr;
            const total = v.constants + v.model_code + v.model_shared_scratchpad + v.runtime_memory + v.tensors;
            if (total > best_mem) {
                best_mem = total;
                best_core = std.fmt.parseInt(u8, entry.key_ptr.*, 10) catch continue;
            }
        }

        // Look up utilization for the matched core
        const util: ?u16 = if (best_core) |core| blk: {
            const counters = rt.report.neuroncore_counters orelse break :blk null;
            var buf: [4]u8 = undefined;
            const key = std.fmt.bufPrint(&buf, "{d}", .{core}) catch break :blk null;
            break :blk if (counters.neuroncores_in_use.map.get(key)) |usage|
                @intFromFloat(@round(usage.neuroncore_utilization))
            else
                null;
        } else null;

        cache[back][n] = .{
            .pid = rt.pid,
            .device_idx = best_core,
            .device_mem_bytes = mem_used.neuron_runtime_used_bytes.neuron_device,
            .util_percent = util,
        };
        n += 1;
    }
    cache_counts[back] = n;
    current.store(back, .release);
}

pub fn enrichProcesses(result: []ProcessInfo) void {
    const idx = current.load(.acquire);
    const entries = cache[idx][0..cache_counts[idx]];
    for (entries) |entry| {
        if (entry.pid == 0) continue;
        for (result) |*info| {
            if (info.pid == entry.pid) {
                info.device_idx = entry.device_idx;
                info.gpu_mem_kib = @intCast(entry.device_mem_bytes / 1024);
                info.gpu_util_percent = entry.util_percent;
            }
        }
    }
}
