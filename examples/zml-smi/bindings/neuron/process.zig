const std = @import("std");
const pi = @import("../../info/process_info.zig");
const schema = @import("schema.zig");

pub fn update(allocator: std.mem.Allocator, list: *std.ArrayList(pi.ProcessInfo), shadow: *std.ArrayList(pi.ProcessInfo), runtimes: []const schema.RuntimeData) void {
    shadow.clearRetainingCapacity();

    for (runtimes) |rt| {
        if (rt.pid == 0) break;
        const mem_used = rt.report.memory_used orelse continue;
        const breakdown = mem_used.neuron_runtime_used_bytes.usage_breakdown orelse continue;

        // Find the neuroncore with the most memory usage for this process because
        // neuron still show every core per process even though the core is not used
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

        const core = best_core orelse continue;

        // Look up utilization for the matched core
        const util: ?u16 = blk: {
            const counters = rt.report.neuroncore_counters orelse break :blk null;
            var buf: [4]u8 = undefined;
            const key = std.fmt.bufPrint(&buf, "{d}", .{core}) catch break :blk null;
            break :blk if (counters.neuroncores_in_use.map.get(key)) |usage|
                @intFromFloat(@round(usage.neuroncore_utilization))
            else
                null;
        };

        shadow.append(allocator, .{
            .pid = rt.pid,
            .device_idx = core,
            .gpu_mem_kib = @intCast(mem_used.neuron_runtime_used_bytes.neuron_device / 1024),
            .gpu_util_percent = util,
        }) catch break;
    }

    const tmp = list.*;
    list.* = shadow.*;
    shadow.* = tmp;
}
