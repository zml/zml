const std = @import("std");
const OneApi = @import("oneapi.zig");
const pi = @import("zml-smi/info").process_info;
const ProcessDoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer(std.ArrayList(pi.ProcessInfo));
const Collector = @import("zml-smi/collector").Collector;

pub fn init(collector: *Collector, list: *ProcessDoubleBuffer, oneapi: *OneApi, dev_offset: u16) !void {
    try collector.spawnPoll(pollOnce, .{ collector.gpa, list, oneapi, dev_offset });
}

fn pollOnce(allocator: std.mem.Allocator, list: *ProcessDoubleBuffer, oneapi: *OneApi, dev_offset: u16) void {
    const back = list.back();
    back.clearRetainingCapacity();

    var processes = oneapi.processList(allocator) catch {
        list.swap();
        return;
    };
    defer processes.deinit(allocator);

    for (processes.items) |proc| {
        back.append(allocator, .{
            .pid = proc.pid,
            .device_idx = @intCast(proc.device_idx + dev_offset),
            .dev_mem_kib = proc.mem_kib,
            .dev_util_percent = proc.util_percent,
        }) catch break;
    }

    list.swap();
}
