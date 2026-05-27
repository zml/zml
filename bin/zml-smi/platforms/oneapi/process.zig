const std = @import("std");
const OneApi = @import("oneapi.zig");
const Monitor = @import("monitor.zig");
const pi = @import("zml-smi/info").process_info;
const ProcessDoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer(std.ArrayList(pi.ProcessInfo));
const Collector = @import("zml-smi/collector").Collector;

const ProcessState = struct {
    allocator: std.mem.Allocator,
    previous: Monitor.ProcessPrevious = .{},
};

pub fn init(collector: *Collector, list: *ProcessDoubleBuffer, oneapi: *OneApi, dev_offset: u16) !void {
    const state = try collector.arena.create(ProcessState);
    state.* = .{ .allocator = collector.arena };

    const previous_usage: ?Monitor.ProcessUsage = Monitor.collectProcessUsage(collector.gpa, collector.io, oneapi.monitor.devices) catch null;
    if (previous_usage) |usage_value| {
        var usage = usage_value;
        defer usage.deinit(collector.gpa);
        Monitor.saveProcessPrevious(state.allocator, &state.previous, &usage) catch {};
    }

    try collector.spawnPoll(pollOnce, .{ collector.gpa, collector.io, list, oneapi, dev_offset, state });
}

fn pollOnce(allocator: std.mem.Allocator, io: std.Io, list: *ProcessDoubleBuffer, oneapi: *OneApi, dev_offset: u16, state: *ProcessState) void {
    const back = list.back();
    back.clearRetainingCapacity();

    var usage = Monitor.collectProcessUsage(allocator, io, oneapi.monitor.devices) catch {
        list.swap();
        return;
    };
    defer usage.deinit(allocator);

    var it = usage.iterator();
    while (it.next()) |entry| {
        const sample = entry.value_ptr.*;
        back.append(allocator, .{
            .pid = sample.pid,
            .device_idx = @intCast(sample.device_idx + dev_offset),
            .dev_mem_kib = sample.mem_kib,
            .dev_util_percent = Monitor.processUtil(state.previous.get(entry.key_ptr.*), sample.engine),
        }) catch break;
    }

    Monitor.saveProcessPrevious(state.allocator, &state.previous, &usage) catch {};
    list.swap();
}
