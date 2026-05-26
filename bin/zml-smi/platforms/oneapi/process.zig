const std = @import("std");
const sysfs = @import("sysfs.zig");
const pi = @import("zml-smi/info").process_info;
const ProcessDoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer(std.ArrayList(pi.ProcessInfo));
const Collector = @import("zml-smi/collector").Collector;

pub const Target = sysfs.Target;
pub const EngineSample = sysfs.EngineSample;
pub const DeviceSample = sysfs.DeviceSample;

pub fn init(collector: *Collector, list: *ProcessDoubleBuffer, targets: []const Target) !void {
    const state = try collector.arena.create(ProcessState);
    state.* = .{ .allocator = collector.arena };

    try collector.spawnPoll(pollOnce, .{ collector.gpa, collector.io, list, targets, state });
}

const ProcessState = struct {
    allocator: std.mem.Allocator,
    previous: std.AutoHashMapUnmanaged(u64, EngineSample) = .{},
};

fn pollOnce(allocator: std.mem.Allocator, io: std.Io, list: *ProcessDoubleBuffer, targets: []const Target, state: *ProcessState) void {
    const back = list.back();
    back.clearRetainingCapacity();

    var usage = sysfs.collectProcessUsage(allocator, io, targets) catch {
        list.swap();
        return;
    };
    defer usage.deinit(allocator);

    var it = usage.iterator();
    while (it.next()) |entry| {
        const sample = entry.value_ptr.*;
        back.append(allocator, .{
            .pid = sample.pid,
            .device_idx = sample.device_idx,
            .dev_mem_kib = sample.mem_kib,
            .dev_util_percent = sysfs.processUtil(state.previous.get(entry.key_ptr.*), sample.engine),
        }) catch break;
    }

    state.previous.clearRetainingCapacity();
    state.previous.ensureTotalCapacity(state.allocator, @intCast(usage.count())) catch {};

    var sample_it = usage.iterator();
    while (sample_it.next()) |entry| {
        if (entry.value_ptr.engine) |engine| {
            state.previous.putAssumeCapacity(entry.key_ptr.*, engine);
        }
    }

    list.swap();
}
