const std = @import("std");
const pi = @import("zml-smi/info").process_info;
const ProcessDoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer(std.ArrayList(pi.ProcessInfo));
const Collector = @import("zml-smi/collector").Collector;

pub fn start(collector: *Collector) !void {
    const arenas: [2]*std.heap.ArenaAllocator = .{ try collector.createPollArena(), try collector.createPollArena() };
    try collector.spawnPoll(pollOnce, .{ arenas, collector.gpa, collector.process_lists.items, &collector.processes });
}

fn pollOnce(arenas: [2]*std.heap.ArenaAllocator, gpa: std.mem.Allocator, lists: []const *const ProcessDoubleBuffer, processes: *ProcessDoubleBuffer) void {
    const arena = arenas[1 - processes.current.load(.acquire)];
    _ = arena.reset(.retain_capacity);

    const back = processes.back();
    back.clearRetainingCapacity();
    for (lists) |list| {
        back.appendSlice(gpa, list.front().items) catch break;
    }

    for (back.items) |*info| {
        info.username = arena.allocator().dupe(u8, info.username) catch "";
        info.comm = arena.allocator().dupe(u8, info.comm) catch "";
    }

    processes.swap();
}
