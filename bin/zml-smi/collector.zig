const std = @import("std");
const DeviceInfo = @import("info/device_info.zig").DeviceInfo;
const pi = @import("info/process_info.zig");
const ProcessDoubleBuffer = @import("utils/double_buffer.zig").DoubleBuffer(std.ArrayList(pi.ProcessInfo));
const Worker = @import("worker.zig").Worker;

pub const Collector = struct {
    device_infos: std.ArrayList(*DeviceInfo) = .empty,
    process_lists: std.ArrayList(*ProcessDoubleBuffer) = .empty,
    poll_arenas: std.ArrayList(*std.heap.ArenaAllocator) = .empty,
    arena: std.mem.Allocator,
    gpa: std.mem.Allocator,
    worker: *Worker,
    io: std.Io,

    pub fn addDevice(self: *Collector, initial: DeviceInfo) !*DeviceInfo {
        const info = try self.arena.create(DeviceInfo);
        info.* = initial;
        try self.device_infos.append(self.arena, info);

        return info;
    }

    pub fn createProcessList(self: *Collector) !*ProcessDoubleBuffer {
        const list = try self.arena.create(ProcessDoubleBuffer);
        list.* = .{ .values = .{ .empty, .empty } };
        try self.process_lists.append(self.arena, list);

        return list;
    }

    pub fn createPollArena(self: *Collector) !*std.heap.ArenaAllocator {
        const poll_arena = try self.arena.create(std.heap.ArenaAllocator);
        poll_arena.* = std.heap.ArenaAllocator.init(self.gpa);
        try self.poll_arenas.append(self.arena, poll_arena);

        return poll_arena;
    }

    pub fn deinit(self: *Collector) void {
        for (self.device_infos.items) |info| {
            self.arena.destroy(info);
        }
        self.device_infos.deinit(self.arena);

        for (self.process_lists.items) |list| {
            list.values[0].deinit(self.gpa);
            list.values[1].deinit(self.gpa);
        }
        self.process_lists.deinit(self.arena);

        for (self.poll_arenas.items) |poll_arena| {
            poll_arena.deinit();
        }
        self.poll_arenas.deinit(self.arena);
    }
};
