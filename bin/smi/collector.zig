const std = @import("std");
const DeviceInfo = @import("info/device_info.zig").DeviceInfo;
const pi = @import("info/process_info.zig");
const ProcessShadowList = @import("utils/shadow_list.zig").ShadowList(pi.ProcessInfo);
const Worker = @import("worker.zig").Worker;

pub const Collector = struct {
    device_infos: std.ArrayList(*DeviceInfo) = .empty,
    process_lists: std.ArrayList(*ProcessShadowList) = .empty,
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

    pub fn createProcessList(self: *Collector) !*ProcessShadowList {
        const list = try self.arena.create(ProcessShadowList);
        list.* = .init();
        try self.process_lists.append(self.arena, list);
        return list;
    }

    pub fn deinit(self: *Collector) void {
        for (self.device_infos.items) |info| self.arena.destroy(info);
        self.device_infos.deinit(self.arena);
        for (self.process_lists.items) |list| list.deinit(self.gpa);
        self.process_lists.deinit(self.arena);
    }
};
