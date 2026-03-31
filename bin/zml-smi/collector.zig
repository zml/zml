const std = @import("std");
const smi_info = @import("zml-smi/info");
const DeviceInfo = smi_info.device_info.DeviceInfo;
const pi = smi_info.process_info;
const ProcessDoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer(std.ArrayList(pi.ProcessInfo));
const Worker = @import("zml-smi/worker").Worker;

pub const Collector = struct {
    device_infos: std.ArrayList(*DeviceInfo) = .empty,
    process_lists: std.ArrayList(*ProcessDoubleBuffer) = .empty,
    poll_arenas: std.ArrayList(*std.heap.ArenaAllocator) = .empty,
    arena: std.mem.Allocator,
    gpa: std.mem.Allocator,
    worker: *Worker,
    io: std.Io,
    poll_only: bool = false,

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

    pub fn spawnPoll(self: *Collector, comptime pollOnce: anytype, args: std.meta.ArgsTuple(@TypeOf(pollOnce))) !void {
        if (self.poll_only) {
            @call(.auto, pollOnce, args);
            return;
        }

        const Args = std.meta.ArgsTuple(@TypeOf(pollOnce));
        try self.worker.spawn(self.io, struct {
            fn f(io: std.Io, w: *const Worker, a: Args) void {
                const interval: std.Io.Duration = .fromMilliseconds(w.poll_interval_ms);
                while (w.isRunning()) {
                    const start: std.Io.Timestamp = .now(io, .awake);

                    @call(.auto, pollOnce, a);

                    const elapsed = start.untilNow(io, .awake);
                    if (elapsed.nanoseconds < interval.nanoseconds) {
                        io.sleep(.fromNanoseconds(interval.nanoseconds - elapsed.nanoseconds), .awake) catch {};
                    }
                }
            }
        }.f, .{ self.io, self.worker, args });
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
