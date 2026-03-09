const std = @import("std");
const DeviceInfo = @import("info/device_info.zig").DeviceInfo;

pub const Worker = struct {
    poll_interval_ms: u16,
    should_stop: std.atomic.Value(bool) = .init(false),
    group: std.Io.Group = .init,

    pub fn shutdown(self: *Worker, io: std.Io) void {
        self.should_stop.store(true, .release);
        self.group.await(io) catch {};
    }

    pub fn isRunning(self: *const Worker) bool {
        return !self.should_stop.load(.acquire);
    }

    pub fn spawnCustomWorker(
        self: *Worker,
        io: std.Io,
        comptime runFn: anytype,
        args: std.meta.ArgsTuple(@TypeOf(runFn)),
    ) !void {
        try self.group.concurrent(io, runFn, args);
    }

    pub fn spawnWorker(
        self: *Worker,
        io: std.Io,
        info: anytype,
        comptime field: []const u8,
        comptime queryFn: anytype,
        device: anytype,
    ) !void {
        const InfoType = @TypeOf(info);
        const DeviceType = @TypeOf(device);
        const S = struct {
            fn run(io_: std.Io, w: *Worker, info_: InfoType, dev: DeviceType) void {
                const interval: std.Io.Duration = .fromMilliseconds(w.poll_interval_ms);
                while (!w.should_stop.load(.acquire)) {
                    const start: std.Io.Timestamp = .now(io_, .awake);

                    @field(info_, field) = queryFn(dev) catch null;

                    const elapsed = start.untilNow(io_, .awake);
                    if (elapsed.nanoseconds < interval.nanoseconds) {
                        io_.sleep(.fromNanoseconds(interval.nanoseconds - elapsed.nanoseconds), .awake) catch {};
                    }
                }
            }
        };

        try self.group.concurrent(io, S.run, .{ io, self, info, device });
    }

    pub fn spawnBatchWorker(
        self: *Worker,
        io: std.Io,
        infos: []*DeviceInfo,
        comptime queryFn: *const fn ([]*DeviceInfo) void,
    ) !void {
        const S = struct {
            fn run(io_: std.Io, w: *Worker, infos_: []*DeviceInfo) void {
                const interval: std.Io.Duration = .fromMilliseconds(w.poll_interval_ms);

                while (!w.should_stop.load(.acquire)) {
                    const start: std.Io.Timestamp = .now(io_, .awake);

                    queryFn(infos_);

                    const elapsed = start.untilNow(io_, .awake);
                    if (elapsed.nanoseconds < interval.nanoseconds) {
                        io_.sleep(.fromNanoseconds(interval.nanoseconds - elapsed.nanoseconds), .awake) catch {};
                    }
                }
            }
        };

        try self.group.concurrent(io, S.run, .{ io, self, infos });
    }
};
