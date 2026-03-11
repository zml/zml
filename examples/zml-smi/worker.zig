const std = @import("std");
const DeviceInfo = @import("info/device_info.zig").DeviceInfo;

pub var poll_interval_ms: u16 = undefined;

var should_stop: std.atomic.Value(bool) = .init(false);
var group: std.Io.Group = .init;

pub fn shutdown(io: std.Io) void {
    should_stop.store(true, .release);
    group.await(io) catch {};
}

pub fn isRunning() bool {
    return !should_stop.load(.acquire);
}

pub fn spawnCustomWorker(
    io: std.Io,
    comptime runFn: anytype,
    args: std.meta.ArgsTuple(@TypeOf(runFn)),
) !void {
    try group.concurrent(io, runFn, args);
}

pub fn spawnWorker(
    io: std.Io,
    info: anytype,
    comptime field: []const u8,
    comptime queryFn: anytype,
    device: anytype,
) !void {
    const InfoType = @TypeOf(info);
    const DeviceType = @TypeOf(device);
    const Worker = struct {
        fn run(io_: std.Io, info_: InfoType, dev: DeviceType) void {
            const interval: std.Io.Duration = .fromMilliseconds(poll_interval_ms);
            while (!should_stop.load(.acquire)) {
                const start: std.Io.Timestamp = .now(io_, .awake);

                @field(info_, field) = queryFn(dev) catch null;

                const elapsed = start.untilNow(io_, .awake);
                if (elapsed.nanoseconds < interval.nanoseconds) {
                    io_.sleep(.fromNanoseconds(interval.nanoseconds - elapsed.nanoseconds), .awake) catch {};
                }
            }
        }
    };

    try group.concurrent(io, Worker.run, .{ io, info, device });
}

pub fn spawnBatchWorker(
    io: std.Io,
    infos: []*DeviceInfo,
    comptime queryFn: *const fn ([]*DeviceInfo) void,
) !void {
    const S = struct {
        fn run(io_: std.Io, infos_: []*DeviceInfo) void {
            const interval: std.Io.Duration = .fromMilliseconds(poll_interval_ms);
            while (!should_stop.load(.acquire)) {
                const start: std.Io.Timestamp = .now(io_, .awake);

                queryFn(infos_);

                const elapsed = start.untilNow(io_, .awake);
                if (elapsed.nanoseconds < interval.nanoseconds) {
                    io_.sleep(.fromNanoseconds(interval.nanoseconds - elapsed.nanoseconds), .awake) catch {};
                }
            }
        }
    };

    try group.concurrent(io, S.run, .{ io, infos });
}
