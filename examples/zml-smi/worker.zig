const std = @import("std");
const DeviceInfo = @import("device_info.zig").DeviceInfo;

pub const poll_interval_ms: u64 = 500;

pub const Signal = struct {
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    dirty: bool = true,

    pub fn notify(self: *Signal, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);

        self.dirty = true;
        self.cond.signal(io);
    }

    pub fn wait(self: *Signal, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);

        while (!self.dirty) {
            self.cond.wait(io, &self.mutex) catch {};
        }

        self.dirty = false;
    }
};

pub fn spawnWorker(
    io: std.Io,
    info: anytype,
    comptime field: []const u8,
    comptime queryFn: anytype,
    device: anytype,
    signal: *Signal,
) !void {
    const InfoType = @TypeOf(info);
    const DeviceType = @TypeOf(device);
    const Worker = struct {
        fn run(io_: std.Io, info_: InfoType, dev: DeviceType, sig: *Signal) void {
            const interval: std.Io.Duration = .fromMilliseconds(poll_interval_ms);
            while (true) {
                const start: std.Io.Timestamp = .now(io_, .awake);

                const new_value = queryFn(dev) catch null;
                const changed = !std.meta.eql(@field(info_, field), new_value);
                @field(info_, field) = new_value;
                if (changed) sig.notify(io_);

                const elapsed = start.untilNow(io_, .awake);
                if (elapsed.nanoseconds < interval.nanoseconds) {
                    io_.sleep(.fromNanoseconds(interval.nanoseconds - elapsed.nanoseconds), .awake) catch {};
                }
            }
        }
    };

    _ = try io.concurrent(Worker.run, .{ io, info, device, signal });
}

pub fn spawnBatchWorker(
    io: std.Io,
    infos: []DeviceInfo,
    comptime queryFn: *const fn ([]DeviceInfo) void,
    signal: *Signal,
) !void {
    const S = struct {
        fn run(io_: std.Io, infos_: []DeviceInfo, sig: *Signal) void {
            const interval: std.Io.Duration = .fromMilliseconds(poll_interval_ms);
            while (true) {
                const start: std.Io.Timestamp = .now(io_, .awake);

                queryFn(infos_);
                sig.notify(io_);

                const elapsed = start.untilNow(io_, .awake);
                if (elapsed.nanoseconds < interval.nanoseconds) {
                    io_.sleep(.fromNanoseconds(interval.nanoseconds - elapsed.nanoseconds), .awake) catch {};
                }
            }
        }
    };

    _ = try io.concurrent(S.run, .{ io, infos, signal });
}
