const std = @import("std");

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

    pub fn spawn(
        self: *Worker,
        io: std.Io,
        comptime runFn: anytype,
        args: std.meta.ArgsTuple(@TypeOf(runFn)),
    ) !void {
        try self.group.concurrent(io, runFn, args);
    }

    pub fn pollMetrics(comptime DB: type, comptime Dev: type, comptime table: anytype) fn (std.Io, *const Worker, DB, Dev) void {
        return struct {
            fn f(io: std.Io, w: *const Worker, db: DB, dev: Dev) void {
                w.pollLoop(io, struct {
                    fn poll(s: DB, d: Dev) void {
                        if (@hasField(Dev, "arena")) {
                            _ = d.arena.reset(.retain_capacity);
                        }
                        const back = s.back();
                        back.* = s.front().*;
                        inline for (table) |m| {
                            @field(back, m.field) = m.query(d) catch null;
                        }
                        s.swap();
                    }
                }.poll, .{ db, dev });
            }
        }.f;
    }

    pub fn pollLoop(self: *const Worker, io: std.Io, comptime func: anytype, args: std.meta.ArgsTuple(@TypeOf(func))) void {
        const interval: std.Io.Duration = .fromMilliseconds(self.poll_interval_ms);
        while (!self.should_stop.load(.acquire)) {
            const start: std.Io.Timestamp = .now(io, .awake);

            @call(.auto, func, args);

            const elapsed = start.untilNow(io, .awake);
            if (elapsed.nanoseconds < interval.nanoseconds) {
                io.sleep(.fromNanoseconds(interval.nanoseconds - elapsed.nanoseconds), .awake) catch {};
            }
        }
    }
};
