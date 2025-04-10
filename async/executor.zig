const std = @import("std");

const libcoro_options = @import("libcoro_options");
const stdx = @import("stdx");

const libcoro = @import("coro.zig");

pub const Executor = struct {
    const Self = @This();

    pub const Func = struct {
        const FuncFn = *const fn (userdata: ?*anyopaque) void;

        func: FuncFn,
        userdata: ?*anyopaque = null,
        next: ?*Func = null,

        pub fn init(func: FuncFn, userdata: ?*anyopaque) Func {
            return .{ .func = func, .userdata = userdata };
        }

        fn run(self: Func) void {
            @call(.auto, self.func, .{self.userdata});
        }
    };

    readyq: stdx.queue.SPSC(Func) = .{},

    pub fn init() Self {
        return .{};
    }

    pub fn runSoon(self: *Self, func: *Func) void {
        self.readyq.push(func);
    }

    pub fn runAllSoon(self: *Self, funcs: stdx.queue.SPSC(Func)) void {
        self.readyq.pushAll(funcs);
    }

    pub fn tick(self: *Self) bool {
        // Reset readyq so that adds run on next tick.
        var now = self.readyq;
        self.readyq = .{};

        libcoro.log(.debug, "Executor.tick readyq.len={d}", .{now.len()});

        var count: usize = 0;
        while (now.pop()) |func| : (count += 1) {
            func.run();
        }

        libcoro.log(.debug, "Executor.tick done", .{});

        return count > 0;
    }
};

pub const Condition = struct {
    exec: *Executor,
    waiters: stdx.queue.SPSC(Executor.Func) = .{},

    pub fn init(exec: *Executor) Condition {
        return .{ .exec = exec };
    }

    pub fn broadcast(self: *Condition) void {
        self.exec.runAllSoon(self.waiters);
    }

    pub fn signal(self: *Condition) void {
        if (self.waiters.pop()) |waiter_func| {
            self.exec.runSoon(waiter_func);
        }
    }

    pub fn wait(self: *Condition) void {
        var cr = CoroResume.init();
        var cb = cr.func();
        self.waiters.push(&cb);
        libcoro.xsuspend();
    }
};

pub const CoroResume = struct {
    const Self = @This();

    coro: libcoro.Frame,

    pub fn init() Self {
        return .{ .coro = libcoro.xframe() };
    }

    pub fn func(self: *Self) Executor.Func {
        return .{ .func = Self.cb, .userdata = self };
    }

    fn cb(ud: ?*anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ud));
        libcoro.xresume(self.coro);
    }
};

pub fn getExec(exec: ?*Executor) *Executor {
    if (exec != null) return exec.?;
    if (libcoro.getEnv().executor) |x| return x;
    @panic("No explicit Executor passed and no default Executor available");
}

pub fn ArrayQueue(comptime T: type, comptime size: usize) type {
    return struct {
        const Self = @This();

        vals: [size]T = undefined,
        head: ?usize = null,
        tail: ?usize = null,

        fn init() Self {
            return .{};
        }

        fn len(self: Self) usize {
            switch (self.state()) {
                .empty => return 0,
                .one => return 1,
                .many => {
                    const head = self.head.?;
                    const tail = self.tail.?;
                    if (tail > head) {
                        return tail - head + 1;
                    }
                    return size - head + tail + 1;
                },
            }
        }

        fn space(self: Self) usize {
            return size - self.len();
        }

        fn push(self: *@This(), val: T) !void {
            if (self.space() < 1) return error.QueueFull;
            switch (self.state()) {
                .empty => {
                    self.head = 0;
                    self.tail = 0;
                    self.vals[0] = val;
                },
                .one, .many => {
                    const tail = self.tail.?;
                    const new_tail = (tail + 1) % size;
                    self.vals[new_tail] = val;
                    self.tail = new_tail;
                },
            }
        }

        fn pop(self: *Self) ?T {
            switch (self.state()) {
                .empty => return null,
                .one => {
                    const out = self.vals[self.head.?];
                    self.head = null;
                    self.tail = null;
                    return out;
                },
                .many => {
                    const out = self.vals[self.head.?];
                    self.head = (self.head.? + 1) % size;
                    return out;
                },
            }
        }

        const State = enum { empty, one, many };
        inline fn state(self: Self) State {
            if (self.head == null) return .empty;
            if (self.head.? == self.tail.?) return .one;
            return .many;
        }
    };
}
