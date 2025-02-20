const std = @import("std");
const stdx = @import("stdx");
const xev = @import("xev");
const libcoro = @import("coro.zig");
const CoroExecutor = @import("executor.zig").Executor;

pub const xasync = libcoro.xasync;
pub const xawait = libcoro.xawait;

const Frame = libcoro.Frame;

const Env = struct {
    exec: ?*Executor = null,
};

pub const EnvArg = struct {
    executor: ?*Executor = null,
    stack_allocator: ?std.mem.Allocator = null,
    default_stack_size: ?usize = null,
};

threadlocal var env: Env = .{};

pub fn initEnv(e: EnvArg) void {
    env = .{ .exec = e.executor };
    libcoro.initEnv(.{
        .stack_allocator = e.stack_allocator,
        .default_stack_size = e.default_stack_size,
        .executor = if (e.executor) |ex| &ex.exec else null,
    });
}

pub const Executor = struct {
    loop: *xev.Loop,
    exec: CoroExecutor = .{},

    pub fn init(loop: *xev.Loop) Executor {
        return .{ .loop = loop };
    }

    fn tick(self: *Executor) !void {
        try self.loop.run(.once);
        _ = self.exec.tick();
    }
};

/// Run a coroutine to completion.
/// Must be called from "root", outside of any created coroutine.
pub fn run(
    exec: *Executor,
    comptime func: anytype,
    args: anytype,
    stack: anytype,
) !stdx.meta.FnSignature(func, @TypeOf(args)).ReturnPayloadT {
    stdx.debug.assert(libcoro.inCoro() == false, "Not in a coroutine", .{});
    const frame = try xasync(func, args, stack);
    defer frame.deinit();
    try runCoro(exec, frame);
    return xawait(frame);
}

/// Run a coroutine to completion.
/// Must be called from "root", outside of any created coroutine.
fn runCoro(exec: *Executor, frame: anytype) !void {
    const f = frame.frame();
    if (f.status == .Start) {
        libcoro.xresume(f);
    }
    while (f.status != .Done) {
        try exec.tick();
    }
}

const SleepResult = xev.Timer.RunError!void;

pub fn sleep(exec: *Executor, ms: u64) !void {
    const loop = exec.loop;
    const Data = XCallback(SleepResult);

    var data = Data.init();
    const w = try xev.Timer.init();
    defer w.deinit();
    var c: xev.Completion = .{};
    w.run(loop, &c, ms, Data, &data, &Data.callback);

    try waitForCompletion(exec, &c);

    return data.result;
}

pub fn waitForCompletionOutsideCoro(exec: *Executor, c: *xev.Completion) !void {
    // which one should it be ?
    // @branchHint(.cold);
    @branchHint(.unlikely);
    while (c.state() != .dead) {
        try exec.tick();
    }
}

pub fn waitForCompletionInCoro(c: *xev.Completion) void {
    while (c.state() != .dead) {
        libcoro.xsuspend();
    }
}

pub fn waitForCompletion(exec: *Executor, c: *xev.Completion) !void {
    if (libcoro.inCoro()) {
        waitForCompletionInCoro(c);
    } else {
        try waitForCompletionOutsideCoro(exec, c);
    }
}

pub const TCP = struct {
    const Self = @This();

    exec: *Executor,
    tcp: xev.TCP,

    pub usingnamespace Stream(Self, xev.TCP, .{
        .close = true,
        .read = .recv,
        .write = .send,
    });

    pub fn init(exec: *Executor, tcp: xev.TCP) Self {
        return .{ .exec = exec, .tcp = tcp };
    }

    fn stream(self: Self) xev.TCP {
        return self.tcp;
    }

    pub fn accept(self: Self) !Self {
        const AcceptResult = xev.TCP.AcceptError!xev.TCP;
        const Data = XCallback(AcceptResult);

        const loop = self.exec.loop;

        var data = Data.init();
        var c: xev.Completion = .{};
        self.tcp.accept(loop, &c, Data, &data, &Data.callback);

        try waitForCompletion(self.exec, &c);

        const result = try data.result;
        return .{ .exec = self.exec, .tcp = result };
    }

    const ConnectResult = xev.TCP.ConnectError!void;
    pub fn connect(self: Self, addr: std.net.Address) !void {
        const ResultT = ConnectResult;
        const Data = struct {
            result: ResultT = undefined,
            frame: ?Frame = null,

            fn callback(
                userdata: ?*@This(),
                l: *xev.Loop,
                c: *xev.Completion,
                s: xev.TCP,
                result: ResultT,
            ) xev.CallbackAction {
                _ = l;
                _ = c;
                _ = s;
                const data = userdata.?;
                data.result = result;
                if (data.frame != null) libcoro.xresume(data.frame.?);
                return .disarm;
            }
        };

        var data: Data = .{ .frame = libcoro.xframe() };
        const loop = self.exec.loop;
        var c: xev.Completion = .{};
        self.tcp.connect(loop, &c, addr, Data, &data, &Data.callback);

        try waitForCompletion(self.exec, &c);

        return data.result;
    }

    const ShutdownResult = xev.TCP.ShutdownError!void;
    pub fn shutdown(self: Self) ShutdownResult {
        const ResultT = ShutdownResult;
        const Data = struct {
            result: ResultT = undefined,
            frame: ?Frame = null,

            fn callback(
                userdata: ?*@This(),
                l: *xev.Loop,
                c: *xev.Completion,
                s: xev.TCP,
                result: ResultT,
            ) xev.CallbackAction {
                _ = l;
                _ = c;
                _ = s;
                const data = userdata.?;
                data.result = result;
                if (data.frame != null) libcoro.xresume(data.frame.?);
                return .disarm;
            }
        };

        var data: Data = .{ .frame = libcoro.xframe() };
        const loop = self.exec.loop;
        var c: xev.Completion = .{};
        self.tcp.shutdown(loop, &c, Data, &data, &Data.callback);

        try waitForCompletion(self.exec, &c);

        return data.result;
    }
};

fn Stream(comptime T: type, comptime StreamT: type, comptime options: xev.stream.Options) type {
    return struct {
        pub usingnamespace if (options.close) Closeable(T, StreamT) else struct {};
        pub usingnamespace if (options.read != .none) Readable(T, StreamT) else struct {};
        pub usingnamespace if (options.write != .none) Writeable(T, StreamT) else struct {};
    };
}

fn Closeable(comptime T: type, comptime StreamT: type) type {
    return struct {
        const Self = T;
        const CloseResult = xev.CloseError!void;
        pub fn close(self: Self) !void {
            const ResultT = CloseResult;
            const Data = struct {
                result: ResultT = undefined,
                frame: ?Frame = null,

                fn callback(
                    userdata: ?*@This(),
                    l: *xev.Loop,
                    c: *xev.Completion,
                    s: StreamT,
                    result: ResultT,
                ) xev.CallbackAction {
                    _ = l;
                    _ = c;
                    _ = s;
                    const data = userdata.?;
                    data.result = result;
                    if (data.frame != null) libcoro.xresume(data.frame.?);
                    return .disarm;
                }
            };

            var data: Data = .{ .frame = libcoro.xframe() };

            const loop = self.exec.loop;
            var c: xev.Completion = .{};
            self.stream().close(loop, &c, Data, &data, &Data.callback);

            try waitForCompletion(self.exec, &c);

            return data.result;
        }
    };
}

fn Readable(comptime T: type, comptime StreamT: type) type {
    return struct {
        const Self = T;
        const ReadResult = xev.ReadError!usize;
        pub fn read(self: Self, buf: xev.ReadBuffer) !usize {
            const ResultT = ReadResult;
            const Data = struct {
                result: ResultT = undefined,
                frame: ?Frame = null,

                fn callback(
                    userdata: ?*@This(),
                    l: *xev.Loop,
                    c: *xev.Completion,
                    s: StreamT,
                    b: xev.ReadBuffer,
                    result: ResultT,
                ) xev.CallbackAction {
                    _ = l;
                    _ = c;
                    _ = s;
                    _ = b;
                    const data = userdata.?;
                    data.result = result;
                    if (data.frame != null) libcoro.xresume(data.frame.?);
                    return .disarm;
                }
            };

            var data: Data = .{ .frame = libcoro.xframe() };

            const loop = self.exec.loop;
            var c: xev.Completion = .{};
            self.stream().read(loop, &c, buf, Data, &data, &Data.callback);

            try waitForCompletion(self.exec, &c);

            return data.result;
        }
    };
}

fn Writeable(comptime T: type, comptime StreamT: type) type {
    return struct {
        const Self = T;
        const WriteResult = xev.WriteError!usize;
        pub fn write(self: Self, buf: xev.WriteBuffer) !usize {
            const ResultT = WriteResult;
            const Data = struct {
                result: ResultT = undefined,
                frame: ?Frame = null,

                fn callback(
                    userdata: ?*@This(),
                    l: *xev.Loop,
                    c: *xev.Completion,
                    s: StreamT,
                    b: xev.WriteBuffer,
                    result: ResultT,
                ) xev.CallbackAction {
                    _ = l;
                    _ = c;
                    _ = s;
                    _ = b;
                    const data = userdata.?;
                    data.result = result;
                    if (data.frame != null) libcoro.xresume(data.frame.?);
                    return .disarm;
                }
            };

            var data: Data = .{ .frame = libcoro.xframe() };

            const loop = self.exec.loop;
            var c: xev.Completion = .{};
            self.stream().write(loop, &c, buf, Data, &data, &Data.callback);

            try waitForCompletion(self.exec, &c);
            return data.result;
        }
    };
}

pub const File = struct {
    const Self = @This();

    exec: *Executor,
    file: xev.File,

    pub usingnamespace Stream(Self, xev.File, .{
        .close = true,
        .read = .read,
        .write = .write,
        .threadpool = true,
    });

    pub fn init(exec: *Executor, file: xev.File) Self {
        return .{ .exec = exec, .file = file };
    }

    fn stream(self: Self) xev.File {
        return self.file;
    }

    const PReadResult = xev.ReadError!usize;
    pub fn pread(self: Self, buf: xev.ReadBuffer, offset: u64) !usize {
        const ResultT = PReadResult;
        const Data = struct {
            result: ResultT = undefined,
            frame: ?Frame = null,

            fn callback(
                userdata: ?*@This(),
                l: *xev.Loop,
                c: *xev.Completion,
                s: xev.File,
                b: xev.ReadBuffer,
                result: ResultT,
            ) xev.CallbackAction {
                _ = l;
                _ = c;
                _ = s;
                _ = b;
                const data = userdata.?;
                data.result = result;
                if (data.frame != null) libcoro.xresume(data.frame.?);
                return .disarm;
            }
        };

        var data: Data = .{ .frame = libcoro.xframe() };

        const loop = self.exec.loop;
        var c: xev.Completion = .{};
        self.file.pread(loop, &c, buf, offset, Data, &data, &Data.callback);

        try waitForCompletion(self.exec, &c);

        return data.result;
    }

    const PWriteResult = xev.WriteError!usize;
    pub fn pwrite(self: Self, buf: xev.WriteBuffer, offset: u64) !usize {
        const ResultT = PWriteResult;
        const Data = struct {
            result: ResultT = undefined,
            frame: ?Frame = null,

            fn callback(
                userdata: ?*@This(),
                l: *xev.Loop,
                c: *xev.Completion,
                s: xev.File,
                b: xev.WriteBuffer,
                result: ResultT,
            ) xev.CallbackAction {
                _ = l;
                _ = c;
                _ = s;
                _ = b;
                const data = userdata.?;
                data.result = result;
                if (data.frame != null) libcoro.xresume(data.frame.?);
                return .disarm;
            }
        };

        var data: Data = .{ .frame = libcoro.xframe() };

        const loop = self.exec.loop;
        var c: xev.Completion = .{};
        self.file.pwrite(loop, &c, buf, offset, Data, &data, &Data.callback);

        try waitForCompletion(self.exec, &c);

        return data.result;
    }
};

pub const Process = struct {
    const Self = @This();

    exec: *Executor,
    p: xev.Process,

    pub fn init(exec: *Executor, p: xev.Process) Self {
        return .{ .exec = exec, .p = p };
    }

    const WaitResult = xev.Process.WaitError!u32;
    pub fn wait(self: Self) !u32 {
        const Data = XCallback(WaitResult);
        var c: xev.Completion = .{};
        var data = Data.init();
        const loop = self.exec.loop;
        self.p.wait(loop, &c, Data, &data, &Data.callback);

        try waitForCompletion(self.exec, &c);

        return data.result;
    }
};

pub const AsyncNotification = struct {
    const Self = @This();

    exec: *Executor,
    notif: xev.Async,

    pub fn init(exec: *Executor, notif: xev.Async) Self {
        return .{ .exec = exec, .notif = notif };
    }

    const WaitResult = xev.Async.WaitError!void;
    pub fn wait(self: Self) !void {
        const Data = XCallback(WaitResult);

        const loop = self.exec.loop;
        var c: xev.Completion = .{};
        var data = Data.init();

        self.notif.wait(loop, &c, Data, &data, &Data.callback);

        try waitForCompletion(self.exec, &c);

        return data.result;
    }
};

pub const UDP = struct {
    const Self = @This();

    exec: *Executor,
    udp: xev.UDP,

    pub usingnamespace Stream(Self, xev.UDP, .{
        .close = true,
        .read = .none,
        .write = .none,
    });

    pub fn init(exec: *Executor, udp: xev.UDP) Self {
        return .{ .exec = exec, .udp = udp };
    }

    pub fn stream(self: Self) xev.UDP {
        return self.udp;
    }

    const ReadResult = xev.ReadError!usize;
    pub fn read(self: Self, buf: xev.ReadBuffer) !usize {
        const ResultT = ReadResult;
        const Data = struct {
            result: ResultT = undefined,
            frame: ?Frame = null,

            fn callback(
                userdata: ?*@This(),
                l: *xev.Loop,
                c: *xev.Completion,
                s: *xev.UDP.State,
                addr: std.net.Address,
                udp: xev.UDP,
                b: xev.ReadBuffer,
                result: ResultT,
            ) xev.CallbackAction {
                _ = l;
                _ = c;
                _ = s;
                _ = addr;
                _ = udp;
                _ = b;
                const data = userdata.?;
                data.result = result;
                if (data.frame != null) libcoro.xresume(data.frame.?);
                return .disarm;
            }
        };

        const loop = self.exec.loop;
        var s: xev.UDP.State = undefined;
        var c: xev.Completion = .{};
        var data: Data = .{ .frame = libcoro.xframe() };
        self.udp.read(loop, &c, &s, buf, Data, &data, &Data.callback);

        try waitForCompletion(self.exec, &c);

        return data.result;
    }

    const WriteResult = xev.WriteError!usize;
    pub fn write(self: Self, addr: std.net.Address, buf: xev.WriteBuffer) !usize {
        const ResultT = WriteResult;
        const Data = struct {
            result: ResultT = undefined,
            frame: ?Frame = null,

            fn callback(
                userdata: ?*@This(),
                l: *xev.Loop,
                c: *xev.Completion,
                s: *xev.UDP.State,
                udp: xev.UDP,
                b: xev.WriteBuffer,
                result: ResultT,
            ) xev.CallbackAction {
                _ = l;
                _ = c;
                _ = s;
                _ = udp;
                _ = b;
                const data = userdata.?;
                data.result = result;
                if (data.frame != null) libcoro.xresume(data.frame.?);
                return .disarm;
            }
        };

        const loop = self.exec.loop;
        var s: xev.UDP.State = undefined;
        var c: xev.Completion = .{};
        var data: Data = .{ .frame = libcoro.xframe() };
        self.udp.write(loop, &c, &s, addr, buf, Data, &data, &Data.callback);

        try waitForCompletion(self.exec, &c);

        return data.result;
    }
};

fn RunT(comptime Func: anytype) type {
    const T = @typeInfo(@TypeOf(Func)).Fn.return_type.?;
    return switch (@typeInfo(T)) {
        .ErrorUnion => |E| E.payload,
        else => T,
    };
}

fn XCallback(comptime ResultT: type) type {
    return struct {
        frame: ?Frame = null,
        result: ResultT = undefined,

        fn init() @This() {
            return .{ .frame = libcoro.xframe() };
        }

        fn callback(
            userdata: ?*@This(),
            _: *xev.Loop,
            _: *xev.Completion,
            result: ResultT,
        ) xev.CallbackAction {
            const data = userdata.?;
            data.result = result;
            if (data.frame != null) libcoro.xresume(data.frame.?);
            return .disarm;
        }
    };
}
