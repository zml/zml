const std = @import("std");
const stdx = @import("stdx");
const xev = @import("xev").Dynamic;
const coro = @import("coro.zig");
const executor = @import("executor.zig");
const channel_mod = @import("channel.zig");
const aio = @import("asyncio.zig");
const stack = @import("stack.zig");

const XevThreadPool = @import("xev").ThreadPool;

pub const Condition = struct {
    inner: executor.Condition,

    pub fn init() Condition {
        return .{ .inner = executor.Condition.init(&AsyncThread.current.executor.exec) };
    }

    pub fn broadcast(self: *Condition) void {
        self.inner.broadcast();
    }

    pub fn signal(self: *Condition) void {
        self.inner.signal();
    }

    pub fn wait(self: *Condition) void {
        self.inner.wait();
    }
};

pub fn Frame(comptime func: anytype) type {
    const Signature = stdx.meta.FnSignature(func, null);
    return FrameExx(func, Signature.ArgsT, Signature.ReturnT);
}

pub fn FrameEx(comptime func: anytype, comptime argsT: type) type {
    const Signature = stdx.meta.FnSignature(func, argsT);
    return FrameExx(func, Signature.ArgsT, Signature.ReturnT);
}

fn FrameExx(comptime func: anytype, comptime argsT: type, comptime returnT: type) type {
    return struct {
        const Self = @This();
        const FrameT = coro.FrameT(func, argsT);

        inner: FrameT,

        pub const wait = await_;
        pub const await_ = awaitt;
        pub fn awaitt(self: *Self) returnT {
            defer {
                self.inner.deinit();
                AsyncThread.current.stack_allocator.destroy(&self.inner._frame.stack);
                self.* = undefined;
            }
            return coro.xawait(self.inner);
        }
    };
}

pub fn asyncc(comptime func: anytype, args: anytype) !FrameEx(func, @TypeOf(args)) {
    const Signature = stdx.meta.FnSignature(func, @TypeOf(args));
    const new_stack = try AsyncThread.current.stack_allocator.create();
    return .{
        .inner = try aio.xasync(func, @as(Signature.ArgsT, args), new_stack),
    };
}

pub fn callBlocking(comptime func: anytype, args: anytype) stdx.meta.FnSignature(func, @TypeOf(args)).ReturnT {
    const Signature = stdx.meta.FnSignature(func, @TypeOf(args));

    const TaskT = struct {
        const Self = @This();

        _task: XevThreadPool.Task = .{ .callback = &Self.run },

        event: threading.ResetEventSingle = .{},
        args: Signature.ArgsT,
        result: Signature.ReturnT = undefined,

        pub fn run(task_: *XevThreadPool.Task) void {
            const task: *Self = @alignCast(@fieldParentPtr("_task", task_));
            task.result = @call(.auto, func, task.args);
            task.event.set();
        }
    };

    var newtask: TaskT = .{
        .args = args,
    };
    AsyncThread.current.thread_pool.schedule(XevThreadPool.Batch.from(&newtask._task));
    newtask.event.wait();

    return newtask.result;
}

pub fn sleep(ms: u64) !void {
    try aio.sleep(AsyncThread.current.executor, ms);
}

pub const threading = struct {
    const Waiter = struct {
        frame: coro.Frame,
        thread: *const AsyncThread,
        next: ?*Waiter = null,
    };

    const WaiterQueue = stdx.queue.MPSC(Waiter);

    pub const ResetEventSingle = struct {
        const State = union(enum) {
            unset,
            waiting: *Waiter,
            set,

            const unset_state: State = .unset;
            const set_state: State = .set;
        };

        waiter: std.atomic.Value(*const State) = std.atomic.Value(*const State).init(&State.unset_state),

        pub fn isSet(self: *ResetEventSingle) bool {
            return self.waiter.load(.monotonic) == &State.set_state;
        }

        pub fn reset(self: *ResetEventSingle) void {
            self.waiter.store(&State.unset_state, .monotonic);
        }

        pub fn set(self: *ResetEventSingle) void {
            switch (self.waiter.swap(&State.set_state, .monotonic).*) {
                .waiting => |waiter| {
                    waiter.thread.waiters_queue.push(waiter);
                    waiter.thread.wake();
                },
                else => {},
            }
        }

        pub fn wait(self: *ResetEventSingle) void {
            var waiter: Waiter = .{
                .frame = coro.xframe(),
                .thread = AsyncThread.current,
            };
            var new_state: State = .{
                .waiting = &waiter,
            };
            if (self.waiter.cmpxchgStrong(&State.unset_state, &new_state, .monotonic, .monotonic) == null) {
                while (self.isSet() == false) {
                    coro.xsuspend();
                }
            }
        }
    };
};

pub const AsyncThread = struct {
    threadlocal var current: *AsyncThread = undefined;

    executor: *aio.Executor,
    stack_allocator: *stack.StackAllocator,
    loop: *xev.Loop,
    thread_pool: *XevThreadPool,
    async_notifier: *xev.Async,
    waiters_queue: *threading.WaiterQueue,

    pub fn wake(self: *const AsyncThread) void {
        self.async_notifier.notify() catch {};
    }

    fn wakerCallback(self: ?*AsyncThread, _: *xev.Loop, _: *xev.Completion, _: xev.Async.WaitError!void) xev.CallbackAction {
        while (self.?.waiters_queue.pop()) |waiter| {
            coro.xresume(waiter.frame);
        }
        return .rearm;
    }

    pub fn main(allocator: std.mem.Allocator, comptime mainFunc: fn () anyerror!void) !void {
        if (xev.dynamic) try xev.detect();
        var thread_pool = XevThreadPool.init(.{});
        defer {
            thread_pool.shutdown();
            thread_pool.deinit();
        }

        var loop = try xev.Loop.init(.{
            .thread_pool = &thread_pool,
        });
        defer loop.deinit();

        var executor_ = aio.Executor.init(&loop);

        var async_notifier = try xev.Async.init();
        defer async_notifier.deinit();

        var waiters_queue: threading.WaiterQueue = undefined;
        waiters_queue.init();

        var stack_allocator = stack.StackAllocator.init(allocator);
        defer stack_allocator.deinit();

        var asyncThread: AsyncThread = .{
            .executor = &executor_,
            .stack_allocator = &stack_allocator,
            .loop = &loop,
            .thread_pool = &thread_pool,
            .async_notifier = &async_notifier,
            .waiters_queue = &waiters_queue,
        };
        AsyncThread.current = &asyncThread;

        var c2: xev.Completion = undefined;
        async_notifier.wait(AsyncThread.current.loop, &c2, AsyncThread, AsyncThread.current, &AsyncThread.wakerCallback);

        // allocate the main coroutine stack, on the current thread's stack!
        var mainStackData: stack.Stack.Data = undefined;
        const mainStack = stack.Stack.init(&mainStackData);

        return try aio.run(&executor_, mainFunc, .{}, mainStack);
    }
};

pub fn getStdIn() !File {
    return File.init(std.io.getStdIn()) catch @panic("Unable to open stdin");
}

pub fn getStdOut() File {
    return File.init(std.io.getStdOut()) catch @panic("Unable to open stdout");
}

pub fn getStdErr() File {
    return File.init(std.io.getStdErr()) catch @panic("Unable to open stderr");
}

pub const File = struct {
    pub const SeekError = stdx.meta.FnSignature(File.seekTo, null).ReturnErrorSet.? || stdx.meta.FnSignature(File.seekBy, null).ReturnErrorSet.?;
    pub const GetSeekPosError = SeekError || stdx.meta.FnSignature(File.stat, null).ReturnErrorSet.?;
    pub const Reader = std.io.GenericReader(File, stdx.meta.FnSignature(File.read, null).ReturnErrorSet.?, File.read);
    pub const Writer = std.io.GenericWriter(File, stdx.meta.FnSignature(File.write, null).ReturnErrorSet.?, File.write);
    pub const SeekableStream = std.io.SeekableStream(
        File,
        SeekError,
        GetSeekPosError,
        seekTo,
        seekBy,
        getPos,
        getEndPos,
    );

    _handle: std.fs.File.Handle,
    inner: aio.File,

    fn asFile(self: File) std.fs.File {
        return .{ .handle = self._handle };
    }

    pub fn handle(self: File) std.fs.File.Handle {
        return self._handle;
    }

    pub fn init(file_: std.fs.File) !File {
        return .{
            ._handle = file_.handle,
            .inner = aio.File.init(AsyncThread.current.executor, try xev.File.init(file_)),
        };
    }

    pub fn open(path: []const u8, flags: std.fs.File.OpenFlags) !File {
        return init(try callBlocking(std.fs.Dir.openFile, .{ std.fs.cwd(), path, flags }));
    }

    pub fn access(path: []const u8, flags: std.fs.File.OpenFlags) !void {
        return try callBlocking(std.fs.Dir.access, .{ std.fs.cwd(), path, flags });
    }

    pub fn read(self: File, buf: []u8) !usize {
        // NOTE(Corentin): Early return is required to avoid error with xev on Linux with io_uring backend.
        if (buf.len == 0) {
            return 0;
        }

        return self.inner.read(.{ .slice = buf }) catch |err| switch (err) {
            // NOTE(Corentin): read shouldn't return an error on EOF, but a read length of 0 instead. This is to be iso with std.fs.File.
            error.EOF => 0,
            else => err,
        };
    }

    pub fn pread(self: File, buf: []u8, offset: u64) !usize {
        // NOTE(Corentin): Early return is required to avoid error with xev on Linux with io_uring backend.
        if (buf.len == 0) {
            return 0;
        }

        return self.inner.pread(.{ .slice = buf }, offset) catch |err| switch (err) {
            // NOTE(Corentin): pread shouldn't return an error on EOF, but a read length of 0 instead. This is to be iso with std.fs.File.
            error.EOF => 0,
            else => err,
        };
    }

    pub fn write(self: File, buf: []const u8) !usize {
        return self.inner.write(.{ .slice = buf });
    }

    pub fn pwrite(self: File, buf: []const u8, offset: u64) !usize {
        return self.inner.pwrite(.{ .slice = buf }, offset);
    }

    pub fn close(self: File) !void {
        return self.inner.close();
    }

    pub fn reader(self: File) Reader {
        return .{ .context = self };
    }

    pub fn seekableStream(file: File) SeekableStream {
        return .{ .context = file };
    }

    pub fn writer(self: File) Writer {
        return .{ .context = self };
    }

    pub fn stat(self: File) !std.fs.File.Stat {
        return try callBlocking(std.fs.File.stat, .{self.asFile()});
    }

    pub fn seekBy(self: File, offset: i64) !void {
        try callBlocking(std.fs.File.seekBy, .{ self.asFile(), offset });
    }

    pub fn seekTo(self: File, offset: u64) !void {
        try callBlocking(std.fs.File.seekTo, .{ self.asFile(), offset });
    }

    pub fn getPos(self: File) !u64 {
        return try callBlocking(std.fs.File.getPos, .{self.asFile()});
    }

    pub fn getEndPos(self: File) !u64 {
        return try callBlocking(std.fs.File.getEndPos, .{self.asFile()});
    }
};

pub const Socket = struct {
    pub fn Listener(comptime T: type) type {
        return struct {
            const Self = @This();

            inner: T.Inner,

            pub fn accept(self: *Self) !T {
                return .{ .inner = try self.inner.accept() };
            }

            pub fn close(self: *Self) !void {
                return self.inner.close();
            }

            pub fn deinit(self: *Self) !void {
                self.inner.shutdown();
            }
        };
    }

    pub const TCP = struct {
        const Inner = aio.TCP;

        pub const Reader = std.io.GenericReader(TCP, stdx.meta.FnSignature(TCP.read, null).ReturnErrorSet.?, TCP.read);
        pub const Writer = std.io.GenericWriter(TCP, stdx.meta.FnSignature(TCP.write, null).ReturnErrorSet.?, TCP.write);

        inner: aio.TCP,

        pub fn listen(addr: std.net.Address) !Listener(TCP) {
            var self: Listener(TCP) = .{
                .inner = aio.TCP.init(AsyncThread.current.executor, try xev.TCP.init(addr)),
            };
            try self.inner.tcp.bind(addr);
            try self.inner.tcp.listen(1024);
            return self;
        }

        pub fn deinit(self: *TCP) void {
            self.inner.shutdown();
        }

        pub fn accept(self: *TCP) !TCP {
            return .{ .inner = try self.inner.accept() };
        }

        pub fn connect(addr: std.net.Address) !TCP {
            var self: TCP = .{
                .inner = aio.TCP.init(AsyncThread.current.executor, try xev.TCP.init(addr)),
            };
            try self.inner.connect(addr);
            return self;
        }

        pub fn read(self: TCP, buf: []u8) !usize {
            return self.inner.read(.{ .slice = buf });
        }

        pub fn write(self: TCP, buf: []const u8) !usize {
            return self.inner.write(.{ .slice = buf });
        }

        pub fn close(self: TCP) !void {
            // defer self.* = undefined;
            return self.inner.close();
        }

        pub fn reader(self: TCP) Reader {
            return .{ .context = self };
        }

        pub fn writer(self: TCP) Writer {
            return .{ .context = self };
        }
    };

    pub const UDP = struct {
        const Inner = aio.TCP;

        pub const Reader = std.io.GenericReader(UDP, stdx.meta.FnSignature(UDP.read, null).ReturnErrorSet.?, UDP.read);
        pub const WriterContext = struct {
            file: UDP,
            addr: std.net.Address,
        };
        pub const Writer = std.io.GenericWriter(WriterContext, stdx.meta.FnSignature(UDP.write, null).ReturnErrorSet.?, struct {
            fn callBlocking(self: WriterContext, buf: []const u8) !usize {
                return self.file.write(self.addr, buf);
            }
        }.call);

        inner: aio.UDP,

        pub fn listen(addr: std.net.Address) !Listener(UDP) {
            var self: Listener(UDP) = .{
                .inner = aio.UDP.init(AsyncThread.current.executor, try xev.UDP.init(addr)),
            };
            try self.inner.udp.bind(addr);
            try self.inner.udp.listen(1024);
            return self;
        }

        pub fn read(self: UDP, buf: []u8) !usize {
            return self.inner.read(.{ .slice = buf });
        }

        pub fn write(self: UDP, addr: std.net.Address, buf: []const u8) !usize {
            return self.inner.write(addr, .{ .slice = buf });
        }

        pub fn close(self: *UDP) !void {
            defer self.* = undefined;
            return self.inner.close();
        }

        pub fn reader(self: File) Reader {
            return .{ .context = self };
        }

        pub fn writer(self: File, addr: std.net.Address) Writer {
            return .{
                .context = .{
                    .file = self,
                    .addr = addr,
                },
            };
        }
    };
};

pub fn Channel(comptime T: type, capacity: usize) type {
    return struct {
        const Self = @This();
        const Inner = channel_mod.Channel(T, capacity);

        inner: Inner,

        pub fn init() Self {
            return .{ .inner = Inner.init(&AsyncThread.current.executor.exec) };
        }

        pub fn initWithLen(len: usize) Self {
            return .{ .inner = Inner.initWithLen(&AsyncThread.current.executor.exec, len) };
        }

        pub fn close(self: *Self) void {
            self.inner.close();
        }

        pub fn try_send(self: *Self, val: T) bool {
            return self.inner.try_send(val);
        }

        pub fn send(self: *Self, val: T) void {
            self.inner.send(val);
        }

        pub fn recv(self: *Self) ?T {
            return self.inner.recv();
        }

        pub fn try_recv(self: *Self) ?T {
            return self.inner.try_recv();
        }
    };
}

pub fn channel(comptime T: type, len: usize, comptime capacity: usize) Channel(T, capacity) {
    return Channel(T, capacity).initWithLen(len);
}

pub const Mutex = struct {
    const VoidChannel = Channel(void, 1);

    inner: VoidChannel,

    pub fn init() Mutex {
        return .{ .inner = VoidChannel.init() };
    }

    pub fn lock(self: *Mutex) void {
        self.inner.send({});
    }

    pub fn unlock(self: *Mutex) void {
        _ = self.inner.recv();
    }
};

pub const LogFn = fn (
    comptime message_level: std.log.Level,
    comptime scope: @TypeOf(.enum_literal),
    comptime format: []const u8,
    args: anytype,
) void;

pub fn logFn(comptime fallbackLogFn: LogFn) LogFn {
    return struct {
        const Self = @This();

        var mu: ?Mutex = null;

        pub fn call(
            comptime message_level: std.log.Level,
            comptime scope: @TypeOf(.enum_literal),
            comptime format: []const u8,
            args: anytype,
        ) void {
            if (coro.inCoro() == false) {
                return fallbackLogFn(message_level, scope, format, args);
            }

            const level_txt = comptime message_level.asText();
            const prefix2 = if (scope == .default) ": " else "(" ++ @tagName(scope) ++ "): ";
            const stderr = getStdErr().writer();
            var bw = std.io.bufferedWriter(stderr);
            const writer = bw.writer();

            var mutex = Self.mu orelse blk: {
                Self.mu = Mutex.init();
                break :blk Self.mu.?;
            };
            mutex.lock();
            defer mutex.unlock();
            nosuspend {
                writer.print(level_txt ++ prefix2 ++ format ++ "\n", args) catch return;
                bw.flush() catch return;
            }
        }
    }.call;
}
