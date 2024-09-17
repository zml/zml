const std = @import("std");
const xev = @import("xev");
const libcoro = @import("libcoro");
const aio = libcoro.asyncio;

/// Normalize from a real tuple to a generic tuple. This is needed because
/// real tuples are reifed tuples are not the same.
fn NormalizedTuple(comptime T: type) type {
    const ti = @typeInfo(T).Struct;
    var types: [ti.fields.len]type = undefined;
    inline for (ti.fields, 0..) |field, i| {
        types[i] = field.type;
    }
    return std.meta.Tuple(&types);
}

pub fn FnSignature(comptime func: anytype, comptime argsT: ?type) type {
    return struct {
        pub const FuncT = if (@TypeOf(func) == type) func else @TypeOf(func);
        pub const ArgsT = blk: {
            if (@typeInfo(FuncT).Fn.params.len == 0) {
                break :blk @TypeOf(.{});
            }
            break :blk argsT orelse std.meta.ArgsTuple(FuncT);
        };
        pub const ReturnT = @TypeOf(@call(.auto, func, @as(ArgsT, undefined)));
        pub const ReturnPayloadT = blk: {
            break :blk switch (@typeInfo(ReturnT)) {
                .ErrorUnion => |u| u.payload,
                else => ReturnT,
            };
        };
        pub const ReturnErrorSet: ?type = blk: {
            break :blk switch (@typeInfo(ReturnT)) {
                .ErrorUnion => |u| u.error_set,
                else => null,
            };
        };
    };
}

pub fn Frame(comptime func: anytype) type {
    const Signature = FnSignature(func, null);
    return FrameEx(func, Signature.ArgsT);
}

pub fn FrameEx(comptime func: anytype, comptime argsT: type) type {
    return FrameExx(func, argsT);
}

fn FrameExx(comptime func: anytype, comptime argsT: type) type {
    return struct {
        const Self = @This();
        const Signature = FnSignature(func, argsT);
        const FrameT = libcoro.FrameT(func, .{ .ArgsT = Signature.ArgsT });

        inner: FrameT,

        pub fn await_(self: *Self) Signature.ReturnT {
            defer {
                self.inner.deinit();
                self.* = undefined;
            }
            return libcoro.xawait(self.inner);
        }

        fn from(other: anytype) !Self {
            return .{ .inner = FrameT.wrap(other.frame()) };
        }
    };
}

pub fn async_(comptime func: anytype, args: anytype) !FrameEx(func, @TypeOf(args)) {
    const frame = try aio.xasync(func, args, null);
    return FrameEx(func, @TypeOf(args)).from(frame);
}

pub fn call(comptime func: anytype, args: std.meta.ArgsTuple(@TypeOf(func))) @TypeOf(callGeneric(func, args)) {
    return callGeneric(func, args);
}

pub fn callGeneric(comptime func: anytype, args: anytype) FnSignature(func, @TypeOf(args)).ReturnT {
    const Signature = FnSignature(func, @TypeOf(args));

    const TaskT = struct {
        const Self = @This();

        _task: xev.ThreadPool.Task = .{ .callback = &Self.run },

        notif: Notification,
        args: *const Signature.ArgsT,
        result: Signature.ReturnT = undefined,

        pub fn run(task_: *xev.ThreadPool.Task) void {
            const task: *Self = @alignCast(@fieldParentPtr("_task", task_));
            task.result = @call(.auto, func, task.args.*);
            task.notif.notify() catch @panic("Unable to notify");
        }
    };

    var newtask: TaskT = .{
        .notif = Notification.init() catch @panic("Notification.init failed"),
        .args = &args,
    };
    defer newtask.notif.deinit();

    AsyncThread.current.thread_pool.schedule(xev.ThreadPool.Batch.from(&newtask._task));
    newtask.notif.wait() catch @panic("Unable to wait for notification");
    return newtask.result;
}

pub fn tick() void {
    AsyncThread.current.executor.exec.tick();
}

pub fn sleep(ms: u64) !void {
    try aio.sleep(AsyncThread.current.executor, ms);
}

pub const Notification = struct {
    inner: aio.AsyncNotification,

    pub fn init() !Notification {
        return .{
            .inner = aio.AsyncNotification.init(AsyncThread.current.executor, try xev.Async.init()),
        };
    }

    pub fn notify(self: *Notification) !void {
        try self.inner.notif.notify();
    }

    pub fn wait(self: *Notification) !void {
        try self.inner.wait();
    }

    pub fn deinit(self: *Notification) void {
        self.inner.notif.deinit();
        self.* = undefined;
    }
};

pub const AsyncThread = struct {
    threadlocal var current: AsyncThread = undefined;

    executor: *aio.Executor,
    loop: *xev.Loop,
    thread_pool: *xev.ThreadPool,

    pub fn main(allocator: std.mem.Allocator, comptime func: anytype, args: anytype) !FnSignature(func, NormalizedTuple(@TypeOf(args))).ReturnPayloadT {
        const Signature = FnSignature(func, NormalizedTuple(@TypeOf(args)));

        var thread_pool = xev.ThreadPool.init(.{});
        defer {
            thread_pool.shutdown();
            thread_pool.deinit();
        }

        var loop = try xev.Loop.init(.{
            .thread_pool = &thread_pool,
        });
        defer loop.deinit();

        var executor = aio.Executor.init(&loop);

        AsyncThread.current = .{
            .executor = &executor,
            .loop = &loop,
            .thread_pool = &thread_pool,
        };

        aio.initEnv(.{
            .stack_allocator = allocator,
            .default_stack_size = 16 * 1024 * 1024,
        });

        if (Signature.ReturnErrorSet) |_| {
            return try aio.run(&executor, func, args, null);
        } else {
            return aio.run(&executor, func, args, null);
        }
    }
};

pub fn StdIn() !File {
    return File.init(std.io.getStdIn()) catch @panic("Unable to open stdin");
}

pub fn StdOut() File {
    return File.init(std.io.getStdOut()) catch @panic("Unable to open stdout");
}

pub fn StdErr() File {
    return File.init(std.io.getStdErr()) catch @panic("Unable to open stderr");
}

pub const File = struct {
    pub const SeekError = FnSignature(File.seekTo, null).ReturnErrorSet.? || FnSignature(File.seekBy, null).ReturnErrorSet.?;
    pub const GetSeekPosError = SeekError || FnSignature(File.stat, null).ReturnErrorSet.?;
    pub const Reader = std.io.GenericReader(File, FnSignature(File.read, null).ReturnErrorSet.?, File.read);
    pub const Writer = std.io.GenericWriter(File, FnSignature(File.write, null).ReturnErrorSet.?, File.write);
    pub const SeekableStream = std.io.SeekableStream(
        File,
        SeekError,
        GetSeekPosError,
        seekTo,
        seekBy,
        getPos,
        getEndPos,
    );

    inner: aio.File,

    fn asFile(self: File) std.fs.File {
        return .{ .handle = self.inner.file.fd };
    }

    pub fn init(file_: std.fs.File) !File {
        return .{ .inner = aio.File.init(AsyncThread.current.executor, try xev.File.init(file_)) };
    }

    pub fn fromFd(fd: std.fs.File.Handle) !File {
        return .{ .inner = aio.File.init(AsyncThread.current.executor, try xev.File.initFd(fd)) };
    }

    pub fn open(path: []const u8, flags: std.fs.File.OpenFlags) !File {
        return init(try call(std.fs.Dir.openFile, .{ std.fs.cwd(), path, flags }));
    }

    pub fn read(self: File, buf: []u8) !usize {
        // NOTE(Corentin): Early return is required to avoid error with xev on Linux with io_uring backend.
        if (buf.len == 0) return 0;

        return self.inner.read(.{ .slice = buf }) catch |err| switch (err) {
            // NOTE(Corentin): read shouldn't return an error on EOF, but a read length of 0 instead. This is to be iso with std.fs.File.
            error.EOF => 0,
            else => err,
        };
    }

    pub fn pread(self: File, buf: []u8, offset: u64) !usize {
        // NOTE(Corentin): Early return is required to avoid error with xev on Linux with io_uring backend.
        if (buf.len == 0) return 0;

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
        return try call(std.fs.File.stat, .{self.asFile()});
    }

    pub fn seekBy(self: File, offset: i64) !void {
        try call(std.fs.File.seekBy, .{ self.asFile(), offset });
    }

    pub fn seekTo(self: File, offset: u64) !void {
        try call(std.fs.File.seekTo, .{ self.asFile(), offset });
    }

    pub fn getPos(self: File) !u64 {
        return try call(std.fs.File.getPos, .{self.asFile()});
    }

    pub fn getEndPos(self: File) !u64 {
        return try call(std.fs.File.getEndPos, .{self.asFile()});
    }
};

pub const Socket = struct {
    pub const TCP = struct {
        pub const Reader = std.io.GenericReader(TCP, FnSignature(TCP.read, null).ReturnErrorSet.?, TCP.read);
        pub const Writer = std.io.GenericWriter(TCP, FnSignature(TCP.write, null).ReturnErrorSet.?, TCP.write);

        inner: aio.TCP,

        pub fn init(addr: std.net.Address) !TCP {
            return .{ .inner = aio.TCP.init(AsyncThread.current.executor, try xev.TCP.init(addr)) };
        }

        pub fn deinit(self: *TCP) void {
            self.inner.shutdown();
        }

        pub fn connect(self: *TCP, addr: std.net.Address) !void {
            return self.inner.connect(addr);
        }

        pub fn read(self: *TCP, buf: []u8) !usize {
            return self.inner.read(.{ .slice = buf });
        }

        pub fn write(self: *TCP, buf: []const u8) !usize {
            return self.inner.write(.{ .slice = buf });
        }

        pub fn close(self: *TCP) !void {
            defer self.* = undefined;
            return self.inner.close();
        }

        pub fn reader(self: File) Reader {
            return .{ .context = self };
        }

        pub fn writer(self: File) Writer {
            return .{ .context = self };
        }
    };

    pub const UDP = struct {
        pub const Reader = std.io.GenericReader(UDP, FnSignature(UDP.read, null).ReturnErrorSet.?, UDP.read);
        pub const WriterContext = struct {
            file: UDP,
            addr: std.net.Address,
        };
        pub const Writer = std.io.GenericWriter(WriterContext, FnSignature(UDP.write, null).ReturnErrorSet.?, struct {
            fn call(self: WriterContext, buf: []const u8) !usize {
                return self.file.write(self.addr, buf);
            }
        }.call);

        inner: aio.UDP,

        pub fn init(addr: std.net.Address) !UDP {
            return .{ .inner = aio.UDP.init(AsyncThread.current.executor, try xev.UDP.init(addr)) };
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

pub const Mutex = struct {
    const VoidChannel = libcoro.Channel(void, .{ .capacity = 1 });

    inner: VoidChannel,

    pub fn init() Mutex {
        return .{ .inner = VoidChannel.init(&AsyncThread.current.executor.exec) };
    }

    pub fn lock(self: *Mutex) !void {
        try self.inner.send({});
    }

    pub fn unlock(self: *Mutex) void {
        _ = self.inner.recv();
    }
};
