const std = @import("std");
const xev = @import("xev");

const FnSignature = @import("meta.zig").FnSignature;

pub fn Frame(comptime func: anytype) type {
    const Signature = FnSignature(func, null);
    return FrameEx(func, Signature.ArgsT);
}

pub fn FrameEx(comptime func: anytype, comptime argsT: type) type {
    return struct {
        const Self = @This();
        const Signature = FnSignature(func, argsT);
        const Task = struct {
            _task: xev.ThreadPool.Task = .{ .callback = &Self.run },
            event: std.Thread.ResetEvent = .{},
            args: Signature.ArgsT,
            result: Signature.ReturnT = undefined,
        };

        _task: *Task,

        fn run(task_: *xev.ThreadPool.Task) void {
            const task: *Task = @alignCast(@fieldParentPtr("_task", task_));
            task.result = @call(.auto, func, task.args);
            task.event.set();
        }

        pub fn await_(self: *Self) Signature.ReturnT {
            defer {
                AsyncThread.current.mutex.lock();
                AsyncThread.current.allocator.destroy(self._task);
                AsyncThread.current.mutex.unlock();
            }
            self._task.event.wait();
            return self._task.result;
        }
    };
}

pub fn asyncc(comptime func: anytype, args: FnSignature(func, null).ArgsT) !FrameEx(func, @TypeOf(args)) {
    return asyncGeneric(func, args);
}

pub fn asyncGeneric(comptime func: anytype, args: anytype) !FrameEx(func, @TypeOf(args)) {
    const FrameT = FrameEx(func, @TypeOf(args));

    AsyncThread.current.mutex.lock();
    defer AsyncThread.current.mutex.unlock();

    const task = try AsyncThread.current.allocator.create(FrameT.Task);
    task.* = .{
        .args = args,
    };

    AsyncThread.current.thread_pool.schedule(xev.ThreadPool.Batch.from(&task._task));
    return .{ ._task = task };
}

pub fn callBlocking(comptime func: anytype, args: FnSignature(func, null).ArgsT) @TypeOf(callBlockingGeneric(func, args)) {
    return callBlockingGeneric(func, args);
}

pub fn callBlockingGeneric(comptime func: anytype, args: anytype) FnSignature(func, @TypeOf(args)).ReturnT {
    return @call(.auto, func, args);
}

pub fn sleep(ms: u64) !void {
    std.time.sleep(ms * std.time.ns_per_ms);
}

pub const AsyncThread = struct {
    var current: AsyncThread = undefined;

    allocator: std.mem.Allocator,
    thread_pool: xev.ThreadPool,
    mutex: std.Thread.Mutex,

    pub fn main(allocator_: std.mem.Allocator, comptime func: anytype, args: anytype) !void {
        current = .{
            .allocator = allocator_,
            .thread_pool = xev.ThreadPool.init(.{}),
            .mutex = .{},
        };

        defer {
            current.thread_pool.shutdown();
            current.thread_pool.deinit();
        }

        return @call(.auto, func, args);
    }
};

pub const Notification = struct {
    inner: std.Thread.ResetEvent,

    pub fn init() !Notification {
        return .{ .inner = .{} };
    }

    pub fn notify(self: *Notification) !void {
        self.inner.set();
    }

    pub fn wait(self: *Notification) !void {
        self.inner.wait();
    }

    pub fn deinit(self: *Notification) void {
        self.inner.set();
        self.* = undefined;
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

    inner: std.fs.File,

    fn asFile(self: File) std.fs.File {
        return self.inner;
    }

    pub fn handle(self: File) std.fs.File.Handle {
        return self.inner.handle;
    }

    pub fn init(file_: std.fs.File) !File {
        return .{ .inner = file_ };
    }

    pub fn open(path: []const u8, flags: std.fs.File.OpenFlags) !File {
        return init(try std.fs.cwd().openFile(path, flags));
    }

    pub fn access(path: []const u8, flags: std.fs.File.OpenFlags) !void {
        return try std.fs.cwd().access(path, flags);
    }

    pub fn read(self: File, buf: []u8) !usize {
        return try self.inner.read(buf);
    }

    pub fn pread(self: File, buf: []u8, offset: u64) !usize {
        return try self.inner.pread(buf, offset);
    }

    pub fn write(self: File, buf: []const u8) !usize {
        return try self.inner.write(buf);
    }

    pub fn pwrite(self: File, buf: []const u8, offset: u64) !usize {
        return try self.inner.pwrite(buf, offset);
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
        return try self.inner.stat();
    }

    pub fn seekBy(self: File, offset: i64) !void {
        try self.inner.seekBy(offset);
    }

    pub fn seekTo(self: File, offset: u64) !void {
        try self.inner.seekTo(offset);
    }

    pub fn getPos(self: File) !u64 {
        return try self.inner.getPos();
    }

    pub fn getEndPos(self: File) !u64 {
        return try self.inner.getEndPos();
    }
};

pub const Mutex = std.Thread.Mutex;
