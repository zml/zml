//! libcoro mutable state:
//! * ThreadState
//!   * current_coro: set in ThreadState.switchTo
//!   * next_coro_id: set in ThreadState.nextCoroId
//!   * suspend_block: set in xsuspendBlock, cleared in ThreadState.switchIn
//! * Coro
//!   * resumer: set in ThreadState.switchTo
//!   * status:
//!     * Active, Suspended: set in ThreadState.switchTo
//!     * Done: set in runcoro
//!   * id.invocation: incremented in ThreadState.switchTo
const std = @import("std");
const builtin = @import("builtin");

const stdx = @import("stdx");

const base = @import("coro_base.zig");
const Executor = @import("executor.zig").Executor;
const stack = @import("stack.zig");
pub const StackT = stack.Stack;

// Public API
// ============================================================================
pub const Error = error{
    StackTooSmall,
    StackOverflow,
    SuspendFromMain,
};
pub const Frame = *Coro;

/// Await the coroutine(s).
/// frame: FrameT: runs the coroutine until done and returns its return value.
pub fn xawait(frame: anytype) @TypeOf(frame).Signature.ReturnT {
    const f = frame.frame();
    while (f.status != .Done) xsuspend();
    std.debug.assert(f.status == .Done);
    return frame.xreturned();
}

/// Create a coroutine and start it
/// stack is {null, usize, StackT}. If null or usize, initEnv must have been
/// called with a default stack allocator.
pub fn xasync(func: anytype, args: anytype, stack_: stack.Stack) !FrameT(func, @TypeOf(args)) {
    const FrameType = CoroT.fromFunc(func, @TypeOf(args));
    const framet = try FrameType.init(args, stack_);
    const frame = framet.frame();
    xresume(frame);
    return FrameType.wrap(frame);
}

pub const FrameT = CoroT.fromFunc;

/// True if within a coroutine, false if at top-level.
pub fn inCoro() bool {
    return thread_state.inCoro();
}

/// Returns the currently running coroutine
pub fn xframe() Frame {
    return thread_state.current();
}

/// Resume the passed coroutine, suspending the current coroutine.
/// When the resumed coroutine suspends, this call will return.
/// Note: When the resumed coroutine returns, control will switch to its parent
/// (i.e. its original resumer).
/// frame: Frame or FrameT
pub fn xresume(frame: anytype) void {
    const f = frame.frame();
    thread_state.switchIn(f);
}

/// Suspend the current coroutine, yielding control back to its
/// resumer. Returns when the coroutine is resumed.
/// Must be called from within a coroutine (i.e. not the top level).
pub fn xsuspend() void {
    xsuspendSafe() catch |e| {
        log(.err, "{any}", .{e});
        @panic("xsuspend");
    };
}

pub fn xsuspendBlock(comptime func: anytype, args: anytype) void {
    const Signature = stdx.meta.FnSignature(func, @TypeOf(args));
    const Callback = struct {
        func: *const Signature.FuncT,
        args: Signature.ArgsT,
        fn cb(ud: ?*anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ud));
            @call(.auto, self.func, self.args);
        }
    };
    var cb = Callback{ .func = func, .args = args };
    thread_state.suspend_block = .{ .func = Callback.cb, .data = @ptrCast(&cb) };
    xsuspend();
}

pub fn xsuspendSafe() Error!void {
    if (thread_state.current_coro == null) {
        return Error.SuspendFromMain;
    }
    const coro = thread_state.current_coro.?;
    try StackOverflow.check(coro);
    thread_state.switchOut(coro.resumer);
}

const Coro = struct {
    /// Coroutine status
    const Status = enum {
        Start,
        Suspended,
        Active,
        Done,
    };
    const Signature = VoidSignature;

    /// Function to run in the coroutine
    func: *const fn () void,
    /// Coroutine stack
    stack: stack.Stack,
    /// Architecture-specific implementation
    impl: base.Coro,
    /// The coroutine that will be yielded to upon suspend
    resumer: *Coro = undefined,
    /// Current status
    status: Status = .Start,
    /// Coro id, {thread, coro id, invocation id}
    id: CoroId.InvocationId,
    /// Caller-specified coro-local storage
    storage: ?*anyopaque = null,

    fn init(func: *const fn () void, stack_: stack.Stack, storage: ?*anyopaque) !Frame {
        return initFromStack(func, stack_, storage);
    }

    pub fn deinit(_: Coro) void {
        // empty
    }

    fn initFromStack(func: *const fn () void, stack_: stack.Stack, storage: ?*anyopaque) !Frame {
        // try StackOverflow.setMagicNumber(stack.full);
        var stack__ = stack_;
        const coro = try stack__.push(Coro);
        const base_coro = try base.Coro.init(&runcoro, stack__.remaining());
        coro.* = .{
            .func = func,
            .impl = base_coro,
            .stack = stack__,
            .storage = storage,
            .id = thread_state.newCoroId(),
        };
        return coro;
    }

    pub fn frame(self: *Coro) Frame {
        return self;
    }

    fn runcoro(from: *base.Coro, this: *base.Coro) callconv(.C) noreturn {
        const from_coro: *Coro = @fieldParentPtr("impl", from);
        const this_coro: *Coro = @fieldParentPtr("impl", this);
        log(.debug, "coro start {any}", .{this_coro.id});
        @call(.auto, this_coro.func, .{});
        this_coro.status = .Done;
        thread_state.switchOut(from_coro);

        // Never returns
        stdx.debug.panic("Cannot resume an already completed coroutine {any}", .{this_coro.id});
    }

    pub fn getStorage(self: Coro, comptime T: type) *T {
        return @ptrCast(@alignCast(self.storage));
    }

    pub fn format(self: Coro, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("Coro{{.id = {any}, .status = {s}}}", .{
            self.id,
            @tagName(self.status),
        });
    }
};

const VoidSignature = CoroT.Signature.init((struct {
    fn func() void {}
}).func, .{});

const CoroT = struct {
    fn fromFunc(comptime Func: anytype, comptime ArgsT: ?type) type {
        return fromSig(stdx.meta.FnSignature(Func, ArgsT));
    }

    fn fromSig(comptime Sig: stdx.meta.Signature) type {
        // Stored in the coro stack
        const InnerStorage = struct {
            args: Sig.ArgsT,
            /// Values that are produced during coroutine execution
            retval: Sig.ReturnT = undefined,
        };

        return struct {
            const Self = @This();
            pub const Signature = Sig;

            _frame: Frame,

            /// Create a Coro
            /// self and stack pointers must remain stable for the lifetime of
            /// the coroutine.
            fn init(args: Sig.ArgsT, stack_: StackT) !Self {
                var coro_stack = stack_;
                const inner = try coro_stack.push(InnerStorage);
                inner.* = .{
                    .args = args,
                };
                return .{ ._frame = try Coro.initFromStack(wrapfn, coro_stack, inner) };
            }

            pub fn wrap(_frame: Frame) Self {
                return .{ ._frame = _frame };
            }

            pub fn deinit(self: Self) void {
                self._frame.deinit();
            }

            pub fn status(self: Self) Coro.Status {
                return self._frame.status;
            }

            pub fn frame(self: Self) Frame {
                return self._frame;
            }

            // Coroutine functions.
            //
            // When considering basic coroutine execution, the coroutine state
            // machine is:
            // * Start
            // * Start->xresume->Active
            // * Active->xsuspend->Suspended
            // * Active->(fn returns)->Done
            // * Suspended->xresume->Active
            //
            // Note that actions in the Active* states are taken from within the
            // coroutine. All other actions act upon the coroutine from the
            // outside.

            /// Returns the value the coroutine returned
            pub fn xreturned(self: Self) Sig.ReturnT {
                const storage = self._frame.getStorage(InnerStorage);
                return storage.retval;
            }

            fn wrapfn() void {
                const storage = thread_state.currentStorage(InnerStorage);
                storage.retval = @call(
                    .auto,
                    Sig.Func.Value,
                    storage.args,
                );
            }
        };
    }
};

/// Estimates the remaining stack size in the currently running coroutine
pub noinline fn remainingStackSize() usize {
    var dummy: usize = 0;
    dummy += 1;
    const addr = @intFromPtr(&dummy);

    // Check if the stack was already overflowed
    const current = xframe();
    StackOverflow.check(current) catch return 0;

    // Check if the stack is currently overflowed
    const bottom = @intFromPtr(current.stack.ptr);
    if (addr < bottom) {
        return 0;
    }

    // Debug check that we're actually in the stack
    const top = @intFromPtr(current.stack.ptr + current.stack.len);
    std.debug.assert(addr < top); // should never have popped beyond the top

    return addr - bottom;
}

// ============================================================================

/// Thread-local coroutine runtime
threadlocal var thread_state: ThreadState = .{};

const ThreadState = struct {
    root_coro: Coro = .{
        .func = undefined,
        .stack = undefined,
        .impl = undefined,
        .id = CoroId.InvocationId.root(),
    },
    current_coro: ?Frame = null,
    next_coro_id: usize = 1,
    suspend_block: ?SuspendBlock = null,

    const SuspendBlock = struct {
        func: *const fn (?*anyopaque) void,
        data: ?*anyopaque,

        fn run(self: SuspendBlock) void {
            @call(.auto, self.func, .{self.data});
        }
    };

    /// Called from resume
    fn switchIn(self: *ThreadState, target: Frame) void {
        log(.debug, "coro resume {any} from {any}", .{ target.id, self.current().id });

        // Switch to target, setting this coro as the resumer.
        self.switchTo(target, true);

        // Suspend within target brings control back here
        // If a suspend block has been set, pop and run it.
        if (self.suspend_block) |block| {
            self.suspend_block = null;
            block.run();
        }
    }

    /// Called from suspend
    fn switchOut(self: *ThreadState, target: Frame) void {
        log(.debug, "coro suspend {any} to {any}", .{ self.current().id, target.id });
        self.switchTo(target, false);
    }

    fn switchTo(self: *ThreadState, target: Frame, set_resumer: bool) void {
        const suspender = self.current();
        if (suspender == target) {
            return;
        }
        if (suspender.status != .Done) {
            suspender.status = .Suspended;
        }
        if (set_resumer) {
            target.resumer = suspender;
        }
        target.status = .Active;
        target.id.incr();
        self.current_coro = target;
        target.impl.resumeFrom(&suspender.impl);
    }

    fn newCoroId(self: *ThreadState) CoroId.InvocationId {
        const out = CoroId.InvocationId.init(.{
            .coro = self.next_coro_id,
        });
        self.next_coro_id += 1;
        return out;
    }

    fn current(self: *ThreadState) Frame {
        return self.current_coro orelse &self.root_coro;
    }

    fn inCoro(self: *ThreadState) bool {
        return self.current() != &self.root_coro;
    }

    /// Returns the storage of the currently running coroutine
    fn currentStorage(self: *ThreadState, comptime T: type) *T {
        return self.current_coro.?.getStorage(T);
    }
};

const CoroId = struct {
    coro: usize,

    pub const InvocationId = if (builtin.mode == .Debug) DebugInvocationId else DummyInvocationId;

    const DummyInvocationId = struct {
        fn init(id: CoroId) @This() {
            _ = id;
            return .{};
        }
        fn root() @This() {
            return .{};
        }
        fn incr(self: *@This()) void {
            _ = self;
        }
    };

    const DebugInvocationId = struct {
        id: CoroId,
        invocation: i64 = -1,

        fn init(id: CoroId) @This() {
            return .{ .id = id };
        }

        fn root() @This() {
            return .{ .id = .{ .coro = 0 } };
        }

        fn incr(self: *@This()) void {
            self.invocation += 1;
        }

        pub fn format(self: @This(), comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            _ = fmt;
            _ = options;
            try writer.print("CoroId{{.cid={d}, .i={d}}}", .{
                self.id.coro,
                self.invocation,
            });
        }
    };
};

const StackOverflow = struct {
    const magic_number: usize = 0x5E574D6D;

    fn check(_: Frame) !void {
        // const stack = coro.stack.ptr;
        // const sp = coro.impl.stack_pointer;
        // const magic_number_ptr: *usize = @ptrCast(stack);
        // if (magic_number_ptr.* != magic_number or //
        //     @intFromPtr(sp) < @intFromPtr(stack))
        // {
        //     return Error.StackOverflow;
        // }
    }

    fn setMagicNumber(_: stack.Stack) !void {
        // if (stack.len <= @sizeOf(usize)) {
        //     return Error.StackTooSmall;
        // }
        // const magic_number_ptr: *usize = @ptrCast(stack.ptr);
        // magic_number_ptr.* = magic_number;
    }
};

var test_idx: usize = 0;
var test_steps = [_]usize{0} ** 8;

fn testSetIdx(val: usize) void {
    test_steps[test_idx] = val;
    test_idx += 1;
}

fn testFn() void {
    std.debug.assert(remainingStackSize() > 2048);
    testSetIdx(2);
    xsuspend();
    testSetIdx(4);
    xsuspend();
    testSetIdx(6);
}

test "basic suspend and resume" {
    // const allocator = std.testing.allocator;

    // const stack_size: usize = 1024 * 4;
    // const stack_ = try stackAlloc(allocator, stack_size);
    // defer allocator.free(stack);

    // testSetIdx(0);
    // const test_coro = try Coro.init(testFn, stack, false, null);

    // testSetIdx(1);
    // try std.testing.expectEqual(test_coro.status, .Start);
    // xresume(test_coro);
    // testSetIdx(3);
    // try std.testing.expectEqual(test_coro.status, .Suspended);
    // xresume(test_coro);
    // try std.testing.expectEqual(test_coro.status, .Suspended);
    // testSetIdx(5);
    // xresume(test_coro);
    // testSetIdx(7);

    // try std.testing.expectEqual(test_coro.status, .Done);

    // for (0..test_steps.len) |i| {
    //     try std.testing.expectEqual(i, test_steps[i]);
    // }
}

test "with values" {
    // const Test = struct {
    //     const Storage = struct {
    //         x: *usize,
    //     };
    //     fn coroInner(x: *usize) void {
    //         x.* += 1;
    //         xsuspend();
    //         x.* += 3;
    //     }
    //     fn coroWrap() void {
    //         const storage = xframe().getStorage(Storage);
    //         const x = storage.x;
    //         coroInner(x);
    //     }
    // };
    // var x: usize = 0;
    // var storage = Test.Storage{ .x = &x };

    // const allocator = std.testing.allocator;
    // const stack = try stackAlloc(allocator, null);
    // defer allocator.free(stack);
    // const coro = try Coro.init(Test.coroWrap, stack, false, @ptrCast(&storage));

    // try std.testing.expectEqual(storage.x.*, 0);
    // xresume(coro);
    // try std.testing.expectEqual(storage.x.*, 1);
    // xresume(coro);
    // try std.testing.expectEqual(storage.x.*, 4);
}

fn testCoroFnImpl(x: *usize) usize {
    x.* += 1;
    xsuspend();
    x.* += 3;
    xsuspend();
    return x.* + 10;
}

test "with CoroT" {
    // const allocator = std.testing.allocator;
    // const stack = try stackAlloc(allocator, null);
    // defer allocator.free(stack);

    // var x: usize = 0;

    // const CoroFn = CoroT.fromFunc(testCoroFnImpl, .{});
    // var coro = try CoroFn.init(.{&x}, stack, false);

    // try std.testing.expectEqual(x, 0);
    // xresume(coro);
    // try std.testing.expectEqual(x, 1);
    // xresume(coro);
    // try std.testing.expectEqual(x, 4);
    // xresume(coro);
    // try std.testing.expectEqual(x, 4);
    // try std.testing.expectEqual(coro.status(), .Done);
    // try std.testing.expectEqual(CoroFn.xreturned(coro), 14);
    // comptime try std.testing.expectEqual(CoroFn.Signature.ReturnT(), usize);
}

test "resume self" {
    xresume(xframe());
    try std.testing.expectEqual(1, 1);
}

pub fn log(comptime level: std.log.Level, comptime fmt: []const u8, args: anytype) void {
    if (comptime !std.log.logEnabled(level, .@"zml/async")) return;

    // Since this logs are to debug the async runtime, we want it to happen synchronously.
    std.log.defaultLog(level, .@"zml/async", fmt, args);
}
