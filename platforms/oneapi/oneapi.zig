const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const platforms_options = @import("platforms/options");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/oneapi");

pub fn isEnabled() bool {
    return platforms_options.oneapi_enabled;
}

fn hasOneApiDevice(io: std.Io) bool {
    var dir = std.Io.Dir.openDirAbsolute(io, "/dev/dri", .{ .iterate = true }) catch return false;
    defer dir.close(io);

    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (std.mem.startsWith(u8, entry.name, "renderD")) {
            return true;
        }
    }

    return false;
}

fn setupOneAPIEnv(driver_path: [:0]const u8) void {
    _ = c.setenv("CCL_LOG_LEVEL", c.getenv("CCL_LOG_LEVEL") orelse "error", 1);
    _ = c.setenv("CCL_ATL_TRANSPORT", "ofi", 1);
    _ = c.setenv("FI_PROVIDER", "shm", 1);
    // collective-permute through oneccl recv/send often relies on sycl d2d memcpy underneath,
    // that doesn't work because of peer residency likely this:
    // https://github.com/intel/compute-runtime/issues/953
    _ = c.setenv("XLA_ONECCL_COLLECTIVE_PERMUTE_BYPASS_SYCL_P2P", "1", 1);
    _ = c.setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:*", 0);
    _ = c.setenv("ZE_ENABLE_ALT_DRIVERS", driver_path, 0);
}

pub fn load(_: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }

    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }

    if (!hasOneApiDevice(io)) {
        return error.Unavailable;
    }

    const r = try bazel.runfiles(bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libzml_oneapi/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for oneAPI runtime", .{});
        return error.FileNotFound;
    };

    var driver_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const driver_path = try stdx.Io.Dir.path.bufJoinZ(&driver_path_buf, &.{ sandbox_path, "lib", "libze_intel_gpu.so.1" });
    setupOneAPIEnv(driver_path);

    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libzml_oneapi.so" });
        break :blk pjrt.Api.loadFrom(path);
    };
}

/// Creates the PJRT client, keeping a SIGSEGV handler installed before it working.
///
/// The Intel GPU driver (NEO) installs a SIGSEGV handler for its CPU page fault
/// manager when the client creates its SYCL contexts. It chains to the handler it
/// replaces, such as the one zio uses to grow coroutine stacks on demand, but:
/// - it is installed without SA_ONSTACK, so a fault on an exhausted coroutine stack
///   cannot be delivered and the kernel kills the process;
/// - its chaining is not thread-safe: concurrent faults on several threads move a
///   shared index into its list of previous handlers past the end of that list.
/// So after creating the client, put a handler in front of NEO's that runs on the
/// alternate signal stack, sends faults next to the stack pointer straight to the
/// handler NEO replaced, and leaves the other faults to NEO.
pub fn createClient(api: *const pjrt.Api, create_options: []const pjrt.NamedValue) pjrt.ClientInitError!*pjrt.Client {
    if (comptime !fault_handler.supported) return pjrt.Client.init(api, create_options);

    var previous: std.posix.Sigaction = undefined;
    std.posix.sigaction(.SEGV, null, &previous);
    // Also when client creation fails: NEO's handler stays installed.
    defer {
        var current: std.posix.Sigaction = undefined;
        std.posix.sigaction(.SEGV, null, &current);
        fault_handler.install(previous, current);
    }
    return pjrt.Client.init(api, create_options);
}

const fault_handler = struct {
    const posix = std.posix;

    const supported = builtin.os.tag == .linux and builtin.cpu.arch == .x86_64;

    /// Faults from this far below the interrupted stack pointer to this far above it
    /// are stack faults. Above covers a frame being set up without stack probes.
    const below_sp = 4 * 1024;
    const above_sp = 64 * 1024;

    var installed: std.atomic.Value(bool) = .init(false);
    var stack_fault_action: posix.Sigaction = undefined;
    var other_fault_action: posix.Sigaction = undefined;

    fn install(previous: posix.Sigaction, current: posix.Sigaction) void {
        if (handlerAddress(current) == handlerAddress(previous)) return;
        // Only a handler that runs on the alternate signal stack, like zio's, is broken
        // by NEO's. This also skips SIG_DFL (0) and SIG_IGN (1).
        if (handlerAddress(previous) <= 1 or previous.flags & posix.SA.ONSTACK == 0) return;
        if (installed.swap(true, .acq_rel)) {
            log.warn("SIGSEGV handler replaced again by the oneAPI runtime, coroutine stack growth may crash", .{});
            return;
        }

        stack_fault_action = previous;
        other_fault_action = current;
        const action: posix.Sigaction = .{
            .handler = .{ .sigaction = handle },
            .mask = current.mask,
            .flags = posix.SA.SIGINFO | posix.SA.ONSTACK,
        };
        posix.sigaction(.SEGV, &action, null);
    }

    fn handlerAddress(action: posix.Sigaction) usize {
        return @intFromPtr(action.handler.handler);
    }

    fn handle(sig: posix.SIG, info: *const posix.siginfo_t, ctx: ?*anyopaque) callconv(.c) void {
        const action = if (isStackFault(info, ctx)) &stack_fault_action else &other_fault_action;
        forward(action, sig, info, ctx);
    }

    fn isStackFault(info: *const posix.siginfo_t, ctx: ?*anyopaque) bool {
        const cpu = std.debug.cpu_context.fromPosixSignalContext(ctx) orelse return false;
        const sp: usize = @intCast(cpu.gprs.get(.rsp));
        const addr = @intFromPtr(info.fields.sigfault.addr);
        return addr >= sp -| below_sp and addr < sp +| above_sp;
    }

    /// Runs the handler of `action` as if the kernel had delivered the signal to it.
    fn forward(action: *const posix.Sigaction, sig: posix.SIG, info: *const posix.siginfo_t, ctx: ?*anyopaque) void {
        if (action.flags & posix.SA.SIGINFO != 0) {
            if (action.handler.sigaction) |handler| return handler(sig, info, ctx);
        } else if (handlerAddress(action.*) > 1) {
            return action.handler.handler.?(sig);
        }
        // Default action: restore it, so the faulting instruction faults again on return.
        posix.sigaction(sig, action, null);
    }
};
