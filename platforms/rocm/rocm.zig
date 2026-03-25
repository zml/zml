const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/rocm");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_ROCM");
}

fn hasRocmDevices(io: std.Io) bool {
    inline for (&.{ "/dev/kfd", "/dev/dri" }) |path| {
        std.Io.Dir.accessAbsolute(io, path, .{ .read = true }) catch return false;
    }
    return true;
}

fn isTraced(io: std.Io) bool {
    const file = std.Io.Dir.openFileAbsolute(io, "/proc/self/status", .{}) catch return false;
    defer file.close(io);

    var read_buf: [1024]u8 = undefined;
    var reader = file.reader(io, &read_buf);
    var status_buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&status_buf);
    _ = reader.interface.streamRemaining(&writer) catch return false;

    var lines = std.mem.splitScalar(u8, writer.buffered(), '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "TracerPid:")) continue;
        const value = std.mem.trim(u8, line["TracerPid:".len..], " \t");
        const tracer_pid = std.fmt.parseUnsigned(u32, value, 10) catch return false;
        return tracer_pid != 0;
    }
    return false;
}

fn setupRocmEnv(io: std.Io, rocm_data_dir: []const u8) !void {
    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    _ = c.setenv("ROCM_PATH", try stdx.Io.Dir.path.bufJoinZ(&buf, &.{rocm_data_dir}), 1); // must be zero terminated
    if (isTraced(io)) {
        // ROCm's async execution path is not stable when the process is running under ptrace/strace.
        _ = c.setenv("HSA_ENABLE_INTERRUPT", "0", 0);
        _ = c.setenv("HIP_LAUNCH_BLOCKING", "1", 0);
        _ = c.setenv("AMD_SERIALIZE_KERNEL", "3", 0);
    }
    _ = c.setenv("AMD_SERIALIZE_KERNEL", "3", 0);

}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = allocator;
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }
    if (!hasRocmDevices(io)) {
        return error.Unavailable;
    }

    const r = try bazel.runfiles(bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libpjrt_rocm/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for ROCm runtime", .{});
        return error.FileNotFound;
    };

    try setupRocmEnv(io, sandbox_path);

    // We must load the PJRT plugin from the main thread.
    //
    // This is because libamdhip64.so use thread local storage as part of the static destructors...
    //
    // This destructor accesses a thread-local variable. If the destructor is
    // executed in a different thread than the one that originally called dlopen()
    // on the library, the thread-local storage (TLS) offset may be resolved
    // relative to the TLS base of the main thread, rather than the thread actually
    // executing the destructor. Accessing this variable results in a segmentation fault...
    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const lib_path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_rocm.so" });
        break :blk .loadFrom(lib_path);
    };
}
