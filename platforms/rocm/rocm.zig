const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const platforms_options = @import("platforms/options");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/rocm");

pub fn isEnabled() bool {
    return platforms_options.rocm_enabled;
}

fn hasRocmDevices(io: std.Io) bool {
    inline for (&.{ "/dev/kfd", "/dev/dri" }) |path| {
        std.Io.Dir.accessAbsolute(io, path, .{ .read = true }) catch return false;
    }
    return true;
}

fn setupRocmEnv(rocm_data_dir: []const u8) !void {
    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    _ = c.setenv("ROCM_PATH", try stdx.Io.Dir.path.bufJoinZ(&buf, &.{rocm_data_dir}), 1); // must be zero terminated

    // Use one HIP graph stream to reduce cross-stream synchronization overhead.
    _ = c.setenv("DEBUG_HIP_FORCE_GRAPH_QUEUES", "1", 0);
    // Share one hardware queue per device so auxiliary barriers stay on the kernel queue.
    _ = c.setenv("GPU_MAX_HW_QUEUES", "1", 0);
    // Disable rocprofiler queue interposition to avoid its dispatch overhead outside profiling.
    _ = c.setenv("ROCPROFILER_QUEUE_INTERPOSITION", "0", 0);
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
    const sandbox_path = try r.rlocation("libzml_rocm/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for ROCm runtime", .{});
        return error.FileNotFound;
    };

    try setupRocmEnv(sandbox_path);

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
        const lib_path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libzml_rocm.so" });
        break :blk .loadFrom(lib_path);
    };
}
