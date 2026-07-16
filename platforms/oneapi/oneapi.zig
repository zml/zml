const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/oneapi");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_ONEAPI");
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
    // oneccl send/recv fail today, this makes it work.
    // _ = c.setenv("SYCL_UR_USE_LEVEL_ZERO_V2", "0", 1);
    // _ = c.setenv("UR_L0_USE_DRIVER_COUNTER_BASED_EVENTS", "0", 1);
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
    const sandbox_path = try r.rlocation("libpjrt_oneapi/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for oneAPI runtime", .{});
        return error.FileNotFound;
    };

    var driver_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const driver_path = try stdx.Io.Dir.path.bufJoinZ(&driver_path_buf, &.{ sandbox_path, "lib", "libze_intel_gpu.so.1" });
    setupOneAPIEnv(driver_path);

    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_oneapi.so" });
        break :blk pjrt.Api.loadFrom(path);
    };
}
