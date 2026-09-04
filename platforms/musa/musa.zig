const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/musa");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_MUSA");
}

fn setupMusaEnv(sandbox_path: []const u8) !void {
    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    _ = c.setenv("MUSA_PATH", try stdx.Io.Dir.path.bufJoinZ(&buf, &.{ sandbox_path, "lib", "musa-sdk" }), 1);
    // Keep the sandbox default while honoring explicit compatibility-test or
    // deployment overrides; the plugin validates the requested shim fail-closed.
    _ = c.setenv("XLA_MUSA_MUBLAS_SHIM_PATH", try stdx.Io.Dir.path.bufJoinZ(&buf, &.{ sandbox_path, "lib", "libxla_musa_mublas_shim.so.1" }), 0);
    _ = c.setenv("XLA_MUSA_MUDNN_SHIM_PATH", try stdx.Io.Dir.path.bufJoinZ(&buf, &.{ sandbox_path, "lib", "libxla_musa_mudnn_shim.so.1" }), 1);
    // muFFT is discovered adjacent to libpjrt_musa in the same sandbox. Do not
    // turn that default into an explicit FFT requirement: SDK 5.1's muFFT is
    // unqualified, and must not prevent loading an unrelated GEMM executable.
    // A user-supplied XLA_MUSA_MUFFT_SHIM_PATH remains untouched and fail-closed.
}

fn probeMusaRuntime(sandbox_path: []const u8) !void {
    var driver_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const driver_path = try stdx.Io.Dir.path.bufJoinZ(&driver_path_buf, &.{ sandbox_path, "lib", "libmusa.so.1" });
    _ = std.c.dlopen(driver_path, .{ .NOW = true, .GLOBAL = true, .NODELETE = true }) orelse {
        const msg = std.c.dlerror();
        if (msg) |err_msg| {
            log.warn("Failed to load MUSA driver from {s}: {s}", .{ driver_path, std.mem.span(err_msg) });
        } else {
            log.warn("Failed to load MUSA driver from {s}", .{driver_path});
        }
        return error.Unavailable;
    };

    var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libmusart.so.5" });
    _ = std.c.dlopen(path, .{ .NOW = true, .GLOBAL = true, .NODELETE = true }) orelse {
        const msg = std.c.dlerror();
        if (msg) |err_msg| {
            log.warn("Failed to load MUSA runtime from {s}: {s}", .{ path, std.mem.span(err_msg) });
        } else {
            log.warn("Failed to load MUSA runtime from {s}", .{path});
        }
        return error.Unavailable;
    };
}

pub fn load(_: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = io;
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }
    if (comptime builtin.cpu.arch != .x86_64) {
        return error.Unavailable;
    }

    const r = try bazel.runfiles(bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libpjrt_musa/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for MUSA runtime", .{});
        return error.FileNotFound;
    };

    try setupMusaEnv(sandbox_path);
    try probeMusaRuntime(sandbox_path);

    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_musa.so" });
        break :blk .loadFrom(path);
    };
}
