const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const stdx = @import("stdx");

const nvidiaLibsPath = "/cuda/";

const log = std.log.scoped(.@"zml/platforms/cuda");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_CUDA");
}

fn hasNvidiaDevice(io: std.Io) bool {
    std.Io.Dir.accessAbsolute(io, "/dev/nvidiactl", .{ .read = true }) catch return false;
    return true;
}

fn hasCudaPathInLDPath() bool {
    const ldLibraryPath = std.c.getenv("LD_LIBRARY_PATH") orelse return false;
    return std.ascii.indexOfIgnoreCase(std.mem.span(ldLibraryPath), nvidiaLibsPath) != null;
}

fn setupXlaGpuCudaDirFlag(allocator: std.mem.Allocator, sandbox: []const u8) !void {
    const xla_flags = std.process.getEnvVarOwned(allocator, "XLA_FLAGS") catch "";
    const new_xla_flagsZ = try std.fmt.allocPrintSentinel(allocator, "{s} --xla_gpu_cuda_data_dir={s}", .{ xla_flags, sandbox }, 0);

    _ = c.setenv("XLA_FLAGS", new_xla_flagsZ, 1);
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }
    if (!hasNvidiaDevice(io)) {
        return error.Unavailable;
    }
    if (hasCudaPathInLDPath()) {
        log.warn("Detected {s} in LD_LIBRARY_PATH. This can lead to undefined behaviors and crashes", .{nvidiaLibsPath});
    }

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const r = try bazel.runfiles(io, bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libpjrt_cuda/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for CUDA runtime", .{});
        return error.FileNotFound;
    };

    // CUDA path has to be set _before_ loading the PJRT plugin.
    // See https://github.com/openxla/xla/issues/21428
    try setupXlaGpuCudaDirFlag(arena.allocator(), sandbox_path);

    {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libnvtx3interop.so" });
        _ = std.c.dlopen(path, .{ .NOW = true, .GLOBAL = true }) orelse {
            log.err("Unable to dlopen libnvtx3interop.so: {s}", .{std.c.dlerror().?});
            return error.DlError;
        };
    }

    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_cuda.so" });
        var future = io.async(pjrt.Api.loadFrom, .{path});
        break :blk future.await(io);
    };
}
