const builtin = @import("builtin");
const std = @import("std");

const asynk = @import("async");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

const nvidiaLibsPath = "/cuda/";

const log = std.log.scoped(.@"zml/runtime/cuda");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_CUDA");
}

fn hasNvidiaDevice() bool {
    asynk.File.access("/dev/nvidiactl", .{ .mode = .read_only }) catch return false;
    return true;
}

fn hasCudaPathInLDPath() bool {
    const ldLibraryPath = c.getenv("LD_LIBRARY_PATH");

    if (ldLibraryPath == null) {
        return false;
    }

    return std.ascii.indexOfIgnoreCase(std.mem.span(ldLibraryPath), nvidiaLibsPath) != null;
}

fn setupXlaGpuCudaDirFlag() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    var r_ = try runfiles.Runfiles.create(.{ .allocator = arena.allocator() }) orelse {
        stdx.debug.panic("Unable to find CUDA directory", .{});
    };

    const source_repo = bazel_builtin.current_repository;
    const r = r_.withSourceRepo(source_repo);
    const cuda_data_dir = (try r.rlocationAlloc(arena.allocator(), "libpjrt_cuda/sandbox")).?;
    const xla_flags = std.process.getEnvVarOwned(arena.allocator(), "XLA_FLAGS") catch "";
    const new_xla_flagsZ = try std.fmt.allocPrintZ(arena.allocator(), "{s} --xla_gpu_cuda_data_dir={s}", .{ xla_flags, cuda_data_dir });

    _ = c.setenv("XLA_FLAGS", new_xla_flagsZ, 1);
}

pub fn load() !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }
    if (!hasNvidiaDevice()) {
        return error.Unavailable;
    }
    if (hasCudaPathInLDPath()) {
        log.warn("Detected {s} in LD_LIBRARY_PATH. This can lead to undefined behaviors and crashes", .{nvidiaLibsPath});
    }

    _ = c.setenv("XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB", "8", 1);
    _ = c.setenv("XLA_PJRT_GPU_HOST_MEMORY_PREALLOCATE", "true", 1);

    // CUDA path has to be set _before_ loading the PJRT plugin.
    // See https://github.com/openxla/xla/issues/21428
    try setupXlaGpuCudaDirFlag();

    return try asynk.callBlocking(pjrt.Api.loadFrom, .{"libpjrt_cuda.so"});
}
