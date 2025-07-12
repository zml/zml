const builtin = @import("builtin");
const std = @import("std");

const asynk = @import("async");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const runfiles = @import("runfiles");
const stdx = @import("stdx");
pub const cudart = @import("cudart.zig");

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

fn setupXlaGpuCudaDirFlag(allocator: std.mem.Allocator, sandbox: []const u8) !void {
    const xla_flags = std.process.getEnvVarOwned(allocator, "XLA_FLAGS") catch "";
    const new_xla_flagsZ = try std.fmt.allocPrintZ(allocator, "{s} --xla_gpu_cuda_data_dir={s}", .{ xla_flags, sandbox });

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

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    var r_ = try runfiles.Runfiles.create(.{ .allocator = arena.allocator() }) orelse {
        stdx.debug.panic("Unable to find CUDA directory", .{});
    };

    const source_repo = bazel_builtin.current_repository;
    const r = r_.withSourceRepo(source_repo);
    const cuda_data_dir = (try r.rlocationAlloc(arena.allocator(), "libpjrt_cuda/sandbox")).?;

    // CUDA path has to be set _before_ loading the PJRT plugin.
    // See https://github.com/openxla/xla/issues/21428
    try setupXlaGpuCudaDirFlag(arena.allocator(), cuda_data_dir);

    {
        const library = try std.fmt.allocPrintZ(arena.allocator(), "{s}/lib/libnvToolsExt.so.1", .{cuda_data_dir});
        const path = try std.posix.toPosixPath(library);
        _ = std.c.dlopen(&path, .{ .NOW = true, .GLOBAL = true }) orelse {
            log.err("Unable to dlopen libnvToolsExt.so.1", .{});
            return error.FileNotFound;
        };
    }

    const library = try std.fmt.allocPrintZ(arena.allocator(), "{s}/lib/libpjrt_cuda.so", .{cuda_data_dir});
    const api = try asynk.callBlocking(pjrt.Api.loadFrom, .{library});

    // Must be done after loading the PJRT plugin, as it will dlsym symbols from libcuda.so
    try cudart.load();

    return api;
}
