const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const stdx = @import("stdx");

const compat_probe = @import("compat_probe.zig");

const nvidiaLibsPath = "/cuda/";

const log = std.log.scoped(.@"zml/platforms/cuda");

fn findCudaSandbox(
    r: anytype,
    buffer: *[std.Io.Dir.max_path_bytes]u8,
) !?[]const u8 {
    const candidate = switch (builtin.cpu.arch) {
        .aarch64 => "libpjrt_cuda_linux_arm64/sandbox",
        .x86_64 => "libpjrt_cuda_linux_amd64/sandbox",
        else => return null,
    };
    return try r.rlocation(candidate, buffer);
}

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_CUDA");
}

pub fn needsCudaCompat(io: std.Io, sandbox_path: []const u8) !bool {
    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const nvidia_compat_path = try stdx.Io.Dir.path.bufJoinZ(&buf, &.{ sandbox_path, "bin", "compat_probe" });

    var child = try std.process.spawn(io, .{
        .argv = &[_][]const u8{nvidia_compat_path},
        .cwd = .{ .path = sandbox_path },
        .stdin = .ignore,
        .stdout = .inherit,
        .stderr = .inherit,
    });
    defer child.kill(io);

    const res = child.wait(io) catch |err| {
        log.err("Failed to run CUDA compatibility probe: {any}", .{err});
        return err;
    };
    const result: compat_probe.ExitCode = @enumFromInt(res.exited);

    return switch (result) {
        .Success => true,
        .SystemDriverMismatch, .CompatNotSupportedOnDevice => false,
        .UnexpectedError => blk: {
            log.err("CUDA compatibility probe returned unexpected error code", .{});
            break :blk false;
        },
    };
}

fn hasNvidiaDevice(io: std.Io) bool {
    for (&[_][]const u8{ "/dev/nvidiactl", "/dev/dxg" }) |dev| {
        std.Io.Dir.accessAbsolute(io, dev, .{ .read = true }) catch continue;
        return true;
    }
    return false;
}

fn hasCudaPathInLDPath() bool {
    const ldLibraryPath = std.c.getenv("LD_LIBRARY_PATH") orelse return false;
    return std.ascii.indexOfIgnoreCase(std.mem.span(ldLibraryPath), nvidiaLibsPath) != null;
}

fn setupXlaGpuCudaDirFlag(allocator: std.mem.Allocator, sandbox: []const u8) !void {
    const xla_flags = std.c.getenv("XLA_FLAGS") orelse "";
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

    const r = try bazel.runfiles(bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try findCudaSandbox(r, &path_buf) orelse {
        log.err("Failed to find sandbox path for CUDA runtime", .{});
        return error.FileNotFound;
    };

    // CUDA path has to be set _before_ loading the PJRT plugin.
    // See https://github.com/openxla/xla/issues/21428
    try setupXlaGpuCudaDirFlag(arena.allocator(), sandbox_path);

    {
        const cudaCompat = needsCudaCompat(io, sandbox_path) catch |err| blk: {
            log.err("Unable to determine wether or not to use CUDA Compat, disabling: {any}", .{err});
            break :blk false;
        };

        if (cudaCompat) {
            log.warn("Detected NVIDIA GPU that requires CUDA compatibility libraries.", .{});
            var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
            const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "compat", "libcuda.so.1" });
            _ = std.c.dlopen(path, .{ .NOW = true }) orelse {
                log.warn("Failed to load CUDA compatibility library from {s}: {any}", .{ path, std.mem.span(std.c.dlerror()) });
            };
            log.info("Loaded CUDA compatibility libraries.", .{});
        }
    }

    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_cuda.so" });
        break :blk .loadFrom(path);
    };
}
