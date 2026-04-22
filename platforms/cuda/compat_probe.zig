const std = @import("std");

pub const ExitCode = enum(u8) {
    Success = 0,
    CompatNotSupportedOnDevice = 1,
    SystemDriverMismatch = 2,
    UnexpectedError = 3,
};

const CUDA_SUCCESS = 0;
const CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803;
const CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804;

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena;
    const io = init.io;

    const self_exe_dir = try std.process.executableDirPathAlloc(io, arena.allocator());
    const libcuda_path = try std.Io.Dir.path.joinZ(arena.allocator(), &.{ self_exe_dir, "..", "lib", "compat", "libcuda.so.1" });

    var libcuda = try std.DynLib.open(libcuda_path);
    const cuInit = libcuda.lookup(*const fn (c_uint) callconv(.c) c_int, "cuInit").?;
    const cuGetErrorString = libcuda.lookup(*const fn (c_int, *[*c]const u8) callconv(.c) c_int, "cuGetErrorString").?;

    return @intFromEnum(switch (cuInit(0)) {
        CUDA_SUCCESS => ExitCode.Success,
        CUDA_ERROR_SYSTEM_DRIVER_MISMATCH => ExitCode.SystemDriverMismatch,
        CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE => ExitCode.CompatNotSupportedOnDevice,
        else => |v| blk: {
            var err_str_ptr: [*c]const u8 = undefined;
            if (cuGetErrorString(v, &err_str_ptr) == CUDA_SUCCESS) {
                std.log.err("cuInit returned unexpected error code: {d}: {s}", .{ v, err_str_ptr });
            } else {
                std.log.err("cuInit returned unexpected error code: {d}", .{v});
            }
            break :blk ExitCode.UnexpectedError;
        },
    });
}
