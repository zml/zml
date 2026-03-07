const std = @import("std");

pub const ExitCode = enum(u8) {
    Success = 0,
    CompatNotSupportedOnDevice = 1,
    UnexpectedError = 2,
};

const CUDA_SUCCESS = 0;
const CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804;

pub fn main(init: std.process.Init) !u8 {
    const arena = init.arena;
    const io = init.io;

    const self_exe_dir = try std.process.executableDirPathAlloc(io, arena.allocator());
    const libcuda_path = try std.Io.Dir.path.joinZ(arena.allocator(), &.{ self_exe_dir, "..", "lib", "compat", "libcuda.so.1" });

    var libcuda = try std.DynLib.open(libcuda_path);
    const cuInit = libcuda.lookup(*const fn (c_uint) callconv(.c) c_int, "cuInit").?;

    return @intFromEnum(switch (cuInit(0)) {
        CUDA_SUCCESS => ExitCode.Success,
        CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE => ExitCode.CompatNotSupportedOnDevice,
        else => |v| blk: {
            std.log.err("cuInit returned unexpected error code: {d}", .{v});
            break :blk ExitCode.UnexpectedError;
        },
    });
}
