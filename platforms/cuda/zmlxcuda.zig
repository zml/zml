const std = @import("std");

const stdx = @import("stdx");

pub export fn zmlxcuda_dlopen(filename: [*c]const u8, flags: c_int) ?*anyopaque {
    const replacements: std.StaticStringMap([:0]const u8) = .initComptime(.{
        .{ "libcuda.so.1", "libcuda.so.1" },
        .{ "libcublas.so", "libcublas.so.13" },
        .{ "libcublasLt.so", "libcublasLt.so.13" },
        .{ "libcudart.so", "libcudart.so.13" },
        .{ "libcudnn.so", "libcudnn.so.9" },
        .{ "libcufft.so", "libcufft.so.12" },
        .{ "libcupti.so", "libcupti.so.13" },
        .{ "libcusolver.so", "libcusolver.so.12" },
        .{ "libcusparse.so", "libcusparse.so.12" },
        .{ "libnccl.so", "libnccl.so.2" },
    });

    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const new_filename: [*c]const u8 = if (filename) |f| blk: {
        const replacement = replacements.get(std.Io.Dir.path.basename(std.mem.span(f))) orelse break :blk f;
        break :blk stdx.Io.Dir.path.bufJoinZ(&buf, &.{
            stdx.process.selfSharedObjectDirPath(),
            replacement,
        }) catch unreachable;
    } else null;

    return std.c.dlopen(new_filename, @bitCast(flags));
}
