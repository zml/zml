const std = @import("std");

const stdx = @import("stdx");

pub export fn zmlxcuda_dlopen(filename: [*c]const u8, flags: c_int) ?*anyopaque {
    const replacements: std.StaticStringMap([:0]const u8) = .initComptime(.{
        .{ "libcublas.so", "libcublas.so.12" },
        .{ "libcublasLt.so", "libcublasLt.so.12" },
        .{ "libcudart.so", "libcudart.so.12" },
        .{ "libcudnn.so", "libcudnn.so.9" },
        .{ "libcufft.so", "libcufft.so.11" },
        .{ "libcupti.so", "libcupti.so.12" },
        .{ "libcusolver.so", "libcusolver.so.11" },
        .{ "libcusparse.so", "libcusparse.so.12" },
        .{ "libnccl.so", "libnccl.so.2" },
    });

    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const new_filename: [*c]const u8 = if (filename) |f| blk: {
        const replacement = replacements.get(std.fs.path.basename(std.mem.span(f))) orelse break :blk f;
        break :blk stdx.fs.path.bufJoinZ(&buf, &.{
            stdx.fs.selfSharedObjectDirPath(),
            replacement,
        }) catch unreachable;
    } else null;

    return std.c.dlopen(new_filename, @bitCast(flags));
}
