const std = @import("std");

const stdx = @import("stdx");

pub export fn zmlxoneapi_dlopen(filename: [*c]const u8, flags: c_int) ?*anyopaque {
    const replacements: std.StaticStringMap([:0]const u8) = .initComptime(.{
        .{ "libOpenCL.so", "libOpenCL.so.1" },
        .{ "libumf.so", "libumf.so.1" },
        .{ "libur_loader.so", "libur_loader.so.0" },
        .{ "libur_adapter_level_zero.so", "libur_adapter_level_zero.so.0" },
        .{ "libur_adapter_level_zero_v2.so", "libur_adapter_level_zero_v2.so.0" },
        .{ "libur_adapter_opencl.so", "libur_adapter_opencl.so.0" },
        .{ "libsycl.so", "libsycl.so.9" },
        .{ "libmkl_core.so", "libmkl_core.so.3" },
        .{ "libmkl_intel_ilp64.so", "libmkl_intel_ilp64.so.3" },
        .{ "libmkl_sequential.so", "libmkl_sequential.so.3" },
        .{ "libmkl_sycl_blas.so", "libmkl_sycl_blas.so.6" },
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
