const std = @import("std");
const c = @cImport({});
const stdx = @import("stdx");

const oneapiUmfVersion = c.ONEAPI_UMF_VERSION.*[0..];
const oneapiUrVersion = c.ONEAPI_UR_VERSION.*[0..];

const libUmf = std.fmt.comptimePrint("libumf.so.{s}", .{oneapiUmfVersion});
const libUrAdapterLevelZero = std.fmt.comptimePrint("libur_adapter_level_zero.so.{s}", .{oneapiUrVersion});
const libUrAdapterOpencl = std.fmt.comptimePrint("libur_adapter_opencl.so.{s}", .{oneapiUrVersion});
const libUrLoader = std.fmt.comptimePrint("libur_loader.so.{s}", .{oneapiUrVersion});

pub export fn zmlxoneapi_dlopen(filename: [*c]const u8, flags: c_int) ?*anyopaque {
    const replacements: std.StaticStringMap([:0]const u8) = .initComptime(.{
        .{ "libOpenCL.so", "libOpenCL.so.1" },
        .{ "libumf.so", libUmf },
        .{ "libumf.so.0", libUmf },
        .{ "libur_loader.so", libUrLoader },
        .{ "libur_adapter_level_zero.so", libUrAdapterLevelZero },
        .{ "libur_adapter_level_zero.so.0", libUrAdapterLevelZero },
        .{ "libur_adapter_opencl.so", libUrAdapterOpencl },
        .{ "libur_adapter_opencl.so.0", libUrAdapterOpencl },
        .{ "libur_loader.so.0", libUrLoader },
        .{ "libsycl.so", "libsycl.so.8" },
        .{ "libmkl_core.so", "libmkl_core.so.2" },
        .{ "libmkl_intel_ilp64.so", "libmkl_intel_ilp64.so.2" },
        .{ "libmkl_sequential.so", "libmkl_sequential.so.2" },
        .{ "libmkl_sycl_blas.so", "libmkl_sycl_blas.so.5" },
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
