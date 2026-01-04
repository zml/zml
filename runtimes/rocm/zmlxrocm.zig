const std = @import("std");

const stdx = @import("stdx");

pub export fn zmlxrocm_dlopen(filename: [*c]const u8, flags: c_int) ?*anyopaque {
    const replacements: std.StaticStringMap([:0]const u8) = .initComptime(.{
        .{ "librocm-core.so", "librocm-core.so.1" },
        .{ "librocm_smi64.so", "librocm_smi64.so.1" },
        .{ "libhsa-runtime64.so", "libhsa-runtime64.so.1" },
        .{ "libhsa-amd-aqlprofile64.so", "libhsa-amd-aqlprofile64.so.1" },
        .{ "libamd_comgr.so", "libamd_comgr.so.3" },
        .{ "librocprofiler-register.so", "librocprofiler-register.so.0" },
        .{ "libMIOpen.so", "libMIOpen.so.1" },
        .{ "libMIOpen.so.1", "libMIOpen.so.1" },
        .{ "librccl.so", "librccl.so.1" },
        .{ "librocblas.so.5", "librocblas.so.5" },
        .{ "librocblas.so", "librocblas.so.5" },
        .{ "libroctracer64.so", "libroctracer64.so.4" },
        .{ "libroctx64.so", "libroctx64.so.4" },
        .{ "libhipblaslt.so", "libhipblaslt.so.1" },
        .{ "libamdhip64.so", "libamdhip64.so.7" },
        .{ "libhiprtc.so", "libhiprtc.so.7" },
    });

    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const new_filename: [*c]const u8 = if (filename) |f| blk: {
        const replacement = replacements.get(std.fs.path.basename(std.mem.span(f))) orelse break :blk f;
        break :blk stdx.fs.path.bufJoinZ(&buf, &.{
            stdx.fs.selfSharedObjectDirPath(),
            replacement,
        }) catch unreachable;
    } else null;

    return std.c.dlopen(new_filename, @bitCast(flags));
}

pub export fn zmlxrocm_fopen64(pathname: [*c]const u8, mode: [*c]const u8) ?*std.c.FILE {
    const replacements: std.StaticStringMap([]const u8) = .initComptime(.{
        .{ "/opt/amdgpu/share/libdrm/amdgpu.ids", "../share/libdrm/amdgpu.ids" },
    });

    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const new_pathname: [*c]const u8 = blk: {
        const replacement = replacements.get(std.mem.span(pathname)) orelse break :blk pathname;
        break :blk stdx.fs.path.bufJoinZ(&buf, &.{
            stdx.fs.selfSharedObjectDirPath(),
            replacement,
        }) catch unreachable;
    };

    return std.c.fopen64(new_pathname, mode);
}
