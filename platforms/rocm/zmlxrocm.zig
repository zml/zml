const std = @import("std");

const stdx = @import("stdx");

pub export fn zmlxrocm_dlopen(filename: [*c]const u8, flags: c_int) ?*anyopaque {
    const replacements: std.StaticStringMap([:0]const u8) = .initComptime(.{
        .{ "libamdhip64.so", "libamdhip64.so.7" },
        .{ "libhsa-amd-aqlprofile64.so", "libhsa-amd-aqlprofile64.so.1" },
        .{ "librocprofiler-sdk-attach.so", "librocprofiler-sdk-attach.so.1" },
        .{ "librocprofiler-sdk.so", "librocprofiler-sdk.so.1" },
        .{ "libroctx64.so", "libroctx64.so.4" },
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

pub export fn zmlxrocm_fopen64(pathname: [*c]const u8, mode: [*c]const u8) ?*std.c.FILE {
    const replacements: std.StaticStringMap([]const u8) = .initComptime(.{
        .{ "/opt/amdgpu/share/libdrm/amdgpu.ids", "../share/libdrm/amdgpu.ids" },
    });

    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const new_pathname: [*c]const u8 = blk: {
        const replacement = replacements.get(std.mem.span(pathname)) orelse break :blk pathname;
        break :blk stdx.Io.Dir.path.bufJoinZ(&buf, &.{
            stdx.process.selfSharedObjectDirPath(),
            replacement,
        }) catch unreachable;
    };

    return std.c.fopen64(new_pathname, mode);
}

extern fn pthread_attr_setstacksize(attr: ?*anyopaque, stacksize: usize) c_int;

// ROCclr asks for small helper-thread stacks: CQ_THREAD_STACK_SIZE is 256 KiB,
// then ROCclr adds the pthread guard size before calling pthread_attr_setstacksize.
// https://github.com/ROCm/clr/blob/1736f3eba9134a5ee4b56475acf15665f065ab05/rocclr/utils/flags.hpp#L25
// https://github.com/ROCm/clr/blob/1736f3eba9134a5ee4b56475acf15665f065ab05/rocclr/os/os_posix.cpp#L378-L385
// glibc 2.28 accepts stack sizes >= PTHREAD_STACK_MIN at attribute-set time,
// but pthread_create can still reject stacks that do not leave room for the
// guard, static TLS, and MINIMAL_REST_STACK:
// https://github.com/bminor/glibc/blob/glibc-2.28/nptl/pthread_attr_setstacksize.c#L35-L38
// https://github.com/bminor/glibc/blob/glibc-2.28/nptl/allocatestack.c#L532-L544
// The 1 MiB floor below is a local compatibility margin for that glibc path;
// it is intentionally larger than ROCclr's request and not a ROCm ABI value.
pub export fn zmlxrocm_pthread_attr_setstacksize(attr: ?*anyopaque, stacksize: usize) c_int {
    const min_stack_size = 1024 * 1024;
    const adjusted_stacksize = blk: {
        if (stacksize < min_stack_size) {
            break :blk min_stack_size;
        } else {
            break :blk stacksize;
        }
    };

    return pthread_attr_setstacksize(attr, adjusted_stacksize);
}
