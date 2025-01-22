const builtin = @import("builtin");
const std = @import("std");
const asynk = @import("async");
const pjrt = @import("pjrt");
const c = @import("c");

const nvidiaLibsPath = "/usr/local/cuda/lib64";

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

    return std.mem.indexOf(u8, std.mem.span(ldLibraryPath), nvidiaLibsPath) != null;
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
        std.log.warn("Detected {s} in LD_LIBRARY_PATH. This can lead to undefined behaviors and crashes", .{nvidiaLibsPath});
    }

    return try asynk.callBlocking(pjrt.Api.loadFrom, .{"libpjrt_cuda.so"});
}
