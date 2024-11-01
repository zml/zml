const builtin = @import("builtin");
const asynk = @import("async");
const pjrt = @import("pjrt");
const c = @import("c");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_CUDA");
}

fn hasNvidiaDevice() bool {
    asynk.File.access("/dev/nvidia0", .{ .mode = .read_only }) catch return false;
    return true;
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

    return try pjrt.Api.loadFrom("libpjrt_cuda.so");
}
