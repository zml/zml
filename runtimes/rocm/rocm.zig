const builtin = @import("builtin");
const asynk = @import("async");
const pjrt = @import("pjrt");
const c = @import("c");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_ROCM");
}

fn hasRocmDevices() bool {
    inline for (&.{ "/dev/kfd", "/dev/dri" }) |path| {
        asynk.File.access(path, .{ .mode = .read_only }) catch return false;
    }
    return true;
}

pub fn load() !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }
    if (!hasRocmDevices()) {
        return error.Unavailable;
    }

    return try pjrt.Api.loadFrom("libpjrt_rocm.so");
}
