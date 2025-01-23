const builtin = @import("builtin");

const asynk = @import("async");
const c = @import("c");
const pjrt = @import("pjrt");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_CPU");
}

pub fn load() !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }

    const ext = switch (builtin.os.tag) {
        .windows => ".dll",
        .macos, .ios, .watchos => ".dylib",
        else => ".so",
    };
    return try asynk.callBlocking(pjrt.Api.loadFrom, .{"libpjrt_cpu" ++ ext});
}
