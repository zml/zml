const builtin = @import("builtin");
const pjrt = @import("pjrt");
const c = @import("c");

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
    return try pjrt.Api.loadFrom("libpjrt_cpu" ++ ext);
}
