const builtin = @import("builtin");
const asynk = @import("async");
const pjrt = @import("pjrt");
const c = @import("c");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_MLX");
}

pub fn load() !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .macos) {
        return error.Unavailable;
    }
    return pjrt.Api.loadFrom("stablehlo_mlx_plugin.so");
}
