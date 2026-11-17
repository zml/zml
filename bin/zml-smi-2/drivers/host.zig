const builtin = @import("builtin");

pub const Host = switch (builtin.os.tag) {
    .linux => @import("host/linux.zig").Host,
    .macos => @import("host/macos.zig").Host,
    else => @compileError("zml-smi-2 runs on Linux and macOS"),
};
