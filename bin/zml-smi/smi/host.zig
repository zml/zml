const builtin = @import("builtin");

pub const Host = switch (builtin.os.tag) {
    .linux => @import("host/linux.zig").Host,
    .macos => @import("host/macos.zig").Host,
    else => @compileError("zml-smi runs on Linux and macOS"),
};
