const std = @import("std");
const builtin = @import("builtin");

const pjrt = @import("pjrt");
const platforms_options = @import("platforms/options");

const log = std.log.scoped(.@"zml/platforms/furiosa");

pub fn isEnabled() bool {
    return platforms_options.furiosa_enabled;
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = allocator;
    _ = io;
    if (comptime !isEnabled() or builtin.os.tag != .linux) return error.Unavailable;
    return loadLibrary("XLA_FURIOSA_PJRT_LIBRARY");
}

pub fn loadLibrary(comptime variable: [:0]const u8) !*const pjrt.Api {
    if (comptime builtin.os.tag != .linux) return error.Unavailable;
    const path = std.c.getenv(variable) orelse {
        log.err("Set {s} to the RNGD PJRT shared library path", .{variable});
        return error.Unavailable;
    };
    return .loadFrom(std.mem.span(path));
}
