const std = @import("std");
const builtin = @import("builtin");

const pjrt = @import("pjrt");
const platforms_options = @import("platforms/options");

const log = std.log.scoped(.@"zml/platforms/furiosa");

pub fn isEnabled() bool {
    return platforms_options.furiosa;
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = allocator;
    _ = io;
    if (comptime !isEnabled() or builtin.os.tag != .linux) return error.Unavailable;

    const path = std.c.getenv("XLA_FURIOSA_PJRT_LIBRARY") orelse {
        log.err("Set XLA_FURIOSA_PJRT_LIBRARY to the RNGD PJRT shared library path", .{});
        return error.Unavailable;
    };
    return .loadFrom(std.mem.span(path));
}
