const std = @import("std");
const furiosa = @import("platforms/furiosa");
const pjrt = @import("pjrt");
const platforms_options = @import("platforms/options");

pub fn isEnabled() bool {
    return platforms_options.furiosa3_enabled;
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = allocator;
    _ = io;
    if (comptime !isEnabled()) return error.Unavailable;
    return furiosa.loadLibrary("XLA_FURIOSA3_PJRT_LIBRARY");
}
