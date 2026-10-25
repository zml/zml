const std = @import("std");
const builtin = @import("builtin");

const pjrt = @import("pjrt");
const options = @import("platforms/options");

const log = std.log.scoped(.@"zml/platforms/tenstorrent");

pub fn isEnabled() bool {
    return options.tenstorrent_enabled;
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = allocator;
    if (comptime !isEnabled() or builtin.os.tag != .linux or builtin.cpu.arch != .x86_64) return error.Unavailable;
    std.Io.Dir.accessAbsolute(io, "/dev/tenstorrent", .{ .read = true }) catch return error.Unavailable;
    const path = std.c.getenv("ZML_TENSTORRENT_PLUGIN_PATH") orelse {
        log.err("Set ZML_TENSTORRENT_PLUGIN_PATH to the Tenstorrent PJRT shared library", .{});
        return error.FileNotFound;
    };
    if (std.c.getenv("TT_METAL_RUNTIME_ROOT") == null) {
        log.err("Set TT_METAL_RUNTIME_ROOT to the Tenstorrent device runtime assets", .{});
        return error.FileNotFound;
    }
    log.info("Loading Tenstorrent plugin: {s}", .{path});
    return .loadFrom(std.mem.span(path));
}
