const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/tt");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_TT");
}

fn hasDevices(io: std.Io) bool {
    std.Io.Dir.accessAbsolute(io, "/dev/tenstorrent", .{ .read = true }) catch return false;
    return true;
}

fn setupEnv(plugin_root: []const u8) !void {
    var tt_metal_home_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const tt_metal_home = try stdx.Io.Dir.path.bufJoinZ(&tt_metal_home_buf, &.{ plugin_root, "tt-metal" });
    _ = c.setenv("TT_METAL_HOME", tt_metal_home, 1);
    _ = c.setenv("TT_METAL_RUNTIME_ROOT", tt_metal_home, 1);
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = allocator;
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }
    if (!hasDevices(io)) {
        return error.Unavailable;
    }

    const r = try bazel.runfiles(bazel_builtin.current_repository);

    var sandbox_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libpjrt_tt/sandbox", &sandbox_path_buf) orelse {
        log.err("Failed to find sandbox path for Tenstorrent runtime", .{});
        return error.FileNotFound;
    };

    var plugin_root_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const plugin_root = try stdx.Io.Dir.path.bufJoin(&plugin_root_buf, &.{ sandbox_path, "pjrt_plugin_tt" });

    try setupEnv(plugin_root);

    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const lib_path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ plugin_root, "pjrt_plugin_tt.so" });
        break :blk pjrt.Api.loadFrom(lib_path);
    };
}
