const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const pjrt = @import("pjrt");
const platforms_options = @import("platforms/options");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/tt");

extern fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

fn setEnv(name: [*:0]const u8, value: [:0]const u8, overwrite: c_int) !void {
    if (setenv(name, value, overwrite) != 0) return error.SetEnvFailed;
}

pub fn isEnabled() bool {
    return platforms_options.tt_enabled;
}

/// Returns true when tt-metal will find a device: either real silicon under
/// `/dev/tenstorrent`, or a ttsim simulator selected by `TT_METAL_SIMULATOR`.
fn hasDevices(io: std.Io) bool {
    if (std.c.getenv("TT_METAL_SIMULATOR") != null) return true;
    std.Io.Dir.accessAbsolute(io, "/dev/tenstorrent", .{ .read = true }) catch return false;
    return true;
}

/// Returns the directory holding `pjrt_plugin_tt.so` and its `tt-metal/` tree.
/// `ZML_TT_PLUGIN_ROOT` overrides the Bazel-fetched wheel, e.g. to load a
/// locally built tt-xla plugin.
fn pluginRoot(allocator: std.mem.Allocator, io: std.Io) ![]const u8 {
    if (std.c.getenv("ZML_TT_PLUGIN_ROOT")) |root| {
        return try allocator.dupe(u8, std.mem.span(root));
    }

    const r = try bazel.runfiles(bazel_builtin.current_repository);
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox = try r.rlocation("libpjrt_tt/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for Tenstorrent runtime", .{});
        return error.FileNotFound;
    };
    const sandbox_real = try std.Io.Dir.cwd().realPathFileAlloc(io, sandbox, allocator);
    defer allocator.free(sandbox_real);
    return try std.fs.path.join(allocator, &.{ sandbox_real, "pjrt_plugin_tt" });
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    if (comptime !isEnabled() or builtin.os.tag != .linux or builtin.cpu.arch != .x86_64) return error.Unavailable;
    if (!hasDevices(io)) return error.Unavailable;

    const root = try pluginRoot(allocator, io);
    defer allocator.free(root);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const tt_metal_home = try stdx.Io.Dir.path.bufJoinZ(&path_buf, &.{ root, "tt-metal" });
    // Not overwritten, so a user-provided tt-metal tree wins.
    try setEnv("TT_METAL_HOME", tt_metal_home, 0);
    try setEnv("TT_METAL_RUNTIME_ROOT", tt_metal_home, 0);

    const library = try stdx.Io.Dir.path.bufJoinZ(&path_buf, &.{ root, "pjrt_plugin_tt.so" });
    log.info("Loading Tenstorrent plugin: {s}", .{library});
    return .loadFrom(library);
}
