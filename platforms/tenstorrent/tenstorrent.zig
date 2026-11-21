const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const pjrt = @import("pjrt");
const platforms_options = @import("platforms/options");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/tenstorrent");

pub fn isEnabled() bool {
    return platforms_options.tenstorrent_enabled;
}

fn isSupportedHost() bool {
    return isEnabled() and builtin.os.tag == .linux and builtin.cpu.arch == .x86_64;
}

/// True when the Tenstorrent kernel driver exposes its device nodes.
pub fn hasDevices(io: std.Io) bool {
    std.Io.Dir.accessAbsolute(io, "/dev/tenstorrent", .{ .read = true }) catch return false;
    return true;
}

/// Loads the plugin for the cards of this host. Without cards the platform is
/// unavailable, so that `Platform.auto` moves on.
pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    if (comptime !isSupportedHost()) return error.Unavailable;
    if (!hasDevices(io)) return error.Unavailable;
    return loadPlugin(allocator, io);
}

/// Loads the plugin without probing for cards: its device-free runtime
/// compiles and runs on any Linux host against a cluster descriptor.
pub fn loadPlugin(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    if (comptime !isSupportedHost()) return error.Unavailable;

    const root = try sandboxPath(allocator, io);
    defer allocator.free(root);

    var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const library = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ root, "lib", "libzml_tenstorrent.so" });
    log.info("Loading Tenstorrent plugin: {s}", .{library});
    return .loadFrom(library);
}

/// Absolute path of the tt-metal runtime root shipped in the sandbox: kernel
/// sources, firmware and the SFPI toolchain that the kernel JIT runs.
pub fn runtimeRoot(allocator: std.mem.Allocator, io: std.Io) ![]u8 {
    if (comptime !isSupportedHost()) return error.Unavailable;

    const root = try sandboxPath(allocator, io);
    defer allocator.free(root);
    return std.Io.Dir.path.join(allocator, &.{ root, "libexec", "tt-metalium" });
}

/// The sandbox resolved to its real directory: tt-metal derives paths from the
/// runtime root, and its kernel JIT cache keys builds by that absolute path.
/// A file inside it is resolved rather than the directory: a Bazel test
/// sandbox recreates the directory per run and links each file to its place
/// in the output tree, so only the file resolves to a path that stays the same
/// across processes and lets the cache hit. (A runfiles manifest lists the
/// directory only, hence the lookup of the directory first.)
fn sandboxPath(allocator: std.mem.Allocator, io: std.Io) ![:0]u8 {
    const r = try bazel.runfiles(bazel_builtin.current_repository);
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_runfile = try r.rlocation("libzml_tenstorrent/sandbox", &path_buf) orelse {
        log.err("Missing Tenstorrent sandbox runfile", .{});
        return error.FileNotFound;
    };
    var plugin_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const plugin = try stdx.Io.Dir.path.bufJoinZ(&plugin_buf, &.{ sandbox_runfile, "lib", "libzml_tenstorrent.so" });
    const real_plugin = try std.Io.Dir.cwd().realPathFileAlloc(io, plugin, allocator);
    defer allocator.free(real_plugin);
    const lib_dir = std.Io.Dir.path.dirname(real_plugin) orelse return error.FileNotFound;
    const sandbox = std.Io.Dir.path.dirname(lib_dir) orelse return error.FileNotFound;
    return allocator.dupeSentinel(u8, sandbox, 0);
}
