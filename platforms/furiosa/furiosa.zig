const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const pjrt = @import("pjrt");
const platforms_options = @import("platforms/options");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/furiosa");

pub fn isEnabled() bool {
    return platforms_options.furiosa_enabled;
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = allocator;
    if (comptime !isEnabled() or builtin.os.tag != .linux or builtin.cpu.arch != .x86_64) return error.Unavailable;

    if (std.c.getenv("XLA_FURIOSA_PJRT_LIBRARY")) |path| {
        log.info("Loading explicit Furiosa plugin: {s}", .{path});
        return .loadFrom(std.mem.span(path));
    }
    const r = try bazel.runfiles(bazel_builtin.current_repository);
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try r.rlocation("libzml_furiosa/lib/libpjrt_c_api_furiosa_plugin.so", &path_buf) orelse {
        log.err("Missing Furiosa plugin runfile; build with --override_repository=libzml_furiosa=/path/to/xla-override", .{});
        return error.FileNotFound;
    };
    // XLA build outputs use $ORIGIN-relative paths for toolchain libraries.
    // Follow the development override's symlink before dlopen so those paths
    // remain relative to the XLA output, not the ZML runfiles directory.
    var resolved_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const resolved_len = std.Io.Dir.cwd().realPathFile(io, path, &resolved_buf) catch |err| {
        log.err("Cannot resolve Furiosa plugin runfile {s}: {}", .{ path, err });
        return err;
    };
    var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const library = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{resolved_buf[0..resolved_len]});
    log.info("Loading Furiosa plugin: {s}", .{library});
    return .loadFrom(library);
}
