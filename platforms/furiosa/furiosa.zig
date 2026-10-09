const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const pjrt = @import("pjrt");
const platforms_options = @import("platforms/options");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/furiosa");

extern fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

fn setEnv(name: [*:0]const u8, value: [:0]const u8, overwrite: c_int) !void {
    if (setenv(name, value, overwrite) != 0) return error.SetEnvFailed;
}

pub fn isEnabled() bool {
    return platforms_options.furiosa_enabled;
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    if (comptime !isEnabled() or builtin.os.tag != .linux or builtin.cpu.arch != .x86_64) return error.Unavailable;

    const r = try bazel.runfiles(bazel_builtin.current_repository);
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox = try r.rlocation("libzml_furiosa/sandbox", &path_buf) orelse {
        log.err("Missing Furiosa sandbox runfile", .{});
        return error.FileNotFound;
    };

    const root = try std.Io.Dir.cwd().realPathFileAlloc(io, sandbox, allocator);
    defer allocator.free(root);

    var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    try setEnv("XLA_FURIOSA_SDK_ROOT", try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{root}), 1);
    try setEnv("FURIOSA_VISIBLE_DEVICES", "0", 0);

    const library = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ root, "lib", "libzml_furiosa.so" });
    log.info("Loading Furiosa plugin: {s}", .{library});
    return .loadFrom(library);
}
