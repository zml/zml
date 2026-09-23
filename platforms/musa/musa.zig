const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/musa");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_MUSA");
}

fn setupMusaEnv(library_directory: []const u8) !void {
    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    _ = c.setenv("MUSA_PATH", try stdx.Io.Dir.path.bufJoinZ(&buf, &.{ library_directory, "musa-sdk" }), 1);
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }
    if (comptime builtin.cpu.arch != .x86_64) {
        return error.Unavailable;
    }

    const r = try bazel.runfiles(bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libzml_musa/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for MUSA runtime", .{});
        return error.FileNotFound;
    };

    var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const runfile_path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libzml_musa.so" });
    // Test sandboxes expose individual files as symlinks. Resolve the plugin
    // before selecting its SDK so compiler files remain inside that SDK root.
    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, runfile_path, allocator);
    defer allocator.free(path);
    const library_directory = std.fs.path.dirname(path) orelse return error.InvalidPath;
    try setupMusaEnv(library_directory);
    return .loadFrom(path);
}
