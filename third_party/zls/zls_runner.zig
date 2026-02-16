/// Custom ZLS launcher.
///
/// Sets up paths to ZLS dependencies with Bazel runfiles.
///
/// This file is used as a template by `zls_write_runner_zig_src.bzl`.
const std = @import("std");

const bazel_builtin = @import("bazel_builtin");
const runfiles = @import("runfiles");

fn getRandomFilename(io: std.Io, buf: []u8, extension: []const u8) ![]const u8 {
    const now = std.Io.Clock.real.now(io);
    return std.fmt.bufPrint(buf, "/tmp/{d}{s}", .{ now.nanoseconds, extension }) catch @panic("OOM");
}

const Config = struct {
    /// Override the Zig library path. Will be automatically resolved using the 'zig_exe_path'.
    zig_lib_path: ?[]const u8 = null,

    /// Specify the path to the Zig executable (not the directory). If unset, zig is looked up in `PATH`. e.g. `/path/to/zig-templeos-armless-1.0.0/zig`.
    zig_exe_path: ?[]const u8 = null,

    /// Specify a custom build runner to resolve build system information.
    build_runner_path: ?[]const u8 = null,

    /// Path to a directory that will be used as zig's cache. Will default to `${KnownFolders.Cache}/zls`.
    global_cache_path: ?[]const u8 = null,
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const io = init.io;

    var r_ = try runfiles.Runfiles.create(.{ .allocator = arena.allocator(), .io = io, .environ_map = init.environ_map, .argv = init.minimal.args }) orelse
        return error.RunfilesNotFound;
    const r = r_.withSourceRepo(bazel_builtin.current_repository);

    const zls_bin_rpath = "@@__ZLS_BIN_RPATH__@@";
    var zls_bin_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const zls_bin_path = try r.rlocation(zls_bin_rpath, &zls_bin_path_buf) orelse
        return error.RLocationNotFound;

    const zig_exe_rpath = "@@__ZIG_EXE_RPATH__@@";
    var zig_exe_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const zig_exe_path = blk2: {
        if (zig_exe_rpath[0] == '/') {
            break :blk2 zig_exe_rpath;
        } else {
            break :blk2 try r.rlocation(zig_exe_rpath, &zig_exe_path_buf) orelse
                return error.RLocationNotFound;
        }
    };

    const zig_lib_path = "@@__ZIG_LIB_PATH__@@";
    const zig_lib_computed_path = blk: {
        if (zig_lib_path[0] == '/') {
            break :blk zig_lib_path;
        } else {
            break :blk null;
        }
    };

    const zls_build_runner_rpath = "@@__ZLS_BUILD_RUNNER_RPATH__@@";
    var zls_build_runner_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const zls_build_runner_path = try r.rlocation(zls_build_runner_rpath, &zls_build_runner_path_buf) orelse
        return error.RLocationNotFound;

    const global_cache_path = "@@__GLOBAL_CACHE_PATH__@@";

    const config: Config = .{
        .zig_exe_path = zig_exe_path,
        .zig_lib_path = zig_lib_computed_path,
        .build_runner_path = zls_build_runner_path,
        .global_cache_path = global_cache_path,
    };

    var buf: [std.Io.Dir.max_name_bytes]u8 = undefined;
    const tmp_file_path = try getRandomFilename(io, &buf, ".json");

    {
        var tmp_file = try std.Io.Dir.createFileAbsolute(io, tmp_file_path, .{});
        defer tmp_file.close(io);

        var out_buf: [4096]u8 = undefined;
        var tmp_file_writer = tmp_file.writer(io, &out_buf);

        const formatter = std.json.fmt(config, .{ .whitespace = .indent_2 });

        try tmp_file_writer.interface.print("{f}", .{formatter});
        try tmp_file_writer.interface.flush();
    }

    const args = init.minimal.args.toSlice(arena.allocator()) catch return error.ArgsAllocFailed;

    const exec_args_len = args.len - 1 + 3; // Skip args[0] and add "zls" + --config-path + tmp_file_path
    var exec_args = try arena.allocator().alloc([]const u8, exec_args_len);

    exec_args[0] = zls_bin_path; // "zls"

    @memcpy(exec_args[1 .. exec_args.len - 2], args[1..]);
    @memcpy(exec_args[exec_args.len - 2 ..], &[_][]const u8{
        "--config-path",
        tmp_file_path,
    });
    var child = try std.process.spawn(io, .{
        .argv = exec_args,
    });
    _ = try child.wait(io);
}
