/// Custom ZLS launcher.
/// 
/// Sets up paths to ZLS dependencies with Bazel runfiles.
///
/// This file is used as a template by `zls_write_runner_zig_src.bzl`.
const std = @import("std");
const runfiles = @import("runfiles");
const bazel_builtin = @import("bazel_builtin");

fn getRandomFilename(buf: *[std.fs.max_name_bytes]u8, extension: []const u8) ![]const u8 {
    const random_bytes_count = 12;
    const sub_path_len = comptime std.fs.base64_encoder.calcSize(random_bytes_count);

    var random_bytes: [random_bytes_count]u8 = undefined;
    std.crypto.random.bytes(&random_bytes);
    var random_name: [sub_path_len]u8 = undefined;
    _ = std.fs.base64_encoder.encode(&random_name, &random_bytes);

    const fmt_template = "/tmp/{s}{s}";
    const fmt_args = .{
        @as([]const u8, &random_name),
        extension,
    };
    return std.fmt.bufPrint(buf, fmt_template, fmt_args) catch @panic("OOM");
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

pub fn main() !void {
    const gpa = std.heap.page_allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    const allocator = arena.allocator();

    var r_ = try runfiles.Runfiles.create(.{ .allocator = allocator }) orelse
        return error.RunfilesNotFound;
    defer r_.deinit(allocator);

    const r = r_.withSourceRepo(bazel_builtin.current_repository);

    const zls_bin_rpath = "@@__ZLS_BIN_RPATH__@@";
    var zls_bin_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const zls_bin_path = try r.rlocation(zls_bin_rpath, &zls_bin_path_buf) orelse
        return error.RLocationNotFound;

    const zig_exe_rpath = "@@__ZIG_EXE_RPATH__@@";
    var zig_exe_path_buf: [std.fs.max_path_bytes]u8 = undefined;
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
    var zls_build_runner_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const zls_build_runner_path = try r.rlocation(zls_build_runner_rpath, &zls_build_runner_path_buf) orelse
        return error.RLocationNotFound;

    const global_cache_path = "@@__GLOBAL_CACHE_PATH__@@";

    const config: Config = .{
        .zig_exe_path = zig_exe_path,
        .zig_lib_path = zig_lib_computed_path,
        .build_runner_path = zls_build_runner_path,
        .global_cache_path = global_cache_path,
    };

    var buf: [std.fs.max_name_bytes]u8 = undefined;
    const tmp_file_path = try getRandomFilename(&buf, ".json");

    {
        var tmp_file = try std.fs.createFileAbsolute(tmp_file_path, .{});
        defer tmp_file.close();

        var out_buf: [4096]u8 = undefined;
        var tmp_file_writer = tmp_file.writer(&out_buf);

        const formatter = std.json.fmt(config, .{ .whitespace = .indent_2 });

        try tmp_file_writer.interface.print("{f}", .{formatter});
        try tmp_file_writer.interface.flush();
    }

    const args = std.process.argsAlloc(allocator) catch return error.ArgsAllocFailed;
    defer std.process.argsFree(allocator, args);

    const exec_args_len = args.len - 1 + 3; // Skip args[0] and add "zls" + --config-path + tmp_file_path
    var exec_args = try allocator.alloc([]const u8, exec_args_len);

    exec_args[0] = zls_bin_path; // "zls"

    // ${@}
    for (args[1..], 0..) |arg, i| {
        exec_args[1 + i] = arg;
    }
    exec_args[exec_args.len - 2] = "--config-path";
    exec_args[exec_args.len - 1] = tmp_file_path;

    return std.process.execv(arena.allocator(), exec_args);
}
