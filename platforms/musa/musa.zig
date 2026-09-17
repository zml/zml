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

fn probeMusaRuntime(library_directory: []const u8) !void {
    // The SDK ships a link-time stub. Use the host driver matching the kernel module.
    const driver_name = "libmusa.so.1";
    _ = std.c.dlopen(driver_name, .{ .NOW = true, .GLOBAL = true, .NODELETE = true }) orelse {
        const msg = std.c.dlerror();
        if (msg) |err_msg| {
            log.warn("Failed to load system MUSA driver {s}: {s}", .{ driver_name, std.mem.span(err_msg) });
        } else {
            log.warn("Failed to load system MUSA driver {s}", .{driver_name});
        }
        return error.Unavailable;
    };

    var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ library_directory, "musa-sdk", "lib", "libmusart.so.5" });
    _ = std.c.dlopen(path, .{ .NOW = true, .GLOBAL = true, .NODELETE = true }) orelse {
        const msg = std.c.dlerror();
        if (msg) |err_msg| {
            log.warn("Failed to load MUSA runtime from {s}: {s}", .{ path, std.mem.span(err_msg) });
        } else {
            log.warn("Failed to load MUSA runtime from {s}", .{path});
        }
        return error.Unavailable;
    };
}

pub fn load(_: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
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
    const sandbox_path = try r.rlocation("libpjrt_musa/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for MUSA runtime", .{});
        return error.FileNotFound;
    };

    var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const runfile_path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_musa.so" });
    // Test sandboxes expose individual files as symlinks. Resolve the plugin
    // before selecting its SDK so compiler files remain inside that SDK root.
    var canonical_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const canonical_len = if (std.fs.path.isAbsolute(runfile_path))
        try std.Io.Dir.realPathFileAbsolute(io, runfile_path, &canonical_buf)
    else
        try std.Io.Dir.cwd().realPathFile(io, runfile_path, &canonical_buf);
    if (canonical_len == canonical_buf.len) return error.NameTooLong;
    canonical_buf[canonical_len] = 0;
    const path = canonical_buf[0..canonical_len :0];
    const library_directory = std.fs.path.dirname(path) orelse return error.InvalidPath;
    try setupMusaEnv(library_directory);
    try probeMusaRuntime(library_directory);
    return .loadFrom(path);
}
