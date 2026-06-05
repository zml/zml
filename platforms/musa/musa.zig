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

fn setupMusaEnv(sandbox_path: []const u8) !void {
    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    _ = c.setenv("MUSA_PATH", try stdx.Io.Dir.path.bufJoinZ(&buf, &.{sandbox_path}), 1);
}

fn probeMusaRuntime(sandbox_path: []const u8) !void {
    var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libmusart.so.1.0" });
    _ = std.c.dlopen(path, .{ .NOW = true }) orelse {
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
    _ = io;
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

    try setupMusaEnv(sandbox_path);
    try probeMusaRuntime(sandbox_path);

    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_musa.so" });
        break :blk .loadFrom(path);
    };
}
