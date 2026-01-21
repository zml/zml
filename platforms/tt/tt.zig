const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/tt");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_TT");
}

fn hasDevices(io: std.Io) bool {
    std.Io.Dir.accessAbsolute(io, "/dev/tenstorrent", .{ .read = true }) catch return false;
    return true;
}

fn setupEnv(sandbox_path: []const u8) !void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    _ = c.setenv("TT_METAL_HOME", try stdx.Io.Dir.path.bufJoinZ(&buf, &.{"/tmp/tt"}), 1); // must be zero terminated
    _ = c.setenv("TT_METAL_RUNTIME_ROOT", try stdx.Io.Dir.path.bufJoinZ(&buf, &.{ sandbox_path, "tt-metal" }), 1); // must be zero terminated
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = allocator; // autofix
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }
    if (!hasDevices(io)) {
        return error.Unavailable;
    }

    const r = try bazel.runfiles(io, bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libpjrt_tt/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for ROCm runtime", .{});
        return error.FileNotFound;
    };

    try setupEnv(sandbox_path);

    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_tt.so" });
        break :blk pjrt.Api.loadFrom(path);
    };
}
