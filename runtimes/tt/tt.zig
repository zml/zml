const std = @import("std");
const builtin = @import("builtin");

const asynk = @import("async");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/runtime/tt");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_TT");
}

fn hasDevices() bool {
    asynk.File.access("/dev/tenstorrent", .{ .mode = .read_only }) catch return false;
    return true;
}

fn setupEnv(sandbox_path: []const u8) !void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    _ = c.setenv("TT_METAL_HOME", try stdx.fs.path.bufJoinZ(&buf, &.{ sandbox_path, "tt-metal" }), 1); // must be zero terminated
}

pub fn load() !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }
    if (!hasDevices()) {
        return error.Unavailable;
    }

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    var r_ = try runfiles.Runfiles.create(.{ .allocator = arena.allocator() }) orelse {
        stdx.debug.panic("Unable to find runfiles", .{});
    };

    const source_repo = bazel_builtin.current_repository;
    const r = r_.withSourceRepo(source_repo);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libpjrt_tt/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for ROCm runtime", .{});
        return error.FileNotFound;
    };

    try setupEnv(sandbox_path);

    return blk: {
        var lib_path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try stdx.fs.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_tt.so" });
        break :blk asynk.callBlocking(pjrt.Api.loadFrom, .{path});
    };
}
