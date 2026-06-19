const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/oneapi");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_ONEAPI");
}

fn hasOneApiDevices(io: std.Io) bool {
    var dir = std.Io.Dir.openDirAbsolute(io, "/dev/dri", .{ .iterate = true }) catch return false;
    defer dir.close(io);

    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (std.mem.startsWith(u8, entry.name, "renderD")) {
            return true;
        }
    }

    return false;
}

fn setupOneAPIEnv() void {
    _ = c.setenv("CCL_LOG_LEVEL", "error", 1);
    _ = c.setenv("CCL_ATL_TRANSPORT", "ofi", 1);
    _ = c.setenv("CCL_TOPO_P2P_ACCESS", "1", 1);
}

pub fn load(_: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }

    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }

    if (!hasOneApiDevices(io)) {
        return error.Unavailable;
    }

    const r = try bazel.runfiles(bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libpjrt_oneapi/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for oneAPI runtime", .{});
        return error.FileNotFound;
    };

    setupOneAPIEnv();

    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_oneapi.so" });
        break :blk pjrt.Api.loadFrom(path);
    };
}
