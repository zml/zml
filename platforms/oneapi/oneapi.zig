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

pub fn load(_: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }

    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }

    if (!hasOneApiDevices(io)) {
        log.warn("oneAPI platform requested but no compatible devices were found; skipping.", .{});
        return error.Unavailable;
    }

    return loadFromRunfiles(io) catch |err| switch (err) {
        error.FileNotFound => {
            log.warn("oneAPI platform requested but no Bazel-packaged plugin was found; skipping.", .{});
            return error.Unavailable;
        },
        else => return err,
    };
}

fn loadFromRunfiles(io: std.Io) !*const pjrt.Api {
    _ = io;
    const r = try bazel.runfiles(bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libpjrt_oneapi/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for oneAPI runtime", .{});
        return error.FileNotFound;
    };

    var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_oneapi.so" });
    return pjrt.Api.loadFrom(path);
}
