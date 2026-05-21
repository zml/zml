const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/oneapi");

const default_xla_flags = "--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0";

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_ONEAPI");
}

fn setupDefaultXlaFlags(allocator: std.mem.Allocator) !void {
    const xla_flags = if (std.c.getenv("XLA_FLAGS")) |flags_z| std.mem.span(flags_z) else "";
    const new_xla_flagsZ = try std.fmt.allocPrintSentinel(allocator, "{s} {s}", .{ xla_flags, default_xla_flags }, 0);
    defer allocator.free(new_xla_flagsZ);
    _ = c.setenv("XLA_FLAGS", new_xla_flagsZ, 1);
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }

    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }

    try setupDefaultXlaFlags(allocator);

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
