const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/metal");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_METAL");
}

fn setMetalToolchainEnv(r: anytype) !void {
    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const bindir = try r.rlocation("zml/platforms/metal/metal_toolchain_bin", &buf) orelse
        return error.ToolchainNotInRunfiles;
    
    var bindir_z: [std.Io.Dir.max_path_bytes]u8 = undefined;
    _ = c.setenv("METAL_TOOLCHAIN", try stdx.Io.Dir.path.bufJoinZ(&bindir_z, &.{bindir}), 0);
}

pub fn load(_: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = io;
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }

    if (comptime builtin.os.tag != .macos) {
        return error.Unavailable;
    }

    const r = try bazel.runfiles(bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("zml/platforms/metal/sandbox/lib", &path_buf) orelse {
        log.err("Failed to find sandbox path for Metal runtime", .{});
        return error.FileNotFound;
    };

    setMetalToolchainEnv(r) catch |err| {
        log.err("Bundled Metal toolchain not found in runfiles ({any})", .{err});
    };

    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "libpjrt_c_api_gpu_plugin.dylib" });
        break :blk pjrt.Api.loadFrom(path);
    };
}
