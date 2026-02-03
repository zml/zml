const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const bazel_runfiles = @import("runfiles");
const c = @import("c");
const pjrt = @import("pjrt");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/metal");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_METAL");
}

fn setupEnv(sandbox_path: []const u8) !void {
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    _ = c.setenv(
        "IREE_PJRT_COMPILER_LIB_PATH",
        @ptrCast(try stdx.Io.Dir.path.bufJoinZ(&path_buf, &.{ sandbox_path, "lib/libIREECompiler.so" })),
        1,
    ); // must be zero terminated
    _ = c.setenv(
        "IREE_LLVM_EMBEDDED_LINKER_PATH",
        @ptrCast(try stdx.Io.Dir.path.bufJoinZ(&path_buf, &.{ sandbox_path, "bin/lld" })),
        1,
    );
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = allocator; // autofix
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .macos) {
        return error.Unavailable;
    }

    const r = try bazel.runfiles(io, bazel_builtin.current_repository);

    var sandbox_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = (r.rlocation("zml/platforms/metal/sandbox", &sandbox_path_buf) catch |err| {
        log.err("unable to resolve sandbox path: {any}", .{err});
        return err;
    }) orelse {
        log.err("unable to resolve sandbox path", .{});
        return error.Unavailable;
    };

    try setupEnv(sandbox_path);

    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "pjrt_plugin_iree_metal.dylib" });
        break :blk pjrt.Api.loadFrom(path);
    };
}
