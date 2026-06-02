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

pub fn load(_: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = io;
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }

    // Metal is macOS-only (the inverse of the oneAPI Linux gate). The plugin is
    // an arm64 Mach-O; on any non-macOS host there is nothing to load.
    if (comptime builtin.os.tag != .macos) {
        return error.Unavailable;
    }

    const r = try bazel.runfiles(bazel_builtin.current_repository);

    // Mirrors the cpu platform: the dylib is staged in-repo under
    // platforms/metal/sandbox/lib via copy_to_directory, so the runfiles path is
    // rooted at the zml workspace (unlike oneapi, which lives in an external repo).
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("zml/platforms/metal/sandbox/lib", &path_buf) orelse {
        log.err("Failed to find sandbox path for Metal runtime", .{});
        return error.FileNotFound;
    };

    return blk: {
        // Extension is constant ".dylib": Metal only exists on macOS, and the
        // plugin's install_name is @rpath/libpjrt_metal.dylib, so the staged file
        // is named exactly libpjrt_metal.dylib.
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "libpjrt_metal.dylib" });
        break :blk pjrt.Api.loadFrom(path);
    };
}
