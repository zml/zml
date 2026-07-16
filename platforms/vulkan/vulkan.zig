const std = @import("std");
const builtin = @import("builtin");
const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/vulkan");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_VULKAN");
}

pub fn load(_: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = io;
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }

    const r = try bazel.runfiles(bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libpjrt_vulkan/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for Vulkan runtime", .{});
        return error.FileNotFound;
    };

    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "pjrt_c_api_gpu_plugin.so" });
        break :blk pjrt.Api.loadFrom(path);
    };
}
