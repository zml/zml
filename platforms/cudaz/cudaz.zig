const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/cudaz");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_CUDAZ");
}

pub fn load(_: std.mem.Allocator, _: std.Io) !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }

    const r = try bazel.runfiles(bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const plugin_path = try r.rlocation("zml/platforms/cudaz/libpjrt_cudaz.so", &path_buf) orelse {
        log.err("Failed to find cudaz PJRT plugin", .{});
        return error.FileNotFound;
    };

    var plugin_path_z_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const plugin_path_z = try stdx.Io.Dir.path.bufJoinZ(&plugin_path_z_buf, &.{plugin_path});
    return .loadFrom(plugin_path_z);
}
