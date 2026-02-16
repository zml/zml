const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/cpu");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_CPU");
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = io; // autofix
    _ = allocator; // autofix
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }

    const r = try bazel.runfiles(bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("zml/platforms/cpu/sandbox/lib", &path_buf) orelse {
        log.err("Failed to find sandbox path for CPU runtime", .{});
        return error.FileNotFound;
    };

    return blk: {
        const ext = switch (builtin.os.tag) {
            .windows => ".dll",
            .macos, .ios, .watchos => ".dylib",
            else => ".so",
        };

        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "libpjrt_cpu" ++ ext });
        break :blk .loadFrom(path);
    };
}
