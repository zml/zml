const builtin = @import("builtin");

const c = @import("c");
const pjrt = @import("pjrt");
const bazel_builtin = @import("bazel_builtin");
const std = @import("std");
const stdx = @import("stdx");
const runfiles = @import("runfiles");

const log = std.log.scoped(.@"zml/runtime/cpu");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_CPU");
}

pub fn load(io: std.Io) !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    var r_ = try runfiles.Runfiles.create(.{ .allocator = arena.allocator(), .io = io }) orelse {
        stdx.debug.panic("Unable to find runfiles", .{});
    };

    const source_repo = bazel_builtin.current_repository;
    const r = r_.withSourceRepo(source_repo);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("zml/runtimes/cpu/sandbox/lib", &path_buf) orelse {
        log.err("Failed to find sandbox path for CPU runtime", .{});
        return error.FileNotFound;
    };

    return blk: {
        const ext = switch (builtin.os.tag) {
            .windows => ".dll",
            .macos, .ios, .watchos => ".dylib",
            else => ".so",
        };

        var lib_path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try stdx.fs.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "libpjrt_cpu" ++ ext });
        var future = io.async(struct {
            fn call(path_: [:0]const u8) !*const pjrt.Api {
                return pjrt.Api.loadFrom(path_);
            }
        }.call, .{path});
        break :blk future.await(io);
    };
}
