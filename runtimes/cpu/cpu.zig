const builtin = @import("builtin");

const asynk = @import("async");
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

pub fn load() !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    var r_ = try runfiles.Runfiles.create(.{ .allocator = arena.allocator() }) orelse {
        std.debug.panic("Unable to create Runfiles isntance.", .{});
    };
    const source_repo = bazel_builtin.current_repository;
    const r = r_.withSourceRepo(source_repo);
    const cuda_data_dir = (try r.rlocationAlloc(arena.allocator(), "zml/runtimes/cpu/sandbox/lib")).?;

    std.debug.print("Loading CPU PJRT API from: {s}\n", .{cuda_data_dir});

    const ext = switch (builtin.os.tag) {
        .windows => ".dll",
        .macos, .ios, .watchos => ".dylib",
        else => ".so",
    };

    const library = try std.fmt.allocPrintZ(arena.allocator(), "{s}/libpjrt_cpu{s}", .{ cuda_data_dir, ext });

    return try asynk.callBlocking(pjrt.Api.loadFrom, .{library});
}
