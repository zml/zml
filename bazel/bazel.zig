const std = @import("std");

const bazel_runfiles = @import("runfiles");
const stdx = @import("stdx");

var runfiles_global: bazel_runfiles.Runfiles = undefined;

export fn _bazel_runfiles_init(argc: c_int, argv: [*c][*:0]u8, envp: [*:null]?[*:0]u8) callconv(.c) void {
    _ = argc;
    _ = envp;
    var threaded: std.Io.Threaded = .init_single_threaded;
    runfiles_global = bazel_runfiles.Runfiles.create(.{
        .allocator = std.heap.c_allocator,
        .io = threaded.io(),
        .argv0 = std.mem.span(argv[0]),
        .directory = if (std.c.getenv("RUNFILES_DIR")) |directory| std.mem.span(directory) else null,
        .manifest = if (std.c.getenv("RUNFILES_MANIFEST_FILE")) |manifest| std.mem.span(manifest) else null,
    }) catch {
        std.debug.panic("Unable to find runfiles", .{});
    } orelse {
        std.debug.panic("Unable to initialize runfiles", .{});
    };
}

pub fn runfiles(source_repository: []const u8) !bazel_runfiles.Runfiles.WithSourceRepo {
    return runfiles_global.withSourceRepo(source_repository);
}
