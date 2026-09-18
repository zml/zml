const std = @import("std");
const builtin = @import("builtin");

const bazel_runfiles = @import("runfiles");
const stdx = @import("stdx");

var runfiles_global: bazel_runfiles.Runfiles = undefined;

const init_array_section = switch (builtin.object_format) {
    .macho => "__DATA,__mod_init_func",
    .elf => ".init_array",
    else => "",
};

export const _bazel_runfiles_init linksection(init_array_section) = &struct {
    fn call(argc: c_int, argv: [*c][*:0]u8, envp: [*:null]?[*:0]u8) callconv(.c) void {
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
}.call;

pub fn runfiles(source_repository: []const u8) !bazel_runfiles.Runfiles.WithSourceRepo {
    return runfiles_global.withSourceRepo(source_repository);
}

// Standalone entry point for the normal and LTO runfiles checks.
pub fn main(init: std.process.Init) !void {
    const fixture = init.environ_map.get("ZML_RUNFILES_TEST_DATA") orelse return error.MissingRunfileFixture;
    const allocator = init.gpa;
    const io = init.io;
    // Do not initialize runfiles here: the linker-retained constructor must do it.
    const r = try runfiles(@import("bazel_builtin").current_repository);
    const path = try r.rlocationAlloc(allocator, fixture) orelse return error.RunfileNotFound;
    defer allocator.free(path);
    const contents = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024));
    defer allocator.free(contents);
    try std.testing.expectEqualStrings("runfiles initializer is retained\n", contents);
}
