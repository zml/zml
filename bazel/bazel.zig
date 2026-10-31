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

const runfiles_init = &struct {
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

comptime {
    if (builtin.os.tag == .linux) {
        // Let the C compiler emit .init_array until Zig's incremental ELF linker
        // handles its section type, flags, and size correctly.
        // TODO: Revert this workaround once Zig issue #37453 is fixed:
        // https://codeberg.org/ziglang/zig/issues/37453
        @export(runfiles_init, .{ .name = "_bazel_runfiles_init_impl" });
    } else {
        @export(&struct {
            const pointer align(@alignOf(usize)) linksection(init_array_section) = runfiles_init;
        }.pointer, .{ .name = "_bazel_runfiles_init" });
    }
}

pub fn runfiles(source_repository: []const u8) !bazel_runfiles.Runfiles.WithSourceRepo {
    return runfiles_global.withSourceRepo(source_repository);
}
