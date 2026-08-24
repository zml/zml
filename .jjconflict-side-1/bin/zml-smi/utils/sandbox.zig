const std = @import("std");
const builtin = @import("builtin");
const bazel_runfiles = @import("runfiles");
const bazel_builtin = @import("bazel_builtin");

var runfiles_global: bazel_runfiles.Runfiles = undefined;

const init_array_section = switch (builtin.object_format) {
    .macho => "__DATA,__mod_init_func",
    .elf => ".init_array",
    else => "",
};

export const _ linksection(init_array_section) = &struct {
    fn call(argc: c_int, argv: [*c][*:0]u8, envp: [*:null]?[*:0]u8) callconv(.c) void {
        _ = argc;
        _ = argv;
        _ = envp;
        var threaded: std.Io.Threaded = .init_single_threaded;
        const io = threaded.io();

        var exe_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const exe_size = std.process.executablePath(io, &exe_path_buf) catch
            std.debug.panic("Unable to get executable path", .{});

        runfiles_global = bazel_runfiles.Runfiles.create(.{
            .allocator = std.heap.c_allocator,
            .io = io,
            .argv0 = exe_path_buf[0..exe_size],
            .directory = if (std.c.getenv("RUNFILES_DIR")) |directory| std.mem.span(directory) else null,
            .manifest = if (std.c.getenv("RUNFILES_MANIFEST_FILE")) |manifest| std.mem.span(manifest) else null,
        }) catch {
            std.debug.panic("Unable to find runfiles", .{});
        } orelse {
            std.debug.panic("Unable to initialize runfiles", .{});
        };
    }
}.call;

pub fn path(buf: *[std.Io.Dir.max_path_bytes]u8) ?[]const u8 {
    const with_repo = runfiles_global.withSourceRepo(bazel_builtin.current_repository);

    return with_repo.rlocation("zml/bin/zml-smi/sandbox", buf) catch null;
}
