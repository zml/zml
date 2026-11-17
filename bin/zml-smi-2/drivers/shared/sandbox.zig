const std = @import("std");
const bazel_runfiles = @import("runfiles");
const bazel_builtin = @import("bazel_builtin");

/// The bundled library directory, found through the runfiles of the binary.
pub fn path(io: std.Io, buffer: *[std.Io.Dir.max_path_bytes]u8) ?[]const u8 {
    var exe: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const exe_len = std.process.executablePath(io, &exe) catch return null;

    var runfiles = (bazel_runfiles.Runfiles.create(.{
        .allocator = std.heap.c_allocator,
        .io = io,
        .argv0 = exe[0..exe_len],
        .directory = env("RUNFILES_DIR"),
        .manifest = env("RUNFILES_MANIFEST_FILE"),
    }) catch return null) orelse return null;
    defer runfiles.deinit(std.heap.c_allocator);

    return runfiles.withSourceRepo(bazel_builtin.current_repository).rlocation("zml/bin/zml-smi-2/sandbox", buffer) catch null;
}

fn env(name: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |value| std.mem.span(value) else null;
}
