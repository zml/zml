const std = @import("std");
const bazel_runfiles = @import("runfiles");
const bazel_builtin = @import("bazel_builtin");

pub fn path(allocator: std.mem.Allocator, io: std.Io, buffer: *[std.Io.Dir.max_path_bytes]u8) ?[]const u8 {
    var exe: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const exe_len = std.process.executablePath(io, &exe) catch return null;

    var runfiles = (bazel_runfiles.Runfiles.create(.{
        .allocator = allocator,
        .io = io,
        .argv0 = exe[0..exe_len],
        .directory = if (std.c.getenv("RUNFILES_DIR")) |directory| std.mem.span(directory) else null,
        .manifest = if (std.c.getenv("RUNFILES_MANIFEST_FILE")) |manifest| std.mem.span(manifest) else null,
    }) catch return null) orelse return null;
    defer runfiles.deinit(allocator);

    return runfiles.withSourceRepo(bazel_builtin.current_repository).rlocation("zml/bin/zml-smi/sandbox", buffer) catch null;
}
