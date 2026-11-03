const std = @import("std");
const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");

test "constructor initializes repository mapping before runfiles lookup" {
    const runfiles = try bazel.runfiles(bazel_builtin.current_repository);
    var buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = try runfiles.rlocation("zml/bazel/runfiles_init.c", &buffer) orelse return error.MissingRunfile;
    const file = try std.Io.Dir.cwd().openFile(std.testing.io, path, .{});
    defer file.close(std.testing.io);
}
