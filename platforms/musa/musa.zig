// Cleanup: keep this file. It loads the PJRT plugin and SDK packaged by XLA.
const std = @import("std");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const pjrt = @import("pjrt");
const options = @import("platforms/options");
const stdx = @import("stdx");

extern fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    if (comptime !options.musa_enabled) return error.Unavailable;

    const runfiles = try bazel.runfiles(bazel_builtin.current_repository);
    var root_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox = try runfiles.rlocation("libzml_musa/sandbox", &root_buf) orelse return error.FileNotFound;
    const root = try std.Io.Dir.cwd().realPathFileAlloc(io, sandbox, allocator);
    defer allocator.free(root);
    // The plugin consults MUSA_PATH before looking next to its shared library.
    // Select the compiler bundled with this plugin even if the host SDK differs.
    if (setenv("MUSA_PATH", root, 1) != 0) return error.SetEnvFailed;
    var library_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const library = try stdx.Io.Dir.path.bufJoinZ(&library_buf, &.{ root, "lib", "libzml_musa.so" });
    return pjrt.Api.loadFrom(library);
}
