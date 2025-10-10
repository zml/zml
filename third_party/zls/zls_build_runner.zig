const builtin = @import("builtin");
const std = @import("std");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    var arena_ = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena_.deinit();
    const arena = arena_.allocator();

    const build_workspace_directory = try std.process.getEnvVarOwned(arena, "BUILD_WORKSPACE_DIRECTORY");
    var child = std.process.Child.init(&.{
        "bazel",
        "run",
        "@@__TARGET__@@",
    }, arena);
    child.stdin_behavior = .Ignore;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Inherit;
    child.cwd = build_workspace_directory;

    try child.spawn();

    var out = child.stdout.?;

    const file = try out.readToEndAlloc(arena, 16 * 1024 * 1024);

    const need = std.mem.replacementSize(
        u8,
        file,
        "@@__BUILD_WORKSPACE_DIRECTORY__@@",
        build_workspace_directory,
    );

    const replaced = try arena.alloc(u8, need);

    _ = std.mem.replace(
        u8,
        file,
        "@@__BUILD_WORKSPACE_DIRECTORY__@@",
        build_workspace_directory,
        replaced,
    );

    if (comptime builtin.zig_version.order(.{ .major = 0, .minor = 15, .patch = 0 }).compare(.gte)) {
        _ = try std.fs.File.stdout().writeAll(replaced);
    } else {
        _ = try std.io.getStdOut().writeAll(replaced);
    }
}
