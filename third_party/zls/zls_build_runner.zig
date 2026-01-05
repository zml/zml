const std = @import("std");
const builtin = @import("builtin");

pub fn main() !void {
    var arena_ = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena_.deinit();
    const arena = arena_.allocator();

    var threaded: std.Io.Threaded = .init(std.heap.smp_allocator, .{});
    defer threaded.deinit();

    const io = threaded.io();

    const build_workspace_directory = try std.process.getEnvVarOwned(arena, "BUILD_WORKSPACE_DIRECTORY");
    var child = std.process.Child.init(&.{
        "bazel",
        "run",
        "@@__TARGET__@@",
    }, arena);
    child.stdin_behavior = .Ignore;
    child.stdout_behavior = .Inherit;
    child.stderr_behavior = .Inherit;
    child.cwd = build_workspace_directory;

    try child.spawn(io);

    // var out = child.stdout.?;
    // _ = out; // autofix

    // const file = if (builtin.zig_version.major == 0 and builtin.zig_version.minor >= 16) blk: {
    //     var threaded = std.Io.Threaded.init(arena);
    //     defer threaded.deinit();
    //     const io = threaded.io();

    //     var read_buffer: [8192]u8 = undefined;
    //     var reader = out.reader(io, &read_buffer);
    //     break :blk try reader.interface.allocRemaining(arena, .unlimited);
    // } else try out.readToEndAlloc(arena, 16 * 1024 * 1024);

    // const need = std.mem.replacementSize(
    //     u8,
    //     file,
    //     "@@__BUILD_WORKSPACE_DIRECTORY__@@",
    //     build_workspace_directory,
    // );

    // const replaced = try arena.alloc(u8, need);

    // _ = std.mem.replace(
    //     u8,
    //     file,
    //     "@@__BUILD_WORKSPACE_DIRECTORY__@@",
    //     build_workspace_directory,
    //     replaced,
    // );

    // if (comptime builtin.zig_version.order(.{ .major = 0, .minor = 15, .patch = 0 }).compare(.gte)) {
    //     _ = try std.fs.File.stdout().writeAll(replaced);
    // } else {
    //     _ = try std.io.getStdOut().writeAll(replaced);
    // }
}
