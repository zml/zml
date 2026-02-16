const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena.allocator());

    const build_workspace_directory = init.environ_map.get("BUILD_WORKSPACE_DIRECTORY").?;
    var output_base_result = try std.process.run(arena.allocator(), init.io, .{
        .argv = &.{
            "bazel",
            "info",
            "execution_root",
        },
        .cwd = .{ .path = build_workspace_directory },
    });

    var output = try std.Io.Dir.cwd().readFileAlloc(
        io,
        args[1],
        arena.allocator(),
        .unlimited,
    );
    output = try std.mem.replaceOwned(
        u8,
        arena.allocator(),
        output,
        "@@__BUILD_WORKSPACE_DIRECTORY__@@",
        build_workspace_directory,
    );
    output = try std.mem.replaceOwned(
        u8,
        arena.allocator(),
        output,
        "@@__BAZEL_EXECUTION_ROOT__@@",
        std.mem.trimEnd(u8, output_base_result.stdout, "\n"),
    );
    try std.Io.File.stdout().writeStreamingAll(io, output);
}
