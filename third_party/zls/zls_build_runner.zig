const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const build_workspace_directory = init.environ_map.get("BUILD_WORKSPACE_DIRECTORY").?;
    var child = try std.process.spawn(init.io, .{
        .argv = &.{
            "bazel",
            "run",
            "@@__TARGET__@@",
        },
        .cwd = .{ .path = build_workspace_directory },
    });
    _ = try child.wait(init.io);
}
