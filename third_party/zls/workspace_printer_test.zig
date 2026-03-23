const std = @import("std");

const bazel_builtin = @import("bazel_builtin");
const runfiles = @import("runfiles");

const workspace_printer = @import("workspace_printer.zig");

fn requireEnv(name: [:0]const u8) ![]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return error.MissingEnvironmentVariable);
}

fn workspaceDirFromConfigPath(config_path: []const u8, package_name: []const u8) ![]const u8 {
    var workspace_dir = std.fs.path.dirname(config_path) orelse
        return error.InvalidWorkspacePath;
    if (package_name.len == 0) {
        return workspace_dir;
    }

    var it = std.mem.splitScalar(u8, package_name, '/');
    while (it.next()) |_| {
        workspace_dir = std.fs.path.dirname(workspace_dir) orelse
            return error.InvalidWorkspacePath;
    }
    return workspace_dir;
}

test "completion print_build_config emits valid json" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var r_ = try runfiles.Runfiles.create(.{
        .allocator = allocator,
        .io = std.testing.io,
        .directory = if (std.c.getenv("RUNFILES_DIR")) |value| std.mem.span(value) else null,
        .manifest = if (std.c.getenv("RUNFILES_MANIFEST_FILE")) |value| std.mem.span(value) else null,
    }) orelse return error.RunfilesNotFound;
    defer r_.deinit(allocator);

    const r = r_.withSourceRepo(bazel_builtin.current_repository);

    const printer_rpath = try requireEnv("COMPLETION_PRINTER_RLOCATION");
    const config_rpath = try requireEnv("COMPLETION_BUILD_CONFIG_RLOCATION");
    const package_name = try requireEnv("COMPLETION_PACKAGE");

    const printer_path = try r.rlocationAlloc(allocator, printer_rpath) orelse
        return error.RLocationNotFound;
    const config_path = try r.rlocationAlloc(allocator, config_rpath) orelse
        return error.RLocationNotFound;
    const workspace_dir = try workspaceDirFromConfigPath(config_path, package_name);
    const execution_root = workspace_dir;

    var child_env_map: std.process.Environ.Map = .init(std.testing.allocator);
    defer child_env_map.deinit();
    try child_env_map.put("BUILD_WORKSPACE_DIRECTORY", workspace_dir);
    try child_env_map.put("BAZEL_EXECUTION_ROOT", execution_root);

    const result = try std.process.run(std.testing.allocator, std.testing.io, .{
        .argv = &.{ printer_path, config_path },
        .environ_map = &child_env_map,
    });
    defer std.testing.allocator.free(result.stdout);
    defer std.testing.allocator.free(result.stderr);

    if (result.stderr.len > 0) {
        std.log.warn("workspace_printer stderr: {s}", .{result.stderr});
    }

    try std.testing.expectEqual(@as(std.process.Child.Term, .{ .exited = 0 }), result.term);

    const config = try std.json.parseFromSliceLeaky(workspace_printer.BuildConfig, allocator, result.stdout, .{});
    try std.testing.expect(config.modules.map.count() > 0);

    var modules = config.modules.map.iterator();
    while (modules.next()) |entry| {
        try std.testing.expect(entry.key_ptr.*.len > 0);

        const module = entry.value_ptr.*;
        for (module.import_table.map.values()) |import_path| {
            try std.testing.expect(import_path.len > 0);
        }
        for (module.include_dirs) |include_dir| {
            try std.testing.expect(include_dir.len > 0);
        }
    }

    for (config.compilations) |compilation| {
        try std.testing.expect(config.modules.map.contains(compilation.root_module));
    }
}
