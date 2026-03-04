const std = @import("std");

const bazel_builtin = @import("bazel_builtin");
const runfiles = @import("runfiles");

// Neovim canonicalizes the path it sends to ZLS, we need to do the same as ZLS will match against the config we generate.
// We canonicalize the following module paths:
//
// {
//   "modules": {
//     "<module-path>": {
//       "import_table": {
//         "<import-name>": "<module-path>"
//       },
//     },
//   },
// }
fn canonicalizeModules(allocator: std.mem.Allocator, io: std.Io, root: *std.json.Value) !void {
    if (root.* != .object) return;
    const modules_value = root.object.getPtr("modules") orelse return;
    if (modules_value.* != .object) return;

    var canonical_modules = std.json.ObjectMap.init(allocator);
    var modules_it = modules_value.object.iterator();
    while (modules_it.next()) |module_entry| {
        const module_path = module_entry.key_ptr.*;
        var module_info = module_entry.value_ptr.*;

        if (module_info == .object) {
            if (module_info.object.getPtr("import_table")) |imports_value| {
                if (imports_value.* == .object) {
                    var canonical_imports = std.json.ObjectMap.init(allocator);
                    var imports_it = imports_value.object.iterator();
                    while (imports_it.next()) |import_entry| {
                        const import_name = import_entry.key_ptr.*;
                        const import_value = import_entry.value_ptr.*;

                        const canonical_import_value = if (import_value == .string)
                            std.json.Value{ .string = try std.Io.Dir.realPathFileAbsoluteAlloc(io, import_value.string, allocator) }
                        else
                            import_value;

                        try canonical_imports.put(import_name, canonical_import_value);
                    }
                    imports_value.* = .{ .object = canonical_imports };
                }
            }
        }

        const canonical_module_path = try std.Io.Dir.realPathFileAbsoluteAlloc(io, module_path, allocator);
        try canonical_modules.put(canonical_module_path, module_info);
    }

    modules_value.* = .{ .object = canonical_modules };
}

fn readBuildConfig(
    allocator: std.mem.Allocator,
    io: std.Io,
    init: std.process.Init,
    arg_path: []const u8,
) ![]u8 {
    if (std.fs.path.isAbsolutePosix(arg_path)) {
        return std.Io.Dir.cwd().readFileAlloc(io, arg_path, allocator, .unlimited);
    }

    var r_ = try runfiles.Runfiles.create(.{
        .allocator = allocator,
        .io = io,
        .environ_map = init.environ_map,
        .argv = init.minimal.args,
    }) orelse return error.RunfilesNotFound;

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const r = r_.withSourceRepo(bazel_builtin.current_repository);

    if (try r.rlocation(arg_path, &path_buf)) |config_path| {
        return std.Io.Dir.cwd().readFileAlloc(io, config_path, allocator, .unlimited);
    }

    const stripped = if (std.mem.startsWith(u8, arg_path, "./")) arg_path[2..] else arg_path;
    if (try r.rlocation(stripped, &path_buf)) |config_path| {
        return std.Io.Dir.cwd().readFileAlloc(io, config_path, allocator, .unlimited);
    }

    // Convenience for testing.
    const output = std.Io.Dir.cwd().readFileAlloc(io, arg_path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    if (output) |content| {
        return content;
    }

    return error.BuildConfigNotFound;
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena.allocator());

    const build_workspace_directory = init.environ_map.get("BUILD_WORKSPACE_DIRECTORY").?;
    const output_base_result = try std.process.run(arena.allocator(), init.io, .{
        .argv = &.{
            "bazel",
            "info",
            "execution_root",
        },
        .cwd = .{ .path = build_workspace_directory },
    });

    var output = try readBuildConfig(
        arena.allocator(),
        io,
        init,
        args[1],
    );
    output = try std.mem.replaceOwned(
        u8,
        arena.allocator(),
        output,
        "@@__BUILD_WORKSPACE_DIRECTORY__@@",
        build_workspace_directory,
    );
    const execution_root = std.mem.trimEnd(u8, output_base_result.stdout, "\n");
    output = try std.mem.replaceOwned(
        u8,
        arena.allocator(),
        output,
        "@@__BAZEL_EXECUTION_ROOT__@@",
        execution_root,
    );
    var root = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), output, .{});
    try canonicalizeModules(arena.allocator(), io, &root);

    var stdout_buf: [4096]u8 = undefined;
    var writer = std.Io.File.stdout().writer(io, &stdout_buf);
    try std.json.Stringify.value(root, .{}, &writer.interface);
    try writer.flush();
}
