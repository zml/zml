const std = @import("std");

const bazel_builtin = @import("bazel_builtin");
const runfiles = @import("runfiles");

/// https://github.com/zigtools/zls/blob/master/src/build_runner/shared.zig#L6 (2026-03-10)
pub const BuildConfig = struct {
    /// The `dependencies` in `build.zig.zon`.
    dependencies: std.json.ArrayHashMap([]const u8),
    /// The key is the `root_source_file`.
    /// All modules with the same root source file are merged. This limitation may be lifted in the future.
    modules: std.json.ArrayHashMap(Module),
    /// List of all compilations units.
    compilations: []Compile,
    /// The names of all top level steps.
    top_level_steps: []const []const u8,
    available_options: std.json.ArrayHashMap(AvailableOption),

    pub const Module = struct {
        import_table: std.json.ArrayHashMap([]const u8),
        c_macros: []const []const u8,
        include_dirs: []const []const u8,
    };

    pub const Compile = struct {
        /// Key in `BuildConfig.modules`.
        root_module: []const u8,

        // may contain additional information in the future like `target` or `link_libc`.
    };

    /// Equivalent to `std.Build.AvailableOption` which is not accessible because it non-pub.
    pub const AvailableOption = @FieldType(@FieldType(std.Build, "available_options_map").KV, "value");
};

fn processConfigLeaky(allocator: std.mem.Allocator, io: std.Io, config_string: []const u8) !BuildConfig {
    var config = try std.json.parseFromSliceLeaky(BuildConfig, allocator, config_string, .{});
    try canonicalizeModulesLeaky(allocator, io, &config);
    return config;
}

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
fn canonicalizeModulesLeaky(allocator: std.mem.Allocator, io: std.Io, config: *BuildConfig) !void {
    var old_modules = config.modules.map.iterator();
    var new_modules: std.StringArrayHashMapUnmanaged(BuildConfig.Module) = .empty;
    try new_modules.ensureTotalCapacity(allocator, old_modules.len);

    while (old_modules.next()) |entry| {
        const module_name = canonicalizePath(allocator, io, entry.key_ptr.*);
        var module: BuildConfig.Module = entry.value_ptr.*;
        for (module.import_table.map.values()) |*imported_module_path| {
            imported_module_path.* = canonicalizePath(allocator, io, imported_module_path.*);
        }
        new_modules.put(allocator, module_name, entry.value_ptr.*) catch return;
    }

    for (config.compilations) |*compile| {
        compile.root_module = canonicalizePath(allocator, io, compile.root_module);
    }

    config.modules.map = new_modules;
}

fn canonicalizePath(allocator: std.mem.Allocator, io: std.Io, path: []const u8) []const u8 {
    return std.Io.Dir.realPathFileAbsoluteAlloc(io, path, allocator) catch path;
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
    const build_execroot = init.environ_map.get("BUILD_EXECROOT") orelse b: {
        std.debug.print("=== BUILD_EXECROOT environment variable not found, using empty string as fallback ===\n", .{});
        break :b "";
    };

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
    output = try std.mem.replaceOwned(
        u8,
        arena.allocator(),
        output,
        "@@__BUILD_EXECROOT__@@",
        build_execroot,
    );

    var stdout_buf: [4096]u8 = undefined;
    var writer = std.Io.File.stdout().writer(io, &stdout_buf);
    if (processConfigLeaky(arena.allocator(), io, output)) |config| {
        try std.json.Stringify.value(config, .{}, &writer.interface);
    } else |_| {
        try writer.interface.writeAll(output);
    }
    try writer.flush();
}
