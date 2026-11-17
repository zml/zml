//! Exposes the Bazel module graph to ZLS through Zig's build server protocol.

const std = @import("std");

const BuildConfig = struct {
    modules: std.json.ArrayHashMap(Module),

    const Module = struct {
        import_table: std.json.ArrayHashMap([]const u8),
    };
};

pub fn build(builder: *std.Build) !void {
    // Bazel tracks changes to the inputs of its completion target.
    builder.graph.poisonCache();
    const target = "//:completion.print_build_config";
    const output = builder.runFallible(&.{ "bazel", "run", target }, .{
        .cwd = .{ .path = builder.fmt("{f}", .{builder.root}) },
        .stdout_limit = .unlimited,
    });
    const metadata = switch (output) {
        .success => |json| json,
        .spawn_failed => |err| std.process.fatal("Could not start Bazel: {t}", .{err}),
        .bad_exit_code => |code| std.process.fatal("Bazel {s} exited with code {d}", .{ target, code }),
        .crashed => std.process.fatal("Bazel {s} terminated unexpectedly", .{target}),
    };
    const config = try std.json.parseFromSliceLeaky(BuildConfig, builder.allocator, metadata, .{
        .ignore_unknown_fields = true,
    });

    // ZLS discovers modules reachable from a compilation step.
    const root_module = builder.createModule(.{ .target = builder.graph.host });
    const modules = try builder.allocator.alloc(*std.Build.Module, config.modules.map.count());
    for (config.modules.map.keys(), modules, 0..) |path, *module, index| {
        module.* = builder.createModule(.{ .root_source_file = builder.graph.cwdRelativePath(path) });
        root_module.addImport(builder.fmt("module_{d}", .{index}), module.*);
    }

    for (config.modules.map.values(), modules) |entry, module| {
        const imports = entry.import_table.map;
        for (imports.keys(), imports.values()) |name, path| {
            const index = config.modules.map.getIndex(path) orelse
                std.process.fatal("Bazel completion is missing module {s}", .{path});
            module.addImport(name, modules[index]);
        }
    }

    const completion = builder.addObject(.{ .name = "zls", .root_module = root_module });
    builder.step("zls", "Expose Bazel modules to the editor").dependOn(&completion.step);
}
