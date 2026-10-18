//! Editor-only std.Build descriptors for the graph prepared by Bazel.
const std = @import("std");
const Build = std.Build;

const Module = struct {
    id: []const u8,
    source: []const u8,
    imports: std.json.ArrayHashMap([]const u8),
    zigopts: []const []const u8,
};
const Compilation = struct { label: []const u8, kind: []const u8, root: []const u8 };
const Manifest = struct {
    schema_version: u32,
    workspace: []const u8,
    target: []const u8,
    settings: struct { mode: []const u8, threaded: []const u8, strip: bool },
    modules: []const Module,
    compilations: []const Compilation,
};
const guidance = "ZML builds with Bazel. Start the language server with tools/zls.sh; restart it after BUILD/dependency/platform changes.";

pub fn build(b: *Build) !void {
    // The environment-selected path is not a tracked configure input in Zig 0.17.
    b.graph.poisonCache();
    const path = b.graph.environ_map.get("ZML_ZLS_BUILD_CONFIG_PATH") orelse {
        std.debug.print("{s}\nMissing ZML_ZLS_BUILD_CONFIG_PATH.\n", .{guidance});
        return error.MissingEditorMetadata;
    };
    if (!std.Io.Dir.path.isAbsolute(path)) return error.MetadataPathMustBeAbsolute;
    b.dependOnFileContents(.{ .cwd_relative = path });
    const content = try std.Io.Dir.cwd().readFileAlloc(b.graph.io, path, b.allocator, .limited(32 * 1024 * 1024));
    const parsed = std.json.parseFromSlice(Manifest, b.allocator, content, .{ .ignore_unknown_fields = true }) catch |err| {
        std.debug.print("Invalid editor metadata: {t}. {s}\n", .{ err, guidance });
        return err;
    };
    const manifest = parsed.value;
    if (manifest.schema_version != 1) return error.UnsupportedEditorSchema;
    const workspace = try std.Io.Dir.cwd().realPathFileAlloc(b.graph.io, try b.root.toString(b.allocator), b.allocator);
    if (!std.mem.eql(u8, workspace, manifest.workspace)) return error.EditorWorkspaceMismatch;
    const query = try std.Target.Query.parse(.{ .arch_os_abi = manifest.target });
    const target = b.resolveTargetQuery(query);
    if (target.result.cpu.arch != b.graph.host.result.cpu.arch or target.result.os.tag != b.graph.host.result.os.tag or target.result.abi != b.graph.host.result.abi)
        return error.CrossTargetEditorMetadataUnsupported;
    const optimize: std.builtin.Optimize = if (std.mem.eql(u8, manifest.settings.mode, "debug")) .debug else if (std.mem.eql(u8, manifest.settings.mode, "release_safe")) .safe else if (std.mem.eql(u8, manifest.settings.mode, "release_fast")) .fast else if (std.mem.eql(u8, manifest.settings.mode, "release_small")) .small else return error.UnsupportedOptimization;

    var modules: std.StringHashMap(*Build.Module) = .init(b.allocator);
    var sources: std.StringHashMap(void) = .init(b.allocator);
    for (manifest.modules) |module| {
        if (modules.contains(module.id)) return error.DuplicateModuleIdentity;
        if (!std.Io.Dir.path.isAbsolute(module.source)) return error.ModulePathMustBeAbsolute;
        if (sources.contains(module.source)) return error.AmbiguousModuleSource;
        try sources.put(module.source, {});
        try std.Io.Dir.cwd().access(b.graph.io, module.source, .{});
        const m = b.createModule(.{
            .root_source_file = .{ .cwd_relative = module.source },
            .target = target,
            .optimize = optimize,
            .single_threaded = std.mem.eql(u8, manifest.settings.threaded, "single"),
            .strip = manifest.settings.strip,
        });
        for (module.zigopts) |option| {
            if (std.mem.eql(u8, option, "-lc")) {
                m.link_libc = true;
            } else if (std.mem.eql(u8, option, "-fno-stack-check")) {
                m.stack_check = false;
            } else if (std.mem.startsWith(u8, option, "-freference-trace=")) {
                // Diagnostic depth does not change module semantics.
            } else {
                std.debug.print("Unsupported editor module option '{s}' in {s}\n", .{ option, module.id });
                return error.UnsupportedModuleOption;
            }
        }
        try modules.put(module.id, m);
    }
    // Keep shared identities and cycles; each module owns its import table.
    for (manifest.modules) |module| {
        const m = modules.get(module.id).?;
        for (module.imports.map.keys(), module.imports.map.values()) |alias, dependency| {
            m.addImport(alias, modules.get(dependency) orelse return error.DanglingModuleImport);
        }
    }
    const guard = b.addFail(guidance);
    b.default_step = &guard.step;
    b.getInstallStep().dependOn(&guard.step);
    const metadata = b.step("editor-metadata", "Editor graph only; build ZML with Bazel");
    var covered: std.AutoHashMap(*Build.Module, void) = .init(b.allocator);
    for (manifest.compilations, 0..) |compilation, i| {
        const root = modules.get(compilation.root) orelse return error.UnknownCompilationRoot;
        const name = b.fmt("editor-{d}", .{i});
        const compile = if (std.mem.eql(u8, compilation.kind, "exe"))
            b.addExecutable(.{ .name = name, .root_module = root })
        else if (std.mem.eql(u8, compilation.kind, "test"))
            b.addTest(.{ .name = name, .root_module = root })
        else if (std.mem.eql(u8, compilation.kind, "static"))
            b.addLibrary(.{ .name = name, .root_module = root, .linkage = .static })
        else if (std.mem.eql(u8, compilation.kind, "shared"))
            b.addLibrary(.{ .name = name, .root_module = root, .linkage = .dynamic })
        else
            return error.UnsupportedCompilationKind;
        compile.step.dependOn(&guard.step);
        metadata.dependOn(&compile.step);
        var pending: std.ArrayList(*Build.Module) = .empty;
        try pending.append(b.allocator, root);
        while (pending.pop()) |m| {
            const entry = try covered.getOrPut(m);
            if (entry.found_existing) continue;
            try pending.appendSlice(b.allocator, m.import_table.values());
        }
    }
    if (covered.count() != modules.count()) return error.UnreachableEditorModules;
}
