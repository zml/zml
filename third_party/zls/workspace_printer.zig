const std = @import("std");

/// Follow a symlink path until it reaches a non-symlink target.
///
/// Why this exists:
/// - In bzlmod, `execution_root/external/<repo>` is usually a symlink chain.
/// - The generated completion config references that execroot path.
/// - But editors open files from the real checkout path on disk.
/// - ZLS path matching is path-string based, so `external/...` and real paths
///   are treated as different files.
/// We resolve the chain so we can rewrite config paths to the same paths the
/// editor uses.
fn resolveSymlinkChain(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u8 {
    var current = try allocator.dupe(u8, path);
    var depth: usize = 0;
    while (depth < 16) : (depth += 1) {
        var link_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const len = std.Io.Dir.readLinkAbsolute(io, current, &link_buf) catch |err| switch (err) {
            error.NotLink, error.FileNotFound => break,
            else => return err,
        };

        const target = link_buf[0..len];
        current = if (target.len > 0 and target[0] == '/')
            try allocator.dupe(u8, target)
        else blk: {
            const base = std.fs.path.dirname(current) orelse ".";
            break :blk try std.fs.path.resolve(allocator, &.{ base, target });
        };
    }
    return current;
}

/// Extract a quoted argument value from a MODULE.bazel line.
///
/// Why this exists:
/// - We only need two fields (`module_name`, `path`) from
///   `local_path_override(...)` blocks.
/// - A tiny helper keeps the override-remap logic readable without pulling in a
///   full Starlark parser.
/// - There is no stable Bazel API in this runner that directly gives
///   local override filesystem paths, so we parse the minimal subset we need.
fn parseQuotedArg(line: []const u8, arg_name: []const u8) ?[]const u8 {
    const arg_idx = std.mem.indexOf(u8, line, arg_name) orelse return null;
    const arg_rest = line[arg_idx + arg_name.len ..];
    const quote_start = std.mem.indexOfScalar(u8, arg_rest, '"') orelse return null;
    const value_start = quote_start + 1;
    const quote_end_rel = std.mem.indexOfScalarPos(u8, arg_rest, value_start, '"') orelse return null;
    return arg_rest[value_start..quote_end_rel];
}

/// Rewrite execroot `external/<repo+>/...` segments for local path overrides.
///
/// Why this exists:
/// - `local_path_override` points a module to a local checkout (e.g. `../zml`).
/// - The completion config still contains execroot paths like
///   `<execution_root>/external/zml+/...`.
/// - If you open `/Users/.../zml/...` in the editor, ZLS cannot connect it to
///   `external/zml+/...`, so imports appear unresolved.
/// - In other words, with current ZLS path identity, we effectively need local
///   filesystem paths in the config unless the editor also opens `external/...`.
/// We read the override mapping and rewrite those prefixes to the local path so
/// both the config and the editor agree on file identity.
fn remapLocalPathOverrides(
    allocator: std.mem.Allocator,
    io: std.Io,
    output: []u8,
    build_workspace_directory: []const u8,
    execution_root: []const u8,
) ![]u8 {
    var module_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const module_path = std.fmt.bufPrint(&module_path_buf, "{s}/MODULE.bazel", .{build_workspace_directory}) catch return output;
    const module_bazel = std.Io.Dir.cwd().readFileAlloc(io, module_path, allocator, .unlimited) catch return output;

    var remapped_output = output;
    var in_local_path_override = false;
    var module_name: ?[]const u8 = null;
    var override_path: ?[]const u8 = null;

    var lines = std.mem.splitScalar(u8, module_bazel, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");

        if (!in_local_path_override) {
            if (std.mem.startsWith(u8, trimmed, "local_path_override(")) {
                in_local_path_override = true;
                module_name = null;
                override_path = null;
            }
            continue;
        }

        // We only need `module_name` and `path` within this block.
        if (module_name == null) {
            module_name = parseQuotedArg(trimmed, "module_name") orelse module_name;
        }
        if (override_path == null) {
            override_path = parseQuotedArg(trimmed, "path") orelse override_path;
        }

        if (std.mem.indexOfScalar(u8, trimmed, ')') != null) {
            if (module_name != null and override_path != null) {
                const repo_name = try std.fmt.allocPrint(allocator, "{s}+", .{module_name.?});
                const resolved_path = if (std.fs.path.isAbsolutePosix(override_path.?))
                    try allocator.dupe(u8, override_path.?)
                else
                    try std.fs.path.resolve(allocator, &.{ build_workspace_directory, override_path.? });

                const external_segment = try std.fmt.allocPrint(
                    allocator,
                    "{s}/external/{s}/",
                    .{ execution_root, repo_name },
                );
                const real_segment = try std.fmt.allocPrint(allocator, "{s}/", .{resolved_path});

                remapped_output = try std.mem.replaceOwned(
                    u8,
                    allocator,
                    remapped_output,
                    external_segment,
                    real_segment,
                );
            }

            in_local_path_override = false;
        }
    }

    return remapped_output;
}

/// Rewrite execroot `external/<repo>` prefixes to their real filesystem target.
///
/// Why this exists:
/// - Even without `local_path_override`, external repos can be symlinked through
///   Bazel's output base.
/// - Rewriting to the canonical real path reduces duplicate identities inside
///   ZLS and improves import resolution consistency.
/// - For local overrides, replacing `/external/<repo>/` with an absolute path can
///   produce `<execution_root>/<absolute_path>/...`; we collapse that too.
fn remapExternalRepos(
    allocator: std.mem.Allocator,
    io: std.Io,
    output: []u8,
    execution_root: []const u8,
) ![]u8 {
    var external_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const external_path = std.fmt.bufPrint(&external_path_buf, "{s}/external", .{execution_root}) catch return output;

    var external_dir = std.Io.Dir.openDir(.cwd(), io, external_path, .{ .iterate = true }) catch return output;
    defer external_dir.close(io);

    var remapped_output = output;
    var it = external_dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .directory) continue;

        var external_repo_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const external_repo_path = std.fmt.bufPrint(
            &external_repo_path_buf,
            "{s}/external/{s}",
            .{ execution_root, entry.name },
        ) catch continue;

        const real_path = resolveSymlinkChain(allocator, io, external_repo_path) catch continue;

        if (std.mem.eql(u8, external_repo_path, real_path)) continue;
        remapped_output = try std.mem.replaceOwned(
            u8,
            allocator,
            remapped_output,
            external_repo_path,
            real_path,
        );
    }

    return remapped_output;
}

fn tryReadFileAlloc(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !?[]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
}

fn readBuildConfig(
    allocator: std.mem.Allocator,
    io: std.Io,
    build_workspace_directory: []const u8,
    execution_root: []const u8,
    arg_path: []const u8,
) ![]u8 {
    if (std.fs.path.isAbsolutePosix(arg_path)) {
        return std.Io.Dir.cwd().readFileAlloc(io, arg_path, allocator, .unlimited);
    }

    if (try tryReadFileAlloc(allocator, io, arg_path)) |output| {
        return output;
    }

    const rel_path = if (std.mem.startsWith(u8, arg_path, "./")) arg_path[2..] else arg_path;

    const path_from_execroot = try std.fs.path.resolve(allocator, &.{ execution_root, rel_path });
    if (try tryReadFileAlloc(allocator, io, path_from_execroot)) |output| {
        return output;
    }

    var bazel_bin_result = try std.process.run(allocator, io, .{
        .argv = &.{
            "bazel",
            "info",
            "bazel-bin",
        },
        .cwd = .{ .path = build_workspace_directory },
    });
    const bazel_bin = std.mem.trimEnd(u8, bazel_bin_result.stdout, "\n");

    const path_from_bazel_bin = try std.fs.path.resolve(allocator, &.{ bazel_bin, rel_path });
    if (try tryReadFileAlloc(allocator, io, path_from_bazel_bin)) |output| {
        return output;
    }

    const path_from_bazel_bin_basename = try std.fs.path.resolve(allocator, &.{
        bazel_bin,
        std.fs.path.basename(rel_path),
    });
    if (try tryReadFileAlloc(allocator, io, path_from_bazel_bin_basename)) |output| {
        return output;
    }

    return error.FileNotFound;
}

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

    var output = try readBuildConfig(
        arena.allocator(),
        io,
        build_workspace_directory,
        std.mem.trimEnd(u8, output_base_result.stdout, "\n"),
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
    output = try remapLocalPathOverrides(arena.allocator(), io, output, build_workspace_directory, execution_root);
    output = try remapExternalRepos(arena.allocator(), io, output, execution_root);
    try std.Io.File.stdout().writeStreamingAll(io, output);
}
