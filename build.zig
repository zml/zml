const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const test_step = b.step("test", "Run unit tests");

    // mlir
    const mlir_c_deps = moduleFromBazelSrcs(
        b,
        null,
        "//mlir:sources",
        "mlir/sources.tar",
        "mlir/test_test_lib_c.zig",
        .{ .link_libcpp = true },
    );
    addObjectFromBazel(mlir_c_deps, "//mlir:mlir_static", "mlir/libmlir_static.a");

    const mlir = b.addModule("mlir", .{
        .root_source_file = b.path("mlir/mlir.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "c", .module = mlir_c_deps },
        },
    });

    const mlir_test = b.addTest(.{ .root_module = mlir });
    const run_mlir_tests = b.addRunArtifact(mlir_test);
    test_step.dependOn(&run_mlir_tests.step);

    // pjrt
    const pjrt_c_deps = moduleFromBazelSrcs(
        b,
        null,
        "//pjrt:sources",
        "pjrt/sources.tar",
        "pjrt/test_test_lib_c.zig",
        .{ .link_libcpp = true },
    );

    const pjrt = b.addModule("pjrt", .{
        .root_source_file = b.path("pjrt/pjrt.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "c", .module = pjrt_c_deps },
        },
    });

    const pjrt_test = b.addTest(.{ .root_module = pjrt });
    const run_pjrt_tests = b.addRunArtifact(pjrt_test);
    test_step.dependOn(&run_pjrt_tests.step);

    // stdx
    const stdx = moduleFromBazelSrcs(
        b,
        "stdx",
        "//stdx:sources",
        "stdx/sources.tar",
        "stdx/stdx.zig",
        .{
            .target = target,
            .optimize = optimize,
        },
    );

    const stdx_test = b.addTest(.{ .root_module = stdx });
    const run_stdx_tests = b.addRunArtifact(stdx_test);
    test_step.dependOn(&run_stdx_tests.step);

    // xev
    const xev = moduleFromBazelSrcs(
        b,
        "xev",
        "//async:sources",
        "async/sources.tar",
        "src/main.zig",
        .{
            .target = target,
            .optimize = optimize,
        },
    );

    const xev_test = b.addTest(.{ .root_module = xev });
    const run_xev_tests = b.addRunArtifact(xev_test);
    test_step.dependOn(&run_xev_tests.step);

    // async
    const async_mod = moduleFromBazelSrcs(
        b,
        "async",
        "//async:sources",
        "async/sources.tar",
        "async/async.zig",
        .{
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "xev", .module = xev },
                .{ .name = "stdx", .module = stdx },
            },
        },
    );

    const async_test = b.addTest(.{ .root_module = async_mod });
    const run_async_tests = b.addRunArtifact(async_test);
    test_step.dependOn(&run_async_tests.step);
}

/// Take the name of a Bazel `cc_static_library` and add it to the given module.
fn addObjectFromBazel(module: *std.Build.Module, name: []const u8, output: []const u8) void {
    const b = module.owner;
    // TODO: consider parsing bazel name to generate output name.
    const bazel_cmd = b.addSystemCommand(&.{ "bazel", "build", "-c", "opt", name });
    const obj_path = b.pathJoin(&.{ "bazel-bin", output });

    // Copy bazel output into zig-cache, cause bazel may remove the file later.
    const cp = b.addWriteFiles();
    cp.step.dependOn(&bazel_cmd.step);
    const obj = cp.addCopyFile(b.path(obj_path), output);

    // Module depends on the copied object.
    module.addObjectFile(obj);
}

fn moduleFromBazelSrcs(
    b: *std.Build,
    module_name: ?[]const u8,
    sources_target: []const u8,
    sources_tar_path: []const u8,
    root: []const u8,
    options: std.Build.Module.CreateOptions,
) *std.Build.Module {
    // TODO: consider parsing bazel name to generate output name.
    const bazel_cmd = b.addSystemCommand(&.{ "bazel", "build", sources_target });
    const srcs_tar = b.path(b.pathJoin(&.{ "bazel-bin", sources_tar_path }));

    const tar_cmd = b.addSystemCommand(&.{ "tar", "-xf" });
    tar_cmd.step.dependOn(&bazel_cmd.step);
    tar_cmd.addFileArg(srcs_tar);
    tar_cmd.addArg("-C");
    const out_dir = tar_cmd.addOutputDirectoryArg("untarred_sources");

    var opts = options;
    if (opts.root_source_file != null) @panic("moduleFromBazelSrcs is already setting the root_source_file option");
    opts.root_source_file = out_dir.path(b, root);
    return if (module_name) |name|
        b.addModule(name, opts)
    else
        b.createModule(opts);
}
