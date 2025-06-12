const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mlir_c_deps = moduleFromBazelSrcs(
        b,
        "//mlir:sources",
        "mlir/sources.tar",
        "mlir/test_test_lib_c.zig",
        .{ .link_libcpp = true },
    );
    addObjectFromBazel(mlir_c_deps, "//mlir:static_c", "mlir/libstatic_c.a");

    const mlir_mod = b.addModule("mlir", .{
        .root_source_file = b.path("mlir/mlir.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "c", .module = mlir_c_deps },
        },
    });

    const mlir_test = b.addTest(.{ .root_module = mlir_mod });

    const run_mlir_tests = b.addRunArtifact(mlir_test);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_mlir_tests.step);
}

/// Take the name of a Bazel `cc_static_library` and add it to the given module.
fn addObjectFromBazel(module: *std.Build.Module, name: []const u8, output: []const u8) void {
    const b = module.owner;
    // TODO: consider parsing bazel name to generate output name.
    const cmd = b.addSystemCommand(&.{ "bazel", "build", "-c", "opt", name });
    const obj = b.path(b.pathJoin(&.{ "bazel-bin", output }));
    obj.addStepDependencies(&cmd.step);
    module.addObjectFile(obj);
}

fn moduleFromBazelSrcs(
    b: *std.Build,
    name: []const u8,
    output: []const u8,
    root: []const u8,
    options: std.Build.Module.CreateOptions,
) *std.Build.Module {
    // TODO: consider parsing bazel name to generate output name.
    const bazel_cmd = b.addSystemCommand(&.{ "bazel", "build", name });
    const srcs_tar = b.path(b.pathJoin(&.{ "bazel-bin", output }));
    srcs_tar.addStepDependencies(&bazel_cmd.step);

    const tar_cmd = b.addSystemCommand(&.{ "tar", "-xf" });
    tar_cmd.addFileArg(srcs_tar);
    tar_cmd.addArg("-C");
    const out_dir = tar_cmd.addOutputDirectoryArg("untarred_sources");

    var opts = options;
    if (opts.root_source_file != null) @panic("moduleFromBazelSrcs is already setting the root_source_file option");
    opts.root_source_file = out_dir.path(b, root);
    return b.addModule(name, opts);
}
