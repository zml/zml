const std = @import("std");

/// !!! This build.zig is work in progress. !!!
///
/// It shows how to bridge Bazel and build.zig.
/// It requires the user to have `bazel` and `tar` installed on their machine.
///
/// Bazel is used to:
///
/// * compile C and C++ deps into .a files.
/// * call zig-translate C
/// * generating Zig files
/// * tarring Zig sources
///
/// build.zig finishes the work by:
/// * untarring Zig sources
/// * copying .a files into zig-cache
/// * creating "zig modules" visible to other build.zig.
///
/// `zig build test --summary all` will run tests for several ZML deps,
/// but not yet ZML itself.
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const test_step = b.step("test", "Run unit tests");

    // mlir
    const mlir_c_deps = moduleFromBazelSrcs(
        b,
        null,
        .canonical(b.allocator, "mlir", "test_test_lib_c.zig"),
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
        .canonical(b.allocator, "pjrt", "test_test_lib_c.zig"),
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
    const stdx = b.addModule("stdx", .{
        .root_source_file = b.path("stdx/stdx.zig"),
        .target = target,
        .optimize = optimize,
    });

    const stdx_test = b.addTest(.{ .root_module = stdx });
    const run_stdx_tests = b.addRunArtifact(stdx_test);
    test_step.dependOn(&run_stdx_tests.step);

    // xev
    const xev = moduleFromBazelSrcs(
        b,
        "xev",
        .{
            // xev sources can be find inside async sources cause it's a dependency.
            .target = "//async:sources",
            .tar_path = "async/sources.tar",
            .directory = "src",
            .root = "main.zig",
        },
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
        .canonical(b.allocator, "async", "async.zig"),
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

    // ffi
    const ffi = moduleFromBazelSrcs(
        b,
        "ffi",
        .fromZml("ffi", "ffi.zig"),
        .{ .target = target, .optimize = optimize },
    );

    const ffi_test = b.addTest(.{ .root_module = ffi });
    const run_ffi_tests = b.addRunArtifact(ffi_test);
    test_step.dependOn(&run_ffi_tests.step);
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

const BazelSrcs = struct {
    target: []const u8,
    tar_path: []const u8,
    directory: []const u8,
    root: []const u8,

    pub fn canonical(allocator: std.mem.Allocator, name: []const u8, root: []const u8) BazelSrcs {
        return .{
            .target = std.mem.concat(allocator, u8, &.{ "//", name, ":sources" }) catch @panic("OOM"),
            .tar_path = std.fs.path.join(allocator, &.{ name, "sources.tar" }) catch @panic("OOM"),
            .directory = name,
            .root = root,
        };
    }

    pub fn fromZml(name: []const u8, root: []const u8) BazelSrcs {
        return .{
            .target = "//zml:sources",
            .tar_path = "zml/sources.tar",
            .directory = name,
            .root = root,
        };
    }
};

/// Ask bazel for the full sources of a zig module.
/// This is needed for module that have generated zig sources,
/// like the output of zig translate-c or protobuf generated sources.
fn moduleFromBazelSrcs(
    b: *std.Build,
    module_name: ?[]const u8,
    srcs: BazelSrcs,
    options: std.Build.Module.CreateOptions,
) *std.Build.Module {
    const bazel_cmd = b.addSystemCommand(&.{ "bazel", "build", srcs.target });
    const srcs_tar = b.path(b.pathJoin(&.{ "bazel-bin", srcs.tar_path }));

    const tar_cmd = b.addSystemCommand(&.{ "tar", "-xf" });
    tar_cmd.step.dependOn(&bazel_cmd.step);
    tar_cmd.addFileArg(srcs_tar);
    tar_cmd.addArg("-C");
    const out_dir = tar_cmd.addOutputDirectoryArg("sources");
    tar_cmd.addArg(srcs.directory);

    var opts = options;
    if (opts.root_source_file != null) @panic("moduleFromBazelSrcs is already setting the root_source_file option");
    opts.root_source_file = out_dir.path(b, b.pathJoin(&.{ srcs.directory, srcs.root }));
    return if (module_name) |name|
        b.addModule(name, opts)
    else
        b.createModule(opts);
}
