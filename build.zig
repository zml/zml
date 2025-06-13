const std = @import("std");

/// !!! This build.zig is experimental !!!
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
/// `zig build test --summary all` will run a lot of tests
/// `zig build test-zml` will stick to ZML test suite
///
/// Caveats:
///
/// * requires a local bazel installed and tar utility
/// * requires to have the CPU pjrt plugin visible in the path when running the executables
/// you can do so eg by find -L ./bazel-out -name libpjrt_cpu.dylib then cp ./bazel-out/.../libpjt_cpu.dylib .
/// * only tested with CPU plugins
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const test_step = b.step("test", "Run all tests across ZML and deps");

    // stdx
    const stdx = b.addModule("stdx", .{
        .root_source_file = b.path("stdx/stdx.zig"),
        .target = target,
        .optimize = optimize,
    });

    const stdx_test = b.addTest(.{ .root_module = stdx });
    const run_stdx_tests = b.addRunArtifact(stdx_test);
    test_step.dependOn(&run_stdx_tests.step);

    // mlir
    const mlir_c_deps = moduleFromBazelSrcs(
        b,
        null,
        .canonical(b.allocator, "mlir/dialects", "test_test_lib_c.zig"),
        .{ .link_libcpp = true },
    );
    const mlir_obj = objectFromBazel(b, "//mlir/dialects:mlir_static", "mlir/dialects/libmlir_static.a");

    const mlir = b.addModule("mlir", .{
        .root_source_file = b.path("mlir/mlir.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "c", .module = mlir_c_deps },
            .{ .name = "stdx", .module = stdx },
        },
    });

    const mlir_test = b.addTest(.{ .root_module = mlir });
    // TODO: I'm not sure what's the best idea: add the object to the compile step or directly to the module.
    mlir_test.addObjectFile(mlir_obj);
    const run_mlir_tests = b.addRunArtifact(mlir_test);
    test_step.dependOn(&run_mlir_tests.step);

    const stablehlo = b.addModule("mlir/dialects/stablehlo", .{
        .root_source_file = b.path("mlir/dialects/stablehlo.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "c", .module = mlir_c_deps },
            .{ .name = "mlir", .module = mlir },
            .{ .name = "stdx", .module = stdx },
        },
    });

    const mlir_dialects = b.addModule("mlir/dialects", .{
        .root_source_file = b.path("mlir/dialects/dialects.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "mlir", .module = mlir },
            .{ .name = "mlir/dialects/stablehlo", .module = stablehlo },
        },
    });

    const mlir_dialects_test = b.addTest(.{ .root_module = mlir_dialects });
    mlir_dialects_test.addObjectFile(mlir_obj);
    const run_mlir_dialects_tests = b.addRunArtifact(mlir_dialects_test);
    test_step.dependOn(&run_mlir_dialects_tests.step);

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
            .{ .name = "stdx", .module = stdx },
        },
    });

    const pjrt_test = b.addTest(.{ .root_module = pjrt });
    const run_pjrt_tests = b.addRunArtifact(pjrt_test);
    test_step.dependOn(&run_pjrt_tests.step);

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

    // hftokenizers
    const hftokenizers = moduleFromBazelSrcs(
        b,
        "//zml/tokenizer/hftokenizers",
        .fromZml("zml/tokenizer/hftokenizers", "hftokenizers.zig"),
        .{ .target = target, .optimize = optimize },
    );

    // sentencepiece
    const sentencepiece_c_deps = moduleFromBazelSrcs(
        b,
        null,
        .canonical(b.allocator, "zml/tokenizer/sentencepiece", "test_test_lib_c.zig"),
        .{ .link_libcpp = true },
    );
    const sentencepiece = moduleFromBazelSrcs(
        b,
        "//zml/tokenizer/sentencepiece",
        .fromZml("zml/tokenizer/sentencepiece", "sentencepiece.zig"),
        .{
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "c", .module = sentencepiece_c_deps },
            },
        },
    );

    // tokenizer
    const tokenizer = moduleFromBazelSrcs(
        b,
        "tokenizer",
        .fromZml("zml/tokenizer", "tokenizer.zig"),
        .{
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "async", .module = async_mod },
                .{ .name = "ffi", .module = ffi },
                // Note: even though we import sentencepiece and hftokenizers,
                // we don't link the corresponding .a cause there is no tests
                // for them directly.
                .{ .name = "hftokenizers", .module = hftokenizers },
                .{ .name = "sentencepiece", .module = sentencepiece },
            },
        },
    );

    const tokenizer_test = b.addTest(.{ .root_module = tokenizer });
    const run_tokenizer_tests = b.addRunArtifact(tokenizer_test);
    test_step.dependOn(&run_tokenizer_tests.step);

    // proto
    const protobuf = moduleFromBazelSrcs(b, "protobuf", .fromZml("src", "protobuf.zig"), .{});
    const xla_compile_proto = xla_proto: {
        // this is horrible, look away.
        const empty: *std.Build.Module = empty: {
            const write = b.addWriteFiles();
            const root = write.add("root.zig", "//! empty module");
            break :empty b.createModule(.{ .root_source_file = root });
        };
        const opts: std.Build.Module.CreateOptions = .{ .imports = &.{.{ .name = "protobuf", .module = protobuf }} };

        const duration = moduleFromBazelSrcs(b, null, .fromZml("_virtual_imports/duration_proto/google/protobuf", "duration.pb.zig"), opts);
        const any = moduleFromBazelSrcs(b, null, .fromZml("_virtual_imports/any_proto/google/protobuf", "any.pb.zig"), opts);
        const wrappers = moduleFromBazelSrcs(b, null, .fromZml("_virtual_imports/wrappers_proto/google/protobuf", "wrappers.pb.zig"), opts);

        const data = moduleFromBazelSrcs(b, null, .fromZml("xla", "xla_data.pb.zig"), opts);
        const service_hlo = moduleFromBazelSrcs(b, null, .fromZml("xla/service", "hlo.pb.zig"), opts);
        const xla = moduleFromBazelSrcs(b, null, .fromZml("xla", "xla.pb.zig"), opts);
        xla.addImport("google_protobuf_any_proto", any);
        xla.addImport("xla_xla_data_proto", data);
        xla.addImport("xla_service_hlo_proto", service_hlo);
        xla.addImport("xla_autotune_results_proto", empty);

        const tsl_dnn = moduleFromBazelSrcs(b, null, .fromZml("xla/tsl/protobuf", "dnn.pb.zig"), opts);
        tsl_dnn.addImport("google_protobuf_wrappers_proto", wrappers);

        const autotuning = moduleFromBazelSrcs(b, null, .fromZml("xla", "autotuning.pb.zig"), opts);
        autotuning.addImport("google_protobuf_duration_proto", duration);
        autotuning.addImport("xla_tsl_protobuf_dnn_proto", tsl_dnn);
        const autotune = moduleFromBazelSrcs(b, null, .fromZml("xla", "autotune_results.pb.zig"), opts);
        autotune.addImport("xla_autotuning_proto", autotuning);

        const stream_executor_cuda_cuda_compute_capability = moduleFromBazelSrcs(b, null, .fromZml("xla/stream_executor/cuda", "cuda_compute_capability.pb.zig"), opts);
        const stream_executor_device_description = moduleFromBazelSrcs(b, null, .fromZml("xla/stream_executor", "device_description.pb.zig"), opts);
        stream_executor_device_description.addImport("xla_autotune_results_proto", autotune);
        stream_executor_device_description.addImport("xla_stream_executor_cuda_cuda_compute_capability_proto", stream_executor_cuda_cuda_compute_capability);

        const compile = moduleFromBazelSrcs(b, null, .fromZml("xla/pjrt/proto", "compile_options.pb.zig"), opts);
        compile.addImport("xla_xla_proto", xla);
        compile.addImport("xla_xla_data_proto", data);
        compile.addImport("xla_stream_executor_device_description_proto", stream_executor_device_description);

        break :xla_proto compile;
    };

    // zml c deps
    const zml_c_deps = moduleFromBazelSrcs(
        b,
        null,
        .fromZml("zml", "test_test_lib_c.zig"),
        .{ .link_libcpp = true },
    );

    // bazel runfiles
    const runfiles = moduleFromBazelSrcs(b, "runfiles", .fromZml("zig/runfiles", "runfiles.zig"), .{});

    // runtimes
    const runtimes = moduleFromBazelSrcs(
        b,
        "runtimes",
        .fromZml("runtimes", "runtimes.zig"),
        .{
            .imports = &.{
                .{ .name = "c", .module = zml_c_deps },
                .{ .name = "pjrt", .module = pjrt },
            },
        },
    );
    const PLATFORMS = [_][]const u8{ "cpu", "cuda", "rocm", "tpu", "neuron" };

    for (&PLATFORMS) |platform| {
        const runtime_path = b.pathJoin(&.{ "runtimes", platform });
        const zig_file = std.mem.concat(b.allocator, u8, &.{ platform, ".zig" }) catch @panic("OOM");
        const platform_runtime = moduleFromBazelSrcs(
            b,
            runtime_path,
            .fromZml(runtime_path, zig_file),
            .{
                .imports = &.{
                    .{ .name = "c", .module = zml_c_deps },
                    .{ .name = "runfiles", .module = runfiles },
                    .{ .name = "pjrt", .module = pjrt },
                    .{ .name = "async", .module = async_mod },
                },
            },
        );
        const import_name = std.mem.concat(b.allocator, u8, &.{ "runtimes/", platform }) catch @panic("OOM");
        runtimes.addImport(import_name, platform_runtime);
    }

    // zml/tools
    const zml_tools = moduleFromBazelSrcs(
        b,
        "zml/tools",
        .fromZml("zml/tools", "tools.zig"),
        .{
            .link_libcpp = target.query.os_tag == .macos,
            .imports = &.{.{ .name = "c", .module = zml_c_deps }},
        },
    );
    const macos_tools_obj = objectFromBazel(b, "//zml/tools:macos_static_tools", "zml/tools/libmacos_static_tools.a");
    if (target.result.os.tag == .macos) {
        zml_tools.addObjectFile(macos_tools_obj);
    }

    // zml
    const zml = moduleFromBazelSrcs(
        b,
        "zml",
        .fromZml("zml", "zml.zig"),
        .{
            .target = target,
            .optimize = optimize,
            .link_libcpp = true,
            .imports = &.{
                .{ .name = "c", .module = zml_c_deps },
                .{ .name = "//xla:xla_proto", .module = xla_compile_proto },
                .{ .name = "async", .module = async_mod },
                .{ .name = "mlir", .module = mlir },
                .{ .name = "mlir/dialects", .module = mlir_dialects },
                .{ .name = "pjrt", .module = pjrt },
                .{ .name = "runtimes", .module = runtimes },
                .{ .name = "stdx", .module = stdx },
                .{ .name = "zml/tokenizer", .module = tokenizer },
                .{ .name = "zml/tools", .module = zml_tools },
                .{ .name = "runfiles", .module = runfiles },
            },
        },
    );

    const zml_test = b.addTest(.{
        .root_module = zml,
        .test_runner = .{ .mode = .simple, .path = b.path("zml/test_runner.zig") },
    });
    const run_zml_tests = b.addRunArtifact(zml_test);
    // TODO let copy the pjrt dylib somewhere Zig can see them.
    const zml_test_step = b.step("test-zml", "Run ZML tests (assumes pjrt.dylib are in the path)");
    zml_test_step.dependOn(&run_zml_tests.step);
    test_step.dependOn(&run_zml_tests.step);
}

/// Take the name of a Bazel `cc_static_library` and create a object LazyPath from it.
/// The object need to be added to a Compile step with `step.addObjectFile(obj)`.
fn objectFromBazel(b: *std.Build, target: []const u8, output: []const u8) std.Build.LazyPath {
    // TODO: consider parsing bazel target to generate output path.
    const bazel_cmd = b.addSystemCommand(&.{ "bazel", "build", "-c", "opt", target });
    const obj_path = b.pathJoin(&.{ "bazel-bin", output });

    // Copy bazel output into zig-cache, cause bazel may remove the file later.
    const cp = b.addWriteFiles();
    cp.step.dependOn(&bazel_cmd.step);
    return cp.addCopyFile(b.path(obj_path), output);
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

var zml_srcs_tar: ?*std.Build.Step.Run = null;

/// Ask bazel for the full sources of a zig module.
/// This is needed for module that have generated zig sources,
/// like the output of zig translate-c or protobuf generated sources.
fn moduleFromBazelSrcs(
    b: *std.Build,
    module_name: ?[]const u8,
    srcs: BazelSrcs,
    options: std.Build.Module.CreateOptions,
) *std.Build.Module {
    const bazel_cmd: *std.Build.Step.Run = cmd: {
        if (std.mem.eql(u8, srcs.target, "//zml:sources")) {
            if (zml_srcs_tar == null) {
                zml_srcs_tar = b.addSystemCommand(&.{ "bazel", "build", srcs.target });
            }
            break :cmd zml_srcs_tar.?;
        }
        break :cmd b.addSystemCommand(&.{ "bazel", "build", srcs.target });
    };
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
