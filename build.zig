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
    const platforms = Platforms.parse(b);

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
    const mlir_srcs = Tarball.sources(b, "mlir/dialects");
    const mlir_c_deps = mlir_srcs.extractModule(
        "mlir/dialects",
        "mlir/dialects",
        "test_test_lib_c.zig",
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
    const pjrt_srcs = Tarball.sources(b, "pjrt");
    const pjrt_c_deps = pjrt_srcs.extractModule(
        "pjrt",
        "pjrt",
        "test_test_lib_c.zig",
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

    var _d: std.BoundedArray(*std.Build.Step.InstallFile, 8) = .{};
    deps: {
        switch (target.result.os.tag) {
            .macos => {
                if (platforms.cpu) {
                    const dep = b.lazyDependency("pjrt_cpu_darwin_arm64", .{}) orelse break :deps;
                    const pjrt_cpu = b.addInstallLibFile(dep.path("libpjrt_cpu.dylib"), "libpjrt_cpu.dylib");
                    _d.appendAssumeCapacity(pjrt_cpu);
                }
            },
            .linux => {
                if (platforms.cpu) {
                    const dep = b.lazyDependency("pjrt_cpu_linux_amd64", .{}) orelse break :deps;
                    const pjrt_cpu = b.addInstallLibFile(dep.path("libpjrt_cpu.so"), "libpjrt_cpu.so");
                    _d.appendAssumeCapacity(pjrt_cpu);
                }

                if (platforms.cuda) {
                    // TODO: this is not enough for libpjrt_cuda, there are a lot more deps needed.
                    const zmlxcuda = objectFromBazel(b, "@libpjrt_cuda//:zmlxcuda_so", "external/+cuda_packages+libpjrt_cuda/libzmlxcuda.so.0");
                    const plugin = objectFromBazel(b, "//runtimes/cuda:libpjrt_cuda", "runtimes/cuda/libpjrt_cuda.so");

                    _d.appendAssumeCapacity(b.addInstallLibFile(zmlxcuda, "libzmlxcuda.so.0"));
                    _d.appendAssumeCapacity(b.addInstallLibFile(plugin, "libpjrt_cuda.so"));
                }
            },
            else => |os| std.debug.panic("Target not supported: {s}", .{@tagName(os)}),
        }
    }
    const pjrt_dynamic_deps = _d.constSlice();

    // xev
    const async_srcs = Tarball.sources(b, "async");
    const xev = async_srcs.extractModule(
        "xev",
        "src",
        "main.zig",
        .{
            .target = target,
            .optimize = optimize,
            .link_libc = target.result.os.tag == .linux,
        },
    );

    const xev_test = b.addTest(.{ .root_module = xev });
    const run_xev_tests = b.addRunArtifact(xev_test);
    test_step.dependOn(&run_xev_tests.step);

    // async
    const async_mod = async_srcs.extractModule(
        "async",
        "async",
        "async.zig",
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

    const zml_srcs = Tarball.zml(b, platforms);
    // ffi
    const ffi = zml_srcs.extractModule(
        "ffi",
        "ffi",
        "ffi.zig",
        .{ .target = target, .optimize = optimize },
    );

    const ffi_test = b.addTest(.{ .root_module = ffi });
    const run_ffi_tests = b.addRunArtifact(ffi_test);
    test_step.dependOn(&run_ffi_tests.step);

    // hftokenizers
    const hftokenizers = zml_srcs.extractModule(
        "//zml/tokenizer/hftokenizers",
        "zml/tokenizer/hftokenizers",
        "hftokenizers.zig",
        .{ .target = target, .optimize = optimize },
    );

    // sentencepiece
    const sentencepiece_c_deps_srcs = Tarball.sources(b, "zml/tokenizer/sentencepiece");
    const sentencepiece_c_deps = sentencepiece_c_deps_srcs.extractModule(
        null,
        "zml/tokenizer/sentencepiece",
        "test_test_lib_c.zig",
        .{ .link_libcpp = true },
    );
    const sentencepiece = zml_srcs.extractModule(
        "//zml/tokenizer/sentencepiece",
        "zml/tokenizer/sentencepiece",
        "sentencepiece.zig",
        .{
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "c", .module = sentencepiece_c_deps },
            },
        },
    );

    // tokenizer
    const tokenizer = zml_srcs.extractModule(
        "tokenizer",
        "zml/tokenizer",
        "tokenizer.zig",
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
    const protobuf = zml_srcs.extractModule("protobuf", "src", "protobuf.zig", .{});
    const xla_compile_proto = xla_proto: {
        // Ideally we should ask bazel to generate this graph of dependencies between the different proto files.
        const empty: *std.Build.Module = empty: {
            const write = b.addWriteFiles();
            const root = write.add("root.zig", "//! empty module");
            break :empty b.createModule(.{ .root_source_file = root });
        };
        const opts: std.Build.Module.CreateOptions = .{ .imports = &.{.{ .name = "protobuf", .module = protobuf }} };

        const duration = zml_srcs.extractModule(null, "_virtual_imports/duration_proto/google/protobuf", "duration.pb.zig", opts);
        const any = zml_srcs.extractModule(null, "_virtual_imports/any_proto/google/protobuf", "any.pb.zig", opts);
        const wrappers = zml_srcs.extractModule(null, "_virtual_imports/wrappers_proto/google/protobuf", "wrappers.pb.zig", opts);

        const data = zml_srcs.extractModule(null, "xla", "xla_data.pb.zig", opts);
        const service_hlo = zml_srcs.extractModule(null, "xla/service", "hlo.pb.zig", opts);
        const xla = zml_srcs.extractModule(null, "xla", "xla.pb.zig", opts);
        xla.addImport("google_protobuf_any_proto", any);
        xla.addImport("xla_xla_data_proto", data);
        xla.addImport("xla_service_hlo_proto", service_hlo);
        xla.addImport("xla_autotune_results_proto", empty);

        const tsl_dnn = zml_srcs.extractModule(null, "xla/tsl/protobuf", "dnn.pb.zig", opts);
        tsl_dnn.addImport("google_protobuf_wrappers_proto", wrappers);

        const autotuning = zml_srcs.extractModule(null, "xla", "autotuning.pb.zig", opts);
        autotuning.addImport("google_protobuf_duration_proto", duration);
        autotuning.addImport("xla_tsl_protobuf_dnn_proto", tsl_dnn);
        const autotune = zml_srcs.extractModule(null, "xla", "autotune_results.pb.zig", opts);
        autotune.addImport("xla_autotuning_proto", autotuning);

        const stream_executor_cuda_cuda_compute_capability = zml_srcs.extractModule(null, "xla/stream_executor/cuda", "cuda_compute_capability.pb.zig", opts);
        const stream_executor_device_description = zml_srcs.extractModule(null, "xla/stream_executor", "device_description.pb.zig", opts);
        stream_executor_device_description.addImport("xla_autotune_results_proto", autotune);
        stream_executor_device_description.addImport("xla_stream_executor_cuda_cuda_compute_capability_proto", stream_executor_cuda_cuda_compute_capability);

        const compile = zml_srcs.extractModule(null, "xla/pjrt/proto", "compile_options.pb.zig", opts);
        compile.addImport("xla_xla_proto", xla);
        compile.addImport("xla_xla_data_proto", data);
        compile.addImport("xla_stream_executor_device_description_proto", stream_executor_device_description);

        break :xla_proto compile;
    };

    // zml c deps
    const zml_c_deps = zml_srcs.extractModule(
        null,
        "zml",
        "test_test_lib_c.zig",
        .{ .link_libcpp = true },
    );

    // bazel runfiles
    const runfiles = zml_srcs.extractModule("runfiles", "zig/runfiles", "runfiles.zig", .{});

    // runtimes
    const runtimes = zml_srcs.extractModule(
        "runtimes",
        "runtimes",
        "runtimes.zig",
        .{
            .imports = &.{
                .{ .name = "c", .module = zml_c_deps },
                .{ .name = "pjrt", .module = pjrt },
            },
        },
    );

    inline for (@typeInfo(Platforms).@"struct".fields) |field| {
        const platform = field.name;
        const runtime_path = b.pathJoin(&.{ "runtimes", platform });
        const zig_file = std.mem.concat(b.allocator, u8, &.{ platform, ".zig" }) catch @panic("OOM");
        const builtin_name = std.fmt.allocPrint(b.allocator, "bazel_builtin_A_S_Sruntimes_S{0s}_C{0s}.zig", .{platform}) catch @panic("OOM");
        const bazel_builtin = zml_srcs.extractModule(null, runtime_path, builtin_name, .{});

        const platform_runtime = zml_srcs.extractModule(
            runtime_path,
            runtime_path,
            zig_file,
            .{
                .imports = &.{
                    .{ .name = "c", .module = zml_c_deps },
                    .{ .name = "runfiles", .module = runfiles },
                    .{ .name = "stdx", .module = stdx },
                    .{ .name = "pjrt", .module = pjrt },
                    .{ .name = "async", .module = async_mod },
                    .{ .name = "bazel_builtin", .module = bazel_builtin },
                },
            },
        );
        const import_name = std.mem.concat(b.allocator, u8, &.{ "runtimes/", platform }) catch @panic("OOM");
        runtimes.addImport(import_name, platform_runtime);
    }

    // zml/tools
    const zml_tools = zml_srcs.extractModule(
        "zml/tools",
        "zml/tools",
        "tools.zig",
        .{
            .link_libcpp = target.result.os.tag == .macos,
            .imports = &.{.{ .name = "c", .module = zml_c_deps }},
        },
    );
    const macos_tools_obj = objectFromBazel(b, "//zml/tools:macos_static_tools", "zml/tools/libmacos_static_tools.a");
    if (target.result.os.tag == .macos) {
        zml_tools.addObjectFile(macos_tools_obj);
    }

    // zml
    const zml_srcs_tar = b.addSystemCommand(&.{ "bazel", "build", "//zml:sources" });
    if (platforms.cpu) zml_srcs_tar.addArg("--@zml//runtimes:cpu=true");
    if (platforms.cuda) zml_srcs_tar.addArg("--@zml//runtimes:cuda=true");
    const zml = zml_srcs.extractModule(
        "zml",
        "zml",
        "zml.zig",
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
    // This is where we will put all the .so needed.
    zml_test.addRPath(.{ .cwd_relative = b.lib_dir });

    const run_zml_tests = b.addRunArtifact(zml_test);
    addRuntimeDeps(run_zml_tests, pjrt_dynamic_deps);

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

const Tarball = struct {
    create_cmd: *std.Build.Step.Run,
    path: std.Build.LazyPath,

    /// Ask bazel for the full sources of a zig module.
    /// This is needed for module that have generated zig sources,
    /// like the output of zig translate-c or protobuf generated sources.
    pub fn sources(b: *std.Build, name: []const u8) Tarball {
        const allocator = b.allocator;
        const target = std.mem.concat(allocator, u8, &.{ "//", name, ":sources" }) catch @panic("OOM");
        const bazel_cmd: *std.Build.Step.Run = b.addSystemCommand(&.{ "bazel", "build", target });

        const srcs_tar = b.path(b.pathJoin(&.{ "bazel-bin", name, "sources.tar" }));
        return .{ .create_cmd = bazel_cmd, .path = srcs_tar };
    }

    /// Zml is a bit different from other targets cause the code of the "runtimes/**.zig"
    /// will change based on the platforms selected.
    pub fn zml(b: *std.Build, platforms: Platforms) Tarball {
        const bazel_cmd: *std.Build.Step.Run = b.addSystemCommand(&.{ "bazel", "build", "//zml:sources" });
        bazel_cmd.addArg(if (platforms.cpu) "--@zml//runtimes:cpu=true" else "--@zml//runtimes:cpu=false");
        bazel_cmd.addArg(if (platforms.cuda) "--@zml//runtimes:cuda=true" else "--@zml//runtimes:cuda=false");

        const srcs_tar = b.path(b.pathJoin(&.{ "bazel-bin", "zml", "sources.tar" }));
        return .{ .create_cmd = bazel_cmd, .path = srcs_tar };
    }

    pub fn extractModule(self: Tarball, module_name: ?[]const u8, directory: []const u8, root: []const u8, options: std.Build.Module.CreateOptions) *std.Build.Module {
        const b = self.create_cmd.step.owner;
        const tar_cmd = b.addSystemCommand(&.{ "tar", "-xf" });
        tar_cmd.step.dependOn(&self.create_cmd.step);
        tar_cmd.addFileArg(self.path);
        tar_cmd.addArg("-C");
        const out_dir = tar_cmd.addOutputDirectoryArg("sources");
        tar_cmd.addArg(directory);

        var opts = options;
        if (opts.root_source_file != null) @panic("moduleFromBazelSrcs is already setting the root_source_file option");
        opts.root_source_file = out_dir.path(b, b.pathJoin(&.{ directory, root }));
        return if (module_name) |name|
            b.addModule(name, opts)
        else
            b.createModule(opts);
    }
};

fn addRuntimeDeps(run_step: *std.Build.Step.Run, libs: []const *std.Build.Step.InstallFile) void {
    for (libs) |lib| {
        run_step.step.dependOn(&lib.step);
    }
}

const Platforms = struct {
    cpu: bool,
    cuda: bool,
    rocm: bool = false,
    tpu: bool = false,
    neuron: bool = false,

    pub fn parse(b: *std.Build) Platforms {
        return .{
            .cpu = b.option(bool, "runtimes:cpu", "Enable cpu runtime (default: true)") orelse true,
            .cuda = b.option(bool, "runtimes:cuda", "Enable cuda runtime") orelse false,
        };
    }
};
