const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Note: setting -Dcuda on CLI fails cause zml_sandbox doesn't yet contains all CUDA deps.
    const cuda = b.option(bool, "cuda", "Enable cuda (broken)") orelse false;
    const zml_pkg = b.dependency("zml", .{
        .@"runtimes:cuda" = cuda,
        .@"runtimes:cpu" = !cuda,
    });
    const zml_sandbox = zml_pkg.namedWriteFiles("zml_sandbox");

    const mnist = b.createModule(.{
        .root_source_file = b.path("mnist.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "async", .module = zml_pkg.module("async") },
            .{ .name = "zml", .module = zml_pkg.module("zml") },
        },
    });

    const exe = b.addExecutable(.{ .name = "mnist", .root_module = mnist });
    exe.addRPath(zml_sandbox.getDirectory());
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    {
        const ggml = b.dependency("ggml", .{});
        const mnist_model = ggml.path("examples/mnist/models/mnist/mnist_model.state_dict");
        const mnist_data = ggml.path("examples/mnist/models/mnist/t10k-images.idx3-ubyte");

        run_cmd.addFileArg(mnist_model);
        run_cmd.addFileArg(mnist_data);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| {
            run_cmd.addArgs(args);
        }
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
