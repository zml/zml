const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{});
    const run = b.step("run", "run");

    {
        const mnist = b.step("mnist", "mnist");
        const mnist_cmd = b.addSystemCommand(&.{ "bazelisk", "run", "//examples/mnist" });
        mnist.dependOn(&mnist_cmd.step);
    }

    {
        const llm = b.step("llm", "llm");
        const llm_cmd = b.addSystemCommand(&.{ "bazelisk", "run", "//examples/llm" });
        if (optimize != .Debug) {
            llm_cmd.addArg("--config=release");
        }
        llm_cmd.addArg("--");
        if (b.args) |cli_args|
            llm_cmd.addArgs(cli_args);

        llm.dependOn(&llm_cmd.step);
        run.dependOn(llm);
    }

    const test_ = b.step("test", "test");

    const stdx_test = b.addSystemCommand(&.{ "bazelisk", "run", "//stdx:test" });
    test_.dependOn(&stdx_test.step);

    const zml_test = b.addSystemCommand(&.{ "bazelisk", "run", "//zml:test" });
    test_.dependOn(&zml_test.step);
}
