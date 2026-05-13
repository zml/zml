const std = @import("std");

/// Disclaimer: this is not a real build.zig,
/// it just defers to bazel to do anything.
/// Our toolchain uses many third party dependencies outside of Zig ecosystem
/// that are using Bazel one way or another.
/// Leveraging Bazel and LLVM sandboxing also allows crazy things,
/// like building a branch of ZML with a local Zig fork built with a local LLVM fork,
/// all of this with a simple `bazel test //zml:test`
///
/// So this is just here to help code editor pick up this project as Zig,
/// and exposing `zig build test` to the user.
pub fn build(b: *std.Build) void {
    const test_ = b.step("test", "Run unit tests");

    const stdx_test = b.addSystemCommand(&.{ "bazelisk", "test", "//stdx/..." });
    test_.dependOn(&stdx_test.step);

    const mlir_test = b.addSystemCommand(&.{ "bazelisk", "test", "//mlir/..." });
    test_.dependOn(&mlir_test.step);

    const kernel_test = b.addSystemCommand(&.{ "bazelisk", "test", "//kernels/..." });
    test_.dependOn(&kernel_test.step);

    const pjrt_test = b.addSystemCommand(&.{
        "bazelisk",
        "test",
        "//pjrt/...",
    });
    test_.dependOn(&pjrt_test.step);

    const zml_test = b.addSystemCommand(&.{ "bazelisk", "test", "//zml/..." });
    test_.dependOn(&zml_test.step);

    _ = b.step("disclaimer",
        \\
        \\    ❗This is not a real build.zig, it just shells out to bazel.
        \\    ❗You should be using `bazel run ...` see README to get started.
        \\    ❗This is intended to help with editor integration.
    );
}
