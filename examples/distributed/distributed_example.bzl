load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@rules_zig//zig:defs.bzl", "zig_binary")

def distributed_example(name, deps = ["//zml"]):
    zig_binary(
        name = name,
        main = "main.zig",
        visibility = ["//visibility:public"],
        deps = deps,
    )
    build_test(
        name = "test",
        targets = [":" + name],
    )
