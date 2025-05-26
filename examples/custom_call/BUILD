load("@zml//bazel:zig.bzl", "zig_cc_binary")

zig_cc_binary(
    name = "custom_call",
    main = "main.zig",
    deps = [
        "@zml//async",
        "@zml//zml",
    ],
)

zig_cc_binary(
    name = "playground",
    copts = ["-mcpu=native"],
    extra_srcs = glob([
        "*.ptx",
        "*.cubin",
    ]),
    main = "playground.zig",
    deps = [
        "@zml//async",
        "@zml//zml",
    ],
)
