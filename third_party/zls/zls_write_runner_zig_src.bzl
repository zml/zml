"""Generates the ZLS runner source file."""

load("@aspect_bazel_lib//lib:paths.bzl", "to_rlocation_path")

def _zls_write_runner_zig_src_impl(ctx):
    zigtoolchaininfo = ctx.toolchains["@rules_zig//zig:toolchain_type"].zigtoolchaininfo
    zlstoolchaininfo = ctx.toolchains["//third_party/zls:toolchain_type"].zlstoolchaininfo

    zls_runner = ctx.outputs.out
    ctx.actions.expand_template(
        output = zls_runner,
        template = ctx.file._runner_tpl,
        substitutions = {
            "@@__ZIG_EXE_RPATH__@@": zigtoolchaininfo.zig_exe_rpath,
            "@@__ZIG_LIB_PATH__@@": zigtoolchaininfo.zig_lib_rpath,
            "@@__ZLS_BIN_RPATH__@@": to_rlocation_path(ctx, zlstoolchaininfo.bin),
            "@@__ZLS_BUILD_RUNNER_RPATH__@@": to_rlocation_path(ctx, ctx.file.build_runner),
            "@@__GLOBAL_CACHE_PATH__@@": zigtoolchaininfo.zig_cache,
        },
    )

    return [
        DefaultInfo(files = depset([zls_runner])),
    ]

zls_write_runner_zig_src = rule(
    implementation = _zls_write_runner_zig_src_impl,
    attrs = {
        "out": attr.output(
            mandatory = True,
        ),
        "build_runner": attr.label(
            allow_single_file = True,
            mandatory = True,
        ),
        "_runner_tpl": attr.label(
            default = "zls_runner.zig",
            allow_single_file = True,
        ),
    },
    toolchains = [
        "@rules_zig//zig:toolchain_type",
        "//third_party/zls:toolchain_type",
    ],
)
