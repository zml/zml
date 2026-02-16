"""Implementation of the zls_completion macro."""

load("@aspect_bazel_lib//lib:utils.bzl", "utils")
load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@rules_zig//zig:defs.bzl", "zig_binary")
load(":zls_write_build_config.bzl", "zls_write_build_config")
load(":zls_write_runner_zig_src.bzl", "zls_write_runner_zig_src")

def zls_completion(name, deps, **kwargs):
    """Entry point for ZLS completion.

    Args:
        name: The name of the completion target.
        deps: The List of Zig modules to include for completion.
        **kwargs: Additional keyword arguments passed to the `zig_binary`
    """

    # Generate the ZLS BuildConfig file.
    # It contains the list of Zig packages alongside their main Zig file paths.
    build_config = name + ".build_config"
    zls_write_build_config(
        name = build_config,
        out = name + ".build_config.json",
        deps = deps,
    )

    # Create a target that will be invoked by ZLS using our customer build_runner.
    build_config_printer = "{}.print_build_config".format(name)
    zig_binary(
        name = build_config_printer,
        main = Label("//third_party/zls:workspace_printer.zig"),
        data = [
            ":{}".format(build_config),
        ],
        args = [
            "$(location :{})".format(build_config),
        ],
        visibility = ["//visibility:private"],
    )

    # Generate the Zig build runner that will be used by ZLS to query the build config.
    expand_template(
        name = name + ".build_runner",
        out = name + ".build_runner.zig",
        substitutions = {
            "@@__TARGET__@@": str(utils.to_label(build_config_printer)),
        },
        template = Label(":zls_build_runner.zig"),
    )

    # Generate the Zig source file for the ZLS runner binary which embeds the
    # rlocationpath of all runtime dependencies of the ZLS runner binary.
    zls_write_runner_zig_src(
        name = name + ".runner",
        out = name + ".runner.zig",
        build_runner = ":" + name + ".build_runner.zig",
    )

    zig_binary(
        name = name,
        main = name + ".runner",
        data = [
            "@rules_zig//zig:resolved_toolchain",
            "@@//third_party/zls:resolved_toolchain",
            name + ".build_runner",
        ],
        deps = [
            "@rules_zig//zig/runfiles",
        ],
        **kwargs
    )
