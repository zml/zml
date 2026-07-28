"""Expose the resolved rules_zig toolchain to the cudaz runtime."""

def _runfile_path(file, workspace_name):
    if file.short_path.startswith("../"):
        return file.short_path[3:]
    return workspace_name + "/" + file.short_path

def _zig_runtime_impl(ctx):
    toolchain = ctx.toolchains["@rules_zig//zig:toolchain_type"].zigtoolchaininfo
    if toolchain.mode != "file":
        fail("cudaz requires a hermetic file-backed Zig toolchain")

    zig_exe = toolchain.zig_exe.file
    zig_lib = toolchain.zig_lib.file
    kernel = ctx.file.kernel
    srcs = ctx.files.srcs
    config = ctx.actions.declare_file(ctx.label.name + ".txt")
    ctx.actions.write(
        output = config,
        content = "\n".join([
            _runfile_path(zig_exe, ctx.workspace_name),
            _runfile_path(zig_lib, ctx.workspace_name),
            _runfile_path(kernel, ctx.workspace_name),
            "",
        ]),
    )

    return DefaultInfo(
        files = depset([config]),
        runfiles = ctx.runfiles(files = [config, zig_exe, zig_lib, kernel] + srcs),
    )

zig_runtime = rule(
    implementation = _zig_runtime_impl,
    attrs = {
        "kernel": attr.label(
            allow_single_file = [".zig"],
            mandatory = True,
        ),
        "srcs": attr.label_list(
            allow_files = [".zig"],
        ),
    },
    toolchains = ["@rules_zig//zig:toolchain_type"],
)
