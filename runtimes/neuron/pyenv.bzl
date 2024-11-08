load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_python//python:defs.bzl", "PyInfo")

def runfile_path(ctx, runfile):
    return paths.normalize(ctx.workspace_name + "/" + runfile.short_path)

def _pyenv_zig_impl(ctx):
    py_toolchain = ctx.toolchains["@rules_python//python:toolchain_type"].py3_runtime
    content = "pub const home: [:0]const u8 = {};\n".format(repr(runfile_path(ctx, py_toolchain.interpreter)))

    modules = []
    for file in py_toolchain.files.to_list():
        if file.basename == "__hello__.py":
            modules.append(repr(runfile_path(ctx, file)))

    imports = depset([], transitive = [
        dep[PyInfo].imports
        for dep in ctx.attr.deps
    ])
    modules.extend([
        repr("{}/__init__.py".format(imp))
        for imp in imports.to_list()
    ])
    content += "pub const modules: []const [:0]const u8 = &.{{ {} }};\n".format(", ".join(modules))

    f = ctx.actions.declare_file("{}.zig".format(ctx.label.name))
    ctx.actions.write(f, content)

    return [
        DefaultInfo(
            files = depset([f]),
        ),
    ]

pyenv_zig = rule(
    implementation = _pyenv_zig_impl,
    attrs = {
        "deps": attr.label_list(providers = [PyInfo]),
    },
    toolchains = ["@rules_python//python:toolchain_type"],
)
