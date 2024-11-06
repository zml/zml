load("@rules_python//python:defs.bzl", "PyInfo")

def _python_bootstrap_impl(ctx):
    py_toolchain = ctx.toolchains["@rules_python//python:toolchain_type"].py3_runtime
    content = "pub const home = {};\n".format(repr(py_toolchain.interpreter.short_path))

    modules = []
    for file in py_toolchain.files.to_list():
        if file.basename == "__hello__.py":
            modules.append(repr(file.short_path.removesuffix("/__hello__.py")))

    imports = depset([], transitive = [
        dep[PyInfo].imports
        for dep in ctx.attr.deps
    ])
    modules.extend([
        repr("../{}".format(imp))
        for imp in imports.to_list()
    ])
    content += "pub const modules = .{{ {} }};\n".format(", ".join(modules))
    content += "pub const main = {};\n".format(repr(ctx.file.main.short_path + "c"))
    content += "pub const main_compiled = {};\n".format(repr(ctx.file.main.short_path))

    f = ctx.actions.declare_file("{}.zig".format(ctx.label.name))
    ctx.actions.write(f, content)

    return [
        DefaultInfo(
            files = depset([f]),
        ),
    ]

python_bootstrap = rule(
    implementation = _python_bootstrap_impl,
    attrs = {
        "main": attr.label(mandatory = True, allow_single_file = True),
        "deps": attr.label_list(providers = [PyInfo]),
    },
    toolchains = ["@rules_python//python:toolchain_type"],
)

# def pyy_binary(name, main, **kwargs):
#     native.py_binary(
#         name = "{}.py".format(name),
#         main = main,
#         **kwargs
#     )
#     python_bootstrap(
#         name = "{}.bootstrap".format(name),
#         main = main,
#         deps = [":{}.py".format(name)],
#     )
#     zig_library(
#         name = "{}_options".format(name),
#         main = ":pymain.zig",
#         deps = [":{}.bootstrap".format(name), "@zml//py:libpython"],
#     )
