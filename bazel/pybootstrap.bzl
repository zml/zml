load("@rules_python//python:defs.bzl", "PyInfo")

def _python_bootstrap_impl(ctx):
    content = []
    py_toolchain = ctx.toolchains["@rules_python//python:toolchain_type"].py3_runtime
    content.append(py_toolchain.interpreter.short_path)

    for file in py_toolchain.files.to_list():
        if file.basename == "__hello__.py":
            content.append(file.short_path.removesuffix("/__hello__.py"))

    imports = depset([], transitive = [
        dep[PyInfo].imports
        for dep in ctx.attr.deps
    ])
    content.extend([
        "../{}".format(imp)
        for imp in imports.to_list()
    ])
    f = ctx.actions.declare_file("{}.txt".format(ctx.label.name))
    ctx.actions.write(f, "\n".join(content))

    return [
        DefaultInfo(
            files = depset([f]),
        ),
    ]

python_bootstrap = rule(
    implementation = _python_bootstrap_impl,
    attrs = {
        "deps": attr.label_list(providers = [PyInfo]),
    },
    toolchains = ["@rules_python//python:toolchain_type"],
)
