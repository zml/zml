def _strip_debug_impl(ctx):
    src = ctx.file.src
    output = ctx.actions.declare_file("{}/{}".format(ctx.attr.name, src.basename))

    ctx.actions.run(
        executable = ctx.file._strip,
        arguments = ["--strip-debug", "-o", output.path, src.path],
        inputs = [src],
        outputs = [output],
    )

    return [
        DefaultInfo(
            files = depset([output]),
        ),
    ]

strip_debug = rule(
    implementation = _strip_debug_impl,
    attrs = {
        "src": attr.label(allow_single_file = True, mandatory = True),
        "_strip": attr.label(
            default = "@llvm//tools:llvm-strip",
            allow_single_file = True,
            cfg = "exec",
        ),
    },
)
