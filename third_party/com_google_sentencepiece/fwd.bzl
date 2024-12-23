def _include_fwd_impl(ctx):
    files = []
    for include in ctx.attr.includes:
        f = ctx.actions.declare_file(include)
        ctx.actions.write(f, '#include "{}"'.format(include))
        files.append(f)
    return [DefaultInfo(files = depset(files))]

include_fwd = rule(
    implementation = _include_fwd_impl,
    attrs = {
        "includes": attr.string_list(),
    },
)
