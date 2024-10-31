def _runfiles_to_default(ctx):
    files = depset([], transitive = [
        dep[DefaultInfo].default_runfiles.files
        for dep in ctx.attr.deps
    ])
    return [
        DefaultInfo(
            files = files,
        ),
    ]

runfiles_to_default = rule(
    implementation = _runfiles_to_default,
    attrs = {
        "deps": attr.label_list(providers = [DefaultInfo]),  # We expect DefaultInfo from dependencies
    },
)
