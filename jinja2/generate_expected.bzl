def _generate_expected_directory_impl(ctx):
    out_dir = ctx.actions.declare_directory(ctx.attr.output_dir_name)

    ctx.actions.run(
        executable = ctx.executable.tool,
        arguments = [
            "--input-root",
            ctx.attr.input_root,
            "--output-root",
            out_dir.path,
        ],
        inputs = depset(ctx.files.srcs),
        outputs = [out_dir],
        tools = [ctx.attr.tool],
        mnemonic = "RenderExpectedFixtures",
        progress_message = "Rendering generated Jinja fixture outputs",
    )

    return [
        DefaultInfo(
            files = depset([out_dir]),
            runfiles = ctx.runfiles(files = [out_dir]),
        ),
    ]

generate_expected_directory = rule(
    implementation = _generate_expected_directory_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
        "tool": attr.label(
            cfg = "exec",
            executable = True,
            mandatory = True,
        ),
        "input_root": attr.string(
            mandatory = True,
        ),
        "output_dir_name": attr.string(
            default = "test_cases_generated",
        ),
    },
)
