def _render_kv(e):
    return e

def _patchelf_impl(ctx):
    output_name = ctx.file.shared_library.basename
    if ctx.attr.soname:
        output_name = ctx.attr.soname
    output = ctx.actions.declare_file("{}/{}".format(ctx.attr.name, output_name))

    commands = [
        "set -e",
        'cp -f "$2" "$3"',
        'chmod +w "$3"',
    ]

    if ctx.attr.soname:
        commands.append('"$1" --set-soname "{}" "$3"'.format(ctx.attr.soname))
    if ctx.attr.remove_needed:
        for v in ctx.attr.remove_needed:
            commands.append('"$1" --remove-needed "{}" "$3"'.format(v))
    if ctx.attr.add_needed:
        for v in ctx.attr.add_needed:
            commands.append('"$1" --add-needed "{}" "$3"'.format(v))

    if ctx.attr.replace_needed:
        for k, v in ctx.attr.replace_needed.items():
            commands.append('"$1" --replace-needed "{}" "{}" "$3"'.format(k, v))

    ctx.actions.run_shell(
        inputs = [ctx.file.shared_library],
        outputs = [output],
        arguments = [ctx.executable._patchelf.path, ctx.file.shared_library.path, output.path],
        command = "\n".join(commands),
        tools = [ctx.executable._patchelf],
    )

    return [
        DefaultInfo(
            files = depset([output]),
        ),
    ]

patchelf = rule(
    implementation = _patchelf_impl,
    attrs = {
        "shared_library": attr.label(allow_single_file = True, mandatory = True),
        "soname": attr.string(),
        "add_needed": attr.string_list(),
        "remove_needed": attr.string_list(),
        "replace_needed": attr.string_dict(),
        "_patchelf": attr.label(
            default = "@patchelf",
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
    },
)
