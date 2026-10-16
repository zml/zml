def _patchelf_impl(ctx):
    output_name = ctx.attr.soname or ctx.file.src.basename
    output = ctx.actions.declare_file("{}/{}".format(ctx.label.name, output_name))

    inputs = [ctx.file.src]
    args = ctx.actions.args()

    # --- soname -----------------------------------------------------------
    if ctx.attr.soname:
        args.add("--set-soname", ctx.attr.soname)

    # --- DT_NEEDED --------------------------------------------------------
    args.add_all(ctx.attr.remove_needed, before_each = "--remove-needed")
    args.add_all(ctx.attr.add_needed, before_each = "--add-needed")
    for old, new in ctx.attr.replace_needed.items():
        args.add("--replace-needed")
        args.add(old)
        args.add(new)

    # --- RPATH / RUNPATH --------------------------------------------------
    # patchelf only accepts one rpath operation per invocation.
    rpath_ops = [
        bool(ctx.attr.set_rpath),
        bool(ctx.attr.add_rpath),
        ctx.attr.remove_rpath,
    ]
    if len([op for op in rpath_ops if op]) > 1:
        fail("patchelf: set_rpath, add_rpath and remove_rpath are mutually exclusive")

    if ctx.attr.set_rpath:
        args.add("--force-rpath")
        args.add("--set-rpath", ctx.attr.set_rpath)
    elif ctx.attr.add_rpath:
        args.add("--force-rpath")
        args.add("--add-rpath", ":".join(ctx.attr.add_rpath))
    elif ctx.attr.remove_rpath:
        args.add("--remove-rpath")

    # --- dynamic symbol renames ------------------------------------------
    if ctx.attr.rename_dynamic_symbols:
        symbols = ctx.actions.declare_file(ctx.label.name + ".symbols")
        ctx.actions.write(symbols, "".join([
            "{} {}\n".format(old, new)
            for old, new in ctx.attr.rename_dynamic_symbols.items()
        ]))
        inputs.append(symbols)
        args.add("--rename-dynamic-symbols", symbols)

    args.add("--output", output)
    args.add(ctx.file.src)

    ctx.actions.run(
        executable = ctx.executable._patchelf,
        inputs = inputs,
        outputs = [output],
        arguments = [args],
        mnemonic = "Patchelf",
        progress_message = "Patching ELF %{output}",
    )
    return [DefaultInfo(files = depset([output]))]

patchelf = rule(
    implementation = _patchelf_impl,
    attrs = {
        "src": attr.label(allow_single_file = True, mandatory = True),
        "soname": attr.string(doc = "Set DT_SONAME; also used as the output file name."),
        "add_needed": attr.string_list(),
        "remove_needed": attr.string_list(),
        "replace_needed": attr.string_dict(doc = "old -> new DT_NEEDED replacements."),
        "rename_dynamic_symbols": attr.string_dict(doc = "old -> new symbol renames."),
        "set_rpath": attr.string(),
        "add_rpath": attr.string_list(doc = "Entries appended to the existing RPATH (joined with ':')."),
        "remove_rpath": attr.bool(default = False, doc = "Strip the DT_RPATH/DT_RUNPATH entry entirely."),
        "_patchelf": attr.label(
            default = "@patchelf",
            executable = True,
            cfg = "exec",
        ),
    },
)
