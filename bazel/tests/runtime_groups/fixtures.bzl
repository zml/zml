"""Small runtime graphs exercising the packaging contract without toolchains."""

load("@rules_runfiles_group//runfiles_group:lib.bzl", "runfiles_groups")
load("//bazel:runtime_groups.bzl", "collect_runtime_groups", "runtime_groups_aspect")

GroupReportInfo = provider("Resolved groups and original runfiles for assertions.", fields = ["groups", "original", "executable_group"])

def _fixture_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name + ".txt")
    ctx.actions.write(output, ctx.label.name)
    files = [output]
    executable = None
    if ctx.attr.make_executable:
        executable = ctx.actions.declare_file(ctx.label.name)
        ctx.actions.write(executable, "#!/bin/sh\nexit 0\n", is_executable = True)
        files.append(executable)
    runfiles = ctx.runfiles(
        files = files,
        symlinks = {ctx.label.name + "/alias": executable or output},
        root_symlinks = {ctx.label.name + "/root_alias": executable or output},
    ).merge_all([
        dep[DefaultInfo].default_runfiles
        for dep in ctx.attr.deps + ctx.attr.keyed_deps.keys()
    ])
    return [DefaultInfo(files = depset(files), executable = executable, runfiles = runfiles)]

_attrs = {
    "deps": attr.label_list(),
    "keyed_deps": attr.label_keyed_string_dict(),
    # Intentionally absent from runfiles, despite using the target configuration.
    "annotation_inputs": attr.label_list(),
    "tool": attr.label(cfg = "exec"),
    "make_executable": attr.bool(),
}

runtime = rule(implementation = _fixture_impl, attrs = _attrs)
binary = rule(implementation = _fixture_impl, attrs = _attrs, executable = True)

def _wrapper_impl(ctx):
    original = ctx.attr.binary[DefaultInfo]
    executable = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.symlink(output = executable, target_file = original.files_to_run.executable)
    return [DefaultInfo(
        executable = executable,
        files = original.files,
        runfiles = ctx.runfiles(files = [executable]).merge(original.default_runfiles),
    )]

# Like with_cfg's executable frontend: retain original outputs and runfiles.
wrapper = rule(
    implementation = _wrapper_impl,
    executable = True,
    attrs = {"binary": attr.label()},
)

def _contents(runfiles):
    return sorted(
        ["file:" + f.basename for f in runfiles.files.to_list()] +
        ["symlink:" + s.path + "=" + s.target_file.basename for s in runfiles.symlinks.to_list()] +
        ["root_symlink:" + s.path + "=" + s.target_file.basename for s in runfiles.root_symlinks.to_list()] +
        ["empty:" + p for p in runfiles.empty_filenames.to_list()],
    )

def _report_impl(ctx):
    source = ctx.attr.binary
    application_files = [ctx.attr.application_binary[DefaultInfo].files_to_run.executable] if ctx.attr.application_binary else []
    info = collect_runtime_groups(ctx, source, application_files = application_files)
    resolved = runfiles_groups.resolve(ctx, info, aspect_hints = [])
    groups = {entry.name: _contents(runfiles_groups.runfiles(ctx, entry)) for entry in resolved.groups}
    original = _contents(source[DefaultInfo].default_runfiles)
    output = ctx.actions.declare_file(ctx.label.name + ".json")
    ctx.actions.write(output, json.encode_indent({
        "groups": groups,
        "original": original,
        "executable_group": resolved.executable_group,
    }))
    return [
        DefaultInfo(files = depset([output])),
        GroupReportInfo(groups = groups, original = original, executable_group = resolved.executable_group),
    ]

report = rule(
    implementation = _report_impl,
    attrs = {
        "binary": attr.label(aspects = [runtime_groups_aspect]),
        "application_binary": attr.label(),
    },
)
