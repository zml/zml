"""Define runtime group annotations and collect annotated dependencies using an aspect."""

load("@rules_runfiles_group//runfiles_group:providers.bzl", "RunfilesGroupInfo")

# Holds the runtime group name declared by an annotation
_RuntimeGroup = provider(fields = ["name"])
# For one target, holds the groups mappings to runfiles collected by the aspect
_RuntimeGroups = provider(fields = ["groups"])

runtime_group = rule(
    implementation = lambda ctx: [_RuntimeGroup(name = ctx.attr.group)],
    attrs = {"group": attr.string(mandatory = True)},
    doc = "Annotate a producer target via aspect_hints with the group name for its runtime files.",
)

def _targets(value):
    """Runtime groups propagate through single-label and label-list attributes only."""
    values = value if type(value) == "list" else [value]
    return [target for target in values if type(target) == "Target"]

def _collect_impl(target, ctx):
    """Collect runtime groups from the target and its dependencies."""

    # Skip build-tool targets because packaging collects dependencis for **running** the application, not for building it.
    if ctx.configuration.is_tool_configuration():
        return [_RuntimeGroups(groups = {})]

    # Read this target runtime-group annotations, ignoring other aspect hints.
    hints = [h for h in getattr(ctx.rule.attr, "aspect_hints", []) if _RuntimeGroup in h]
    if len(hints) > 1:
        fail("Only one runtime group annotation is allowed on " + str(target.label))

    # hints contains only this target annotations, not its dependencies
    # LLMD (no annotation): receives dependency groups via `else`
    # └── ROCm Zig library (no annotation): forwards rocm via `else`
    #     └── libzml_rocm (annotated): defines rocm via if hints
    # Groups propagate upward
    if hints:
        # Use this annotated target runfiles (replacing any groups from its dependencies)
        groups = {hints[0][_RuntimeGroup].name: target[DefaultInfo].default_runfiles}
    else:
        groups = {}
        for edge in dir(ctx.rule.attr):
            for dep in _targets(getattr(ctx.rule.attr, edge)):
                if _RuntimeGroups in dep:
                    for name, runfiles in dep[_RuntimeGroups].groups.items():
                        groups.setdefault(name, []).append(runfiles)
        # Merge contributions with the same group name at unannotated targets.
        groups = {name: ctx.runfiles().merge_all(parts) for name, parts in groups.items()}
    return [_RuntimeGroups(groups = groups)]

# To be used on the consumer side to collect runtime groups from dependencies
runtime_groups_aspect = aspect(
    implementation = _collect_impl,
    attr_aspects = ["*"],
    provides = [_RuntimeGroups],
)

def _entries(runfiles):
    """ Index runfile entries so assigned group entries can be removed when computing shared. Runfiles depsets are immutable """
    return {
        "files": {f: f for f in runfiles.files.to_list()},
        "symlinks": {(s.path, s.target_file): s for s in runfiles.symlinks.to_list()},
        "root_symlinks": {(s.path, s.target_file): s for s in runfiles.root_symlinks.to_list()},
        "empty_filenames": {p: p for p in runfiles.empty_filenames.to_list()},
    }

def collect_runtime_groups(ctx, binary):
    """Collect intact producer groups and assign remaining binary runfiles to shared.

    Every file and alias in a annotated group must also exist in the binary runfiles. Merge runfiles unassigned to any group into a `shared` group (preserving any existing shared group). Exclude the application executable and executable symlinks from shared group.
    Executable targets must expose all application executable artifacts through DefaultInfo.files plus files_to_run.executable, with no runtime dependencies in those outputs.
    Wrappers that hide original executables are unsupported.
    The binary attribute must have runtime_groups_aspect applied.
    """
    collected = binary[_RuntimeGroups]
    executable = binary[DefaultInfo].files_to_run.executable
    excluded = {f: True for f in binary[DefaultInfo].files.to_list() + [executable]} if executable != None else {}
    remaining = _entries(binary[DefaultInfo].default_runfiles)
    groups = dict(collected.groups)

    for runfiles in groups.values():
        for field, entries in _entries(runfiles).items():
            for key in entries:
                remaining[field].pop(key, None)
    if remaining["empty_filenames"]:
        fail("Unassigned synthetic empty runfiles require a producer annotation")

    # Collect remaining unassigned entries to a shared runfiles group.
    shared = ctx.runfiles(
        files = [f for f in remaining["files"].values() if f not in excluded],
        symlinks = {s.path: s.target_file for s in remaining["symlinks"].values() if s.target_file not in excluded},
        root_symlinks = {s.path: s.target_file for s in remaining["root_symlinks"].values() if s.target_file not in excluded},
    )
    groups["shared"] = groups.get("shared", ctx.runfiles()).merge(shared)

    # Return the final set of runtime groups. Including the shared group
    return RunfilesGroupInfo(**groups)
