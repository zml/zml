"""Annotate assembled runtimes and group a binary's runfiles for packaging."""

load("@rules_runfiles_group//runfiles_group:lib.bzl", "runfiles_groups")
load("@rules_runfiles_group//runfiles_group:providers.bzl", "RunfilesGroupInfo")

_RuntimeGroupInfo = provider("Runtime ownership annotation.", fields = ["name"])

# This is a partial collection: unannotated runfiles are added at the consumer.
_RuntimeGroupsInfo = provider("Partial transitive collection of annotated runtimes.", fields = ["entries"])

def _runtime_group_impl(ctx):
    if not ctx.attr.group or ctx.attr.group == "application":
        fail("Runtime group names must be nonempty and must not be 'application'")
    return [_RuntimeGroupInfo(name = ctx.attr.group)]

runtime_group = rule(
    implementation = _runtime_group_impl,
    attrs = {"group": attr.string(mandatory = True)},
    doc = "Assign a producer's complete runfiles to a named group through aspect_hints. Nested annotations are superseded.",
)

def _targets(value):
    """Extract targets from label attributes, including keyed and list dictionaries."""
    values = value.keys() + value.values() if type(value) == "dict" else [value]
    targets = []
    for item in values:
        targets.extend([target for target in (item if type(item) == "list" else [item]) if type(target) == "Target"])
    return targets

def _collect_impl(target, ctx):
    """Collect runtime groups from the target and its dependencies."""

    # Build tools do not contribute to the application's runtime package.
    if ctx.configuration.is_tool_configuration():
        return [_RuntimeGroupsInfo(entries = runfiles_groups.entries())]

    # Read this target runtime-group annotations, ignoring other aspect hints.
    hints = [h for h in getattr(ctx.rule.attr, "aspect_hints", []) if _RuntimeGroupInfo in h]
    if len(hints) > 1:
        fail("Only one runtime group annotation is allowed on " + str(target.label))

    if hints:
        # The annotation owns the assembled runtime, including its dependencies.
        return [_RuntimeGroupsInfo(entries = runfiles_groups.entries(direct = [
            runfiles_groups.entry(
                name = hints[0][_RuntimeGroupInfo].name,
                content = target[DefaultInfo].default_runfiles,
                rank = runfiles_groups.RANK_FOUNDATION,
            ),
        ]))]

    return [_RuntimeGroupsInfo(entries = runfiles_groups.entries(transitive = [
        dep[_RuntimeGroupsInfo].entries
        for edge in dir(ctx.rule.attr)
        for dep in _targets(getattr(ctx.rule.attr, edge))
        if _RuntimeGroupsInfo in dep
    ]))]

# To be used on the consumer side to collect runtime groups from dependencies
runtime_groups_aspect = aspect(
    implementation = _collect_impl,
    attr_aspects = ["*"],
    provides = [_RuntimeGroupsInfo],
)

def _entries(runfiles):
    """Index each runfiles component without losing aliases or synthetic files."""
    return {
        "files": {f: f for f in runfiles.files.to_list()},
        "symlinks": {(s.path, s.target_file): s for s in runfiles.symlinks.to_list()},
        "root_symlinks": {(s.path, s.target_file): s for s in runfiles.root_symlinks.to_list()},
        "empty_filenames": {p: p for p in runfiles.empty_filenames.to_list()},
    }

def _runfiles(ctx, entries):
    # Synthetic empty files cannot be constructed with ctx.runfiles(). They must
    # stay in the intact annotated runfiles object that supplies them.
    if entries["empty_filenames"]:
        fail("Unassigned synthetic empty runfiles require a producer annotation")
    return ctx.runfiles(
        files = entries["files"].values(),
        symlinks = {s.path: s.target_file for s in entries["symlinks"].values()},
        root_symlinks = {s.path: s.target_file for s in entries["root_symlinks"].values()},
    )

def collect_runtime_groups(ctx, binary, *, application_files = []):
    """Return complete RunfilesGroupInfo for a target with runtime_groups_aspect.

    Annotations own complete runtime bundles; every annotated entry must also
    exist in the binary's runfiles. Groups may overlap. Unassigned entries join
    `shared`, including any existing contributions to that group. The root
    executable, explicitly selected application files, and their aliases, when
    present in runfiles, belong to `application`.
    Other DefaultInfo.files are not assumed to be executables.

    Consumers resolve this provider with runfiles_groups.resolve(). They own
    executable replacement, repository mappings, and the final package layout.
    Collect once per binary and share the result between layer selectors.

    Args:
        ctx: The consuming rule's context.
        binary: A target with runtime_groups_aspect applied.
        application_files: Additional Files owned by the application, such as the
            original executable behind a wrapper. Never inferred from default outputs.

    Returns:
        RunfilesGroupInfo whose entries cover DefaultInfo.default_runfiles.
    """
    collected = binary[_RuntimeGroupsInfo]
    executable = binary[DefaultInfo].files_to_run.executable
    application_files = {f: True for f in application_files}
    if executable != None:
        application_files[executable] = True
    original = _entries(binary[DefaultInfo].default_runfiles)
    remaining = {field: dict(entries) for field, entries in original.items()}

    # Visiting all attributes can discover target-configured dependencies that
    # are not runtime dependencies. Never silently add their files to a package.
    for entry in collected.entries.to_list():
        for field, entries in _entries(runfiles_groups.runfiles(ctx, entry)).items():
            for key in entries:
                if key not in original[field]:
                    fail("Runtime group '{}' contains {} entry {} absent from {} runfiles".format(entry.name, field, key, binary.label))
                file = key if field == "files" else key[1] if field in ["symlinks", "root_symlinks"] else None
                if file in application_files:
                    fail("Runtime group '{}' contains an application file of {}; annotate its runtime dependencies instead".format(entry.name, binary.label))
                remaining[field].pop(key, None)

    own = []
    if application_files:
        application = {
            "files": {f: f for f in remaining["files"] if f in application_files},
            "symlinks": {key: s for key, s in remaining["symlinks"].items() if s.target_file in application_files},
            "root_symlinks": {key: s for key, s in remaining["root_symlinks"].items() if s.target_file in application_files},
            "empty_filenames": {},
        }
        for field, entries in application.items():
            for key in entries:
                remaining[field].pop(key)
        own.append(runfiles_groups.entry(
            name = "application",
            content = _runfiles(ctx, application),
            kind = "first_party",
        ))

    own.append(runfiles_groups.entry(
        name = "shared",
        content = _runfiles(ctx, remaining),
        rank = runfiles_groups.RANK_SHARED_DEPS,
    ))
    return RunfilesGroupInfo(
        entries = runfiles_groups.entries(direct = own, transitive = [collected.entries]),
        executable_group = "application" if executable != None else None,
    )
