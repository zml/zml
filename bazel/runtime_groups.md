# Runtime groups

ZML annotates each assembled backend runtime with a group name (`cuda`, `rocm`,
`tpu`, or `oneapi`). Packaging rules can put those files in separate image layers
without knowing the backend's internal dependency graph. Ordinary builds need no
grouping aspect.

## Producers

Attach a `runtime_group` hint to the target whose **complete default runfiles**
form the runtime bundle:

```starlark
load("@zml//bazel:runtime_groups.bzl", "runtime_group")

runtime_group(
    name = "runtime_group",
    group = "cuda",
)

cc_library(
    name = "libzml_cuda",
    aspect_hints = [":runtime_group"],
    data = [":sandbox"],
)
```

The backend wrappers are the ownership boundary: their runfiles contain the
assembled, patched runtime, rather than the inputs used to construct it. No
change to `rules_cc` is needed. An outer annotation supersedes annotations below
it. Several producers may contribute to the same group; different groups may
overlap. A group name does not establish version compatibility or choose one
runtime version over another.

## Consumers

Apply `runtime_groups_aspect` to the binary attribute of a packaging rule. Call
`collect_runtime_groups(ctx, binary)` once for that binary and share its result
between layer selectors. The result uses the `rules_runfiles_group` **0.1.0** API:

```starlark
load("@rules_runfiles_group//runfiles_group:lib.bzl", "runfiles_groups")
load("@zml//bazel:runtime_groups.bzl", "collect_runtime_groups", "runtime_groups_aspect")

def _impl(ctx):
    info = collect_runtime_groups(ctx, ctx.attr.binary)
    resolved = runfiles_groups.resolve(ctx, info, aspect_hints = [])
    entry = resolved.by_name[ctx.attr.group]
    runfiles = runfiles_groups.runfiles(ctx, entry)
    # Package runfiles using the consumer's existing layout.
```

The union of all groups equals the binary's `DefaultInfo.default_runfiles`:

- Annotated runtime entries retain their producer's group name.
- Unassigned entries join `shared`, alongside any annotated `shared` entries.
- The root executable and its runfiles aliases belong to `application`, named
  by `executable_group`. Other default outputs remain ordinary runtime data.
  Executables in runtime dependencies, such as Crashpad, retain their group.

The consumer still owns executable stripping/replacement, the repository mapping
manifest, and the final filesystem layout. These are packaging decisions, not
producer annotations.

The aspect traverses all dependency attributes, skipping execution configurations.
Because a target-configured dependency need not be a runtime dependency, the
collector rejects annotated entries absent from the binary's runfiles. Annotate
runtime bundles that the consuming binary actually includes. Unassigned synthetic
empty runfiles also cause an error: Starlark cannot recreate them with
`ctx.runfiles()`, so their producer must be annotated to preserve its runfiles
object intact. `application` is reserved; annotate runtime dependencies instead
of the root executable.

## Migrating the image consumer

The former API exposed groups as provider fields (`getattr(info, group)`). Use
`resolve(...).by_name[group]` and `runfiles(...)` instead. The executable is now
explicitly represented in `application`; replace it with the stripped executable
and add the repository mapping there without packaging the original a second
time. Keep dependency groups independent of that application layer.

`with_cfg` executable wrappers keep the original executable in runfiles as well
as adding a wrapper symlink. Pass that known original artifact explicitly:

```starlark
info = collect_runtime_groups(
    ctx,
    ctx.attr.binary,
    application_files = [original_executable],
)
```

Both executables and their aliases then belong to `application`; unrelated
default outputs remain in `shared`. Select the known input to the stripping
operation, not every file in `DefaultInfo.files`. This parameter only classifies
entries already present in the binary's runfiles; it does not add files.

For monorepo's image pipeline, upgrade `rules_img` from 0.3.17 to **at least
0.3.18** together with this change. Version 0.3.17 imports APIs removed by
`rules_runfiles_group` 0.1.0, including when loading `layer_from_tar`.
