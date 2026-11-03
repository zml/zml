# Runtime groups

Runtime groups let packaging rules place CUDA, ROCm, TPU, and oneAPI runtimes in separate image layers. Producers name the groups. Consumers choose how to package them. Ordinary builds need no extra configuration.

## Producers

Attach a hint to the target that supplies the runtime. Its complete default runfiles become part of the named group:

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

Several targets can contribute to the same group: for instance both `flashinfer_cutlass_moe` and `libzml_cuda` currently are both producers to group `cuda`. An annotation includes the target transitive runfiles and supersedes annotations below it.

## Consumers

Apply `runtime_groups_aspect` to the `binary` attribute of the rule that consumes runtime groups:

```starlark
load("@zml//bazel:runtime_groups.bzl", "collect_runtime_groups", "runtime_groups_aspect")

attrs = {
    "binary": attr.label(aspects = [runtime_groups_aspect]),
}
```

Inside the consuming packaging rule’s implementation function, collect and resolve the binary runtime groups, then select the group to package.

```starlark
load("@rules_runfiles_group//runfiles_group:lib.bzl", "runfiles_groups")

info = collect_runtime_groups(ctx, ctx.attr.binary)
groups = runfiles_groups.resolve(ctx, info, aspect_hints = [])
runfiles = runfiles_groups.runfiles(ctx, groups.by_name["cuda"])
```

Together, the groups preserve all of the binary default runfiles:

- Named groups contain the annotated runtimes.
- `shared` contains remaining runtime files.
- `application` contains the binary's executable and its aliases. This name is reserved and cannot be used in producer hints.

For wrappers such as `with_cfg` that also retain the original executable, include that file explicitly in `application`:

```starlark
info = collect_runtime_groups(
    ctx,
    ctx.attr.binary,
    application_files = [original_executable],
)
```

Grouping does not build image layers or strip executables. The packaging rule chooses the layout and supplies the repository mapping manifest.
