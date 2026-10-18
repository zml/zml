# Zig editor support

Use `tools/zls.sh` as the language-server executable from the development
shell (`devenv shell` or direnv). VS Code and Zed workspace settings already select
it and disable build-on-save. The script accepts ordinary ZLS arguments and works
when invoked outside the checkout or through a symlink. `bazel run //:completion`
is also supported.

Bazel prepares the source graph and generated inputs before ZLS starts. Initial
startup can take a few minutes: source generators such as protoc, TableGen,
translate-c, and uucode must be available. It does not build the ZML applications.
ZLS then uses Zig's native `zig build --listen=-` protocol to load editor metadata.
Compilation and tests remain Bazel operations.

Restart the language server after changing BUILD files, dependencies, or platform
flags. Ordinary source edits use LSP document updates. Native BSP does **not**
watch or regenerate the Bazel graph.

## Configuration

Separate Bazel arguments from ZLS arguments using JSON arrays in the environment:

```sh
export ZML_ZLS_BAZEL_FLAGS='["--@zml//platforms:cuda=false"]'
tools/zls.sh --log-level debug
```

All preparation commands receive the same configuration flags. The launcher asks
Bazel for its output base, then uses a **private sibling with `-zls` appended**.
Only editor launchers should use that sibling. This separates preparation from
ordinary terminal builds, which do not participate in the launcher's lock.
The private cache persists between sessions; its first build is cold even when
the ordinary Bazel cache is warm. Editor builds disable convenience symlinks so
they do not repoint the checkout's `bazel-bin`.

To select another cache location:

```sh
export ZML_ZLS_BAZEL_STARTUP_FLAGS='["--output_base=/tmp/zml-editor-bazel"]'
export ZML_ZLS_BAZEL_FLAGS='["--@zml//platforms:cuda=false"]'
tools/zls.sh
```

This uses `/tmp/zml-editor-bazel-zls` for editor preparation. Other startup flags
are retained; the launcher overrides `--output_base` and `--symlink_prefix` for
its preparation commands.

One native Bazel target configuration is supported per server session, on macOS
and Linux. Cross-target CPU/OS/ABI configurations, custom dynamic linkers, and
unrepresented Zig options are rejected explicitly. The bridge preserves module
aliases, shared modules, compilation kinds, optimization, threading, stripping,
and supported module flags. It does not describe the C/C++ linker graph or offer
a replacement `zig build` command. The launcher generates a session-local
`builtin.zig` with the selected Zig target and settings, including optimization
and threading, and supplies it to ZLS explicitly. ZLS still uses one builtin per
session; compilation-specific builtin values such as `is_test` are not obtained
from BSP. The current editor roots are all executables.

Build-on-save must remain disabled; the
workspace settings use `zig.buildOnSaveProvider = "off"` in VS Code and
`lsp.zls.settings.zls.enable_build_on_save = false` in Zed.

Each session has its own temporary directory. Startup serializes metadata
preparation, copies generated inputs (including relative imports and embedded
files), then atomically publishes an immutable `manifest.json`. Concurrent
sessions can use different platform options without replacing each other's
generated sources. Terminal builds use the ordinary output base and cannot
replace inputs while an editor snapshot is being published. Workspace source
files remain live. Do not delete the private editor output base or replace its
external repositories while sessions using it are active; restart those sessions
after dependency upgrades.

ZLS discovers workspace build roots during initialization, so an external or
generated document can be opened first. Files inside a workspace retain their
ancestor build-file precedence. External dependencies prefer a containing
workspace graph; other known graphs are a membership-checked fallback.

The launcher prints its session directory and all preparation diagnostics to
stderr. ZLS logs are also written to `<session>/zls.log`; pass `--log-file /path`
to retain a log after shutdown. stdout is reserved for the server's LSP stream.
Normal exit and handled termination remove the session directory, propagate the
server's status, and terminate remaining processes in its process group.

## Metadata contract and maintenance

`//:completion_metadata` is defined in `bazel/zls.bzl`. It consumes
`ZigModuleInfo`, reconstructs descriptors for the explicitly selected real
compilation roots, and reuses rules_zig's C translation machinery. Its default
outputs include every module's transitive source inputs, including generated
options, `bazel_builtin`, translated C, external generated Zig, and embedded
data. No application executable is included as preparation data. The `manifest`
output group selects only the raw JSON descriptor.

The version 1 manifest contains:

| Field | Meaning |
| --- | --- |
| `schema_version` | Contract version; currently `1`. |
| `workspace` | Canonical absolute workspace path, added at publication. |
| `execution_root` | Resolved Bazel execution root, added at publication. |
| `configuration`, `target`, `target_args`, `settings`, `bazel_flags` | Active Bazel/Zig configuration and preparation arguments. |
| `modules` | Deterministically ordered module IDs, source paths, per-module alias-to-ID imports, and Zig options. |
| `compilations` | Actual target labels, root module IDs, and kinds (`exe`, `test`, `static`, `shared`). |
| `inputs` | All required source/extra-source inputs, in Bazel execution-root notation. |
| `tools` | Resolved Zig executable, Zig library, Zig version, and ZLS executable. |

The raw exporter uses execution-root-relative paths for external/generated files
and workspace-relative paths for source files. Publication resolves and validates
all inputs, replaces generated module paths with their session copies, and
rejects duplicate identities, conflicting source roots, dangling edges, missing
files, and modules unreachable from the real compilation roots.

The root `build.zig` delegates to `bazel/zls_bridge.zig`. It reads the absolute
`ZML_ZLS_BUILD_CONFIG_PATH`, checks the schema and workspace, creates all modules
before wiring imports, and exposes compilation descriptors under
`editor-metadata`. Its default and compilation steps are guarded with an
explanation that builds use Bazel. Configuration neither runs Bazel nor downloads
or generates dependencies. The manifest is a `dependOnFileContents` input; the
bridge also calls `poisonCache()` because environment-based snapshot selection is
not tracked by Zig 0.17's configure cache.

The tested toolchain pair is Zig **0.17.0**, rules_zig **0.17.0**, and ZLS
**4e83599939e9723a8cf7961bb77214549c885116**. lsp-kit is pinned to
**3e1058db66079911fdde82f833fdce6856e0a706**; the diffz and known-folders pins match
ZLS's `build.zig.zon`. Tracy remains disabled. Two independently removable generic
ZLS corrections are described in [the patch notes](../../third_party/zls/PATCHES.md).
There is no private ZLS JSON loader, obsolete build runner, or custom BSP server.

## Validation

Run the full editor integration probe outside Bazel, to avoid recursive builds:

```sh
ZML_ZLS_BAZEL_FLAGS='["--@zml//platforms:cuda=false"]' python3 tools/zls_navigation_test.py
bazel test //tools:zls_launcher_test
bazel build @zls//:zls //:completion
bash -n tools/zls.sh
```

The probe opens `platforms/capabilities.zig` first. It exercises declaration
ranges, transitive and generated imports, external uucode files, hover,
completion, relative imports, `std`, `builtin`, and build-on-save configuration.
It also runs a small native BSP fixture with a shared module and two real roots,
checks invalid metadata and manifest changes with a shared Zig cache, and tests
launcher exit, termination, concurrent configurations, and symlinked workspaces.
Regression checks also open external documents without warming workspace
discovery, exercise overlapping build graphs in different orders, inspect
nondefault builtin settings, and interleave an ordinary Bazel build between
editor generation and snapshot publication.
Set `ZML_ZLS_TEST_VERBOSE=1` to stream server diagnostics while it runs.

Run the probe with a **new private editor output base** before broad builds or
tests. Do not clean the developer's existing output base. Toolchain upgrades must rerun both the
native fixture and real navigation; compilation alone is insufficient. For
shared Bazel changes also run `bazel build //...` and `bazel test //...`.

Validation on 2026-10-18 used macOS arm64, Bazel 9.1.1, Zig 0.17.0, and CUDA
disabled. The full probe passed with the previously nonexistent private output
base `/tmp/zml-zls-fix-20261018-zls`, with no application binaries materialized by
startup. The native fixture, restart, lifecycle, external-first, overlapping
graph, nondefault builtin, and interleaved terminal-build checks passed.

The earlier VS Code smoke test with `ziglang.vscode-zig@0.6.18` resolved the first
named import using its actual definition-provider API and reported build-on-save
`off`. The follow-up fixes were verified through LSP; the editor UI was not rerun.

The following also passed (all Bazel commands ran through `devenv shell`):

```sh
ZML_ZLS_BAZEL_STARTUP_FLAGS='["--output_base=/tmp/zml-zls-fix-20261018"]' ZML_ZLS_BAZEL_FLAGS='["--@zml//platforms:cuda=false"]' python3 tools/zls_navigation_test.py
bazel build @zls//:zls --@zml//platforms:cuda=false
bazel build //... --@zml//platforms:cuda=false
bazel test //... --@zml//platforms:cuda=false
```

The broad build covered 232 targets; all 29 tests passed (27 cached, 2 executed).
Zig formatting, Python and shell syntax, and patch application checks passed.
Linux, GPU execution, and Zed's UI were not tested here. The two generic ZLS fixes
remain local patches pending upstream submission.
