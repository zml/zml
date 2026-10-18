# Restore ZLS through Zig's native build-system protocol

Reviewed on 2026-10-18. This is an implementation plan; the repository integration
and dependency update have not been implemented. A temporary Zig protocol probe
was run during review; its limited results are recorded below.

## Goal and design decision

Restore working ZLS through `tools/zls.sh`, with an integration that follows Zig's
build-system protocol (BSP) as it evolves. Bazel remains responsible for the
configured dependency graph, generated sources, compilation, and tests.

**Use a small `std.Build` bridge over metadata exported by Bazel.** Replace the
empty root `build.zig` with this bridge and let the pinned Zig toolchain implement
`zig build --listen=-`. ZLS consumes its normal upstream protocol. The bridge
describes editor metadata; it is not a second implementation of the ZML build.

```text
editor -> tools/zls.sh
       -> Bazel: build editor tools and generate required sources
       -> export a versioned, workspace-scoped metadata snapshot
       -> launch updated ZLS with the selected Zig toolchain

ZLS -> Zig's native `zig build --listen=-`
    -> root build.zig -> Bazel metadata -> std.Build modules and compilation roots
    <- Zig's native protocol/configuration output
```

This replaces the earlier proposal to inject JSON into ZLS's private
`bsp.BuildConfig`. JSON can remain the Bazel-to-bridge transport, but ZLS must not
know about that format. Do not maintain a custom binary protocol server, resurrect
`build_runner_path`, or hand-maintain a second module/dependency list.

Keep compatibility work at the Bazel exporter and public `std.Build` boundary.
Prefer upstream ZLS unchanged. Any necessary ZLS correction must fix a generic
upstream issue, have a regression test and upstream reference, and be removable
independently of this integration. A ZML-specific JSON-loader patch is not the
planned architecture.

## Acceptance criteria

All required checks must use the real editor entry point, with CUDA disabled and
without a GPU. Successful compilation or valid JSON alone is insufficient.

| Check | Required result |
| --- | --- |
| Open `platforms/capabilities.zig` first | Definition on `@import("platforms/cuda")` resolves to `platforms/cuda/cuda.zig`, without opening the module root first. |
| Follow `cuda.Cuda` | Reach its actual declaration, following intermediate aliases where necessary. |
| Follow transitive imports | `platforms/cuda/capabilities` resolves with the importing module's own dependency table. |
| Open generated options | `platforms/options` resolves to an existing generated file; navigation to an option declaration works. |
| Navigate inside an external dependency | Resolve `@import("tables")` in uucode's `src/get.zig`, then `@import("build_config")` in its generated `tables.zig`. Derive paths from the graph. |
| Preserve basic editor behavior | Relative imports, `std`, and `builtin` resolve; hover and completion produce useful results for the CUDA capability types. |
| Exercise native BSP | ZLS obtains the real graph through `zig build --listen=-`; no obsolete runner or private JSON loader is involved. |
| Start without existing outputs | Startup produces generated sources and passes navigation without first building ZML or running a test that warms those outputs. |
| Restart after a graph change | A new session sees updated dependencies/options; no stale Zig configure cache or shared snapshot wins. |
| Exercise process lifecycle | LSP stdout remains clean; startup failures, nonzero server exits, normal shutdown, termination, and concurrent sessions behave correctly. |

One active Bazel configuration per server session is sufficient initially. Source
edits use normal LSP document updates. BUILD/dependency/platform changes initially
require a documented server restart. Automatic regeneration is follow-up work;
do not imply that native BSP automatically watches Bazel's graph.

## Current setup and review findings

| Location | Current responsibility / planned change |
| --- | --- |
| `tools/zls.sh` | Editor entry point used by VS Code and Zed; will prepare metadata and launch the server. |
| Root `build.zig` | Empty discovery marker; becomes the thin native build bridge. |
| Root `BUILD.bazel`, `completion` | Selects editor roots; will expose metadata/source preparation and a compatible launcher. |
| `rules_zig`'s `zig/zls/` | Existing module-map exporter, printer, old build runner, and launcher. Reuse provider information, not the obsolete runner interface. |
| `third_party/zls/repo.bzl`, `zls.bazel`, `lsp.bazel` | Pins and builds ZLS and its dependencies. |
| `third_party/zls/BUILD.bazel`, `build_options.zig` | ZLS toolchain registration and version metadata. |
| `MODULE.bazel`, `MODULE.bazel.lock` | Zig/rules_zig selection and dependency resolution; update lockfile through Bazel when required. |
| `bazel/` | Natural home for a reusable exporter and bridge helper. |

### Established facts

- The current ZLS pin is `eab2be0fd74443a27662da809b771c4ad0d2afcf`.
  Its `DocumentStore` disables build integration and its `BuildConfig` is an
  `@compileError`. Flipping a boolean is insufficient. The original investigation
  reported named-import definition returning `null` with the existing executable,
  while relative imports and `std` worked; reproduce this on a fresh build.
- Upstream PR #3270 restored integration using Zig's native BSP. Candidate
  `4e83599939e9723a8cf7961bb77214549c885116` declares Zig 0.17.0 as its minimum
  compiler version. It is a candidate until its build and integration tests pass.
- The candidate discovers modules from reachable compilation steps. Its code for
  exported modules is commented out. Merely calling `b.addModule` is insufficient.
  Its BSP loader requests only steps needed to materialize module source paths;
  already-generated absolute paths avoid requesting ZML compilation.
- The existing printer was run during this review, using existing outputs. Its
  128 module roots all existed, and its per-module imports included the required
  CUDA and generated options edges. `compilations` was empty. The five actual
  application roots already selected by `completion` reached all 128 modules:
  `//bin/zml-smi`, `//examples/io:playground`, `//examples/llm`, `//examples/mnist`,
  and `//examples/sharding`. Derive this information from Bazel in the implementation.
- A temporary two-module fixture using the repository's Zig 0.17.0 completed
  handshake, configuration, an empty build-step request, and clean protocol exit.
  Its compilation step had a failing guard, which was not executed. This supports
  the native metadata approach; it does **not** establish ZLS navigation success.

### Gaps the implementation must close

- **Generated source dependencies:** the existing exporter returns the JSON and
  translated-C inputs, without explicitly including every module's
  `ZigModuleInfo.transitive_inputs`. The printer test includes `data = deps`, which
  can hide this gap by building applications first. A warm checkout is not proof
  that editor startup materializes options, generated Zig files, or embedded data.
- **External source association:** the candidate searches ancestor directories
  for `build.zig`. Canonical Bazel external/generated paths can be outside the
  workspace. Correct paths in the graph alone do not guarantee those documents
  inherit its imports. Test this before finalizing the integration.
- **Launcher lifecycle:** the existing rules_zig launcher copies the environment
  but discards its child's exit status and leaves its temporary configuration file.
  A shell trap cannot recover a status discarded by an inner launcher. Fix this
  at the layer owning the child, or replace that launcher.
- **Configuration caching:** a bridge that reads an environment-selected snapshot
  must not reuse a configuration created for a previous snapshot/session.
- **Version compatibility:** Zig's protocol and `std.Build` APIs evolve. Pin and
  test Zig/ZLS together; avoid coupling repository code to ZLS's internal arena
  layout or Zig's binary configuration encoding.

## Implementation sequence

### 1. Capture the regression and update the ZLS toolchain

- Work from the development environment (`devenv shell` / direnv), confirm Bazel
  and the repository's Zig are available, and record versions. Do not substitute
  an arbitrary system Zig. Preserve unrelated workspace changes.
- Add a standalone Python LSP probe, e.g. `tools/zls_navigation_test.py`, that
  launches `tools/zls.sh`. Keep it outside a Bazel test that would invoke Bazel
  recursively. Capture the expected named-import failure on the original pin.
- Update ZLS to the candidate above, including `lsp-kit` at
  `3e1058db66079911fdde82f833fdce6856e0a706`. Check all pins against its
  `build.zig.zon`; `diffz` and `known-folders` currently match. Review generated
  sources, imports, executable options, and disabled Tracy support against
  upstream's build definitions.
- Synchronize `build_options.zig` and the ZLS toolchain version. Build
  `@zls//:zls` independently before adding the bridge, so compiler/dependency
  failures remain distinguishable from integration failures.

### 2. Prove the native path before expanding the implementation

- Build a minimal bridge with the updated ZLS and pinned Zig, using Bazel's
  existing map plus actual application roots for the initial experiment.
  Validate both root and non-root import navigation through native BSP.
- Create modules first, then wire each module's imports in a second pass. Preserve
  shared module identity and cycles; do not flatten import tables.
- Expose actual compilation roots through a named metadata step so the selected
  ZLS discovers them. Do not invent a compilation unit for every library: doing
  so changes `@import("root")` semantics. Assert that the real roots cover the
  selected graph. If a future library is uncovered, add an appropriate real
  compilation/test root or obtain upstream exported-module support.
- Probe generated and external navigation immediately. If ZLS's ancestor-only
  discovery prevents association, prefer a compatible upstream fix. Otherwise
  carry a small, generic, upstreamable correction that considers known build
  graphs for documents actually belonging to those graphs. It must preserve
  unrelated project resolution, handle asynchronous loading, and avoid caching a
  premature "no build file" result. Do not route every file into ZML's graph.
- Confirm that loading metadata executes no ZML compile/link/runtime steps and
  starts no recursive Bazel invocation. This is the gate for the architecture;
  a passing handshake alone is insufficient.

### 3. Export a complete, versioned Bazel metadata contract

Implement a small repository-owned rule/aspect over `ZigModuleInfo`, or a focused
upstreamable rules_zig change. Reuse existing provider and translate-C machinery.
Do not copy the entire rules_zig implementation or query/parse BUILD text in Zig.

- Export a schema version, canonical workspace identity, active configuration
  information, modules, their import edges, and actual compilation roots/kinds.
  Include target and relevant module settings available from Bazel; never silently
  describe a cross-target graph as the host configuration. Start with native
  configurations if necessary and explicitly reject unsupported combinations.
- Treat the current legacy JSON as an input reference, not a sufficient contract:
  it has neither compilation roots nor a complete target/configuration model.
- Preserve configured module identity and per-module aliases. Use deterministic
  ordering, validate dangling edges/duplicate identities, and detect conflicting
  import tables for one source root. The candidate ZLS merges modules sharing a
  source root; do not silently advertise support for ambiguous configurations.
- Add all required transitive source/extra-source inputs to a metadata preparation
  target, including generated options, `bazel_builtin`, translated C, generated
  external modules, and `@embedFile` inputs. Request those files as Bazel outputs
  or runfiles so they exist locally, including with remote caching. Do not put
  application binaries in `data` merely to generate their sources.
- Expand workspace/execroot paths only after outputs exist; canonicalize source,
  external, and generated paths consistently. Reject unresolved placeholders,
  missing required sources, or a snapshot for a different workspace.
- Ensure export and source generation use identical Bazel flags/configuration.
  Pass a resolved execution root to any retained printer rather than relying on
  its internal `bazel info` subprocess or default-config `bazel-bin` symlinks.
- Test this target on a fresh output base before running the existing printer
  test or broad application builds, which could mask missing dependencies.

### 4. Implement the thin `std.Build` bridge

- Keep root `build.zig` small; put parsing/translation in a reusable helper under
  `bazel/`. It reads the prepared manifest and constructs normal `std.Build.Module`
  objects and real compilation descriptors. It neither runs Bazel nor downloads
  dependencies nor regenerates sources during configuration.
- Use a proposed `ZML_ZLS_BUILD_CONFIG_PATH` environment variable to select an
  absolute, immutable session snapshot. Validate its schema/workspace before use.
  Missing or invalid metadata must produce an actionable error directing the user
  to `tools/zls.sh`, rather than an empty successful graph.
- Use public `std.Build` APIs and supported absolute lazy paths for pre-generated
  files. Keep the two-pass module construction and root-coverage checks from the
  prototype. Build metadata must reflect Bazel semantics, including root kinds.
- Make accidental `zig build` / metadata-step execution explain that ZML builds
  with Bazel. Guard descriptive compilation steps against execution, as in the
  protocol fixture, while allowing configure and metadata-only BSP requests.
  Do not add a `check` step that advertises an incomplete Zig build.
- Register manifest contents as a configure dependency using Zig 0.17's
  `dependOnFileContents`. Environment-based snapshot selection is also an input:
  use `b.graph.poisonCache()` until selection is demonstrably tracked by Zig's
  configure cache. Tracking only the previously selected file is insufficient.
  Test two different manifests in successive sessions sharing a Zig cache.
- Add focused tests for schema errors, workspace mismatch, graph fidelity,
  compilation roots, and configuration invalidation. Avoid tests that merely
  compare the bridge's implementation to itself; inspect the resulting native
  configuration and exercise it with ZLS.
  Include a shared module used by two real compilation roots, verifying
  `@import("root")` without adding artificial roots. Retain a small native BSP
  fixture as an upgrade regression test.

### 5. Integrate startup and process ownership

- Preserve `tools/zls.sh` as the VS Code/Zed entry point, including invocation
  from outside the repository and argument forwarding. Use fail-fast workspace
  resolution and keep stdout exclusively for LSP.
- Prepare tools and metadata before starting ZLS. Keep preparation invocations
  sequential where dependent; do not nest Bazel inside an active build action or
  inside `build.zig`. Provide a clear route to pass the same Bazel configuration
  to all preparation commands, separate from server arguments.
- Create a unique per-session temporary directory for the manifest and ZLS
  configuration; publish the snapshot only after successful generation. Keep it
  for the complete session and never overwrite another session's manifest.
  A snapshot's referenced generated files must also remain valid for its active
  configuration; a unique JSON filename alone does not isolate shared outputs.
- Replace or adapt the legacy launcher using resolved Bazel toolchain/runfiles
  paths for Zig, its library, ZLS, and caches. Keep `//:completion` as a compatible
  entry point if practical. Remove the old runner path and generation from the
  new launch path instead of depending on ZLS ignoring an obsolete setting.
- Set `enable_build_on_save = false`. Verify effective configuration after editor
  initialization and configuration updates, because editor settings can override
  launcher defaults. Keep this setting local to the Bazel workspace.
- Use one clear owner for the server process and temporary directory. Preserve
  stdin/stdout, propagate exit status, forward termination, reap children, and
  remove temporary files on normal exit or handled termination. Do not combine
  `exec` with a shell EXIT trap that is expected to run after the replaced process.
- Exercise clean LSP shutdown, closed input, invalid server arguments, preparation
  failure, SIGTERM, and concurrent launches. Confirm no orphaned ZLS/BSP children.

### 6. Validate, document, and prepare upstream maintenance

The LSP test must implement Content-Length framing, initialization, document
opening/changes, definition/hover/completion requests, and shutdown. Drain stderr
concurrently; handle server requests and asynchronous notifications. Bound startup,
configuration, request, and shutdown waits separately. Retry unresolved navigation
only during configuration, and report the last response plus server logs on failure.

Derive cursor positions from source snippets, use the negotiated position encoding,
and handle `Location` and `LocationLink` results. Assert canonical file paths and
actual declaration ranges; bound alias-following to avoid cycles or false success.
Run the direct non-root-file test first, before other requests warm association.
Include an unrelated/nested Zig project to check graph isolation.

Run narrow implementation tests and native integration checks first, for example:

```sh
bazel build @zls//:zls //:completion
python3 tools/zls_navigation_test.py
bazel test //:completion.print_build_config_test
bash -n tools/zls.sh
```

Add exact exporter/bridge test target names once implemented; replace the printer
test command if the old printer is retired. Use the repository's Zig for formatting
and buildifier for changed Bazel/Starlark files. Check any patch applies cleanly to
its pin. Do not count an old `bazel-bin/completion` executable as fresh validation.

Run the startup/navigation check in an isolated fresh output base **before** tests
that materialize application dependencies; do not clean the developer's existing
output base. Test restart after metadata changes, symlinked workspace paths, and
concurrent sessions. Smoke-test VS Code or Zed when available. Validate macOS and
Linux where available and record untested platforms. For shared Bazel rule changes,
run `bazel build //...` and `bazel test //...` as required by AGENTS.md.

Document the schema, supported configurations, Zig/ZLS pins, launcher command,
restart requirement, diagnostic locations, and any remaining upstream limitation.
Keep exporter, bridge, and launcher responsibilities separable so the general
integration can move into rules_zig. A version upgrade must rerun both a native
protocol fixture and real LSP navigation; compile-only checks are insufficient.

## Scope and follow-up

The initial deliverable is reliable navigation and editor analysis using native
BSP, with accurate module/compilation metadata and startup generation. Runtime
compilation, linking, and tests remain Bazel operations. Preserve existing named
imports, Zig 0.17.0, and platform selection; avoid unrelated runtime changes.

Follow-up work may add automatic Bazel metadata regeneration, a supported
build-on-save path that invokes Bazel, and richer target-specific analysis as
upstream ZLS consumes more protocol information. Do not implement these by
emulating Zig's wire format or by accumulating ZML-specific branches inside ZLS.

Completion means the real launcher passes the acceptance matrix through native
BSP, including a startup without pre-existing generated outputs. Remaining
limitations must be explicit; neither the protocol fixture nor this plan itself
constitutes a restored ZLS integration.

## Evidence and upstream references

- [Pinned ZLS with disabled integration](https://github.com/zigtools/zls/blob/eab2be0fd74443a27662da809b771c4ad0d2afcf/src/DocumentStore.zig)
- [Upstream build-system migration issue #3208](https://github.com/zigtools/zls/issues/3208)
- [Restored integration, PR #3270](https://github.com/zigtools/zls/pull/3270)
- [Candidate dependency manifest](https://github.com/zigtools/zls/blob/4e83599939e9723a8cf7961bb77214549c885116/build.zig.zon)
- [Candidate native BSP client and module collection](https://github.com/zigtools/zls/blob/4e83599939e9723a8cf7961bb77214549c885116/src/bsp.zig)
- [Candidate document/build association](https://github.com/zigtools/zls/blob/4e83599939e9723a8cf7961bb77214549c885116/src/DocumentStore.zig)
- [Candidate configuration precedence](https://github.com/zigtools/zls/blob/4e83599939e9723a8cf7961bb77214549c885116/src/configuration.zig)
- [rules_zig launcher](https://github.com/hermeticbuild/rules_zig/blob/v0.17.0/zig/zls/zls_runner.zig)
- [rules_zig module-map generation](https://github.com/hermeticbuild/rules_zig/blob/v0.17.0/zig/zls/zls_write_build_config.bzl)
- [rules_zig completion targets and test data](https://github.com/hermeticbuild/rules_zig/blob/v0.17.0/zig/zls/zls_completion.bzl)
- [rules_zig path expansion and canonicalization](https://github.com/hermeticbuild/rules_zig/blob/v0.17.0/zig/zls/workspace_printer.zig)

Zig API/protocol details were checked against the repository-resolved Zig 0.17.0
toolchain sources: `lib/std/Build.zig`, `lib/std/Build/Module.zig`,
`lib/std/Build/Configuration.zig`, `lib/std/zig/Client.zig`,
`lib/std/zig/Server.zig`, and `lib/compiler/Maker.zig`. Use the Bazel toolchain's
sources when implementing; old `build_runner.zig` documentation does not describe
this version's configure/make split.
