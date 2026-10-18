# ZLS native build integration patches

Both patches apply to `4e83599939e9723a8cf7961bb77214549c885116` and correct generic
problems in the native integration introduced by
[upstream PR #3270](https://github.com/zigtools/zls/pull/3270), following
[the build-system migration](https://github.com/zigtools/zls/issues/3208).
These references describe the upstream integration; the corrections below have
not yet been submitted upstream.

- `absolute-lazy-paths.patch`: Zig permits absolute `cwd_relative` lazy paths.
  Joining such a path to the build directory creates a nonexistent doubled path.
  Resolve relative lazy paths instead, retaining absolute paths as absolute.
- `known-build-graphs.patch`: discover workspace build roots during
  initialization, before a source document needs association. Files inside a
  workspace first consult their ancestor builds; external dependencies prefer a
  workspace graph that contains them. Other known graphs are a fallback only
  when their module roots or relative imports actually contain the document.
  Wait for higher-priority configurations instead of caching an early match
  from another project. Failed configurations do not block fallback forever,
  and negative association is retried as graphs become available.

Neither patch refers to ZML, Bazel, or the editor manifest. Each can be removed
independently when its fix is available upstream. Keep compatibility changes at
this generic boundary; do not add a ZML loader to ZLS.

`tools/zls_bridge_test.py` is the regression fixture, invoked by
`tools/zls_navigation_test.py`. Its absolute-path modules, external non-root file
opened before graph discovery, shared module reachable from two actual roots,
and unrelated source exercise these corrections using a real ZLS/BSP session.
The external-first assertion opens no workspace document to trigger discovery.
Overlapping graphs with different imports for one shared source are exercised
in both load orders, with and without a build file at the workspace root.
The repository navigation test additionally checks generated uucode imports
when its external source is the first document opened.
