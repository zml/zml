# Use ZML alongside a `build.zig` project

ZML is currently built and packaged with Bazel. A native Zig package that can be
added directly to another project's `build.zig.zon` is not available yet.

If your application already uses Zig's standard build system, keep the ZML
component behind an explicit artifact boundary. Bazel is the source of truth for
building ZML and for selecting its toolchain, MLIR, PJRT, and platform-specific
dependencies.

## Current options

The most isolated option is an executable boundary: build and run the ZML
component with Bazel, then communicate with it through a command-line or
application-specific interface.

A `build.zig` project may also consume a Bazel-built library through a C ABI
facade. In that setup, the application owns the facade, invokes Bazel to build
the target, and links the produced artifact from the Zig build. This is a
workaround rather than a packaged integration: ZML does not currently provide a
maintained C ABI package, generated manifest, or helper for discovering and
packaging all required runtime dependencies.

## Artifact-boundary repro shape

A minimal external repro can exercise this shape today: a `build.zig` wrapper
invokes Bazel, discovers the produced artifact with
`bazel cquery --output=files`, stages the shared library and header, and
links/tests a Zig consumer against the staged artifact. The same pattern can
also stage a Bazel-built executable and run it from the `build.zig` project.

Binary dependency discovery remains the important unresolved part. A repro can
record the staged shared library's direct ELF `DT_NEEDED` entries with
`readelf` into a text file or JSON manifest, which makes the artifact's declared
dynamic dependencies visible to the consumer. That does not resolve transitive
runtime dependencies and does not validate a real model execution path or PJRT
runtime loading.

## Why not import ZML directly?

ZML depends on generated MLIR/XLA/PJRT bindings, platform libraries, and Bazel
toolchain setup. Those pieces are selected and wired by Bazel rules. Importing
ZML source files directly from another `build.zig` project skips that setup and
is not supported.

## Future direction

A better integration could let Bazel generate a small manifest describing the
artifacts, module paths, and runtime dependencies needed by a `build.zig`
project. Until such a manifest exists, treat Bazel as the source of truth for
building ZML.
