# Use ZML from a `build.zig` project

ZML is currently built and packaged with Bazel. A native Zig package that can be
added directly to another project's `build.zig.zon` is not available yet.

If your application already uses Zig's standard build system, keep the ZML build
as a Bazel step and consume its output from your `build.zig` project. This keeps
ZML's toolchain, MLIR, PJRT, and platform-specific dependencies under Bazel,
while your application can stay on the standard Zig build flow.

## Current integration model

1. Build the ZML target you need with Bazel.
2. Export the produced library or executable artifact from `bazel-bin`.
3. Link or run that artifact from your Zig project.
4. Keep the boundary explicit, usually through the C ABI or a command-line
   executable, instead of trying to import ZML internals as Zig modules.

This is less ergonomic than a native Zig package, but it avoids duplicating the
large Bazel-managed dependency graph in a second build system.

## Why not import ZML directly?

ZML depends on generated MLIR/XLA/PJRT bindings, platform libraries, and Bazel
toolchain setup. Those pieces are selected and wired by Bazel rules. Importing
ZML source files directly from another `build.zig` skips that setup and is
expected to fail or produce a different runtime environment.

## Future direction

A better integration could let Bazel generate a small manifest describing the
artifacts and module paths needed by a `build.zig` project. Until such a
manifest exists, treat Bazel as the source of truth for building ZML.
