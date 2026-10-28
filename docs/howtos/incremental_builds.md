# Incremental Zig builds

Build or run MNIST with a compiler that stays alive between Bazel invocations:

```sh
bazel build --config=incremental //examples/mnist
bazel run --config=incremental //examples/mnist
```

These commands build or run once. To automatically rebuild and rerun whenever
you save a source file, start the explicit
[iBazel](https://github.com/bazelbuild/bazel-watcher) launcher:

```sh
./tools/ibazel run --config=incremental //examples/mnist
```

Leave this command running while you edit. iBazel watches the target's
dependencies and asks Bazel to rebuild on changes, reusing the existing compiler.
Compile errors appear in the terminal; fix them and save again to rebuild. MNIST
exits after recognizing a digit, but the watcher remains running for the next
save. Ctrl-C stops the watcher (iBazel can require a second Ctrl-C while a
subprocess is running).

To use the shorter `ibazel` command with the repository launcher, add `tools` to
your current shell's PATH from the repository root:

```sh
export PATH="$PWD/tools:$PATH"
ibazel run --config=incremental //examples/mnist
```

`tools/ibazel` downloads the pinned iBazel release on first use, verifies its
SHA-256, and caches it under `${XDG_CACHE_HOME:-$HOME/.cache}/zml/ibazel`. This
needs `curl` and either `sha256sum` or `shasum`. Linux and macOS on x86_64 and
arm64 are supported by the launcher; the incremental compiler's platform
restrictions below still apply. An existing iBazel installation also works.

The same launcher can watch builds or tests:

```sh
./tools/ibazel build --config=incremental //examples/mnist
./tools/ibazel test --config=incremental //zml:test
```

Use `--` to separate program arguments, as with ordinary `bazel run`. iBazel
accepts a subset of Bazel command-line flags (see `./tools/ibazel --help`); put
additional flags in a named `.bazelrc` configuration and pass `--config=NAME`.

An unchanged target is still a normal Bazel cache hit. When a source changes,
Bazel sends another request to the existing compiler. Stopping the watcher leaves
the Bazel server and compiler workers available for the next invocation.
`bazel shutdown` releases the workers and their compiler state.

The configuration uses Debug Zig code and optimized C/C++ dependencies, like
`--config=debug`. Zig uses its self-hosted code generator (`-fno-llvm`) and its
incremental ELF linker (`-fno-lld -fnew-linker`). `use_cc_common_link=false` makes
Zig build the executable directly, retaining frontend, code generation, and
linker state in the same compiler process. C/C++ dependencies still use the
configured C/C++ toolchain. The Linux x86_64 target uses glibc 2.28, matching the
C/C++ toolchain's default baseline instead of rules_zig's glibc 2.17 default.

## How it works

[Bazel persistent workers](https://bazel.build/remote/persistent) receive one
request for each compile action. The patch in
`third_party/rules_zig/incremental-worker.patch` adds an optional executable
adapter to the pinned `rules_zig` 0.17.0 release. The adapter is selected only by
the incremental configuration; host tools continue to use ordinary Zig actions.
No separate checkout or local module override is needed.

`bazel/zig_worker/worker.zig` translates Bazel's JSON worker messages into Zig's
binary compiler protocol using `std.zig.Client`, `Server`, and `ErrorBundle`.
It launches `zig ... -fincremental --listen=-`, sends
an `update` message for each build, and waits for the terminating error bundle.
This is the compiler-server interface used by Zig's own build system. `--watch`
belongs to the build runner. iBazel watches files and starts Bazel builds; Bazel
then detects changed inputs and schedules compiler updates. The compiler itself
does not watch the filesystem or rebuild outside Bazel's control.

The adapter is compiled with the same Zig toolchain it manages. Its small
bootstrap rule invokes the compiler directly, avoiding a dependency cycle in
which building the worker would itself require the worker.

The rules patch links libc and the C toolchain's static C++ runtime libraries,
translates C driver linker flags into Zig options, preserves versioned shared
library paths, and passes object members of Bazel's `alwayslink` libraries
directly to Zig. Bazel names these archives `.lo`, which Zig interprets as object
files.
Two dependency fixes avoid duplicate definitions rejected by the linker:
libbacktrace selects one implementation for each platform and operation, and
LLVM and protobuf share zlib instead of linking both zlib and zlib-ng. LLVM's
expected `zlib-ng` target name is an alias to that shared zlib library.

The incremental configuration also disables protobuf's optional descriptor weak
messages feature through a small protobuf patch. That feature requires linker
generated `__start_pb_defaults` and `__stop_pb_defaults` symbols, which Zig's
incremental linker does not yet provide. Normal protobuf messages are supported;
`--descriptor_implicit_weak_messages` is unavailable in this configuration.

Runfiles initialization uses a C constructor that calls the exported Zig
initializer. Zig 0.17's incremental linker handles constructor sections from C
objects, but falls back to an ordinary data section for the previous Zig
`linksection(".init_array")` declaration when its section attributes conflict.

Zig rejects explicit output paths with `--listen=-`. Each compiler therefore owns
a private temporary cache, and the adapter copies successful outputs from the
reported cache digest into Bazel's declared output paths. These are independent
copies, since Zig may mutate its cache artifacts during the next update. Failed
compilations return diagnostics and never publish cached outputs.

The output path is part of Bazel's worker key, with one worker per key, so each
compile action retains its own compiler. Changes to arguments, input membership,
or non-Zig inputs restart that compiler. Source edits reuse it, except when an
edit preserves the file metadata Zig uses to detect changes. Compiler crashes
discard the session; a later build can start a fresh compiler.

## Scope and debugging

This is an experimental local development configuration tested with Zig 0.17.0
on Linux x86_64. The adapter requires its compiler's exact Zig version; that
internal protocol must be reviewed before upgrading the toolchain.

The full `//...` build is not yet supported: `//bin/zml-smi` reaches linker
limitations involving type-only references to external driver functions and
upb's `__start_linkarr_upb_AllExts` / `__stop_linkarr_upb_AllExts` section symbols.
The dependent completion build-config test also fails to build. MNIST and the
core CPU tests use the self-hosted path successfully; there is no LLVM fallback.

Worker sandboxing is disabled in this configuration because Zig retains open
input directories across requests. Bazel 9.3 applies that switch to all workers
in the invocation. Other action sandboxing remains enabled. Incremental Zig
actions are explicitly local and do not use remote execution or remote caching.
Use the ordinary build configurations for remote builds.

Workers consume memory until Bazel evicts them or shuts down. There can be one
compiler per distinct output, even with `--worker_max_instances=ZigIncremental=1`.
Changing configuration, worker code, or compiler tools can also discard workers.
Normal worker EOF removes its private cache. Forced termination can leave
`/tmp/zml-zig-worker-*` directories behind.

To confirm reuse after a source edit:

```sh
bazel build --config=incremental --worker_verbose //examples/mnist
```

The adapter reports `pid=... update=... restarted=...` for executed actions.
The same PID and an increasing update count demonstrate reuse. A build with no
changed inputs does not send a request, so it does not print this line.

The real-compiler integration tests exercise edited imports, error recovery,
deleted outputs, input and argument invalidation, crashes, and JSON framing:

```sh
bazel test --config=incremental //bazel/zig_worker:worker_test
```

The protocol definitions are `lib/std/zig/Client.zig`, `Server.zig`, and
`ErrorBundle.zig`; completion ordering is defined by `serveUpdateResults` in
`src/main.zig` in the [Zig 0.17.0 sources](https://ziglang.org/download/0.17.0/zig-0.17.0.tar.xz).
