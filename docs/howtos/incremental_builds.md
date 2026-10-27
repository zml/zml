# Incremental Zig builds

Build or run MNIST with a compiler that stays alive between Bazel invocations:

```sh
bazel build --config=incremental //examples/mnist
bazel run --config=incremental //examples/mnist
```

Edit a Zig source file and repeat the command. An unchanged target is still a
normal Bazel cache hit. When a source changes, Bazel sends another request to the
existing compiler. `bazel shutdown` releases the workers and their compiler state.

The configuration uses Debug Zig code and optimized C/C++ dependencies, like
`--config=debug`. It selects LLVM because Zig 0.17's self-hosted backend produced
incremental MNIST archives that the C linker rejected. Zig's frontend retains its
state; LLVM code generation and the final C link still take time.

## How it works

[Bazel persistent workers](https://bazel.build/remote/persistent) receive one
request for each compile action. The patch in
`third_party/rules_zig/incremental-worker.patch` adds an optional executable
adapter to the pinned `rules_zig` 0.17.0 release. The adapter is selected only by
the incremental configuration; host tools continue to use ordinary Zig actions.
No separate checkout or local module override is needed.

`bazel/zig_worker/worker.py` translates Bazel's JSON worker messages into Zig's
binary compiler protocol. It launches `zig ... -fincremental --listen=-`, sends
an `update` message for each build, and waits for the terminating error bundle.
This is the compiler-server interface used by Zig's own build system. `--watch`
belongs to the build runner: Bazel already detects changes and schedules actions,
so an independent filesystem watcher would rebuild outside Bazel's control.

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
on Linux x86_64. The adapter checks the compiler protocol version; that internal
protocol must be reviewed before upgrading to another Zig minor version.

Worker sandboxing is disabled in this configuration because Zig retains open
input directories across requests. Bazel 9.3 applies that switch to all workers
in the invocation. Other action sandboxing remains enabled. Incremental Zig
actions are explicitly local and do not use remote execution or remote caching.
Use the ordinary build configurations for remote builds.

Workers consume memory until Bazel evicts them or shuts down. There can be one
compiler per distinct output, even with `--worker_max_instances=ZigIncremental=1`.
Changing configuration, worker code, or compiler tools can also discard workers.

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
