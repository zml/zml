# Furiosa3: native TCL MLIR backend

Enable with `--@zml//platforms:furiosa3=true`, and point
`XLA_FURIOSA3_PJRT_LIBRARY` at the XLA checkout's
`bazel-bin/xla/pjrt/furiosa3/libpjrt_c_api_furiosa3_plugin.so`.
This platform has a distinct PJRT identity. It shares device allocation,
layouts, asynchronous execution and sharding behavior with the other Furiosa
platforms, and selects vanilla attention. PE count defaults to four; use
`--furiosa-pe-count=8` for the whole card.

Llama retains one XLA executable per whole forward and separate weight
arguments. Cached RoPE and declined-donation cleanup apply to Furiosa3 too.
The plugin lowers post-fusion HLO through verified TCL MLIR operations and
outlines private native units within that public executable.

The companion XLA checkout contains `xla/pjrt/furiosa3/run_llama.sh`, which
builds both components and selects the SDK and plugin paths for this host:

```sh
/home/steeve/xla-private/xla/pjrt/furiosa3/run_llama.sh \
  --prompt='Count from 1 to 100, separated by commas.'
```

Validation: `bazel build //... --@zml//platforms:furiosa3=true --jobs=16`
builds all 169 targets. Whole 32-layer Llama 3.1 8B position-0 forward passes
CPU argmax, KV tolerance and untouched-cache checks on four and eight PEs.
Eight-PE decode harness trials measure 59.4, 59.3 and 60.1 tok/s with original
BF16 weights, vanilla attention, argmax, cache length 128 and both cache
fusion options enabled. These do not establish long-context CPU equivalence.
Further generation and compiler evidence is recorded in the companion XLA
`xla/pjrt/furiosa3/experiments/2026-09-27-llama` directory.
