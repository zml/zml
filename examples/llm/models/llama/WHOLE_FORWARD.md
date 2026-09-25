# Whole-model Llama forward

Llama now compiles embedding, all transformer layers, final normalization,
output projection and sampling together. Prefill and decode each have one XLA
executable. There is no host-side layer grouping. Greedy sampling is the CLI
default (`--topk=1`). The final normalization runs once in `LmHead`.

Weights now use the ordinary loader and remain **291 separate tensors**. The
packing plan, packed tensor type, concatenation writer and graph-side unpacking
slices have been removed. `Forward.Input.weights` is `model.Model`, and inference,
session, component tests and logits all use `model.Buffers` and `loadBuffers`.
The entire forward pass still compiles together. Argmax remains the default.

Furiosa bridge/11 handles address indirection internally when a program exceeds
119 inline arguments. It supports up to 480 arguments including runtime statics
on one card. The wrapper reconstructs the task argument array from a device
address table; it does not concatenate or copy weight payloads. Detailed ABI,
limits, test evidence and failed experiments are committed in the XLA checkout:
`xla/stream_executor/furiosa/opt_runtime/INDIRECT_ARGUMENTS.md`.

## Validation after packing removal

- `//examples/llm`, `//examples/llm:llama_tests` and
  `//examples/llm:llama_logits` build with Bazel 9.1.1,
  `--@zml//platforms:furiosa=true --config=debug`.
- A forced-indirect eight-layer bridge comparison passes at layers 24–31,
  query length 1, cache length 128 and position 127, checking hidden states, KV
  updates and untouched cache entries. This is a component validation, not
  grouped production inference.
- The bridge's independent PJRT hardware test passes with 150 separate inputs
  and 150 separate outputs, across changing data and reversed input bindings.
- The new complete 32-layer forward with 291 separate weight arguments is
  compiling at this commit. Full-model correctness and throughput are pending.
  No new rate is claimed. The captured TCL SHA256 is
  `748a8f3972ffb90bce92db7a331f9064c69cf8419b00b22186d4fcc8384597c8`;
  the log is `/home/steeve/.local/state/xla-rngd/internals/indirect-args/llama-full.log`.

## Historical packed-weight validation

The removed packed whole-forward graph passed the unpacked CPU reference at
query length 1, cache length 128 and position 127: exact argmax, existing KV
tolerances (absolute 0.03, relative 0.02, all elements), and exact preservation
of untouched cache entries. CPU reference and actual weights were loaded and
released sequentially to bound host memory; that sequencing remains in the
comparison harness.

The packed RNGD executable compiled and ran but **failed the CPU comparison**:
it returned token 323 where CPU returned 311. No tolerance was relaxed. The
following measurements describe that removed implementation. Neither they nor
the successful component tests establish correct current full-model inference.

## Provisional decode benchmark, 2026-09-25

Original BF16 `/var/models/meta-llama/Llama-3.1-8B-Instruct`, one RNGD card,
vanilla attention, argmax, cache length 128, one complete forward executable.
Staged fallback, execution tracing and profiling were disabled. No concurrent
build or compiler job ran during timing.

Three trials of 100 autoregressive decode steps measured **52.93, 52.73 and
52.81 tok/s**, median **52.81 tok/s**. Each trial starts from BOS with zeroed KV,
runs four warmup steps, then advances positions and feeds each predicted token
back to the executable. Timing includes synchronous token readback. EOS does
not terminate these fixed-length trials. Compilation, uploads, prefill,
tokenization and terminal rendering are excluded. This is a decode benchmark,
not a validated chat-generation rate; the correctness failure above remains.

The same benchmark command now exercises separate weights and requires bridge/11.
After sourcing the SDK environment and setting `XLA_FURIOSA_PJRT_LIBRARY` and
`XLA_FURIOSA_COMPILER_CACHE` as usual:

```sh
bazel-bin/examples/llm/llama_tests \
  --platform=furiosa \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct \
  --compare-cpu --forward-only --seqlen=1 --cache-seqlen=128 \
  --token-offset=127 --benchmark-iterations=100
```

The command logs throughput before performing the correctness comparison.
The historical packed run exited nonzero; the new result is pending. Evidence is retained locally in
`/home/steeve/.local/state/xla-rngd/internals/full-forward/decode-benchmark.log`.
The successful CPU-only check is `packing-cpu-sequential.log` in that directory.

Build validation passed for `//examples/llm`, `//examples/llm:llama_tests` and
`//examples/llm:llama_logits` with `--@zml//platforms:furiosa=true --config=debug`.
The benchmark changes were rebuilt with the `llama_tests` target. The captured
decode TCL SHA256 is
`efa80d98893c6605d47af8b8291bd1604e9ae79c0b6834575247c91b55ad8848`.

## Initial slowdown diagnosis

A separate diagnostic run enabled `XLA_FURIOSA_TRACE_EXECUTION=1`,
`XLA_FURIOSA_PROFILE=1`, `XLA_FURIOSA_PROFILE_LEVEL=info` and
`XLA_FURIOSA_PROFILE_COUNT=1`, using eight timed iterations. It is not the
unprofiled throughput measurement above. The first invocation was profiled;
subsequent invocations provided host launch timing.

The EDF has binary shape `[1, 2, 5, 784128]`: five internal task chunks per
cluster in a single executable. The first invocation's task spans sum to
35,122,644/35,114,919 cycles across the two clusters. Including gaps between
chunks gives 36,247,632/36,240,036 cycles. Subsequent native launches have median
wait time 18,625 microseconds; argument preparation and submission together are
typically about 15 microseconds. Thus native execution dominates the roughly
18.9 ms per token, rather than host argument binding or the removed layer loop.

The prior eight-layer candidate had two chunks and about 6.4 million task cycles
per invocation, repeated four times plus separate embedding/head execution.
The new larger graph and packed weight arguments changed the compiled program;
the specific operations or layout choices responsible have not been isolated.
Internal task chunking is compiler scheduling inside one EDF, not multiple XLA
executables. Evidence: local `internals/full-forward/decode-profile.log`.

## Comparison diagnostics after removing packing

The historical packed run stopped at its first argmax mismatch, so it did not
establish whether the transformer KV outputs matched CPU. The comparison now
checks both KV components after an argmax mismatch and reports each layer's
maximum absolute error and number of updated entries outside tolerance. This
helps distinguish errors already present in the transformer from differences
in the final normalization, projection or argmax. Nonfinite entries count as
failures. Untouched cache storage is still checked bit for bit.

The acceptance criteria are unchanged: exact argmax, absolute KV tolerance
0.03 plus relative tolerance 0.02, and every KV element must pass. A mismatch
still returns `TestUnexpectedResult`; other errors are propagated immediately.
The added reporting does not change the compiled model or benchmark timing.

Validation on 2026-09-25:

| Check | Why | Result |
|---|---|---|
| Zig formatting | Keep the diagnostic changes consistent with repository style | Applied |
| `bazel-9.1.1 build //examples/llm:llama_tests --@zml//platforms:furiosa=true --jobs=16 --config=debug` | Compile the revised error handling and per-layer BF16 cache inspection | Pass; 7.858 seconds, log in [testdata/diagnostic-build.log](testdata/diagnostic-build.log) |
| Full separate-weight forward CPU comparison and three 100-token decode trials | Verify correctness and measure performance without packed weight slices | Still compiling when this diagnostic change was committed; no result yet |

The running full-model process started before this diagnostic rebuild. If it
fails, rerun the rebuilt test using the compiler cache to obtain per-layer
reports. The separate-weight TCL SHA256 is
`748a8f3972ffb90bce92db7a331f9064c69cf8419b00b22186d4fcc8384597c8`;
its live log is
`/home/steeve/.local/state/xla-rngd/internals/indirect-args/llama-full.log`.
Hardware execution of the new failure-reporting path is not yet verified.
