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
- The complete 32-layer forward with 291 separate weights compiles and runs.
  It passes CPU comparison at position 0, but fails at position 127. Three
  unprofiled 100-token decode trials measured 61.83, 62.34 and 62.45 tok/s.
  These are provisional decode timings, not validated text-generation rates.
  Details and evidence follow below. The captured TCL SHA256 is
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


## Separate-weight full-forward results (2026-09-26 review)

The SDK compile completed on September 25 in **1136.8638 seconds** (18m57s).
The main `prelower -> postlower` phase took 886.66724 seconds. The source hash
above identifies the program. The EDF is about 57 MiB. No staged fallback or
forced-indirect flag was enabled: the 291 separate weights select bridge/11's
address table automatically.

| Experiment | Why | Result |
|---|---|---|
| Unprofiled, three 100-token autoregressive decode trials | Measure the full executable after removing weight packing, with no concurrent compiler or build | 61.83 / 62.34 / 62.45 tok/s; median 62.34, about 18% above the historical packed median |
| CPU comparison, position 127, deterministic nonzero prior KV | Exercise the complete transformer with attention to the populated cache | **Fail**: device argmax 323, CPU 311; both KV components also exceed the existing tolerance |
| Diagnostic rerun, position 127, cached executable | Locate differences rather than stopping at argmax | Layer 0 keys exact; key tolerance first exceeded at layer 5, value tolerance at layer 22; all untouched cache bits preserved |
| CPU comparison, position 0 | Check the same complete program without attending to prior cache positions | **Pass**: exact argmax 76944, KV tolerance and untouched cache bits |
| Info-level device profile plus launch tracing, position-0 run | Separate native execution from address-table and host overhead | Five internal chunks; task cycles 28,283,178 / 28,280,253; spans including gaps 30,037,186 / 30,035,322 across the two clusters |

The position-127 updated-key maximum absolute error is 0.125, and the
updated-value maximum is 0.056152344. Differences accumulate across layers;
this does not yet identify an incorrect primitive or establish acceptable
full-model numerical accuracy. Acceptance tolerances were not changed.
The improved failure-reporting path was exercised and returned a failing status.

The separate profile run used `XLA_FURIOSA_PROFILE=1`,
`XLA_FURIOSA_PROFILE_LEVEL=info`, `XLA_FURIOSA_PROFILE_COUNT=1`,
`XLA_FURIOSA_TRACE_EXECUTION=1`, `--token-offset=0` and
`--benchmark-iterations=8`. Median ordinary launch measurements were 55.31 us
for arguments, 2.925 us for submission and 15,565.637 us waiting for completion.
The address table remains small compared with execution. Profiled wall time is
not used as the throughput claim. Removing packing also changes compiler layout
choices, so these results do not isolate the cost of graph slices alone.

Raw logs are committed in `testdata/separate-{compile,benchmark,diagnostics,profile-position0}.log`.
The next real-generation run uses the normal LLM CLI with `--seqlen=128`,
`--backend=vanilla`, `--topk=1` and prompt
`Count from 1 to 100, separated by commas.` Its full prefill graph is compiling;
no generation result is claimed yet. The approximately 100 tok/s objective and
position-127 correctness investigation remain open.

## CPU logit-margin diagnostic

The position-127 token mismatch alone does not show whether the winning tokens
have nearly equal logits or a large score difference. The independently
composed CPU reference now also returns its logits. For a single-token query,
the test reports the CPU score of both the expected token and the device's
chosen token, their difference, and how many CPU vocabulary scores are higher
than or tied with the device token. These are CPU scores, not device logits.
The Furiosa forward graph and its outputs remain unchanged, so this diagnostic
can reuse the already compiled EDF. The test still fails on the same argmax
and KV criteria.

`bazel-9.1.1 build //examples/llm:llama_tests --@zml//platforms:furiosa=true
--jobs=1 --config=debug` passed in 56.489 seconds; Zig formatting and
`git diff --check` passed. The build log is committed as
`testdata/logit-diagnostic-build.log`. Runtime validation is pending: the
full-prefill SDK compile currently uses about 52 GiB, so a CPU full-model
reference must wait to avoid overlapping their memory demands. The diagnostic
build ran during prefill compilation, not during any throughput measurement.
The idle Bazel server was shut down after the build to release memory.


## BF16 bandwidth estimate

Furiosa's official RNGD specification lists 1.5 TB/s HBM3 bandwidth:
https://developer.furiosa.ai/latest/en/overview/rngd.html (checked 2026-09-26).
Checkpoint safetensors headers give 13,958,643,712 bytes for transformer
projection weights, 1,050,673,152 bytes for the output head and 532,480 bytes for
normalizations. Reading these once per token totals 15,009,849,344 bytes.
At the advertised bandwidth this alone takes 10.0066 ms, or 99.9344 tok/s.
The 62.34 tok/s measurement corresponds to 935.714 logical weight GB/s.

This is an analytical estimate, **not measured HBM traffic or utilization**.
It excludes embedding lookup, KV/activation/scratch transfers and any retained
on-chip weights; it assumes ordinary single-token dense inference without
compression. Thus roughly 100 tok/s on the original BF16 checkpoint is near the
ideal weight-read limit. This does not change the target or authorize switching
precision. Evidence is `testdata/bf16-bandwidth-estimate.json`.


## Logit-margin result and prefill memory failure

The rebuilt position-127 comparison completed in 28.110 seconds and still
failed the original assertions. CPU logits for both token 311 (its argmax)
and token 323 (the device result) are **2.328125**. No CPU score is higher and
exactly two scores share that value. Thus the device token belongs to the CPU's
tied maximum; this does not establish whether the device logits also tie or
whether small accumulated differences changed their ordering. The KV errors
are unchanged. No tolerance or tie-breaking criterion has been relaxed.
The command, exit status and full diagnostics are committed in
`testdata/llama-full-logit-margin127*`.

The first complete 128-token prefill compilation failed during
`prelower -> postlower` when the kernel OOM killer terminated `furiosa-tcc`
PID 1287605 at 2026-09-26 07:03:10 UTC. The kernel recorded 61,490,924 KiB
anonymous RSS; host swap was exhausted. No prefill EDF or generation result
was produced. A manual SIGTERM was attempted after observing resource
exhaustion, but the compiler had already exited; the command returned
`No such process`. Its OOM score had been raised to 1000 to prefer that process
over the session. Failed CLI/compiler logs and the kernel OOM lines are
committed as `testdata/prefill-default-oom-*`.

The logit-margin comparison was queued using a Linux process descriptor for the
existing generation process and started only after that process exited, keeping
the large CPU reference separate from SDK compilation.

A new prefill attempt uses the same TCL source, shape, model, eight-candidate
search policy, compiler and cache with **RAYON_NUM_THREADS=16** set only in the
compiler wrapper. The inference process does not inherit this override. Early
compiler resident memory was about 20.5 GiB versus roughly 50 GiB in the first
attempt. Completion, final peak memory and generation throughput are pending.
The log is `internals/indirect-args/llama-generation-worker16.log`. The compiler
PID 1291658 also has OOM score 1000. No jemalloc preload is enabled.
