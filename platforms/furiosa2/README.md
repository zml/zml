# Furiosa2 vISA platform

`furiosa2` loads the separate vISA PJRT plugin while sharing Furiosa's device
memory, sharding, layout queries, vanilla attention selection, and asynchronous
execution behavior. Enable it with `--@zml//platforms:furiosa2=true` and set
`XLA_FURIOSA2_PJRT_LIBRARY`. The initial compiler supports four PEs, so
`CreateOptions.furiosa2.pe_count` defaults to four; the original Furiosa default
remains eight.

## Verified ZML benchmark

From this checkout, with the companion XLA repository built:

```sh
export XLA_FURIOSA2_PJRT_LIBRARY=/home/steeve/xla-private/bazel-bin/xla/pjrt/furiosa2/libpjrt_c_api_furiosa2_plugin.so
export XLA_FURIOSA2_COMPILER=/home/steeve/xla-private/xla/pjrt/furiosa2/compile_visa.sh
export XLA_FURIOSA_RUNTIME_LIBRARY=/home/steeve/.local/share/xla-rngd-sdk/libdevice_runtime.so
bazel run //examples/benchmark --@zml//platforms:furiosa2=true -- \
  --operation=add_negate --size=2048 --dtype=f32 --iterations=1000
```

The benchmark logs the selected platform and generated StableHLO, completes a
warmup before timing, waits for every measured result, and checks every final
element against the uploaded inputs outside the timed region. The measured
duration includes PJRT dispatch, device execution, completion, and output-buffer
management; it is not a device-only kernel duration. It leaves the matmul
benchmark as the default operation.

On 2026-09-26, three separate invocations measured:

| Run | Cached compilation/load | Mean per execution, 1000 iterations |
| --- | --- | --- |
| 1 | 113.246 ms | 36.401 us |
| 2 | 103.918 ms | 24.736 us |
| 3 | 109.199 ms | 31.210 us |

All reported platform `furiosa2` and passed the output check, with
`XLA_FURIOSA_COMPILER=/nonexistent/tcl-compiler` to verify independence from TCL
compilation. A fresh compiler/Cargo output cache took 17.907 s for compilation
and load, but ran concurrently with the all-target build. Its 136.613 us runtime
mean is retained as a contended observation, not a comparable performance
baseline. An earlier 100-iteration smoke run measured 35.413 us.

Logs are retained at
`/home/steeve/.local/state/xla-rngd/furiosa2/zml/{cold,warm-1,warm-2,warm-3}.log`
and copied into the companion XLA backend's experiments directory.

With the subsequent BF16 lowering and bridge/13, the same benchmark also
accepts `--dtype=bf16`. Its finite-input reference rounds the addition to BF16
using round-to-nearest-even before negating; it checks output bits after timing.
For `--size=14336 --dtype=bf16 --iterations=1000`, three warmed invocations
measured 51.711, 49.040 and 59.449 us per execution, with 94.327–99.850 ms cached
compilation/load. Each selected furiosa2 and passed correctness checks with the
TCL compiler path disabled. These are microbenchmark latencies, not Llama tok/s.
The companion XLA `2026-09-26-bf16` experiment retains the logs and SDK layout
failures found while extending the emitter.

## Llama status, 2026-09-27

Llama 3.1 8B Instruct now compiles both prefill and decode and generates text
through the vISA plugin. Original BF16 weights, vanilla attention, argmax and
one public XLA executable per forward are retained. No TCL fallback or weight
packing is used. With a 128-position cache on four PEs, three warmed 64-token
full-forward decode trials measured **14.23, 14.27 and 14.35 tok/s**, including
synchronous token readback but excluding compilation and weight upload. This
is an untuned baseline, substantially below the original TCL backend.

A subsequent vISA emitter change redistributes large computed BF16 tensors
in device SRAM before writing them to HBM. Two processes, each with three
64-token trials, measure **15.21–15.45 tok/s** with the same four-PE model setup.
The position-zero comparison and real chat example pass; the position-eight
history test retains the same 109 key-cache values outside tolerance. The XLA
`2026-09-27-hbm-stores` experiment records all results, including an isolated
candidate's faster 16.2 tok/s result that is not the integrated performance claim.

Retaining decode projection inputs and outputs in SRAM subsequently raises
the integrated result to **17.27–17.31 tok/s**, measured in six 64-token trials
across two processes. A fresh preceding baseline measured 15.37–15.38 tok/s.
All 119 hardware tests and six host targets pass; the position-zero comparison
and chat example pass. The history comparison retains the same 109 failing
keys and per-layer maxima. The XLA `2026-09-27-projection-sram` experiment
records controlled variants, schedule evidence and integrated results.

The shared runtime's bridge/14 adds native eight-PE copy/zero helpers, with
hardware validation of queued lifetimes and a direct projection across both
clusters (XLA commit `22c1280188`, `2026-09-27-eight-pe-runtime` experiment).
Furiosa2's production compiler still requires four PEs. ZML regression trials
with this runtime measure **17.29, 17.31, 17.30 tok/s**; position zero passes,
and populated history retains the same 109 key-cache failures. This runtime
foundation does not yet enable eight-PE model compilation.

```sh
export XLA_FURIOSA_COMPILER=/nonexistent/tcl-compiler
bazel run //examples/llm --@zml//platforms:furiosa2=true -- \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct \
  --backend=vanilla --seqlen=128 --topk=1 \
  --prompt='What is the capital of France?'

bazel run //examples/llm:llama_tests --@zml//platforms:furiosa2=true -- \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct \
  --platform=furiosa2 --compare-cpu --forward-only \
  --seqlen=1 --cache-seqlen=128 --layers=32 --benchmark-iterations=64
```

The position-zero full-forward comparison passes exact argmax, the existing
KV tolerances, and untouched-cache checks. At position eight with identical
CPU-computed history, argmax still matches but some key-cache elements exceed
tolerance. Disabling CPU excess precision reduces but does not eliminate that
failure. Neither tolerances nor defaults have changed. The companion XLA
`xla/pjrt/furiosa2/experiments/2026-09-27-llama` directory retains both passing
and failing results and the benchmark's binary provenance.

The opt-in `llama_tests --layerwise --layerwise-stages` diagnostic now exposes
Q/K/V projections, rotary outputs, scaled keys, attention scores, softmax
probabilities, context and output projection. It verifies that its diagnostic
attention expansion exactly matches the existing production attention on
each backend. These exact-match statistics identify rounding differences;
they do not replace or waive the whole-forward tolerance gate. Use
`--layerwise-dump-dir=<path>` to save the raw BF16/F32 intermediates.

`--forward-only --reference-f64-dots` adds a separate CPU diagnostic that
accumulates projections and attention contractions in F64, then rounds their
results to BF16. The default reference and tolerances are unchanged. It first
checks 195,841 BF16 rounding cases; direct CPU F64-to-BF16 conversion double
rounds at some midpoints, so this diagnostic rounds the significand explicitly.
The CPU reference executes one layer at a time to bound widened-weight memory;
the accelerator still executes its complete compiled forward.

At position eight with CPU-computed history and
`XLA_FLAGS=--xla_allow_excess_precision=false`, furiosa2 versus this reference
has 79 key and one value elements outside tolerance. The ordinary CPU forward
versus the same reference also fails (64 key elements); both match argmax 1001.
These results establish that this accumulated-error tolerance failure also
occurs without RNGD. They do not clear the existing model gate or establish
whole-model accuracy. See the XLA `2026-09-27-f64-reference` experiment for
commands, calibration failures and memory reports.

`--layerwise --prefill-history` now uses that same CPU-computed token prefix
for each layer's CPU, local-device and propagated-device comparisons. With
32 layers at position eight, all 96 local-input hidden/key/value checks pass
with both default and strict CPU precision. With propagated inputs, strict
precision produces 61 key failures and no hidden/value failures; its per-layer
key failure counts and maxima match the recorded whole-forward comparison.
Default precision produces 194 key, two value and one hidden failures in this
layerwise diagnostic, so its metrics do not match the default whole forward.
Both runs correctly exit 1. These results reproduce accumulated drift using
model-computed history, without clearing the original whole-forward gate.
The XLA `2026-09-27-layerwise-history` experiment retains commands and logs.

## Validation

- `bazel build //... --@zml//platforms:furiosa2=true`: all 168 targets passed.
- `bazel test //... --jobs=16 --config=debug --@zml//platforms:furiosa2=false`:
  all 24 CPU test targets passed.
- The hardware add/negate benchmark above passed all three runs.
- Llama compilation failure is retained in `llama-first-attempt.log`; there is
  no fallback to TCL or host computation.
