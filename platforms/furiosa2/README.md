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

## Llama status

The existing one-XLA-executable forward pass, vanilla attention, argmax sampling,
and original weights remain the target. The initial attempt used:

```sh
bazel run //examples/llm --@zml//platforms:furiosa2=true -- \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct \
  --backend=vanilla --seqlen=128 --topk=1 \
  --prompt='What is the capital of France?'
```

Model/tokenizer selection succeeded and the whole prefill forward reached the
new compiler. It stopped with `FURIOSA2 requires an F32 array result`, because
the initial vISA emitter only handles a small F32 elementwise subset. BF16,
multi-output units, contractions, reductions, indexing, and native memory helpers
were still backend work at that point. BF16 elementwise arithmetic, conversions,
tuple outputs and native memory helpers now pass hardware tests. The latest
Llama attempt stops at the embedding unit's `u32[128]` token indices; gather,
RMSNorm reduction/broadcast and contraction lowering remain unfinished.
**Llama does not generate tokens on furiosa2 yet; these
microbenchmark numbers are not tok/s measurements.**

## Validation

- `bazel build //... --@zml//platforms:furiosa2=true`: all 168 targets passed.
- `bazel test //... --jobs=16 --config=debug --@zml//platforms:furiosa2=false`:
  all 24 CPU test targets passed.
- The hardware add/negate benchmark above passed all three runs.
- Llama compilation failure is retained in `llama-first-attempt.log`; there is
  no fallback to TCL or host computation.
