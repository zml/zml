# LLM

`//examples/llm` runs interactive or one-shot text generation from a model repository from HuggingFace.

We support the following models, automatically detected from the `model_type` in the `config.json`:

- Llama 3.1
- Qwen 3.5
- LFM 2.5

## Run

To load a model from HuggingFace directly:

```bash
# CPU
bazel run //examples/llm -- --model=hf://meta-llama/Llama-3.1-8B-Instruct
# CUDA
bazel run //examples/llm --@zml//platforms:cuda=true -- --model=hf://meta-llama/Llama-3.1-8B-Instruct
# ROCm
bazel run //examples/llm --@zml//platforms:rocm=true -- --model=hf://meta-llama/Llama-3.1-8B-Instruct
```

From a local directory:

```bash
bazel run //examples/llm --@zml//platforms:cuda=true -- --model=/var/models/meta-llama/Llama-3.1-8B-Instruct/
```

For a single non-interactive prompt:

```bash
bazel run //examples/llm --@zml//platforms:cuda=true -- --model=hf://meta-llama/Llama-3.1-8B-Instruct --prompt="Write a haiku about Zig"
```

## Options

- `--model=<path>`: Required. Model repository to load. This can be a local path or a huggingface/S3 URI such as `hf://...` or `s3://...`.
- `--prompt=<string>`: Optional. Runs a single prompt instead of opening the interactive chat loop.
- `--seqlen=<number>`: Optional. Maximum sequence length. Defaults to `2048`.
- `--backend=<stablehlo|cute|nki|metal>`: Optional. Attention backend. `cute` uses CuTe flash attention on CUDA GPUs with compute capability 8.0 or newer. If omitted, the program auto-selects one for the current platform, including `cute` on supported CUDA GPUs.

- `--warmup-runs=<n>`: Optional. With `--prompt`, generate and discard `n` responses before the measured run. Defaults to `0`. Each run gets a fresh conversation and cache state while sharing compiled executables and loaded weights. Warm-up durations are logged separately; only measured responses and decode rates are printed.

- `--benchmark-runs=<n>`: Optional. Generate and time `n` fresh sessions after warm-up, printing each response and decode rate. Defaults to `1`; must be positive, and requires `--prompt` for multiple runs. Executables and weights are reused, while conversation, cache, recurrent state, and RNG are reset for every session. Use, for example, `--warmup-runs=1 --benchmark-runs=5 --topk=1` to measure repeated greedy generations.

- `--dump-token-ids`: Print both prompt and generated token IDs after timed decode, so benchmark comparisons can verify the actual input and output sequences.

For a warm generation measurement, use `--warmup-runs=1` with a fixed prompt,
sequence length, and sampling strategy. Compilation, weight loading, prefill,
and warm-up are outside the printed decode duration. That duration still
includes token transfer, text decoding, and output, as it does without warm-up.
Report the warm-up count alongside throughput when comparing configurations.
The published-token count includes the first token selected during prefill and
excludes EOS; the timed loop includes the decode call that produces EOS.
For Llama, the first generated token is consumed at cache position equal to the
prompt length. The session must not skip that position after publishing the token.

## Teacher-forced Llama verifier experiment

The manual `llama_verifier` target compares BF16 greedy Llama execution with
one or two explicit query rows against ordinary serial decode. Two rows mean
one current token plus one candidate taken from the serial reference. This is
a numerical harness; it has no proposer, acceptance policy, or production
session flag.

Build the harness and its focused rowwise head test:

```bash
./bazel.sh build //examples/llm:llama_verifier //examples/llm:llama_verifier_head_test \
  --@zml//platforms:tenstorrent=true --@zml//platforms:cpu=false --jobs=2
```

Use the configured Tenstorrent PJRT plugin and runtime environment. The XLA
integration's wider BF16 schedules require `XLA_TT_EXPERIMENTAL_WIDE_DECODE=1`.
Keep these flags identical for paired comparisons:

```bash
export XLA_TT_EXPERIMENTAL_WIDE_DECODE=1
export XLA_TT_EXPERIMENTAL_TRACE=1
export XLA_TT_EXPERIMENTAL_BF16_PREFETCH=1
export XLA_TT_EXPERIMENTAL_BF8_WEIGHTS=0

bazel-bin/examples/llm/llama_verifier \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct \
  '--prompt=What is 7 plus 5? Explain briefly.' \
  --rows=1 --offset=0 --window=1 --continuation=4 --repeats=5
```

First require the one-row gate to pass. Then use `--rows=2 --window=2` for two
consecutive teacher-forced blocks. `--serial-attention` keeps projections,
normalization, and feed-forward layers wide while invoking the existing
single-query decode attention separately for each query row. Without that flag,
the verifier uses native offset multi-query attention.

`--offset=0` starts at the ordinary session's first decode position, printed
alongside the prompt token IDs. Explicit offsets such as `31`, `32`, or `63`
must follow the prompt and fit the selected `--seqlen` and `--window`.
`--continuation` requests ordinary serial steps from the verifier-produced
cache, capped by remaining capacity and reference EOS; the log reports the
actual continuation length. A window that would consume EOS is rejected; EOS
as the final prediction is allowed. `--repeats=0` disables timing.
The head test checks distinct row winners, smallest-index ties, and parity with
the original last-active-row head.

The harness loads weights once, snapshots the serial cache before verification,
and uploads independent cache allocations for each comparison. It first checks
the all-row head at sequence length one against the unchanged serial head.
It then compares every numerical-window prediction and serial continuation
token exactly. Full K/V snapshots are compared byte for byte outside timing; a
mismatch reports zero-based layer/head/row coordinates and maximum absolute
BF16 error.
Any token or cache mismatch fails `exact_serial_gate` and the process exits with
an error, even when the immediate predicted IDs agree.

`teacher_forced_upper_bound` reports mean warmed executable-call latency for
repeating the final fixed-input block. It excludes uploads, prediction readback,
acceptance, and cache diagnostics. The last timed-repeat output is retained
until the timer stops; its prediction IDs and the repeated cache state are
then checked against the serial reference. It is not useful-token generation
throughput and does not measure a speculative decoder.

The verifier trials below predate the session's first-decode-position correction
from `prompt.len + 1` to `prompt.len`. The verifier now follows the corrected
session boundary. These historical corpus results have not been revalidated
with the corrected starting position; the new native cache regression and
matched TTNN/ZML counting benchmark cover the ordinary session fix.

On one P150a with Llama-3.1-8B-Instruct BF16 and capacity 128 (2026-10-28),
the initial arithmetic trial passed the one-row gate at 32.35 ms/block.
Two-row offset multi-query attention (41.90 ms/block) and decode attention per
query (41.63 ms/block) initially matched tested token IDs but failed full-cache
comparison, first at layer 1, row 23; layer 0 K/V were identical.

The TTNN compiler's `LinearOpRewritePattern` was splitting a multi-row fused
matmul/residual operation into a BF16 matmul followed by a separate add,
introducing a rounding boundary absent from serial decode. The XLA integration
now preserves the fused multi-row residual operation under the wider-decode
opt-in (XLA fork commit `db8fe2c86c`; native regression tests `bbb36e8914`).
The successful results below require a plugin built with that fix and
`--serial-attention`; merely splitting attention did not fix the cache drift.

With the corrected plugin, all predictions, final timed-repeat predictions,
full K/V snapshots, and serial continuation tokens passed the unchanged exact
gate in these trials. Each run used two consecutive two-row blocks:

| Prompt | Block base positions | Continuation steps | Mean block call | Exact gate |
| --- | --- | --- | --- | --- |
| Arithmetic prompt above | 23, 25 | 4 | 41.51 ms, 10 repeats | Passed |
| Count from 1 to 20, separated by commas. | 31, 33 | 16 | 41.66 ms, 3 repeats | Passed |
| Explain why ice floats in water. Use two short paragraphs. | 32, 34 | 16 | 41.62 ms, 3 repeats | Passed |
| Write a short story about a robot learning to paint. | 63, 65 | 16 | 41.62 ms, 3 repeats | Passed |

For the corpus runs, use `--rows=2 --serial-attention --window=2
--continuation=16 --repeats=3`, the exact prompt text in the table, and
`--offset` equal to the first block position. All used the flags above.
These are teacher-forced block latencies, not speculative generation rates.
The full prompt/offset matrix and acceptance/rollback behavior remain untested;
these finite trials do not prove universal serial equivalence.

The corrected plugin's default offset multi-query attention mode still failed
the arithmetic cache gate at 41.61 ms/block, despite matching all tested IDs.
After two blocks, 98,450 K elements and 112,207 V elements differed, first at
layer 1. The validated path therefore requires `--serial-attention`; the
default multi-query mode remains a failing diagnostic.

A separate diagnostic with XLA fusion/codegen disabled aborted before the
numerical comparison because a feed-forward matmul's static circular buffers
clashed with existing L1 allocations. That run provides no numerical result
and no comparable block latency. Acceptance, wrong candidates, and cache
rollback are outside this harness.
