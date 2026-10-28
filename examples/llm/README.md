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

For a warm generation measurement, use `--warmup-runs=1` with a fixed prompt,
sequence length, and sampling strategy. Compilation, weight loading, prefill,
and warm-up are outside the printed decode duration. That duration still
includes token transfer, text decoding, and output, as it does without warm-up.
Report the warm-up count alongside throughput when comparing configurations.

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
acceptance, and cache diagnostics. Timed-repeat output IDs are not read back;
repeated cache state is checked afterward. It is not useful-token generation
throughput and does not measure a speculative decoder.

Recorded on one P150a, Llama-3.1-8B-Instruct BF16, capacity 128, the prompt above,
first decode position 23, and four continuation steps (2026-10-28):

| Query rows | Attention | Mean block call | Exact serial gate |
| --- | --- | --- | --- |
| 1 | Native decode | 32.35 ms, 5 repeats | Passed |
| 2 | Native offset multi-query | 41.90 ms, 10 repeats | Failed: K/V drift |
| 2 | Native decode per query | 41.63 ms, 10 repeats | Failed: K/V drift |

Both two-row runs matched all four teacher-forced predictions and four
continuation IDs in this small trial, but caches first differed at layer 1,
row 23; layer 0 K/V were identical. Splitting attention changed the differences
without restoring exact cache state. These results do not establish serial
state equivalence or authorize a production speculative path. Offset coverage
and broader prompt validation remain pending while the drift is investigated.

A separate diagnostic with XLA fusion/codegen disabled aborted before the
numerical comparison because a feed-forward matmul's static circular buffers
clashed with existing L1 allocations. That run provides no numerical result
and no comparable block latency. Acceptance, wrong candidates, and cache
rollback are outside this harness.
