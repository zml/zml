# Queued decode diagnostic

`llm --benchmark-decode-queue=N` compares serial token readback with queue
depths 2 and 8 using the same compiled Llama forward. It requires a prompt,
Furiosa3, and the `furiosa_fa` attention backend. Normal generation is unchanged.

The diagnostic resets the complete KV cache to zero and the RNG to seed 0
before each run, then prefills the same prompt. It executes four untimed
warmups and N timed autoregressive decode steps. Every sampled token feeds the
next execution on device. Token readbacks for every rank are requested before
the next execution can donate their buffers, and event waits are deferred to
the end of each bounded batch. Host destinations remain alive through all
copies, including error cleanup.

Three trials rotate queue depths as 1/2/8, 2/8/1, 8/1/2. After timing, the
program verifies every generated token on every rank against the first serial
run, rejects out-of-vocabulary/poison values, and compares the complete K/V
cache and RNG bytes exactly. This checks queue and donation ordering against
a serial device reference; it is not an independent CPU model oracle.

The workload deliberately continues after EOS and excludes terminal output,
prefill, cache reset and final state comparisons from timing. It reports actual
decode **steps/s**, with no prefill token in the numerator. Do not report this
as the normal interactive CLI's throughput or enable queued normal generation
without addressing EOS and session-state semantics.

Build with Furiosa3 enabled (the default CPU-only build cannot run TCL attention):

```sh
bazel build --repo_env=RULES_ZIG_CACHE_PREFIX=/tmp/zig-cache-steeve \
  --//platforms:furiosa3=true //examples/llm:llm
```

With the same SDK/plugin environment as the usual Llama command:

```sh
bazel-bin/examples/llm/llm \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct \
  --furiosa-pe-count=8 --backend=furiosa_fa --topk=16 --seqlen=256 \
  --benchmark-decode-queue=192 \
  --prompt='List every integer from 1 to 200, separated by commas. Do not skip any numbers.'
```

The prompt length plus four warmups plus N must fit in `--seqlen`.
For the September 29 two-card BF16 experiment, balanced-order medians were
95.011 steps/s at depth 1, 95.992 at depth 2, and 96.681 at depth 8 (+1.76%
versus serial). All token/state comparisons passed. This is a modest
benchmark-only improvement, not the 130 tok/s goal. Exact settings, binary
hashes, compiler logs, first attempts and all trial rates are retained in
`xla-private/xla/pjrt/furiosa3/experiments/2026-09-29-queued-decode/`.
