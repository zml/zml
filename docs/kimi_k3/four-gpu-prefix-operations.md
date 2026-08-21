# Kimi K3 four-layer GPU0 and TP4 operations

The `kimi_k3_prefix_cli` is a development diagnostic for the resident S4
prefix: layers 0 through 3 only. It accepts arbitrary UTF-8 prompts and performs
deterministic greedy decoding, but its text is not a reliable Kimi K3 answer.
Production `llm` initialization remains configuration-driven and full-model-only.

## Supported layouts

- `gpu0`: exactly one visible CUDA device, selected with
  `CUDA_VISIBLE_DEVICES=0`; this is the default CLI mode and GPU-0 oracle.
- `tp4_ep1`: exactly four visible CUDA devices, selected with
  `CUDA_VISIBLE_DEVICES=0,1,2,3`, plus `--distributed`; tensors and KDA state
  are sharded across the model axis and the complete expert bank is replicated.
- `tp2_ep2` and `tp1_ep4`: logical ownership contracts only on the current
  one-axis ZML mesh. They are rejected as physical prompt layouts rather than
  silently emulated.

The canonical node has four NVIDIA GB300 GPUs with 284,208 MiB each on a
complete NV18 fabric. Keep at least 200,000 MiB free on every rank before a
conformance run so unrelated processes cannot distort memory or latency data.
The Milestone 26 runner rechecks this threshold before every TP4 workload.

## Offline build

From the ZML repository:

```bash
bazel --batch build --repository_disable_download \
  --@zml//platforms:cuda=true --@zml//platforms:cpu=false \
  //examples/llm:kimi_k3_prefix_cli \
  //examples/llm:kimi_k3_session_tests
```

Set these paths for the examples below:

```bash
workspace=/path/to/kimi-k3
weights="$workspace/artifacts/checkpoints/S4"
tokenizer="$workspace/artifacts/tokenizers/milestone-16/tokenizer.json"
cli="$workspace/zml/bazel-bin/examples/llm/kimi_k3_prefix_cli"
session="$workspace/zml/bazel-bin/examples/llm/kimi_k3_session_tests"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
export TF_DETERMINISTIC_OPS=1
```

Project runners never download weights.

## Prompt execution

GPU-0 oracle:

```bash
CUDA_VISIBLE_DEVICES=0 "$cli" \
  --weights="$weights" --tokenizer="$tokenizer" \
  --prompt="What is the capital of France?" \
  --max-new-tokens=2 --context-limit=512
```

Physical TP4:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 "$cli" \
  --weights="$weights" --tokenizer="$tokenizer" \
  --prompt="What is the capital of France?" \
  --max-new-tokens=2 --context-limit=512 --distributed
```

A valid run prints the permanent diagnostic warning, prompt token count,
generated token IDs, decoded bytes, timings, resident payload counters, and
`devices=1 layout=gpu0` or `devices=4 layout=tp4_ep1`. The GPU-0 and TP4 token
IDs and decoded byte stream must match; output quality is not an acceptance
criterion.

Use `--validate-only` to exercise UTF-8, tokenization, and the
`prompt_tokens + max_new_tokens <= context_limit <= 4096` contract without
initializing CUDA or opening checkpoint payloads. Empty/whitespace prompts,
invalid UTF-8, zero or more than 32 generated tokens, and capacity overflow
fail closed.

## Continuation, EOS, cache, and reset gates

The fixed raw-token continuation gate is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 "$session" \
  --weights="$weights" --tokenizer="$tokenizer" \
  --layer-limit=4 --token-count=4 --repeats=1 \
  --resident --decode-one --distributed
```

This must preserve shard-local KDA cache placement between the prefill and
continuation executables. Milestone 26 also dumps GPU-0 and TP4 prefill/decode
caches at capacity two and compares all 14 named segments with the centralized
BF16/F32 tolerances. A forced-EOS test proves that history and cache remain
unchanged when `<|end_of_msg|>` is already selected. Repeated fixed vectors
prove reset determinism.

## Bounded profiling

Pass `--profile-dir=<directory>` to capture one XSpace protobuf and converted
Perfetto trace around prefill plus decode. Profiled generation is capped at two
tokens. Do not commit large traces; the milestone manifest records their hashes
and sizes.

## Failure triage

1. Confirm exactly the intended devices are visible and otherwise idle.
2. Rebuild with the CUDA flag; a CPU-only build reports the backend unavailable.
3. Run the fixed raw-token continuation gate before an arbitrary prompt.
4. A full-width KDA cache where a shard-local cache is expected indicates lost
   output partitioning; do not copy or reshape it on the host to hide the bug.
5. Compare GPU-0 and TP4 cache reports before changing tolerances. Never derive
   a new tolerance from the distributed result under test.
6. Check resident counters: three staged layer loads and zero steady-state
   payload reloads are required.
7. Reproduce the complete gate from the workspace root with
   `./scripts/reproduce.sh 26`. If an unrelated workload takes a GPU after an otherwise successful early
   phase, rerun `scripts/milestones/milestone-26.sh --resume`; it validates the
   retained evidence before resuming at TP4 reset.

## Full-model boundary

S4 is about 54.6 GiB and fits comfortably in this scope. The complete checkpoint
is about 1.56 TB, while the four GPUs expose about 1.192 TB aggregate HBM before
runtime overhead. Therefore no four-layer or four-GPU PASS satisfies Gate F.
Full 93-layer logits, 32-token decoding, long-context progression, and deployment
SLOs remain mandatory on an eight-GPU-class node.
