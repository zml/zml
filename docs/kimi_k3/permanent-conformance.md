# Kimi K3 permanent conformance and operations

Milestone 20 froze the historical boundary between full-model readiness and
differential validation. The full-model readiness path remains configuration-driven
and binds all 93 text layers. Partial layer selection, compact expert fixtures,
recorded route alignment, and expanded intermediate tensors remain isolated in
dedicated conformance executables.

The normal Kimi `//examples/llm` path is now a separate fixed diagnostic: it
selects layers 0-16 and keeps them resident on exactly four CUDA devices. It is
not full-model initialization and its decoded text is not a reliable answer.

All commands below are offline with respect to model data. Project scripts never
download weights. Historical and full-model conformance use physical NVIDIA GPU 0. The
isolated
four-layer diagnostic additionally supports physical TP4 as documented in
`four-gpu-prefix-operations.md`:

```bash
export CUDA_VISIBLE_DEVICES=0
workspace=/path/to/kimi-k3
cd "$workspace/zml"
```

## Permanent production boundary

The reusable production executables return only the state consumed by the next
step:

- layer 0: hidden output, block residual, and KDA cache;
- KDA/MLA MoE: hidden output and cache, plus block state at official boundaries;
- output head: generated token.

Expanded KDA, MLA, routing, MoE, layer, and head results remain permanent
numerical oracles in isolated conformance binaries. They are not compiled as the
result arity of the production session. Production token execution contains no
activation transfer, per-token wall clock, or expert-staging progress log.
Use standard ZML profiling for performance attribution.

The dependency-free policy audit is:

```bash
.venv/bin/python tools/kimi_k3/test_cleanup_audit.py
.venv/bin/python tools/kimi_k3/cleanup_audit.py
```

## Fixed 47-layer example

The normal Kimi `//examples/llm` command executes layers 0-46 and keeps every
selected weight resident while generating tokens. It requires exactly four CUDA
devices and targets four NVIDIA GB300 GPUs. Ordinary tensors retain TP4 while
the same physical ranks each own a contiguous 224-expert shard of the six
routed-expert value/scale banks. This is shared-axis TP4+EP4, not a 16-rank
Cartesian topology. Routed-expert HBM is exactly 180,807,008,256 bytes per rank
for the 46 resident MoE layers; expected total allocator use is approximately
200-210 GB per rank.

The fixed selection performs 46 resident MoE loads, 248,518 checkpoint payload
reads, and 776,886,773,760 source payload bytes. The read total is the measured
family sum `35*5,404 + 11*5,398`; it corrects the earlier 248,492 estimate
without changing the source-byte accounting.

Create the model-local tokenizer link once:

```bash
cd /home/kevin/kimi-k3
ln -s ../../artifacts/tokenizers/milestone-16/tokenizer.json \
  moonshot/kimi-k3/tokenizer.json
```

Invoke Bazel with platform options before the target and allow enough total
prompt-plus-generation capacity. The formatted France prompt is approximately
95 tokens, so `--seqlen=10` is expected to fail as too small.

```bash
cd /home/kevin/kimi-k3/zml

CUDA_VISIBLE_DEVICES=0,1,2,3 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
XLA_FLAGS=--xla_gpu_deterministic_ops=true \
TF_DETERMINISTIC_OPS=1 \
bazel run \
  --@zml//platforms:cuda=true \
  --@zml//platforms:cpu=false \
  //examples/llm -- \
  --model=/home/kevin/kimi-k3/moonshot/kimi-k3 \
  --prompt="What is the capital of France?" \
  --seqlen=128
```

The process emits `KIMI_K3_DIAGNOSTIC_WARNING layers=47 full_model=false
reliable_answer=false`. Output is deliberately truncated-model diagnostic text;
it must not be presented as a factual response or as full Kimi K3 inference.

## Conformance groups

| Group | Permanent targets or tools |
| --- | --- |
| Config/checkpoint | `kimi_k3_tests`, `test_checkpoint_tools.py`, `test_full_model_preflight.py` |
| Tokenizer | `kimi_k3_tokenizer_tests` |
| Primitives and AttnRes | `kimi_k3_primitives_tests`, `kimi_k3_attn_res_tests` |
| KDA | `kimi_k3_kda_tests`, `kimi_k3_kda_prefill_tests`, `kimi_k3_kda_optimized_tests`, layer-0 cache tests |
| MLA | `kimi_k3_mla_tests`, `kimi_k3_mla_cache_tests`, `kimi_k3_mla_optimized_tests` |
| Router and experts | `kimi_k3_router_tests`, `kimi_k3_grouped_mxfp4_tests`, `kimi_k3_moe_tests`, `kimi_k3_expert_parallel_tests` |
| Integrated layers | `kimi_k3_layer0_tests`, `kimi_k3_prefix_tests`, `kimi_k3_layer_family_tests`, `kimi_k3_prefix4_tests` |
| Session/readiness | `kimi_k3_runtime_weights_tests`, `kimi_k3_session_tests`, `kimi_k3_readiness_tests` |

Build the production CLI and all permanent Kimi K3 executables with:

```bash
CUDA_VISIBLE_DEVICES=0 bazel --batch build \
  --repository_disable_download \
  --@zml//platforms:cuda=true --@zml//platforms:cpu=false \
  //examples/llm:llm '//examples/llm:kimi_k3*'
```

The workspace-level `./scripts/reproduce.sh 20` performs the canonical cleanup
gate and records logs, a machine-readable report, and artifact hashes.

## Minimum-checkpoint execution

S1, S2, and S4 are generated indexes and local symlinks, never weight copies.
S4 uses official text shards 1, 2, 3, 4, and 94 and covers every layer family.
After the checkpoint tiers and fixtures have been generated by their historical
milestone runners, use the S4 session gate for reset and decode coverage:

```bash
CUDA_VISIBLE_DEVICES=0 bazel-bin/examples/llm/kimi_k3_session_tests \
  --weights="$workspace/artifacts/checkpoints/S4" \
  --tokenizer="$workspace/artifacts/tokenizers/milestone-16/tokenizer.json" \
  --layer-limit=4 --token-count=4 --repeats=2
```

`--layer-limit` belongs to this test executable only. The normal `llm` command
has no configurable partial-model option; Kimi uses the fixed internal
47-layer diagnostic selection described above.

## Fixture regeneration

Fixture exporters require the pinned root uv environment because they execute
the Moonshot/PyTorch reference. Restore it with `./scripts/bootstrap.sh` only
when dependency downloads are authorized. Then run the milestone that owns the
fixture, for example:

```bash
./scripts/reproduce.sh 9   # layer 0 and S2 prefix
./scripts/reproduce.sh 14  # KDA/MLA layer families
./scripts/reproduce.sh 15  # four-layer S4 prefix
```

Exporters always write manifests with source/checkpoint identity and tensor
semantic hashes. They no longer expose verbose activation-print switches.
Regeneration must not overwrite accepted evidence unless the corresponding lock
file and SITREP are intentionally updated and reviewed.

## Profiling

Use the existing `--profile`, `--profile-repository`, and `--profile-session`
options on conformance executables. A performance run must use warm-up,
synchronize by consuming a result, keep load/compile measurements separate,
and omit diagnostic tensor copies. Convert XSpace output with the repository's
`xspace_to_perfetto` target. Hardware, driver, backend, dtype, batch, sequence
length, sample distribution, and trace hash belong in the report.

## Failure triage

1. Run `cleanup_audit.py` and the metadata/checkpoint preflight first.
2. Confirm `CUDA_VISIBLE_DEVICES=0` and record `nvidia-smi --id=0` output.
3. Re-run the narrow failing family target before the integrated session.
4. Compare the first named failing boundary with its locked fixture; do not
   loosen a global tolerance to hide a local error.
5. For missing tensors, inspect the generated tier index and source symlinks.
   Never synthesize zeros or reuse a different layer.
6. For memory failures, separate resident head/layer-0 memory, staged expert
   banks, and persistent caches. Do not call a partial run full-model proof.

## Historical evidence and full-model gate

Milestones 0–19 retain immutable SITREPs, logs, reports, and hash manifests at
the workspace root. Their runners are workspace-relative. A historical runner
may require the original pinned root uv environment; when it is not installed,
the accepted artifact manifest is the reproducibility authority.

Milestone 21 starts only after text shards 1 through 94 are present and their
hashes pass. Vision/projector shards 95 and 96 are optional for text-only
acceptance. The full gate also requires sufficient distributed NVIDIA topology
and completed deployment performance budgets. With the current one-GPU scope,
metadata, hash, family compilation, and streamed staging can be exercised, but
the distributed 93-layer Moonshot/ZML equivalence and performance gate remains
deferred until the hardware scope expands.
