# Kimi K3 optimized KDA and MLA comparison

This is the Milestone 18 source and semantic map. The immutable Moonshot
revision is `c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721`, recorded in
`docs/kimi_k3/revisions.lock.json`. All official paths below are relative to
the existing local checkpoint `/dev/shm/kimi-k3/moonshot/kimi-k3`; no model
file is downloaded by the reproduction runner.

## KDA map

Moonshot's `KimiKDA.forward` is in `modeling_kimi_linear.py:543-650`. It
selects `fused_recurrent` for cached single-token execution at lines 559-563,
projects and applies the three short convolutions at lines 580-600, constructs
channel-wise decay and beta inputs at lines 601-607, calls `chunk_kda` for
prefill at lines 609-627 or `fused_recurrent_kda` for decode at lines 628-645,
and stores convolution/recurrent state at lines 646-649.

| Semantic boundary | Moonshot | ZML production | ZML reference/test |
| --- | --- | --- | --- |
| Mode selection | `chunk_kda` / `fused_recurrent_kda` | fused `recurrentOptimized` for prefill and decode | `recurrentReference` sequential StableHLO scan |
| Convolution state | three `ShortConvolution` caches | `convSequence` / `convStep`, three BF16 `[b,channel,4]` windows | same cache layout and official M8 fixture |
| Q/K normalization | `use_qk_l2norm_in_kernel=True` | explicit FP32 `normalizeL2` before recurrence | identical inputs to both recurrence paths |
| Channel decay | `g`, `A_log`, `dt_bias`, safe lower bound | FP32 `alpha[b,s,h,k]` applied independently per K channel | NumPy and sequential ZML use the same equation |
| State | transposed layout requested by `transpose_state_layout=True` | FP32 `[b,h,v,k]`, updated in-place by the Triton kernel | readable `[b,h,v,k]` scan state |
| Update | FLA KDA recurrence | `S = alpha*S + beta*(v-S@k) outer k`; output `S@q/sqrt(k)` | the same equation, one StableHLO loop iteration per token |
| Production result | output and final cache | compact output/cache; fused recurrence is default | diagnostic and reference functions remain public to tests |

The fused CUDA kernel is `zml/attention/triton_kernels/kda_recurrent.zig`.
One program owns a value-row tile for one batch/head pair, retains FP32 state
across the statically sized sequence loop, and writes both output and final
state. This removes the per-token StableHLO loop/launch structure without
changing the recurrence ordering. `BLOCK_V=32` is retained; measured 8 and 16
variants were slower.

Validation is deliberately three-way: deterministic NumPy fixture,
`recurrentReference`, and `recurrentOptimized`. Cases cover 1, 3, 4, 5, 31,
32, 33, 63, 64, 65, and 257 tokens plus production H96/K128/V128 decode and
prefill64. Fixture states are non-zero. The independent official M8 full-KDA
fixture additionally covers full 1/4/8/16, token decode, 15 split points, and
continuation.

## MLA map

Moonshot's readable `KimiMLAAttention.forward` is in
`modeling_kimi_linear.py:405-472`. It forms 128 non-rotary plus 64 extra query
dimensions at lines 418-424, forms a 512 compressed projection plus 64 extra
key dimensions at lines 426-435, expands per-head key/value tensors at lines
430-440, and sends expanded K/V through the configured attention interface at
lines 442-463. That implementation is the correctness oracle, not the ZML
production cache representation.

| Semantic boundary | Moonshot/readable oracle | ZML production | ZML reference/test |
| --- | --- | --- | --- |
| Temporal cache | expanded H96 K192 + V128 BF16 | normalized compressed 512 + extra key 64 BF16 | `ExpandedCache` retains the Moonshot layout |
| Values/token/layer | 30,720 values / 61,440 bytes | 576 values / 1,152 bytes | expanded and latent paths compared in M13 |
| Content score | `q_pass @ key_up(cache)` | absorb `key_up` into q, then `q_absorbed @ compressed_cache` | explicit expanded score |
| Extra score | 64-D query/key dot | identical 64-D query/key dot | identical |
| Readout | probabilities times expanded value | probabilities times compressed cache, then `value_up` | expanded value aggregation |
| Cache update | append expanded K/V | exact-length concatenate or fixed-capacity `dynamicUpdateSlice` with buffer reuse | split/decode/session cache harness |
| Production result | output and cache | `CompactResult` contains output and latent cache only | `LatentResult` retains probabilities and absorbed/readout diagnostics |

The key algebra is `(q Wk) C^T = q (C Wk^T)^T` for scoring and
`(P C) Wv^T = P (C Wv^T)` for readout. ZML changes the association, not the
mathematics. The fixed-capacity session mask excludes unwritten slots, and the
cache allocation shape remains invariant across decode positions.

`latentAttentionStableHlo` isolates the production absorbed score, FP32
softmax, and latent readout. Its CUDA fixture covers page/tile boundaries
31/32/33, 63/64/65, 127/128, partial valid capacity, and a finite 4096-token
context. H100 synchronized-mean ceilings are 750 microseconds at 64 and 900
microseconds at 4096.

Two custom Triton experiments were rejected because they were slower than
XLA StableHLO while numerically correct: scalar reductions measured 474 vs
443 microseconds at 64 and 1180 vs 525 at 4096; tensor-core dots measured 554
vs 521 at 64 and 782 vs 499 at 4096. Rejected code is not retained.

## Cache and performance invariants

- KDA reference and optimized paths use the same 6,586,368-byte batch-1 cache:
  6,291,456 bytes of FP32 recurrent state plus 294,912 bytes of BF16
  convolution windows. Optimization changes execution, not persistent state.
- Latent MLA reduces cache bytes by 53.333x and never constructs expanded K/V
  in a production function.
- KDA decode and prefill synchronized means may not exceed 105% of the
  sequential reference. The official full-KDA fixture remains the end-to-end
  semantic check.
- MLA stage ceilings are regression alarms, not claims that diagnostic whole
  layer timings are production benchmarks.
- CUDA is required explicitly. There is no CPU inference fallback.

## Reproduce

From `/dev/shm/kimi-k3`:

```bash
./scripts/reproduce.sh 18
```

The runner uses `/dev/shm/kimi-k3/.venv`, forces offline Hugging Face mode,
builds with CUDA enabled and CPU disabled, verifies the five locally present
checkpoint shards without downloading the other 91, executes all prior and
new KDA/MLA gates, creates both family Perfetto traces, and writes
`artifacts/reports/milestone-18-comparisons.json`.

Temporary activation samples, synchronized diagnostic timing, and family
profiling spans are marked `KIMI_K3_TEMP_REMOVE_M20` in source and are removed
or replaced by permanent instrumentation during cleanup.
