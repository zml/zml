# Kimi K3 full-model and distributed readiness

Milestone 19 validates the complete logical text model without opening
unavailable checkpoint payloads. The authoritative offline preflight is:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/kimi_k3/full_model_preflight.py \
  --model=/path/to/kimi-k3 \
  --inventory=/path/to/tensor-name-inventory.jsonl.gz \
  --hash-manifest=/path/to/checkpoint-sha256.json \
  --scenario-devices=1 \
  --output=/path/to/report.json \
  --quiet
```

Add `--verify-present-hashes --require-complete` only for the Milestone 21
full-checkpoint gate. The preflight never downloads a shard.

## Frozen ownership contract

The full official index contains 497,220 names across 96 shards. Exactly
497,052 names belong to the text model and 168 vision/projector names are
explicitly ignored. The logical text schedule is:

- one KDA+dense layer;
- 68 KDA+MoE layers;
- 24 MLA+MoE layers;
- 69 packed KDA caches and 24 packed MLA caches;
- 92 complete 896-expert banks with six packed/scale tensors per expert;
- eight Attention Residual source slots and a final output Attention Residual.

Every layer is validated against its exact non-expert suffix set. Every MoE
component must contain global expert IDs 0 through 895 exactly once.

## Bounded loading

The head and layer 0 remain resident. Layers 1 through 92 are staged in logical
order and released after execution. Each packed expert component is read one
expert tensor at a time into a bounded host buffer and transferred as a single
typed bank; no dequantized full-model host copy is created.

For the available representative layers, the largest staged layer is
16,990,092,800 bytes. The maximum host expert-component staging allocation is
4,932,501,504 bytes, the packed six-component expert bank is 15,722,348,544
bytes before sharding, and the resident head plus layer 0 is 7,038,876,672
bytes.

The runtime loader has separate `model_sharding` and `expert_sharding`
channels. Dense/attention tensors use the tensor-parallel model sharding while
expert banks use the expert sharding. `DistributedPlan` validates a
tensor-parallel degree against every contracted model dimension and assigns all
896 experts to contiguous, balanced expert-parallel ranges without gaps.

## Cache and storage estimates

The official index reports 1,560,860,324,864 bytes for the complete
multimodal checkpoint. This is used as a conservative upper bound for text
storage because unavailable shard headers cannot be opened to subtract the
ignored vision payload.

At batch 1, all KDA caches require 454,459,392 bytes. A one-million-token
latent MLA cache requires 27,648,000,000 bytes, for a combined persistent cache
of 28,102,459,392 bytes. Attention Residual workspace is forward-local and
contributes no persistent cache bytes.

The report includes 1/8/16/24/32-device storage, streaming, cache, tensor
parallel, and expert partition scenarios, plus conservative per-token
communication bounds.

## Current execution scope

All CUDA compilation and execution is pinned to physical GPU 0 with
`CUDA_VISIBLE_DEVICES=0`. The 93-layer family set compiles on that device, and
the available layers prove sequential substitution of layers 1 and 2 through
the same KDA+MoE family shape. Physical multi-device equivalence is deferred by
the current single-GPU scope; distributed ownership is validated structurally.

The live preflight records the current shard count while user-supplied files may
still be arriving. Full text validation starts only after all required shards 1
through 94 are present and hashed locally. Shards 95 and 96 contain the ignored
projector/vision payload and are optional for the text-only gate.
