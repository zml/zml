# Transformer Threading Progress (Post-Block0)

This file tracks progress after Block-0 M1-M6 parity was completed.

## Why this file exists

- `block0_reverse_engineering_map.md` is now the historical record for Block-0 reverse engineering and parity bring-up.
- This file continues the same stagewise tracking style for model-level threading and multi-block validation.
- Goal: keep chronology, decisions, and validation evidence easy to audit.

## Scope

Track work related to:

1. Threading `LTXModel.forward` from legacy FF-boundary path to full AV block semantics.
2. Validating inline AdaLN / AV cross-attn computations against Python reference.
3. Extending parity from block 0 to block slices and then full stack.
4. Consolidating or retiring legacy checker paths once full path is stable.

## Milestones

### T0 - Baseline snapshot

- Status: DONE
- Notes:
  - Block-0 canonical full forward (`BasicAVTransformerBlock.forward`) integrated.
  - Existing validation matrix (token 128/256/512 x LoRA 0.0/0.5) passed after fixture refresh.

### T1 - Native block threading API in Zig

- Status: DONE
- Target:
  - `BasicAVTransformerBlock.forwardNative` computes AdaLN values inline from scale-shift tables.
  - `LTXModel.forwardNative` threads both streams across all blocks.
- Exit criteria:
  - Compiles cleanly. ✓
  - Block-0 native checker parity against Python fixture passes. ✓

### T2 - Block-0 native parity checker

- Status: DONE
- Target:
  - Add fixture export + checker that validates `forwardNative` end-to-end for one block.
- Exit criteria:
  - PASS at token 128/256/512 for LoRA 0.0/0.5. ✓

### T3 - Multi-block slice parity

- Status: TODO
- Target:
  - Validate contiguous block slices (for example 0-7, 8-15) with replay traces.
- Exit criteria:
  - Slice checks pass with stable tolerances.

### T4 - Full transformer parity

- Status: TODO
- Target:
  - Validate complete stage-2 transformer pass using threaded native path.
- Exit criteria:
  - Full-pass outputs match reference within agreed tolerances.

### T5 - Cleanup and consolidation

- Status: TODO
- Target:
  - Remove or demote legacy FF-boundary-only entrypoints/checkers when no longer needed.
- Exit criteria:
  - Docs and targets reflect one canonical path.

## Validation Log

Use one entry per executed check (append chronologically).

| Date | Milestone | Command/Target | Fixture/Checkpoint | Result | Notes |
|---|---|---|---|---|---|
| 2026-03-23 | T1 | `bazel build //examples/ltx:block0_full_check //examples/ltx:block0_ff_boundary_check` | local compile only | PASS | `forwardNative` scaffolding compiles; runtime parity for native path pending dedicated checker. |
| 2026-03-23 | T2 | `bazel run //examples/ltx:block0_native_check --@zml//platforms:cuda=true -- <ckpt> <fixture>` | `stage2_block0_lora0.0_merged.safetensors` + `block0_native_lora0.0_t128.safetensors` | PASS | Native video parity PASSED; native audio parity PASSED; full native block0 parity PASSED for token_limit=128, LoRA=0.0. |
| 2026-03-23 | T2 | `run_native_t2_matrix.sh` (LoRA 0.0, t256) | `stage2_block0_lora0.0_merged.safetensors` + `block0_native_lora0.0_t256.safetensors` | PASS | Matrix combo lora0.0_t256 parity PASSED. |
| 2026-03-23 | T2 | `run_native_t2_matrix.sh` (LoRA 0.0, t512) | `stage2_block0_lora0.0_merged.safetensors` + `block0_native_lora0.0_t512.safetensors` | PASS | Matrix combo lora0.0_t512 parity PASSED. |
| 2026-03-23 | T2 | `run_native_t2_matrix.sh` (LoRA 0.5, t128) | `stage2_block0_lora0.5_merged.safetensors` + `block0_native_lora0.5_t128.safetensors` | PASS | Matrix combo lora0.5_t128 parity PASSED. Fixed: `adaValueAt` now casts SST to timestep dtype (`.convert(ts.dtype())`) — iso with Python `get_ada_values`. |
| 2026-03-23 | T2 | `run_native_t2_matrix.sh` (LoRA 0.5, t256) | `stage2_block0_lora0.5_merged.safetensors` + `block0_native_lora0.5_t256.safetensors` | PASS | Matrix combo lora0.5_t256 parity PASSED. |
| 2026-03-23 | T2 | `run_native_t2_matrix.sh` (LoRA 0.5, t512) | `stage2_block0_lora0.5_merged.safetensors` + `block0_native_lora0.5_t512.safetensors` | PASS | Matrix combo lora0.5_t512 parity PASSED. All 6 combos (t128/256/512 × LoRA 0.0/0.5) now PASS — T2 milestone complete. |

## Technical Notes

### SST dtype casting in `adaValueAt`

When merging LoRA weights, the SST (`scale_shift_table`) tensors may remain in `f32` in the checkpoint, while activation embeddings (`timestep`) are always `bf16`. Python's `get_ada_values` defensively casts SST to the activation dtype:

```python
scale_shift_table[indices].to(device=timestep.device, dtype=timestep.dtype) + timestep[...]
```

The Zig implementation mirrors this with `.convert(ts.dtype())` to ensure dtype compatibility before the add operation, matching the Python reference exactly. This is not just a workaround — it's the canonical pattern used elsewhere in the model for activation-dtype promotion.

## Open Questions

1. Should perturbation masks remain out-of-scope for first native checker (assume identity masks), then added in a separate milestone?

## Change Log

- 2026-03-23: Initialized post-block0 tracking file and milestone framework.
- 2026-03-23: Added native checker/export/capture wiring for cross-attention AdaLN (rows 6..8 + prompt modulation) and recorded first runtime PASS (t128, LoRA 0.0).
- 2026-03-23: Fixed `adaValueAt` dtype mismatch: SST is now explicitly cast to timestep dtype via `.convert(ts.dtype())` (iso with Python `get_ada_values` .to(dtype=timestep.dtype)). Deployed `run_native_t2_matrix.sh` batch script with checkpoint validation. T2 matrix validation complete: all 6 combos (LoRA 0.0/0.5 × t128/256/512) PASS.
