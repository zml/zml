# Quick Start: Diagnostic Pipeline

## One-command summary

```bash
# Step 1: Capture with diagnostics (30-60 sec)
python3 examples/ltx/replay_stage2_transformer_step.py \
    --pass-label diag --capture-kwargs --capture-inputs \
    --include '^velocity_model\.transformer_blocks\.0\.attn1(\.|$)' \
    --distilled-lora-strength 0.0 --step-idx 0

# Step 2: Export fixture (5-10 sec)
python3 examples/ltx/export_attention_fixture.py \
    trace_run/acts_stage2_transformer_step_000_diag.pt \
    /root/models/ltx-2.3/attn1_diag.safetensors \
    --mode attn1

# Step 3: Generate Python reference (5-10 sec, on GPU server)
python3 examples/ltx/diagnostic_rope_validation.py \
    /root/models/ltx-2.3/attn1_diag.safetensors \
    /root/models/ltx-2.3/attn1_reference.safetensors \
    --attn-name attn1 --num-heads 32 --token-limit 256
```

## What each step produces:

| Step | Output | Purpose |
|------|--------|---------|
| 1 | `acts_stage2_transformer_step_000_diag.pt` | Captured q, k, v, pe, mask from LTX forward |
| 2 | `attn1_diag.safetensors` | Exported fixture with diagnostics (q/k/v post-proj, RoPE params) |
| 3 | `attn1_reference.safetensors` | Python reference: q_rotated, k_rotated (post-rope ground truth) |

## Key files modified:

- **replay_stage2_transformer_step.py**: Added hooks for to_q, to_k, to_v outputs
- **export_attention_fixture.py**: Exports captured q, k, v as `to_q_diag0`, `to_k_diag0`, `to_v_diag0`
- **diagnostic_rope_validation.py** (NEW): Python reference RoPE validator
- **DIAGNOSTIC_PIPELINE.md**: Full documentation

## Next: Compare in Zig

Once you have `attn1_reference.safetensors`, load it in `attention_forward_check.zig` to:
1. Compare ZML's q_head_split vs. reference (validates head-split reshape)
2. Compare ZML's q_rotated vs. reference (validates RoPE application)
3. Compare attention output vs. reference (validates SDPA)

This isolates which operation diverges from LTX.
