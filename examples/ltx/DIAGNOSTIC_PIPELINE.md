# Diagnostic Pipeline: Intermediate Tensor Validation

This guide explains how to validate RoPE (rotary positional embedding) application and intermediate tensor values between LTX Python and ZML Zig implementations.

## Overview

The diagnostic pipeline captures and compares intermediate tensors at three stages:

1. **q, k, v (post-projection, pre-rope)**: Capture raw projection outputs to verify linear transformations
2. **q_rotated, k_rotated (post-rope)**: Apply RoPE in Python reference implementation and compare
3. **Attention output**: Final comparison at attention module output

## Pipeline Steps

### Step 1: Replay with Diagnostic Capture

Run the replay script with flags to capture intermediate tensors:

```bash
python3 examples/ltx/replay_stage2_transformer_step.py \
    --pass-label diagnostic_full \
    --capture-kwargs \
    --capture-inputs \
    --include '^velocity_model\.transformer_blocks\.0\.attn1(\.|$)' \
    --distilled-lora-strength 0.0 \
    --step-idx 0
```

This will:
- Capture pe, k_pe tuples via kwargs hooks
- Capture to_q, to_k, to_v outputs via forward hooks
- Save to `trace_run/acts_stage2_transformer_step_000_diagnostic_full.pt`

### Step 2: Export Fixture with Diagnostics

Export the captured activations to safetensors:

```bash
python3 examples/ltx/export_attention_fixture.py \
    trace_run/acts_stage2_transformer_step_000_diagnostic_full.pt \
    trace_run/attn1_fixture_diagnostic_full.safetensors \
    --mode attn1
```

Expected output includes:
- `attn1.input0`: [B, T, D] input to attention
- `attn1.output0`: [B, T, D] output from to_out
- `attn1.pe_cos0`, `attn1.pe_sin0`: Rotary embedding factors
- `attn1.to_q_diag0`, `attn1.to_k_diag0`, `attn1.to_v_diag0`: **NEW** post-projection intermediates

### Step 3: Validate RoPE with Python Reference

Run the diagnostic validation script to compute reference post-rope values:

```bash
python3 examples/ltx/diagnostic_rope_validation.py \
    trace_run/attn1_fixture_diagnostic_full.safetensors \
    trace_run/attn1_diagnostics_full.safetensors \
    --attn-name attn1 \
    --num-heads 32 \
    --token-limit 256
```

This will:
- Load q, k, v, pe_cos, pe_sin from fixture
- Apply head-split: [B, T, D] -> [B, H, T, HD]
- Detect RoPE layout (interleaved vs split)
- Apply RoPE: produce q_rotated, k_rotated
- Save reference values to output file

Output file contains:
- `attn1.q_head_split`: [B, H, T, HD] after head-split
- `attn1.k_head_split`: [B, H, T, HD] after head-split
- `attn1.v_head_split`: [B, H, T, HD] after head-split
- `attn1.q_rotated`: [B, H, T, HD] after RoPE (reference)
- `attn1.k_rotated`: [B, H, T, HD] after RoPE (reference)

### Step 4: Compare ZML Against Reference (Next Phase)

- Zig checker loads diagnostic reference values
- Compute same operations in ZML
- Compare at each stage to isolate divergence source

## Checkpoint: What Gets Validated?

After step 3, you have:
- ✅ **q, k, v validity**: Compare raw projection outputs in LTX vs. loaded in fixture
- ✅ **Head-split correctness**: Compare reshape/transpose logic
- ✅ **RoPE application**: Compare interleaved/split rotation in Python reference
- ✅ **Layout detection**: Verify correct rope_layout inference (interleaved vs split)

## Token Limiting for Development

Use `--token-limit 256` (or 512, 1024) to reduce memory and compilation overhead while testing:
- Replay: no impact (captures full)
- Export: exports full, but subsequent scripts can limit
- Diagnostic validation: slices to limit for comparison
- Zig checker: compiles reduced graph

## Troubleshooting

**Missing intermediates in fixture?**
- Ensure `--capture-kwargs` and `--include` regex both specified in replay
- Verify repo includes attn1 module name

**Rope layout detected incorrectly?**
- Check pe_cos final dimension matches rotary factor size
- LTX split rope uses hd=64 for heads_dim=128 (so 2x multiplier)

**Parity still poor after RoPE validation?**
- If q_rotated/k_rotated match reference but attention output doesn't match:
  - Problem is in SDPA (scaled dot-product attention) math
  - Check mask handling, softmax normalization, output scaling
- If q_rotated/k_rotated don't match:
  - Problem is in head-split or RoPE application
  - Debug individual rotation formula

## Files Generated

```
trace_run/
├── acts_stage2_transformer_step_000_diagnostic_full.pt    # Replay captures
├── attn1_fixture_diagnostic_full.safetensors               # Exported fixture
└── attn1_diagnostics_full.safetensors                      # Reference values (post-rope)
```

## Next: Zig Integration

Once Python reference validates correctly, the Zig checker can:
1. Load reference values from diagnostics file
2. Compute same operations using ZML tensor library
3. Report parity metrics at each stage (pre-rope, post-rope, post-sdpa)
4. Identify exact divergence point

This isolates the problem to a specific operation rather than looking at final attention output alone.
