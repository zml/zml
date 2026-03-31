## Diagnostic Pipeline Implementation Summary

I've implemented a complete diagnostic infrastructure to validate intermediate tensors between LTX Python and ZML Zig at three critical stages. Here's what was added:

### **Modified Files**

#### 1. **replay_stage2_transformer_step.py**
**Changes:**
- Added `captured_intermediates` dict to store q, k, v outputs from projection layers
- Registered forward hooks on `to_q`, `to_k`, `to_v` modules (when inside attn1 and using --capture-kwargs)
- Hooks capture `.to_q.__output__`, `.to_k.__output__`, `.to_v.__output__` tensors
- These intermediates are merged into activations before saving

**Key code pattern:**
```python
# Hooks registered for to_q, to_k, to_v in attn1
# They capture [B, T, D] post-projection tensors
captured_intermediates[f"{name}.__output__"] = output.detach().cpu().contiguous()
```

#### 2. **export_attention_fixture.py**
**Changes:**
- Extract q, k, v intermediates from captured activations
- Export as `{attn_name}.to_q_diag0`, `{attn_name}.to_k_diag0`, `{attn_name}.to_v_diag0` in fixture
- Save metadata flag `has_diagnostics` to indicate presence
- Print diagnostic tensor info on completion

**Exported tensors:**
- `attn1.to_q_diag0`: [B, T, D_Q] post-projection q
- `attn1.to_k_diag0`: [B, T, D_K] post-projection k
- `attn1.to_v_diag0`: [B, T, D_V] post-projection v

#### 3. **diagnostic_rope_validation.py** (NEW FILE)
**Purpose:** Python reference implementation that:
- Loads q, k, v, pe_cos, pe_sin from fixture
- Detects RoPE layout (interleaved vs split) by comparing dimensions
- Applies head-split: [B, T, D] → [B, H, T, HD]
- Applies RoPE using correct rotation formula
- Exports post-rope reference values for Zig comparison

**Key functions:**
- `apply_head_split()`: [B, T, D] → [B, H, T, HD] reshape with permute
- `apply_rope_interleaved()`: x * cos + rotate(x) * sin (interleaved pairs)
- `apply_rope_split()`: split-format rotation (cos/sin are half dimension)
- `detect_rope_layout()`: auto-detect based on hd_rope vs hd_qkv

**Exported diagnostics:**
- `attn1.q_head_split`: [B, H, T, HD] after head-split
- `attn1.k_head_split`: [B, H, T, HD] after head-split
- `attn1.v_head_split`: [B, H, T, HD] after head-split
- `attn1.q_rotated`: [B, H, T, HD] after RoPE (ground truth)
- `attn1.k_rotated`: [B, H, T, HD] after RoPE (ground truth)

### **New Documentation Files**

#### 1. **DIAGNOSTIC_PIPELINE.md**
- Complete step-by-step guide for running diagnostic pipeline
- Explains each stage: capture → export → validate
- Troubleshooting section
- Integration with Zig checker (coming next phase)

#### 2. **DIAGNOSTIC_QUICKSTART.md**
- One-command summary of all three steps
- Output file summary table
- Key files modified list
- Next steps for Zig integration

### **Validation Stages**

The pipeline isolates divergence by comparing at three levels:

```
Stage 1: q, k, v (post-projection, pre-rope)
         ↓ Validate linear transformation correctness

Stage 2: q_head_split, k_head_split, q_rotated, k_rotated
         ↓ Validate head-split reshape and RoPE application

Stage 3: Attention output (post-SDPA, pre-to_out)
         ↓ Validate scaled dot-product attention semantics
```

### **Usage on GPU Server**

```bash
# After syncing updated .py files and running replay/export on local:
# (or run replay+export directly on GPU server)

cd /root/repos/LTX-2

# Generate Python reference values
python3 examples/ltx/diagnostic_rope_validation.py \
    /root/models/ltx-2.3/attn1_fixture_base_nolora.safetensors \
    /root/models/ltx-2.3/attn1_diagnostics.safetensors \
    --attn-name attn1 --num-heads 32 --token-limit 256

# Output: diagnostics with q_rotated, k_rotated (ground truth for comparison)
```

### **Next Phase: Zig Integration**

The Zig checker (`attention_forward_check.zig`) will be enhanced to:
1. Load reference diagnostics from `.safetensors`
2. Compute q_head_split, k_head_split using ZML tensor ops
3. Compute q_rotated, k_rotated by applying RoPE using ZML helpers
4. Compare each stage independently:
   - *Stage 1 divergence?: Head-split reshape issue*
   - *Stage 2 divergence?: RoPE application bug*
   - *Stage 3 divergence?: SDPA semantics mismatch*

### **Benefits of This Approach**

✅ **Isolates problems**: Know exactly which operation diverges
✅ **Validates incrementally**: Build confidence stage-by-stage
✅ **Minimal Zig changes**: Can validate Python correctness first
✅ **Debugging friendly**: Export intermediate values for inspection
✅ **Token-limit compatible**: Reduces compile memory for development

### **What to Do Now**

On the GPU server:

1. **Sync updated Python scripts:**
   ```bash
   rsync -av examples/ltx/replay_stage2_transformer_step.py root@dev-oboulant:/root/repos/LTX-2/scripts/
   rsync -av examples/ltx/export_attention_fixture.py root@dev-oboulant:/root/repos/LTX-2/scripts/
   scp examples/ltx/diagnostic_rope_validation.py root@dev-oboulant:/root/repos/LTX-2/scripts/
   ```

2. **Run diagnostic pipeline:**
   ```bash
   # Re-capture with new hooks
   python3 /root/repos/LTX-2/scripts/replay_stage2_transformer_step.py \
       --pass-label diag --capture-kwargs --capture-inputs \
       --include '^velocity_model\.transformer_blocks\.0\.attn1(\.|$)' \
       --distilled-lora-strength 0.0 --step-idx 0

   # Export with diagnostics
   python3 /root/repos/LTX-2/scripts/export_attention_fixture.py \
       trace_run/acts_stage2_transformer_step_000_diag.pt \
       /root/models/ltx-2.3/attn1_diag.safetensors \
       --mode attn1

   # Generate Python reference
   python3 /root/repos/LTX-2/scripts/diagnostic_rope_validation.py \
       /root/models/ltx-2.3/attn1_diag.safetensors \
       /root/models/ltx-2.3/attn1_reference.safetensors \
       --attn-name attn1 --num-heads 32 --token-limit 256
   ```

3. **Verify output files exist:**
   ```bash
   ls -lh /root/models/ltx-2.3/attn1_*.safetensors
   ```

4. **Share results:**
   - Size of diagnostic fixture
   - Metadata from reference file (rope_layout, token_limit)
   - Any errors or warnings

### **Important Notes**

- Intermediate hooks only register if **both** `--capture-kwargs` AND `--include` regex are specified
- The `--include` regex must match attn1 modules (e.g., `'^velocity_model\.transformer_blocks\.0\.attn1(\.|$)'`)
- Rope layout is auto-detected: "interleaved" (rope_hd == qkv_hd) or "split" (rope_hd * 2 == qkv_hd)
- `--token-limit` is applied in diagnostic script, not during capture (capture is always full)

### **Files Created/Modified**

```
Modified:
  examples/ltx/replay_stage2_transformer_step.py        (+110 lines)
  examples/ltx/export_attention_fixture.py              (+30 lines)

Created:
  examples/ltx/diagnostic_rope_validation.py            (270 lines, full RoPE validator)
  examples/ltx/DIAGNOSTIC_PIPELINE.md                   (documentation)
  examples/ltx/DIAGNOSTIC_QUICKSTART.md                 (quick reference)
```
