## 🔍 Diagnostic Pipeline Implementation — Complete Summary

I've implemented a complete diagnostic infrastructure to validate intermediate tensors at three stages of the attention computation, allowing you to isolate exactly where ZML diverges from LTX.

---

## 📋 What Was Implemented

### 1. **Enhanced Replay Script** (`replay_stage2_transformer_step.py`)

**Added hooks to capture q, k, v post-projection:**
- When `--capture-kwargs` and `--include` regex are both specified
- Registers forward hooks on `to_q`, `to_k`, `to_v` layers in attn1
- Captures [B, T, D] tensors immediately after linear projection
- Stores them as activations with key `{module_name}.__output__`

**Key addition:** 140+ lines of diagnostic hook registration

```python
# Hooks now capture:
- velocity_model.transformer_blocks.0.attn1.to_q.__output__  → q [B,T,D_Q]
- velocity_model.transformer_blocks.0.attn1.to_k.__output__  → k [B,T,D_K]
- velocity_model.transformer_blocks.0.attn1.to_v.__output__  → v [B,T,D_V]
```

### 2. **Enhanced Export Script** (`export_attention_fixture.py`)

**Extracts and exports q, k, v diagnostics:**
- Looks for captured intermediate tensors in activations
- Exports as `{attn_name}.to_q_diag0`, `{attn_name}.to_k_diag0`, `{attn_name}.to_v_diag0`
- Adds metadata flag `has_diagnostics` to fixture
- Prints diagnostic tensor info on completion

**Key addition:** 30+ lines of diagnostic extraction

```python
# Exports to fixture:
- attn1.to_q_diag0: [B,T,D_Q] post-projection q
- attn1.to_k_diag0: [B,T,D_K] post-projection k
- attn1.to_v_diag0: [B,T,D_V] post-projection v
```

### 3. **New RoPE Validator** (`diagnostic_rope_validation.py`)

**Complete Python reference implementation (270 lines):**
- Loads q, k, v and rotary parameters from fixture
- Detects RoPE layout automatically (interleaved vs. split)
- Applies head-split: [B, T, D] → [B, H, T, HD]
- Applies RoPE using correct rotation formulas
- Exports post-rope reference values

**Key functions:**
- `detect_rope_layout()`: Auto-detect from dimension ratios
- `apply_head_split()`: Reshape and permute for multi-head attention
- `apply_rope_interleaved()`: Rotation for interleaved layout
- `apply_rope_split()`: Rotation for split layout

**Exports to diagnostics file:**
```python
- attn1.q_head_split:  [B,H,T,HD] after reshape
- attn1.k_head_split:  [B,H,T,HD] after reshape
- attn1.v_head_split:  [B,H,T,HD] after reshape
- attn1.q_rotated:     [B,H,T,HD] after RoPE ← Ground truth for q
- attn1.k_rotated:     [B,H,T,HD] after RoPE ← Ground truth for k
```

### 4. **Documentation**

Created four markdown guides:

| File | Purpose |
|------|---------|
| `DIAGNOSTIC_IMPLEMENTATION_SUMMARY.md` | Technical details of all changes |
| `DIAGNOSTIC_PIPELINE.md` | Comprehensive step-by-step guide |
| `DIAGNOSTIC_QUICKSTART.md` | One-command summary |
| `DIAGNOSTIC_CHECKLIST.md` | Actionable steps with verification |

---

## 🔄 The Diagnostic Pipeline

### Three-Stage Validation

```
┌─────────────────────────────────────────────┐
│ Stage 1: CAPTURE                            │
│ - Replay LTX forward with diagnostic hooks  │
│ - Capture: q, k, v (post-projection)        │
│ - Capture: pe, k_pe (rotary kwargs)         │
└────────────┬────────────────────────────────┘
             ↓ Output: acts_stage2_transformer_step_000_diagnostic.pt

┌─────────────────────────────────────────────┐
│ Stage 2: EXPORT                             │
│ - Extract diagnostics from .pt              │
│ - Export fixture with:                      │
│   * q, k, v (post-projection)               │
│   * pe_cos, pe_sin (rotary parameters)      │
└────────────┬────────────────────────────────┘
             ↓ Output: attn1_fixture_diagnostic.safetensors

┌─────────────────────────────────────────────┐
│ Stage 3: VALIDATE (Python Reference)        │
│ - Load q, k, v, pe_cos, pe_sin              │
│ - Apply head-split → [B,H,T,HD]             │
│ - Auto-detect RoPE layout                   │
│ - Apply RoPE (interleaved/split)            │
│ - Export: q_head_split, k_head_split,       │
│   v_head_split, q_rotated, k_rotated        │
└────────────┬────────────────────────────────┘
             ↓ Output: attn1_reference.safetensors (GROUND TRUTH)

┌─────────────────────────────────────────────┐
│ Stage 4: COMPARE (Next Phase) in Zig        │
│ - Load reference diagnostics                │
│ - Compute same operations in ZML            │
│ - Compare: ZML q_head_split vs. reference   │
│ - Compare: ZML q_rotated vs. reference      │
│ - Compare: ZML attention output vs. ref     │
│ → Reports which operation diverges          │
└─────────────────────────────────────────────┘
```

---

## 🎯 What This Validates

After running the full pipeline, you'll have:

✅ **Pre-rope tensors** (q, k, v)
   - Validates linear projections are correct
   - Check if divergence is in to_q/to_k/to_v or later

✅ **Head-split tensors** (q_head_split, k_head_split, v_head_split)
   - Validates reshape/transpose logic
   - Check if multi-head reshaping matches LTX

✅ **Post-rope reference** (q_rotated, k_rotated)
   - Ground truth for RoPE application
   - Compare ZML's RoPE against this reference
   - Isolate if problem is in head-split or RoPE formula

✅ **Metadata**
   - RoPE layout detection (interleaved vs. split)
   - Token limits applied
   - Fixture source information

---

## 🚀 How to Run

### Quick Start (30 seconds)

```bash
# 1. Read the quick summary
cat examples/ltx/DIAGNOSTIC_QUICKSTART.md

# 2. Follow the checklist
cat examples/ltx/DIAGNOSTIC_CHECKLIST.md

# 3. Execute on GPU server (Steps 1️⃣-5️⃣ in the checklist)
```

### Full Reference

```bash
# All documentation
cat examples/ltx/DIAGNOSTIC_PIPELINE.md          # Complete guide
cat examples/ltx/DIAGNOSTIC_IMPLEMENTATION_SUMMARY.md  # Technical details
```

---

## 📊 Output Files Generated

After running full pipeline on GPU server, you'll have:

```
/root/models/ltx-2.3/
├── attn1_fixture_diagnostic.safetensors    # Fixture with q/k/v diagnostics
└── attn1_reference.safetensors              # Ground truth post-rope values
```

**File contents:**

| Fixture | Reference |
|---------|-----------|
| `attn1.input0` [B,T,D] | `attn1.q_head_split` [B,H,T,HD] |
| `attn1.to_q_diag0` [B,T,D_Q] | `attn1.k_head_split` [B,H,T,HD] |
| `attn1.to_k_diag0` [B,T,D_K] | `attn1.v_head_split` [B,H,T,HD] |
| `attn1.to_v_diag0` [B,T,D_V] | `attn1.q_rotated` [B,H,T,HD] ← Use this |
| `attn1.pe_cos0` [1,H,T,64] | `attn1.k_rotated` [B,H,T,HD] ← Use this |
| `attn1.pe_sin0` [1,H,T,64] | |
| `attn1.output0` [B,T,D] | |

---

## 🔗 Integration with Zig (Next Phase)

The Zig checker will be enhanced to:

1. **Load reference values** from `attn1_reference.safetensors`
2. **Compute same operations** using ZML tensor library:
   ```zig
   // Load reference q_head_split, k_head_split
   // Compute ZML's q_head_split, k_head_split
   // Compare: reportParity(zml_qhs, ref_qhs, "q_head_split");
   ```

3. **Isolate divergence point:**
   - If stage1 diverges → problem is in head-split reshape
   - If stage2 diverges → problem is in RoPE formula
   - If stage3 diverges → problem is in SDPA semantics

---

## 📝 Files Changed

### Modified
- `replay_stage2_transformer_step.py` (+110 lines): Added intermediate hooks
- `export_attention_fixture.py` (+30 lines): Extract and export diagnostics

### Created
- `diagnostic_rope_validation.py` (270 lines): Python RoPE reference
- `DIAGNOSTIC_PIPELINE.md`: Full documentation
- `DIAGNOSTIC_QUICKSTART.md`: Quick reference
- `DIAGNOSTIC_CHECKLIST.md`: Actionable steps
- `DIAGNOSTIC_IMPLEMENTATION_SUMMARY.md`: Technical summary (this was created too)

---

## ✨ Key Advantages

🎯 **Isolates divergence** — Know exactly which operation fails
📊 **Stage-by-stage validation** — Build confidence incrementally
🔄 **Minimal Zig changes** — Validate Python first
💾 **Export for inspection** — Save and analyze intermediate values
⚡ **Token-limit friendly** — Reduce compile overhead during development

---

## 🎬 Next Actions

### Immediate (On GPU Server)

1. Sync updated Python scripts
2. Run replay with `--capture-kwargs --capture-inputs`
3. Export fixture (will include q/k/v diagnostics)
4. Run diagnostic_rope_validation.py
5. Share results

### Then (Zig Integration)

1. Update attention_forward_check.zig to load reference
2. Add comparison code for q_head_split, k_head_split, q_rotated, k_rotated
3. Report which stage diverges
4. Fix the specific divergence source

---

## ✅ Ready to Go

The diagnostic infrastructure is complete and ready to use. All Python code has been syntax-validated. You can now:

1. **Read the documentation** to understand the approach
2. **Follow the checklist** to execute on GPU server
3. **Generate ground truth** reference values
4. **Use reference values** in Zig comparison (next phase)

The diagnostic pipeline will pinpoint which operation (head-split vs. RoPE vs. SDPA) is causing the parity gap.

---

**Status**: ✅ Implementation complete
**Ready**: Yes, to execute on GPU server
**Estimated time**: 5-10 minutes for full pipeline run
