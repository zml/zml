# Stage 1 — 30-Step Divergence Analysis

## Current Status (March 30, 2026)

### 30-Step Free-Run Results

Step-0 velocity: **PASS** (cos_sim=0.9996 video, 0.9999 audio)
Step-1 latent: **PASS** (cos_sim=0.99994, mae=0.0003)
Step-5 latent: **PASS** (cos_sim=0.99981, mae=0.0099)
Step-15 latent: **FAIL** (cos_sim=0.995, mae=0.059)
Step-30 final: **FAIL** (cos_sim=0.658 video, 0.797 audio)

### Conclusion

The Zig implementation is **numerically correct**. The 30-step divergence is
caused by iterative accumulation of irreducible XLA vs CUDA backend differences,
amplified by the guidance formula. There are no remaining fixable bugs.

---

## 1. Empirical Dtype Trace (trace_dtype_chain.py)

Traced with **f32 denoise_mask** (matching real pipeline — NOT bf16 as initially assumed).

### Key findings:

| Operation | Python dtype chain |
|---|---|
| **Guider combine** | ALL bf16: `pred = cond + 2*(cond−neg) + 1*(cond−ptb) + 2*(cond−iso)`, `std()` in bf16, rescale in bf16 |
| **to_denoised (X0Model)** | `timesteps = mask(f32) * sigma(f32) → f32` (no bf16 bottleneck), arithmetic in f32, output bf16 |
| **post_process_latent** | `denoised(bf16) * mask(f32) → f32` (PyTorch promotes), `clean.float() * (1−mask)(f32) → f32`, sum f32, `.to(bf16)` |
| **to_velocity** | f32 arithmetic → `.to(sample.dtype)` = **bf16 roundtrip** |
| **Euler step** | velocity(bf16).to(f32) * dt(f32) + sample.to(f32) → `.to(bf16)` |

### Critical correction: denoise_mask is f32, not bf16

The initial dtype trace used `torch.ones(...).to(torch.bfloat16)` for masks.
The real pipeline uses **f32** masks (confirmed from `export_stage1_inputs.py` output:
`video_denoise_mask: [1, 6144, 1] torch.float32`).

This changed the analysis for:
- `to_denoised`: `mask(f32) * sigma(f32) → f32` — **no bf16 quantization of timesteps**
- `post_process_latent`: `denoised(bf16) * mask(f32) → f32` — PyTorch promotes to f32

### Zig dtype chain (model.zig) — matches Python:

- `guiderCombineSingle`: all bf16 arithmetic ✓
- `forwardToDenoised`: `denoise_mask.mul(sigma.convert(denoise_mask.dtype()))` → adapts to mask dtype ✓
- `forwardDenoisingStepFromX0`: explicit `mask_f32 = denoise_mask.convert(.f32)`, all f32 arithmetic ✓
- bf16 velocity roundtrip present ✓

### Export dtype validation

Both `export_stage1_inputs.py` and `export_stage1_step0_reference.py` preserve
runtime dtypes faithfully (`.detach().cpu().clone()`, no explicit dtype conversion).
Verified via safetensors inspection:
```
video_latent       torch.bfloat16  [1, 6144, 128]
video_denoise_mask torch.float32   [1, 6144, 1]
video_clean_latent torch.bfloat16  [1, 6144, 128]
v_context_pos      torch.bfloat16  [1, 1024, 4096]
```

---

## 2. Per-Pass Comparison at Step 0

Isolated each sub-component at step 0 to identify the error budget.

### Raw velocities (before to_denoised):

| Pass | Video cos_sim | Video MAE | Audio cos_sim | Audio MAE |
|---|---|---|---|---|
| **Conditional** | 0.999627 | 0.0215 | 0.999877 | 0.0130 |
| **Negative/CFG** | 0.998986 | 0.0353 | 0.999721 | 0.0259 |
| **STG/Perturbed** | 0.999729 | 0.0175 | 0.999776 | 0.0193 |
| **Isolated** | 0.999391 | 0.0276 | 0.999979 | 0.0055 |

### After guider combine (guided x0):

| | Video cos_sim | Video MAE | Audio cos_sim | Audio MAE |
|---|---|---|---|---|
| **Guided x0** | 0.996855 | **0.0599** | 0.997103 | **0.0595** |

### After Euler step (step-1 latent):

| | Video cos_sim | Video MAE |
|---|---|---|
| **Step-1 latent** | 0.999938 | **0.0003** |

### Error budget analysis:

```
  cond          mae=0.0215  ██████████
  neg           mae=0.0353  █████████████████
  ptb           mae=0.0175  ████████
  iso           mae=0.0276  █████████████
  guided_x0     mae=0.0599  █████████████████████████████
  step1_lat     mae=0.0003
```

**Key insights:**
1. All 4 raw velocities are in the same ~0.02–0.035 error band — irreducible
   XLA vs CUDA backend difference across 48 transformer blocks.
2. Guider combine amplifies ~3x — mathematically inevitable with
   `pred = cond + 2*(cond−neg) + 1*(cond−ptb) + 2*(cond−iso)`.
3. Step-1 latent shrinks to 0.0003 because `dt ≈ −0.005` damps the x0 error by ~200x.
4. `forwardToDenoised`, `guiderCombineSingle`, `forwardDenoisingStepFromX0`,
   `post_process_latent` — **all correct**.

---

## 3. Per-Step Reset Test

Fed Python's exact latent into Zig at each step to isolate per-step error
from accumulation.

### Reset test results (Zig starts from Python's latent each step):

| Step | Video cos_sim | Video MAE | |dt| | MAE/|dt| |
|---|---|---|---|---|
| 1 | 0.999938 | 0.000293 | 0.005 | 0.058 |
| 5 | 0.999924 | 0.003472 | 0.006 | 0.536 |
| 10 | 0.999928 | 0.002504 | 0.009 | 0.268 |
| 15 | 0.999912 | 0.003938 | 0.015 | 0.269 |
| 20 | 0.999858 | 0.006883 | 0.026 | 0.264 |
| 25 | 0.999479 | 0.014989 | 0.059 | 0.254 |
| 30 | 0.999596 | 0.018537 | 0.100 | 0.185 |

**Average MAE/|dt| ratio = 0.27** (stable across steps 8–27).

### Interpretation:

The per-step error follows:

$$\text{mae}_{\text{step}} \approx 0.26 \times |dt|$$

This means:
1. The **guided x0 error** (~0.06) is roughly constant regardless of step.
2. The **Euler update** scales it by `|dt|`: `Δlatent = Δvelocity × |dt|`.
3. Since `|dt|` grows from 0.005 (step 1) to 0.166 (step 29) due to the
   LTX-2 sigma schedule, the per-step latent error grows proportionally.
4. This is **not a bug** — it's the schedule geometry.

### Free-run vs reset comparison:

| Step | Free-run MAE (accumulated) | Reset MAE (isolated) | Ratio |
|---|---|---|---|
| 1 | 0.0003 | 0.0003 | 1.0x |
| 5 | 0.0099 | 0.0035 | 2.9x |
| 15 | 0.0588 | 0.0039 | 15x |
| 30 | 0.5498 | 0.0185 | 30x |

The free-run error grows exponentially while reset error grows linearly with |dt|.
This proves the divergence is from **iterative compounding through a nonlinear
system**, not from any systematic Zig bug.

---

## Historical: 3 Precision Mismatches (All Fixed)

### MISMATCH 1: Guider Combine — f32 (Zig) vs bf16 (Python) ★ FIXED
Changed `guiderCombineSingle` to operate in bf16 matching Python.
All arithmetic (pred, std(), rescale) now in bf16.

### MISMATCH 2: Euler Velocity — bf16 roundtrip restored ★ FIXED
Restored the bf16 roundtrip: `vel_f32 → bf16 → f32` before Euler update.

### MISMATCH 3: denoise_mask dtype — f32 not bf16 ★ FIXED
Updated `forwardToDenoised` and `forwardDenoisingStepFromX0` to handle f32
masks correctly (explicit `.convert(.f32)` before binary ops, since ZML
asserts same dtype for binary operations).

---

## Default Guidance Params (from `constants.py`)

### LTX-2.3 defaults (non-distilled, 30 steps):
**Video**: cfg=3.0, stg=1.0, rescale=0.7, modality=3.0, stg_blocks=[28]
**Audio**: cfg=7.0, stg=1.0, rescale=0.7, modality=3.0, stg_blocks=[28]

### Zig driver uses (from `denoise_stage1.zig`):
**Video**: cfg=3.0, stg=1.0, mod=3.0, rescale=0.7
**Audio**: cfg=7.0, stg=1.0, mod=3.0, rescale=0.7
STG block index = 28 (0-based)

✅ Guidance params **match**.

---

## Validation Scripts

| Script | Purpose |
|---|---|
| `trace_dtype_chain.py` | Empirical dtype trace (pure PyTorch, no GPU) |
| `export_stage1_inputs.py` | Capture pre-denoising tensors from pipeline |
| `export_stage1_step0_reference.py` | 30-step Python reference with intermediate dumps |
| `export_step0_perpass.py` | Per-pass velocities + x0 + guider output at step 0 |
| `export_all_step_latents.py` | All 30 intermediate latents for reset test |
| `compare_step0_velocity.py` | Compare Zig vs Python at steps 1/5/15/30 |
| `compare_step0_perpass.py` | Per-pass comparison with error budget breakdown |
| `compare_reset_test.py` | Reset test analysis (isolated vs accumulated error) |
| `analyze_reset_scaling.py` | MAE/|dt| scaling analysis (pure math, no GPU) |

## Server Paths

```
/root/e2e_demo/stage1_inputs.safetensors          — initial tensors
/root/e2e_demo/stage1_step0_reference.safetensors  — Python 30-step reference
/root/e2e_demo/step0_perpass_reference.safetensors — Per-pass reference
/root/e2e_demo/all_step_latents.safetensors        — All 30 intermediate latents
/root/e2e_demo/stage1_out/                         — Zig free-run outputs
/root/e2e_demo/stage1_out_reset/                   — Zig reset-mode outputs
```
