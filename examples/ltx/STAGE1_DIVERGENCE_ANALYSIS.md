# Stage 1 — 30-Step Divergence Analysis

## Current Status (March 27, 2026)

Step-0 velocity: **PASS** (cos_sim=0.9996 video, 0.9999 audio)  
Step-1 latent: **PASS** (cos_sim=0.99994, max_abs=0.031)  
Step-5 latent: **PASS** (cos_sim=0.99980, max_abs=0.219)  
Step-15 latent: **FAIL** (cos_sim=0.995, close_fraction=0.964)  
Step-30 final: **FAIL** (cos_sim=0.669 video, 0.793 audio)

The error grows smoothly — no jumps — confirming accumulated numerical drift.

---

## Root Causes: 3 Precision Mismatches In Zig vs Python

### MISMATCH 1: Guider Combine — f32 (Zig) vs bf16 (Python) ★ BIGGEST

**Python** (`MultiModalGuider.calculate()` in `guiders.py:244`):
All arithmetic stays in **bfloat16** (tensor dtype):
```python
pred = (
    cond                                           # bf16
    + (self.params.cfg_scale - 1) * (cond - uncond_text)    # bf16 arithmetic
    + self.params.stg_scale * (cond - uncond_perturbed)     # bf16 arithmetic
    + (self.params.modality_scale - 1) * (cond - uncond_modality)  # bf16 arithmetic
)
```
Scalar multipliers (cfg_scale=3.0, etc.) are Python floats which promote to bf16 when multiplied with bf16 tensors.

Rescaling also stays in bf16:
```python
factor = cond.std() / pred.std()    # bf16 std, bf16 division
factor = rescale_scale * factor + (1 - rescale_scale)
pred = pred * factor                # bf16 * bf16
```

**Zig** (`guiderCombineSingle` in `model.zig:1512`):
Everything is upcast to **f32** before arithmetic:
```zig
const cond_f32 = cond.convert(.f32);
const neg_f32 = neg.convert(.f32);
// ... all inputs → f32
var pred = cond_f32.add(cfg_term).add(stg_term).add(mod_term);  // f32
const cond_std = tensorStdAll(cond_f32);   // f32 std!
const pred_std = tensorStdAll(pred);       // f32 std!
```

**Impact**: With cfg=3, stg=1, mod=3, the guidance formula amplifies input
differences by ~6x. Doing this in f32 vs bf16 produces systematically different
rounding, especially in the std-rescale factor (0.7 * cond.std()/pred.std()).
bf16 has only 7 bits of mantissa → std computations differ significantly.

**Fix**: Change `guiderCombineSingle` to operate in bf16, matching Python.
Or at minimum, convert pred back to bf16 before computing std/rescaling.

### MISMATCH 2: Euler Velocity — no bf16 roundtrip (Zig) vs bf16 roundtrip (Python)

**Python** `to_velocity()` in `utils.py:21`:
```python
return ((sample.to(f32) - denoised.to(f32)) / sigma).to(sample.dtype)
#                                                     ^^^^^^^^^^^^^^^^^^
#                                              ROUNDS TO BF16!
```
Then in `EulerDiffusionStep.step()`:
```python
velocity = to_velocity(sample, sigma, denoised_sample)  # returns bf16!
return (sample.to(f32) + velocity.to(f32) * dt).to(sample.dtype)
#                         ^^^^^^^^^^^^^^^^^^^^
#               velocity was bf16, upcast back to f32 — LOSSY ROUNDTRIP
```

**Zig** `forwardDenoisingStepFromX0` (after recent "precision fix"):
```zig
const euler_vel_f32 = sample_f32.sub(blended_f32).div(sigma_f32);  // stays f32
const next_f32 = sample_f32.add(euler_vel_f32.mul(dt));  // stays f32
// NO bf16 roundtrip on velocity
```

**Impact**: The velocity bf16 roundtrip in Python loses ~8 bits of precision per
step. Zig skips this, so the resulting velocity values differ slightly from Python.

**Fix**: Revert the "precision fix" — add the bf16 roundtrip back:
```zig
const euler_vel_f32 = sample_f32.sub(blended_f32).div(sigma_f32);
const euler_vel_bf16 = euler_vel_f32.convert(out_dtype);  // roundtrip to bf16
const next_f32 = sample_f32.add(euler_vel_bf16.convert(.f32).mul(dt));
```

### MISMATCH 3: post_process_latent — all f32 (Zig) vs partial bf16 (Python)

**Python** (`helpers.py:282`):
```python
return (denoised * denoise_mask + clean.float() * (1 - denoise_mask)).to(denoised.dtype)
```
Breakdown:
- `denoised * denoise_mask` → bf16 * bf16 → **bf16** (NO upcast!)
- `clean.float()` → f32
- `(1 - denoise_mask)` → 1.0 (f32 literal) - bf16 → f32
- `f32 * f32` → f32
- `bf16 + f32` → f32 (promotion)
- `.to(denoised.dtype)` → bf16

Key: the `denoised * denoise_mask` multiplication happens in bf16.

**Zig** (`forwardDenoisingStepFromX0`):
```zig
const blended_f32 = denoised.convert(.f32).mul(mask_f32).add(clean_f32.mul(one_minus_mask));
// denoised is upcast to f32 BEFORE multiplication — different rounding
```

**Fix**: Match Python's partial-bf16 chain:
```zig
const denoised_times_mask = denoised.mul(denoise_mask);  // bf16 * bf16 → bf16
const blended_f32 = denoised_times_mask.convert(.f32).add(clean_f32.mul(one_minus_mask));
```

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

## Priority-Ordered Fix Plan

### Fix A: Guider combine in bf16 (HIGHEST IMPACT)
Change `guiderCombineSingle` to operate in bf16 matching Python.
The std-rescale computation is especially sensitive — bf16 std() produces
significantly different values than f32 std().

### Fix B: Restore Euler velocity bf16 roundtrip
The "precision fix" from earlier was COUNTERPRODUCTIVE — it made Zig
diverge MORE from Python by skipping a roundtrip that Python does.
Need to restore: vel_f32 → bf16 → f32 before Euler update.

### Fix C: post_process_latent partial bf16
Match Python: `denoised * mask` in bf16 first, then upcast to f32
for the clean blending.

---

## Complete Python Dtype Chain Per Step (Reference)

```
denoise_fn(video_state, audio_state, sigmas, step_idx):
  │
  ├─ [Pass 1: Conditional] LTXModel → velocity (bf16)
  │   └─ X0Model: to_denoised(latent_bf16, vel_bf16, timesteps_f32) → x0 (bf16)
  │
  ├─ [Pass 2: CFG/Negative] LTXModel → velocity (bf16)
  │   └─ X0Model: to_denoised(...) → neg_x0 (bf16)
  │
  ├─ [Pass 3: STG/Perturbed] LTXModel → velocity (bf16)
  │   └─ X0Model: to_denoised(...) → ptb_x0 (bf16)
  │
  ├─ [Pass 4: Modality Isolated] LTXModel → velocity (bf16)
  │   └─ X0Model: to_denoised(...) → iso_x0 (bf16)
  │
  └─ MultiModalGuider.calculate(x0, neg_x0, ptb_x0, iso_x0):
      │  ALL IN BF16:
      │  pred = x0 + 2*(x0-neg) + 1*(x0-ptb) + 2*(x0-iso)
      │  factor = 0.7 * (x0.std()/pred.std()) + 0.3
      │  pred = pred * factor
      └─ return pred (bf16)

post_process_latent(guided_x0_bf16, mask_bf16, clean_bf16):
  │  (guided_x0 * mask)_bf16 + clean.float() * (1-mask)  → f32 → bf16
  └─ return blended (bf16)

EulerDiffusionStep.step(sample_bf16, blended_bf16, sigmas_f32, idx):
  │  velocity = to_velocity(sample, sigma_scalar, blended):
  │    ((sample_f32 - blended_f32) / sigma_float).to(bf16)  ← BF16 ROUNDTRIP
  │  
  │  next = (sample_f32 + velocity_bf16.to(f32) * dt_f32).to(bf16)
  └─ return next (bf16)
```

---

## Validation Strategy After Fixes

1. Apply Fixes A+B+C to `model.zig`
2. Rebuild: `bazel build --config=release --@zml//platforms:cuda=true //examples/ltx:denoise_stage1`
3. Re-run Zig driver (30 steps)
4. Compare step 1/5/15/30 — expect significantly tighter agreement

**Expected improvement**:
- Step-1: should stay ~identical
- Step-5: should improve from cos_sim=0.99980 → ~0.99995
- Step-15: should improve from cos_sim=0.995 → ~0.999+
- Step-30: should improve from cos_sim=0.669 → ~0.99+

**If still diverging after fixes**, the residual error is from the NN forward
pass itself (48-block transformer in bf16) — that per-step cos_sim=0.9996 is
the irreducible floor from weight loading / attention implementation differences.

---

## Files to Edit

1. `examples/ltx/model.zig` — `guiderCombineSingle`, `forwardDenoisingStepFromX0`
2. No Python changes needed
3. No BUILD.bazel changes needed

## Server Commands (After Fixes)

```bash
# 1. Deploy updated model.zig
scp examples/ltx/model.zig root@dev-oboulant:/root/repos/zml/examples/ltx/

# 2. Build and run
cd /root/repos/zml && bazel run --config=release --@zml//platforms:cuda=true \
    //examples/ltx:denoise_stage1 -- \
    /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    /root/e2e_demo/stage1_inputs.safetensors \
    /root/e2e_demo/stage1_out

# 3. Compare (Python reference already saved)
cd /root/repos/LTX-2 && uv run python scripts/compare_step0_velocity.py --full \
    --zig-dir /root/e2e_demo/stage1_out \
    --ref /root/e2e_demo/stage1_step0_reference.safetensors
```
