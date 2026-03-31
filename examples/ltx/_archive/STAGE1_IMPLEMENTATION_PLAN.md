# Stage 1 Denoising — Implementation Plan

## Overview

Stage 1 runs the **full (non-distilled) denoising loop** at half resolution with
**classifier-free guidance (CFG)**, **spatiotemporal guidance (STG)**, and
**modality isolation guidance**. This requires 4 transformer forward passes per
denoising step (vs 1 for Stage 2).

**Same model architecture as Stage 2** — confirmed by parameter comparison:
identical module tree, same shapes, only weight matrices differ (LoRA deltas,
max diff ~0.002–0.009 in bf16; biases unchanged). The existing Zig model code
works unchanged — just load different weights.

---

## Stage 1 vs Stage 2 Comparison

| Property | Stage 1 | Stage 2 (distilled) |
|----------|---------|---------------------|
| Resolution | half (384×256 for 768×512) | full (768×512) |
| Denoising steps | 30–40 (configurable) | 3 (fixed) |
| Sigma schedule | Dynamic (shifted cosine) | Fixed `[0.909375, 0.725, 0.421875, 0.0]` |
| Forward passes/step | **4** (CFG+STG+modality) | 1 (no guidance) |
| Text context | Positive AND negative | Positive only |
| Perturbations | STG (block 29) + modality isolation | None |
| Weights | Base checkpoint | Base + distilled LoRA |
| Total transformer calls | ~120–160 (30–40 × 4) | 3 |

---

## 4 Guidance Passes Per Step

Each denoising step runs the transformer 4 times with the same noisy latents but
different contexts/perturbations:

| Pass | Text context | A2V/V2A cross-attn | Self-attn | Purpose |
|------|-------------|-------------------|-----------|---------|
| 1. **Positive** | Positive prompt | Normal | Normal | Conditional generation |
| 2. **Negative (CFG)** | Negative prompt | Normal | Normal | Unconditional baseline |
| 3. **Perturbed (STG)** | Positive prompt | Normal | **V-passthrough at block 29** | Spatiotemporal reference |
| 4. **Isolated (modality)** | Positive prompt | **Zeroed (all blocks)** | Normal | Modality isolation reference |

### Guidance combine formula

Per modality (video and audio separately):

```
pred = cond
     + (cfg_scale - 1) * (cond - neg)         # CFG term
     + stg_scale * (cond - ptb)               # STG term
     + (modality_scale - 1) * (cond - iso)    # Modality isolation term

if rescale_scale != 0:
    factor = rescale_scale * (cond.std() / pred.std()) + (1 - rescale_scale)
    pred = pred * factor
```

### Default guidance parameters

| Parameter | Video | Audio |
|-----------|-------|-------|
| `cfg_scale` | 3.0 | 7.0 |
| `stg_scale` | 1.0 | 1.0 |
| `rescale_scale` | 0.7 | 0.7 |
| `modality_scale` | 3.0 | 3.0 |
| `stg_blocks` | [29] | [29] |

---

## What Exists in Zig Already

| Component | Status | Reuse for Stage 1? |
|-----------|--------|---------------------|
| `forwardBlock0Native` | ✅ | Passes 1, 2 (swap text context) |
| `forwardPreprocess` | ✅ | ✅ (produces adaln + RoPE from sigma) |
| `forwardOutputProjection` | ✅ | ✅ |
| `forwardDenoisingStep` | ✅ | ✅ (Euler step + post_process_latent) |
| A2V/V2A mask support | ✅ | Pass 4: set masks to zero |
| STG V-passthrough | ✅ | `forwardBlock0NativeSTG` — comptime V-passthrough variant |
| Guider combine | ✅ | **New**: `forwardGuiderCombine` — CFG+STG+modality formula + std-rescale |
| Dynamic sigma schedule | ✅ | **New**: `computeSigmaSchedule` — native Zig LTX2Scheduler |

---

## Implementation Steps

### Step 1: STG block variant (V-passthrough at self-attention) — ✅ DONE

**What**: When STG perturbation is active, block 29's self-attention replaces the
full Q·K·V attention with a V-only passthrough: `to_out(to_v(x))`. The AdaLN
gating and residual add still happen normally.

**How**: Added comptime flags to `forwardNativeImpl`:
- `comptime skip_video_self_attn: bool, comptime skip_audio_self_attn: bool`
- When `true`, calls `self.attn1.forwardValuePassthrough(norm_vx, ...)` — computes
  `to_out(to_v(x))` with per-head gating, skipping Q/K projections, RoPE, and SDPA.

New entry point: `forwardBlock0NativeSTG(...)` — identical signature to
`forwardBlock0Native`, calls `forwardNativeImpl(skip_video_self_attn=true, skip_audio_self_attn=true, ...)`.

This compiles to a **second block exe** that's used only for block 29 during Pass 3.

**Validation results** (synthetic inputs, real block-0 weights from base checkpoint):

| Metric | Video | Audio |
|--------|-------|-------|
| Zig vs Python cos_sim | 0.999995 | 0.999991 |
| max_abs | 0.5000 | 0.5000 |
| mean_abs | 0.003835 | 0.009674 |
| STG vs Normal cos_sim | 0.581 | 0.533 |

Pass threshold: `expectClose(atol=0.2, rtol=0.01, min_close=0.999)` — **PASSED**.

The "STG vs Normal" row is a **sanity check** — it compares V-passthrough output against
normal (full attention) output. These are *supposed* to differ substantially because
V-passthrough skips Q·K attention entirely. A cos_sim of ~0.58 confirms the V-passthrough
code path is actually active and changing behavior (identical outputs would indicate a bug).

**Files**:
- `model.zig`: `Attention.forwardValuePassthrough`, `forwardNativeImpl` with skip flags,
  `forwardBlock0NativeSTG` / `forwardBlock0NativeSTGWithAVMasks` entrypoints
- `stg_block_check.zig`: Zig parity checker
- `export_stg_block_fixture.py`: Self-contained fixture generator (synthetic inputs +
  real weights, monkeypatches attn1/audio_attn1 for V-passthrough reference)
- Checkpoint key prefix: `model.diffusion_model.transformer_blocks.{idx}.`

### Step 2: Guider combine function — ✅ DONE

**What**: Combine the 4 denoised outputs using the CFG+STG+modality formula.

**How**: New compiled function `forwardGuiderCombine` (16 Tensor inputs):
- 4 per-modality outputs × 2 modalities (cond/neg/ptb/iso for video and audio, bf16)
- 4 scalar guidance params × 2 modalities (cfg/stg/mod/rescale, f32)
- Returns `GuiderCombineResult { guided_v, guided_a }`

Internal helper `guiderCombineSingle` implements per-modality:
```
pred = cond + (cfg-1)*(cond-neg) + stg*(cond-ptb) + (mod-1)*(cond-iso)
pred *= rescale * (cond.std() / pred.std()) + (1 - rescale)
```

`tensorStdAll` computes std across all elements via flatten→mean→variance→sqrt.

**Validation results** (synthetic inputs, default guidance params):

| Metric | Video | Audio |
|--------|-------|-------|
| Zig vs Python cos_sim | 1.000000 | 1.000000 |
| max_abs | 0.0000 | 0.0000 |
| mean_abs | 0.000000 | 0.000000 |

**Bit-identical** on synthetic inputs. On real pipeline data, matches the pipeline's
own `MultiModalGuider.calculate` output:

| Metric | Video | Audio |
|--------|-------|-------|
| Zig vs Python (synthetic) cos_sim | 1.000000 | 1.000000 |
| Our formula vs pipeline's guider cos_sim | 0.999993 | 0.999989 |
| Our formula vs pipeline's guider max_abs | 0.046875 | 0.031250 |

The tiny residual (~0.03-0.05 max_abs) is bf16 rounding from different cast ordering
in the pipeline's internal path. Formula is confirmed correct.

**Files**:
- `model.zig`: `forwardGuiderCombine`, `guiderCombineSingle`, `tensorStdAll`, `GuiderCombineResult`
- `guider_combine_check.zig`: Zig parity checker
- `export_guider_combine_fixture.py`: Self-contained fixture generator (synthetic bf16 inputs,
  default params: video cfg=3/stg=1/mod=3/rescale=0.7, audio cfg=7/stg=1/mod=3/rescale=0.7)
- `export_guider_combine_live_fixture.py`: Live pipeline capture — hooks `transformer.forward`
  and `MultiModalGuider.calculate` to capture real 4-pass outputs + pipeline's guided result

### Step 3: Sigma schedule — ✅ DONE

**What**: Stage 1 uses `LTX2Scheduler` which generates a shifted/stretched cosine
schedule dynamically. Parameters: `max_shift=2.05`, `base_shift=0.95`,
`terminal=0.1`, `num_steps=30`.

**How**: Implemented natively in Zig as `computeSigmaSchedule` — a host-side function
(not a compiled graph op) that produces `num_steps + 1` f32 values at startup.

Algorithm (matching `LTX2Scheduler.execute`):
1. `linspace(1.0, 0.0, num_steps + 1)`
2. Compute token-dependent shift: `sigma_shift = tokens * mm + b` where
   `mm = (max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR)`
3. Apply logistic shift: `sigma = exp(s) / (exp(s) + 1/sigma - 1)`
4. Stretch so last non-zero sigma = terminal

Constants: `BASE_SHIFT_ANCHOR=1024`, `MAX_SHIFT_ANCHOR=4096`.

Also added `stage1_default_schedule` struct with the default parameters.

**Validation results** (Zig vs Python `LTX2Scheduler().execute(steps=30)`):

| Metric | Value |
|--------|-------|
| max_abs_diff | 1.788e-7 |
| mean_abs_diff | 5.0e-8 |

Within f32 epsilon (1.19e-7 per op). All 31 values match to full f32 precision.

**Files**:
- `model.zig`: `computeSigmaSchedule`, `stage1_default_schedule`
- `sigma_schedule_check.zig`: Comparison against hardcoded Python reference values

### Step 4: Driver (`denoise_stage1.zig`) — 🔄 IN PROGRESS (Phase 1 done)

**What**: Outer loop driving the full Stage 1 denoising.

**Incremental phases**:
- **Phase 1** ✅: Single-pass skeleton (conditional only, no guidance)
- **Phase 2**: Add 4-pass structure (negative context, STG block swap, isolation masks)
- **Phase 3**: Wire in `forwardGuiderCombine` and validate against live fixture

**Per step** (30 iterations):
1. Run `forwardPreprocess` (1×) — sigma → adaln timesteps + RoPE + patchify
2. Run Pass 1 (positive): 48 blocks via `block_normal_exe` + output projection
3. Run Pass 2 (negative): 48 blocks via `block_normal_exe` + output projection
   (same exe, different text context buffers)
4. Run Pass 3 (STG): 47 blocks via `block_normal_exe` + 1 block (29) via
   `block_stg_exe` + output projection
5. Run Pass 4 (isolated): 48 blocks via `block_iso_exe` (WithAVMasks, zero masks)
   + output projection
6. Run `forwardGuiderCombine` — combine 4 denoised outputs
7. Run `forwardDenoisingStep` — Euler step + post_process_latent

**Compiled exes needed** (3 block variants):
- `preprocess_exe` (reuse pattern from Stage 2)
- `block_normal_exe` — `forwardBlock0Native` (24 args, no masks) — Passes 1 & 2
- `block_stg_exe` — `forwardBlock0NativeSTG` (24 args) — Pass 3, block 28 only
- `block_iso_exe` — `forwardBlock0NativeWithAVMasks` (26 args, zero mask buffers) — Pass 4
- `proj_v_exe`, `proj_a_exe` (reuse pattern from Stage 2)
- `guider_combine_exe` (new)
- `denoise_v_exe`, `denoise_a_exe` (reuse pattern from Stage 2)

**Note**: Stage 1 runs at half resolution → different token counts → recompiled exes.

**Phase 1 status**: Skeleton builds and compiles. Uses dynamic sigma schedule from
`computeSigmaSchedule` (30 steps). Single-pass loop with `block_normal_exe` +
output projection + Euler step. Needs `export_stage1_inputs.py` to run on GPU.

**Step-0 velocity validation** (48-block full forward pass, distilled checkpoint,
real pipeline inputs, Zig raw velocity vs Python `velocity_model.forward` output):

| Metric | Video | Audio |
|--------|-------|-------|
| cos_sim | 0.999604 | 0.999872 |
| close_fraction(atol=0.2, rtol=0.01) | 1.000000 | 1.000000 |
| max_abs_err | 0.175 | 0.083 |
| mean_abs_err | 0.022 | 0.013 |
| rmse | 0.029 | 0.018 |
| p50 | 0.016 | 0.011 |
| p90 | 0.047 | 0.031 |
| p99 | 0.080 | 0.052 |
| p999 | 0.109 | 0.070 |

Using `expectClose(atol=0.2, rtol=0.01, min_close=0.999)` — the same thresholds as
Step 1's single-block validation — should **PASS** since max_abs_err (0.175) is below
the single-block atol (0.2). This is a good result: bf16 error through 48 blocks
stays within the tolerance budget for a single block.

Note: initial comparison was x0-vs-velocity (cos_sim = -0.42) — Python's `X0Model.forward`
returns x0 predictions (x0 = sample - v·σ), while Zig outputs raw velocity from
`forwardOutputProjection`. Fixed by hooking `transformer.velocity_model.forward`
instead of `transformer.forward`.

**30-step final latent comparison** (Phase 1, single-pass conditional only — NO guidance):

| Metric | Video | Audio |
|--------|-------|-------|
| cos_sim | 0.063 | -0.016 |
| close_fraction(atol=0.2, rtol=0.01) | 0.028 | 0.005 |
| max_abs_err | 24.6 | 61.1 |
| Zig range | [-6.8, 7.3] | [-5.1, 6.4] |
| Python range | [-17.9, 21.1] | [-59.8, 49.8] |

**Expected FAIL** — Zig Phase 1 runs 1 pass (conditional only), Python runs 4 passes
with full guidance (CFG scale 3.0/7.0, STG, modality isolation). Guidance amplifies
the velocity signal 3–10×, so after step 1 the latents diverge exponentially through
30 steps. This confirms guidance is the missing piece — validates that Phase 2 + Phase 3
are required before 30-step comparison can pass. The step-0 velocity PASS already
confirms the per-step transformer forward pass is correct.

**Files**:
- `denoise_stage1.zig`: Stage 1 driver (Phase 1 — single-pass skeleton)
- `export_stage1_inputs.py`: Captures pre-denoising tensors from real pipeline
- `export_stage1_step0_reference.py`: Python reference (step-0 velocity + full 30-step)
- `compare_step0_velocity.py`: Comparison with expectClose metrics + cos_sim
- **[STAGE1_DIVERGENCE_ANALYSIS.md](STAGE1_DIVERGENCE_ANALYSIS.md)**: Deep dive into
  30-step divergence root causes (3 dtype mismatches) and priority-ordered fix plan

### Step 5: Weight loading

Stage 1 uses the **base checkpoint** (without distilled LoRA). The existing Zig
weight loading code should work — just point to the base safetensors file.

Need to verify: is the base checkpoint a separate file, or is it the same file
with LoRA already merged? If the distilled checkpoint has LoRA pre-merged, we
need the un-merged base. Check what files exist in `/root/models/ltx-2.3/`.

---

## Validation Plan

### Phase 1: Per-component validation

**Fixture export** (`export_stage1_fixture.py`):
Run the full Python Stage 1 pipeline and capture:
- Initial noised latent states (after `noise_video_state` / `noise_audio_state`)
- Positive + negative text contexts
- Sigma schedule (all 31 values)
- Per-step intermediates for 1–2 representative steps:
  - 4 denoised outputs (cond, neg, ptb, iso) for both video and audio
  - Guided combined output
  - Euler-stepped next latent
- Final denoised latents after all steps

**Checkers**:
1. `stage1_stg_block_check.zig` — Validate V-passthrough variant at block 29
   against Python reference
2. `stage1_guider_check.zig` — Validate guider combine formula against Python
   reference (single step)
3. `stage1_single_step_check.zig` — Validate one full step (4 passes + combine +
   Euler) against Python reference

### Phase 2: Full loop validation

Run all 30 steps and compare final latents against Python reference. Expect
similar degradation pattern as Stage 2 (cos_sim compounding over steps), but
potentially better per-step since Stage 1 has more steps (each step is a smaller
change).

### Phase 3: E2E integration

Replace the Python export step with:
1. **Python**: Text encoding only → export positive/negative contexts + sigma schedule
2. **Zig**: Full Stage 1 denoising (30 steps × 4 passes × 48 blocks)
3. **Zig**: Noise initialization for Stage 2 (trivial math)
4. **Zig**: Full Stage 2 denoising (3 steps × 1 pass × 48 blocks)
5. **Python**: Unpatchify + VAE decode → MP4

This makes the Zig portion handle both denoising stages — the compute-heaviest
part of the pipeline.

---

## Perturbation Mechanics (Python Reference)

### STG: `SKIP_VIDEO_SELF_ATTN` / `SKIP_AUDIO_SELF_ATTN` at blocks=[29]

In Python's `Attention.forward`:
```python
if all_perturbed:
    # Skip Q/K entirely, just pass V through
    v = self.to_v(context)
    out = v  # no attention computation
else:
    # Normal attention: Q, K, V → scaled dot product
    q = self.to_q(x); k = self.to_k(context)
    out = attention(q, k, v)
    if perturbation_mask is not None:
        out = out * mask + v * (1 - mask)  # blend

return self.to_out(out)
```

Since we run STG as a **separate forward pass** (all samples perturbed), the code
path is simply: `to_out(to_v(x))` for self-attention at block 29. All other
blocks run normally.

### Isolated: `SKIP_A2V_CROSS_ATTN` / `SKIP_V2A_CROSS_ATTN` at blocks=None

Cross-attention perturbations use multiply-by-zero:
```python
a2v_mask = perturbations.mask_like(...)  # 0.0 for perturbed
vx = vx + attn_output * gate * a2v_mask  # → 0.0, residual adds zero
```

Since `blocks=None` means all blocks, the effect is that **no audio-video
cross-attention happens** — each modality is processed in isolation.

The existing Zig code already supports this via `a2v_mask` and `v2a_mask` in
`SharedInputs`. Pass 4 just needs to set these masks to all-zeros tensors.

---

## Noise Initialization (Stage 2 Setup)

This is a separate, trivial task that can be done independently:

```
latent = noise * mask * sigma_0 + clean * (1 - mask * sigma_0)
```

where:
- `noise` = Gaussian random tensor (same shape as latent)
- `mask` = denoise_mask (per-token strength, 0–1)
- `sigma_0` = first sigma value (0.909375 for Stage 2)
- `clean` = clean_latent (conditioning reference)

This eliminates the Python `export_stage2_inputs.py` dependency for Stage 2 setup.
Once Stage 1 produces denoised latents + the spatial upsampler runs, noise init
connects Stage 1 output to Stage 2 input.

**Recommendation**: Implement noise init **before** Stage 1 — it's a 30-minute
task, can be validated with existing Stage 2 fixtures, and eliminates a Python
dependency. Stage 1 is a multi-day effort.

---

## Ordering Recommendation

1. **Noise init** (trivial, validates immediately) ← do first
2. **STG block variant** (Step 1 above)
3. **Guider combine** (Step 2 above)
4. **Fixture export** (Python script for Stage 1 intermediates)
5. **Per-component validation** (STG block, guider combine)
6. **Driver + full loop** (Steps 3–4 above)
7. **Full loop validation**
8. **E2E integration**
