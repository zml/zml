# LTX-2 Pipeline Status

Reference Python pipeline: `ti2vid_two_stages.py` (two-stage text/image-to-video generation).

## Full Pipeline Stages (Stage 2 — distilled)

| # | Component | Status | Notes |
|---|-----------|--------|-------|
| **1** | **Text encoding** (Gemma → embeddings) | Not started | Gemma2 text encoder + embeddings processor produces `v_context` and `a_context` [B, T, 4096]. Separate ~2B param model. |
| **2** | **Noise initialization** | **DONE** ✅ | `noised = noise * mask * sigma_0 + clean * (1 - mask * sigma_0)`. Validated: video cos_sim=1.000000 close=1.000000, audio cos_sim=1.000000 close=1.000000. Implemented in `model.zig:forwardNoiseInit`. |
| **3** | **Denoising loop** (3 Euler steps) | **DONE (Step 3)** ✅ | Sigma schedule: `[0.909375, 0.725, 0.421875, 0.0]` → 3 steps. Validated: video cos_sim=0.978–0.982, audio cos_sim=0.9999 |
| 3a | velocity_model.forward | **DONE (Step 2)** ✅ | Preprocessing + 48 blocks + output projection. Validated: cos_sim=0.9965, close=0.9762 |
| 3b | post_process_latent | **DONE (Step 3)** ✅ | `denoised * mask + clean * (1 - mask)` — implemented in `forwardDenoisingStep` |
| 3c | Euler step | **DONE (Step 3)** ✅ | `velocity = (sample - denoised) / sigma; x_next = sample + velocity * dt` — implemented in `forwardDenoisingStep` |
| **4** | **Video VAE decode** | Not started | Separate model: latent [B,C,T,H,W] → pixel frames. Tiled decoding for memory. |
| **5** | **Audio VAE decode** | Not started | Separate model: audio latent → mel spectrogram → vocoder → waveform |
| **6** | **Video upsampler** (Stage 1 → Stage 2) | Not started | Spatial 2× upscaler applied to Stage 1 output before Stage 2 denoising |
| **7** | **Video encoding** (MP4 + audio mux) | Not started | ffmpeg-level, not ML |

## Denoising Loop Pseudocode (Step 3)

```
sigmas = [0.909375, 0.725, 0.421875, 0.0]

for step_idx in 0..3:
    sigma = sigmas[step_idx]           # [0.909375, 0.725, 0.421875]
    sigma_next = sigmas[step_idx + 1]  # [0.725, 0.421875, 0.0]

    # 3a: Run velocity model (ALREADY DONE)
    denoised_v, denoised_a = velocity_model(noisy_latent, sigma, context, positions)

    # 3b: Blend with clean latent where mask says "don't denoise"
    denoised_v = denoised_v * v_mask + v_clean * (1 - v_mask)
    denoised_a = denoised_a * a_mask + a_clean * (1 - a_mask)

    # 3c: Euler step
    velocity_v = (v_latent - denoised_v) / sigma
    v_latent = v_latent + velocity_v * (sigma_next - sigma)
    # same for audio
```

## Key Python Reference Functions

- **`euler_denoising_loop`** — `ltx-pipelines/src/ltx_pipelines/utils/samplers.py:18`
- **`denoise_audio_video`** — `ltx-pipelines/src/ltx_pipelines/utils/helpers.py:457`
- **`simple_denoising_func`** — `ltx-pipelines/src/ltx_pipelines/utils/helpers.py:324`
- **`post_process_latent`** — `ltx-pipelines/src/ltx_pipelines/utils/helpers.py:313`
- **`EulerDiffusionStep.step`** — `ltx-core/src/ltx_core/components/diffusion_steps.py:6`
- **`to_velocity`** — `ltx-core/src/ltx_core/utils.py:21` — `(sample - denoised) / sigma`
- **`GaussianNoiser`** — `ltx-core/src/ltx_core/components/noisers.py:15`
- **`STAGE_2_DISTILLED_SIGMA_VALUES`** — `ltx-pipelines/src/ltx_pipelines/utils/constants.py:15` — `[0.909375, 0.725, 0.421875, 0.0]`
- **`Modality`** — `ltx-core/src/ltx_core/model/transformer/modality.py:7`
- **`LatentState`** — `ltx-core/src/ltx_core/types.py:159` — fields: `latent`, `denoise_mask`, `positions`, `clean_latent`, `attention_mask`

## End-to-End Demo ✅ DONE

Hybrid Python→Zig→Python pipeline validated end-to-end (March 2026):

1. **Python** (`e2e/export_stage2_inputs.py`): Text encoding + Stage 1 (30 steps) + upsample → exports clean latents, noise, masks, contexts as safetensors
2. **Zig** (`denoise_e2e.zig`): Noise init (`forwardNoiseInit`) + 3-step Stage 2 denoising loop (48 blocks × 3 Euler steps) → raw binary latents
3. **Python** (`e2e/decode_latents.py`): Unpatchify + Video VAE decode + Audio VAE decode + Vocoder → MP4

Produces playable video+audio (768×512, 121 frames @ 25fps). Verified against pure-Python
reference — same semantic content, confirming no divergence from the Zig denoiser.

**Noise init integration (March 27):** The Zig denoiser now applies `forwardNoiseInit` on device
rather than receiving pre-noised latents from Python. The export script provides clean latent +
back-computed noise, and Zig computes `noised = noise * mask * sigma_0 + clean * (1 - mask * sigma_0)`.
E2E re-validated — produces identical video output.

## Step 2 Validation Results

| Metric | Video | Audio |
|--------|-------|-------|
| cos_sim | 0.9965 | 0.9999 |
| close | 0.9762 | 1.0000 |

Thresholds: cos_sim ≥ 0.995, close ≥ 0.96. Per-block analysis confirmed no logic bugs —
divergence is pure floating-point accumulation over 48 sequential blocks.

## Step 3 Validation Results

| Metric | Video | Audio |
|--------|-------|-------|
| cos_sim | 0.978–0.982 | 0.9999 |
| close | 0.907–0.919 | 1.0000 |

Thresholds: video cos_sim ≥ 0.975, close ≥ 0.90; audio cos_sim ≥ 0.995, close ≥ 0.96.
Error compounds across 3 denoising steps (each step’s input is previous step’s output).
Run-to-run variance: ±0.002 on cos_sim due to non-deterministic GPU GEMM scheduling.

bf16 attention experiment showed f32 attention is strictly better for parity (XLA f32 matmuls
closer to PyTorch cuBLAS bf16 tensor core behavior than XLA bf16 matmuls).

## Noise Init Validation Results

| Metric | Video | Audio |
|--------|-------|-------|
| cos_sim | 1.000000 | 1.000000 |
| close | 1.000000 | 1.000000 |
| max_abs_err | 1.56e-2 | 7.81e-3 |
| mean_abs_err | 1.32e-6 | 1.35e-6 |

Perfect parity. The small max_abs_err is from bf16 rounding in the recovered noise tensor
(original noise was generated in bf16, recovered via f32 division) — not a Zig computation error.
Fixture: `export_noise_init_fixture.py` extracts from `trace_run/11_stage2_steps.pt`.
Checker: `check_noise_init.zig` compiles+runs `forwardNoiseInit` on device.

---

## Stage 1 — Full Denoising (non-distilled, 30 steps × 4 guidance passes)

See [STAGE1_IMPLEMENTATION_PLAN.md](STAGE1_IMPLEMENTATION_PLAN.md) for full details.

| # | Component | Status |
|---|-----------|--------|
| **1** | STG block variant (V-passthrough) | **DONE** ✅ — cos_sim 0.999995 / 0.999991 |
| **2** | Guider combine (CFG+STG+modality) | Not started |
| **3** | Sigma schedule | Not started |
| **4** | Stage 1 driver (30 steps × 4 passes) | Not started |
| **5** | Weight loading validation | Not started |
