# Task: Implement the Stage 1 → Stage 2 Bridge in Zig/ZML

## Context

We are building a Zig/ZML port of the LTX-2.3 two-stage video generation pipeline. The current state:

**Completed Zig drivers:**
- `denoise_stage1.zig` — Stage 1 30-step guided denoiser → outputs `video_latent.bin` + `audio_latent.bin` (raw bf16, patchified)
- `upsample.zig` — Latent upsampler → outputs `upsampled_video.bin` (raw bf16, 5D `[1, 128, F, H*2, W*2]`)
- `denoise_e2e.zig` — Stage 2 3-step distilled denoiser → reads `stage2_inputs.safetensors`

**Python bridge (to be replaced):**
- `bridge_s1_to_s2.py` — Takes Zig Stage 1 outputs + metadata, runs Python upsampler, computes Stage 2 positions/masks/noised latents, writes `stage2_inputs.safetensors` for `denoise_e2e.zig`

**Goal:** Implement a Zig `bridge.zig` binary that replaces `bridge_s1_to_s2.py`, making the full pipeline Python-free between Stage 1 and Stage 2.

## What the bridge does

The bridge transforms Stage 1 denoised outputs into Stage 2 denoiser inputs. It performs 6 distinct steps:

```
Stage 1 outputs:
  video_latent.bin  [1, T_v1, 128] bf16 patchified
  audio_latent.bin  [1, T_a1, 128] bf16 patchified

         │
         ▼
  ┌──────────────────────────────────┐
  │  1. Unpatchify video             │  [1, T_v1, 128] → [1, 128, F, H_s1, W_s1]
  │  2. Upsample video 2x            │  [1, 128, F, H, W] → [1, 128, F, H*2, W*2]  (already in model.zig)
  │  3. Re-patchify video for S2     │  [1, 128, F, H*2, W*2] → [1, T_v2, 128]
  │  4. Audio: passthrough           │  [1, T_a, 128] → [1, T_a, 128]  (T_a unchanged, no upsample)
  │  5. Compute positions + masks    │  video_positions, audio_positions, denoise_masks
  │  6. Noise latents with sigma_0   │  clean * (1 - mask*σ) + noise * mask*σ
  └──────────────────────────────────┘
         │
         ▼
Stage 2 inputs (stage2_inputs.safetensors — 12 tensors):
  video_latent         [1, T_v2, 128]  bf16  (noised)
  audio_latent         [1, T_a2, 128]  bf16  (noised)
  video_noise          [1, T_v2, 128]  bf16
  audio_noise          [1, T_a2, 128]  bf16
  video_clean_latent   [1, T_v2, 128]  bf16
  audio_clean_latent   [1, T_a2, 128]  bf16
  video_denoise_mask   [1, T_v2, 1]    f32
  audio_denoise_mask   [1, T_a2, 1]    f32
  video_positions      [1, 3, T_v2, 2] bf16
  audio_positions      [1, 1, T_a2, 2] f32
  v_context            [1, S, 4096]    bf16
  a_context            [1, S, 2048]    bf16
```

## Step-by-step architecture

### Step 1: Video unpatchify (already done)

`forwardUnpatchifyVideo` in `model.zig`:
```
[1, F*H*W, 128] → reshape [1, F, H, W, 128] → transpose [0, 4, 1, 2, 3] → [1, 128, F, H, W]
```

### Step 2: Video upsample 2x (already done)

`forwardUpsample` in `model.zig`:
```
[1, 128, F, H, W] → un_normalize → CNN → normalize → [1, 128, F, H*2, W*2]
```

### Step 3: Re-patchify video for Stage 2 (NEW — pure tensor ops)

After upsampling, the video is `[1, 128, F, H_s2, W_s2]` where H_s2 = H_s1*2, W_s2 = W_s1*2.

**Video patchify** (VideoLatentPatchifier, patch_size=1):
```
[1, 128, F, H_s2, W_s2] → permute [0, 2, 3, 4, 1] = [1, F, H_s2, W_s2, 128]
                         → reshape [1, F*H_s2*W_s2, 128] = [1, T_v2, 128]
```

### Step 4: Audio passthrough

Audio is not upsampled. With `patch_size=1`, the patchified format is the same between stages:
- Stage 1 output: `[1, T_a, 128]` bf16 patchified
- Stage 2 input: `[1, T_a, 128]` bf16 patchified (identical)

The `audio_clean_latent` for Stage 2 is just the Stage 1 `audio_latent.bin` as-is.

**Masks** — Stage 2 uses full denoising (all ones):
```
video_denoise_mask: [1, T_v2, 1] = ones(f32)
audio_denoise_mask: [1, T_a2, 1] = ones(f32)
```

### Step 5: Compute positions (NEW — the main new logic)

#### Video positions: `[1, 3, T_v2, 2]` bf16

Three axes (time, height, width), each with [start, end) in pixel coordinates.

**Patch grid bounds** (Stage 2, F=16, H_s2=32, W_s2=48, patch_size=1):
```
For each patch index (f, h, w):
  time_start  = f,     time_end  = f + 1
  height_start = h,    height_end = h + 1
  width_start  = w,    width_end  = w + 1
```

Flatten in order (f, h, w) via meshgrid indexing="ij" to get `[3, T_v2, 2]`.

**Scale to pixel coords** (scale_factors = (8, 32, 32)):
```
pixel_coords = latent_coords * scale_factor_per_axis
  time:   start*8, end*8     → [0, 8, 16, ..., 120] / [8, 16, ..., 128]
  height: start*32, end*32   → [0, 32, 64, ..., 992] / [32, 64, ..., 1024]
  width:  start*32, end*32   → [0, 32, 64, ..., 1504] / [32, 64, ..., 1536]
```

**Causal fix** (time axis only):
```
pixel_coords[:, 0, :, :] = (pixel_coords[:, 0, :, :] + 1 - 8).clamp(min=0)
  Example: start: [0*8+1-8, 1*8+1-8, ...] = [-7, 1, 9, ...].clamp(0) = [0, 1, 9, 17, ...]
           end:   [1*8+1-8, 2*8+1-8, ...] = [1, 9, 17, ...].clamp(0) = [1, 9, 17, 25, ...]
```

**Divide time by FPS** (time axis only):
```
pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :] / fps
```

Final shape: `[1, 3, T_v2, 2]` converted to bf16.

**bf16 conversion**: Must use round-to-nearest-even (not truncation) to match
PyTorch's `.to(torch.bfloat16)`. Simple truncation (drop lower 16 bits) produces
1-ULP bf16 differences.

**GPU/CPU rounding note**: The reference is generated on CUDA GPU, which uses
FMA-based Newton-Raphson for f32 division (`/ fps`). This can differ by 1 ULP
from CPU division. However, the bf16 conversion discards the lower 16 mantissa
bits where such 1-ULP f32 differences live, so GPU and CPU results are bitwise
identical after bf16 quantization.

**Concrete example** (F=16, H=32, W=48, fps=24):
- T_v2 = 16 × 32 × 48 = 24576
- Each patch (f, h, w) at flat index `f*32*48 + h*48 + w`:
  - time: `[clamp(f*8-7, 0)/24, clamp((f+1)*8-7, 0)/24]`
  - height: `[h*32, (h+1)*32]`
  - width: `[w*32, (w+1)*32]`

#### Audio positions: `[1, 1, T_a2, 2]` f32

Single axis (time), with [start_sec, end_sec) per audio latent frame.

**Audio timing computation** (`_get_audio_latent_time_in_sec` in patchifiers.py):

The PyTorch code performs the following f32 operation chain:
```python
audio_latent_frame = torch.arange(shift, T + shift, dtype=torch.float32, device=device)
audio_mel_frame = audio_latent_frame * 4                    # f32 multiply
audio_mel_frame = (audio_mel_frame + 1 - 4).clip(min=0)     # causal fix, f32
result = audio_mel_frame * hop_length / sample_rate          # f32 * 160 / 16000
```

**IMPORTANT**: The final step is two separate f32 operations (`* 160` then `/ 16000`),
NOT a single `* 0.01`. This matters for bitwise fidelity because `0.01` is not exactly
representable in IEEE 754 float, and the two-op chain produces different f32 rounding
than a single multiply by `0.01`.

Simplified per-frame formula:
```
For frame i in [0, T_a):
  mel_start = (float(i) * 4 + 1 - 4).clamp(0)
  mel_end   = (float(i+1) * 4 + 1 - 4).clamp(0)
  start_sec = mel_start * 160 / 16000     (two f32 ops, NOT * 0.01)
  end_sec   = mel_end * 160 / 16000       (two f32 ops, NOT * 0.01)
```

**Concrete example** (T_a=126):
- Frame 0: mel_start=clamp(-3,0)=0, mel_end=clamp(1,0)=1 → [0.0, 0.01]
- Frame 1: mel_start=clamp(1,0)=1, mel_end=clamp(5,0)=5 → [0.01, 0.05]
- Frame 2: mel_start=clamp(5,0)=5, mel_end=clamp(9,0)=9 → [0.05, 0.09]
- Frame i≥1: mel_start = 4*i-3, mel_end = 4*i+1

Shape: `[1, 1, T_a, 2]` f32 (stays f32, no bf16 conversion).

**GPU/CPU rounding note**: The reference is generated on CUDA GPU, where f32
division (`/ 16000`) uses FMA-based Newton-Raphson and can produce results that
differ by 1 ULP from CPU division. Since audio_positions stays in f32 (unlike
video_positions which quantizes to bf16), this 1-ULP difference is preserved in
the output file. The Zig bridge computes positions on CPU, so bitwise mismatch
vs the GPU-generated reference is expected but numerically inconsequential
(cosine similarity = 1.0, max absolute difference = 0.0).

### Step 6: Noise the latents (NEW — simple arithmetic)

**Noising formula** (same for video and audio):
```
mask_sigma = denoise_mask * sigma_0           # [B, T, 1] * scalar → [B, T, 1]
noised = clean * (1 - mask_sigma) + noise * mask_sigma
```

Since `denoise_mask` is all-ones for Stage 2:
```
noised = clean * (1 - sigma_0) + noise * sigma_0
```

`sigma_0` comes from `pipeline_meta.json` → `stage2.sigma_0` (typically 0.909375).

The noise tensors come from `stage2_noise.safetensors` (keys: `video_noise_s2`, `audio_noise_s2`).

### Text contexts (passthrough)

Loaded from `stage1_inputs.safetensors`:
- `v_context_pos` → saved as `v_context` in Stage 2 inputs
- `a_context_pos` → saved as `a_context` in Stage 2 inputs

Stage 2 uses only positive context (distilled, no CFG/guidance).

## Implementation plan

### What already exists in Zig

| Component | Location | Status |
|-----------|----------|--------|
| Video unpatchify | `model.zig` `forwardUnpatchifyVideo` | Done |
| Video upsample | `model.zig` `forwardUpsample` + `upsample.zig` | Done |
| Stage 2 input loading | `denoise_e2e.zig` | Done (reads safetensors) |

### What needs to be implemented

| Component | Complexity | Notes |
|-----------|-----------|-------|
| Video re-patchify | Trivial | transpose + reshape, 3 lines |
| Audio passthrough | None | Stage 1 patchified output used directly |
| Video position computation | Medium | meshgrid + scale + causal_fix + /fps |
| Audio position computation | Medium | loop over T_a, causal timing formula |
| Denoise masks | Trivial | all-ones tensors |
| Noising | Trivial | `clean*(1-σ) + noise*σ`, 3 lines |
| Safetensors writing | Medium | Need to write safetensors format from Zig |
| CLI + orchestration | Low | Extend from upsample.zig pattern |

### Key decisions

**Safetensors writing**: `denoise_e2e.zig` reads safetensors via `zml.safetensors.TensorRegistry`. For writing, we need to check if ZML has a safetensors writer. If not, the format is simple enough to write manually:
- 8-byte LE header length
- JSON header with tensor metadata
- Raw tensor data concatenated

**Host vs device computation**: Positions and masks are small tensors with simple arithmetic. They can be computed entirely on the host (CPU) and uploaded to device only for the noising step.

**Compiled functions**: The noising step (`clean*(1-σ) + noise*σ`) should be a compiled ZML function since it operates on large tensors (T_v2=24576). Positions and masks are computed on host.

## New functions to add to model.zig

```
// Video patchify: [1, 128, F, H, W] → [1, F*H*W, 128]
pub fn forwardPatchifyVideo(input: Tensor) Tensor

// Noise init: clean*(1-mask*σ) + noise*mask*σ  (same as denoise_e2e forwardNoiseInit)
// Already exists in denoise_e2e.zig — may share or duplicate
```

Note: Audio does not need unpatchify/re-patchify — `audio_latent.bin` from Stage 1 is already in the correct `[1, T_a, 128]` format for Stage 2.

## bridge.zig — Driver structure

```
CLI args:
  --stage1-video     path to video_latent.bin (Stage 1 output)
  --stage1-audio     path to audio_latent.bin (Stage 1 output)
  --stage2-noise     path to stage2_noise.safetensors
  --stage1-inputs    path to stage1_inputs.safetensors (for text contexts)
  --meta             path to pipeline_meta.json
  --upsampler-ckpt   upsampler checkpoint
  --main-ckpt        main checkpoint (for per_channel_statistics)
  --output           output path for stage2_inputs.safetensors
  --ref              optional reference stage2_inputs.safetensors for validation

Flow:
  1. Parse args, load metadata from pipeline_meta.json
  2. Open checkpoint stores (upsampler + main)
  3. Init platform

  4. Load video_latent.bin → [1, T_v1, 128] bf16
  5. Compile+run unpatchify → [1, 128, F, H_s1, W_s1]
  6. Compile+run upsample → [1, 128, F, H_s2, W_s2]    (reuses model.forwardUpsample)

  7. Load audio_latent.bin → [1, T_a, 128] bf16 (used directly as audio_clean_latent)

  8. Re-patchify video → [1, T_v2, 128]                 (device: transpose+reshape)

  9. Compute video positions on host → [1, 3, T_v2, 2] bf16
  10. Compute audio positions on host → [1, 1, T_a, 2] f32
  11. Create denoise masks on host → all-ones

  12. Load noise from stage2_noise.safetensors
  13. Load text contexts from stage1_inputs.safetensors
  14. Compile+run noising → noised latents

  15. Write all 12 tensors to stage2_inputs.safetensors

  16. Optional: compare against --ref per-tensor
```

## Validation strategy

**Reference**: The Python `bridge_s1_to_s2.py` already outputs `stage2_inputs.safetensors` with all 12 tensors. We compare tensor-by-tensor:

| Tensor | Expected match | Notes |
|--------|---------------|-------|
| `video_clean_latent` | cosim > 0.995 | Depends on upsampler accuracy (already validated at 0.9978) |
| `audio_clean_latent` | Exact bitwise | Passthrough from Stage 1 output |
| `video_denoise_mask` | Exact bitwise | All ones |
| `audio_denoise_mask` | Exact bitwise | All ones |
| `video_positions` | Exact or near-exact | Pure arithmetic, but bf16 rounding possible |
| `audio_positions` | Near-exact (1 ULP) | f32 arithmetic; GPU/CPU division rounding differs by 1 ULP (see GPU/CPU rounding note above) |
| `video_noise` | Exact bitwise | Passthrough from stage2_noise.safetensors |
| `audio_noise` | Exact bitwise | Passthrough from stage2_noise.safetensors |
| `v_context` | Exact bitwise | Passthrough from stage1_inputs.safetensors |
| `a_context` | Exact bitwise | Passthrough from stage1_inputs.safetensors |
| `video_latent` (noised) | cosim > 0.995 | Depends on video_clean_latent accuracy |
| `audio_latent` (noised) | Exact bitwise | clean*(1-σ) + noise*σ with exact inputs |

Most tensors should be bitwise exact. Only video tensors that depend on the upsampler output will have bf16 drift — and those are already validated.

## Input files summary

The bridge needs these files from the earlier pipeline stages:

| File | Source | Keys/Format |
|------|--------|-------------|
| `video_latent.bin` | `denoise_stage1.zig` output | Raw bf16, `[1, T_v1, 128]` |
| `audio_latent.bin` | `denoise_stage1.zig` output | Raw bf16, `[1, T_a1, 128]` |
| `stage2_noise.safetensors` | `export_mixed_pipeline.py` | `video_noise_s2` `[1, T_v2, 128]` bf16, `audio_noise_s2` `[1, T_a2, 128]` bf16 |
| `stage1_inputs.safetensors` | `export_mixed_pipeline.py` | `v_context_pos`, `a_context_pos` (text embeddings) |
| `pipeline_meta.json` | `export_mixed_pipeline.py` | Stage 1/2 latent dims, sigma_0, fps, etc. |
| Upsampler checkpoint | Model weights | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` |
| Main checkpoint | Model weights | `ltx-2.3-22b-dev.safetensors` (for per_channel_stats) |

## Concrete example dimensions

For the "beach" prompt (1024×1536, 121 frames @ 24fps):

| Quantity | Stage 1 | Stage 2 |
|----------|---------|---------|
| F_lat | 16 | 16 |
| H_lat | 16 | 32 |
| W_lat | 24 | 48 |
| T_video | 6144 | 24576 |
| T_audio | 126 | 126 |
| sigma_0 | — | 0.909375 |
| num_steps | 30 | 3 |

## Important notes

- The position computation has no learnable parameters — it's purely geometric. This means the implementation is a one-time effort that doesn't depend on checkpoint loading.
- The patchify/unpatchify operations with `patch_size=1` are all pure reshape+transpose — no actual spatial patching.
- Stage 2 audio tokens (T_a2) are the same as Stage 1 (T_a1 = T_a = 126) — audio is not upsampled.
- The `sigma_0` for noising comes from `pipeline_meta.json`, not from a hardcoded schedule. It must be read and used dynamically.
- `stage2_noise.safetensors` contains noise drawn from the Python pipeline's deterministic seed. Using the exact same noise is essential for reproducibility.
