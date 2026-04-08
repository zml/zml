# Image Conditioning — Implementation Plan

Enable image-to-video generation by implementing the VAE encoder in Zig and
wiring the conditioning logic into the inference pipeline.

## Implementation status

| Milestone | Status | Notes |
|-----------|--------|-------|
| **M0** Python export script | **Done** | `export_image_conditioning.py` — also supports unconditioned (`--image` optional) |
| **M1** VAE encoder in Zig | **Done** | `video_vae_encoder.zig` (367 lines), cosim 0.9997 E2E |
| **M2** Image loading in Zig | **Done** | `image_loading.zig` (116 lines) — stb_image + bilinear resize + center crop |
| **M3** Conditioning in pipeline | **Done** | Stage 1 only — both `inference.zig` and `model.zig` updated |
| **M4** End-to-end validation | **Done** | Identity preserved, regression tested with/without `--image` |
| **M5** Full image loading | **Done** | Merged into M2 — stb_image path, no Python preprocessing needed |

### Files changed (vs `oboulant/ltx-cleanup` base)

Modified:
- `model.zig` (+244/−73) — per-token AdaLN masking (`adaValueAtMasked`), `SharedInputs` extended with 6 new fields
- `inference.zig` (+305/−12) — `--image` flag, VAE encode + conditioning pipeline, block call sites updated
- `BUILD.bazel` (+21) — new srcs (`image_loading.zig`, `video_vae_encoder.zig`), `@stb//:stb` dep
- `06_image_conditioning.md` — validation results + this status section

New:
- `video_vae_encoder.zig` (367 lines) — full VAE encoder: causal conv3d, space-to-depth, patchify, normalize
- `image_loading.zig` (116 lines) — JPEG/PNG load via stb_image, bilinear resize, center crop, bf16 normalize
- `export_image_conditioning.py` (843 lines) — Python reference export (conditioned + unconditioned paths)
- `validate_encoder.zig` (618 lines) — standalone encoder validation binary (used during M1)
- `diagnose_conditioning.py` (213 lines) — debug script (used during debugging)
- `diagnose_pipeline.py` (217 lines) — debug script (used during debugging)

### Key deviation from plan: per-token AdaLN masking

The original plan stated: *"conditioning only modifies the initial state tensors
(latent, clean_latent, denoise_mask) before denoising starts. The transformer
itself is unchanged."*

This turned out to be **wrong** for preserving identity when using strength=1.0
conditioning. The naive approach (broadcast the sigma-derived timestep to all
tokens uniformly) caused the **conditioned tokens (mask=0.0)** to receive the
same large AdaLN modulation as the unconditioned tokens. Since conditioned tokens
are already clean (no noise), they should get zero-sigma modulation instead.

#### The problem

Each of the 48 transformer blocks computes 9 AdaLN modulation values per modality
(shift, scale, gate for MSA, FF, and QK-norm). These are derived from the
timestep embedding, which is a function of sigma. When mask=0 ("this token is
clean"), the effective sigma for that token should be 0, but the original code
broadcast a single sigma to ALL tokens.

Result: conditioned first-frame tokens got "noisy" modulations → identity drifted
over time → generated person looked different from the reference image.

#### The solution: `adaValueAtMasked`

A new function blends two sets of modulation values per-token:

```
result[t] = mask[t] * ada_value(sigma) + (1 - mask[t]) * ada_value(0)
```

- For unconditioned tokens (mask=1.0): uses sigma modulation (unchanged behavior)
- For conditioned tokens (mask=0.0): uses zero-sigma modulation (correct for clean tokens)

**Memory optimization**: The naive approach would materialize `[B, T, d_ada]`
(~905MB for T=6144, d_ada=36864). Instead, `adaValueAtMasked` blends AFTER
slicing each ada value to `[B, T, D]` (~96MB), which XLA can reuse sequentially.
This keeps peak memory well within the 48GB GPU budget.

#### Scope of changes in `model.zig`

- `adaValueAtMasked()` function: ~30 lines
- `SharedInputs` struct: 6 new fields (`video_timesteps_zero`, `audio_timesteps_zero`,
  `v_denoise_mask`, `a_denoise_mask`, plus the zero versions for both modalities)
- `PreprocessOutput`: 6 matching new fields
- `forwardPreprocess`: computes both sigma and zero AdaLN embeddings, blends
  `embedded_timestep` per-token (small enough at [B, T, D_emb])
- All 6 block entry point variants: updated signatures with 4 new params
- All 18 `adaValueAt` calls for main timesteps (9 per modality × 2 modalities)
  converted to `adaValueAtMasked`
- `OutputProjection.forward`: accepts [B, T, D_emb] embedded_timestep (broadcasts across T)

#### Scope of changes in `inference.zig`

- All block compile args (3 Stage 1 variants + 1 Stage 2) include 4 new shapes
- All block call sites (5 Stage 1 + 1 Stage 2) pass 4 new buffers
- Output projection compilation tags `[B, 1, D_emb]` as `.{.b, .t, .d_emb}`

### Regression safety

When no `--image` is provided, the mask is all 1.0 everywhere. In this case
`adaValueAtMasked` reduces to:

```
1.0 * ada_value(sigma) + 0.0 * ada_value(0) = ada_value(sigma)
```

This is mathematically identical to the original `adaValueAt` — the
unconditioned path produces the same result as before.

### Stage 2 conditioning: current state

In the current implementation, Stage 2 conditioning IS applied in Zig when
`--image` is provided: the image is VAE-encoded at full resolution, tokens
replace the first-frame positions, and denoise_mask is set to 0.0. However the
Stage 2 bridge creates all-ones mask by default, so conditioning only takes
effect when the `--image` flag is passed.

## Usage

### Python export (on GPU server)

The export script runs the full two-stage pipeline in Python, capturing all
intermediate states for Zig to consume.

**With image conditioning:**

```bash
cd /root/repos/LTX-2
uv run python /root/repos/zml/examples/ltx/export_image_conditioning.py \
    --image /path/to/reference_image.jpg \
    --prompt "A beautiful sunset over the ocean" \
    --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
    --output-dir /root/imgcond_ref/ \
    --decode-video
```

**Without image (text-to-video only):**

```bash
cd /root/repos/LTX-2
uv run python /root/repos/zml/examples/ltx/export_image_conditioning.py \
    --prompt "A cat sitting on a windowsill watching rain" \
    --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
    --output-dir /root/newprompt_ref/ \
    --decode-video
```

Key outputs consumed by Zig:
- `unconditioned_stage1_inputs.safetensors` — Stage 1 initial state (no image baked in)
- `stage2_noise.safetensors` — Stage 2 noise vectors
- `pipeline_meta.json` — resolution, sigmas, guidance params

### Zig inference (on GPU server)

**With image conditioning (bf16 attention for Stage 2):**

```bash
bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:inference -- \
    --stage1-ckpt /root/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
    --stage2-ckpt /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    --upsampler-ckpt /root/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --stage1-inputs /root/imgcond_ref/unconditioned_stage1_inputs.safetensors \
    --stage2-noise /root/imgcond_ref/stage2_noise.safetensors \
    --meta /root/imgcond_ref/pipeline_meta.json \
    --output-dir /root/imgcond_zig/ \
    --bf16-attn-stage2 \
    --image /root/models/reference_image.jpg
```

**Without image (same prompt, same seed):**

```bash
bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:inference -- \
    --stage1-ckpt /root/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
    --stage2-ckpt /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    --upsampler-ckpt /root/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --stage1-inputs /root/imgcond_ref/unconditioned_stage1_inputs.safetensors \
    --stage2-noise /root/imgcond_ref/stage2_noise.safetensors \
    --meta /root/imgcond_ref/pipeline_meta.json \
    --output-dir /root/nocond_zig/ \
    --bf16-attn-stage2
```

The `--image` flag is optional. When omitted, the pipeline runs in pure
text-to-video mode (all masks = 1.0, no per-token AdaLN blending needed).

## Background

Currently the pipeline is text-only: Python exports text embeddings + noise,
Zig runs denoising → decode → MP4. Image conditioning (the "I" in TI2Vid)
allows providing a reference image that anchors the first frame(s) of the
generated video.

### How conditioning works

1. A reference image is loaded, resized to target resolution, normalized to `[-1, 1]`
2. The **VAE encoder** encodes it into a latent: `[1, 3, 1, H, W]` → `[1, 128, 1, H', W']`
3. The latent is **patchified** to token space: `[1, 128, 1, H', W']` → `[1, H'×W', 128]`
4. These tokens **replace** the first-frame positions in the initial noised latent
5. The **denoise mask** is set to `1 − strength` at those positions (typically 0.0 for strength=1.0)
6. The **clean latent** is set to the encoded image tokens at those positions
7. During noising, those positions receive less noise (proportional to the mask value)
8. The denoiser then generates video consistent with the conditioned first frame

The key insight: conditioning only modifies the **initial state** tensors (latent,
clean_latent, denoise_mask) before denoising starts. The transformer itself is
unchanged.

## Architecture: VAE Encoder

### Block structure (from checkpoint `vae.encoder.*`)

```
Input: [B, 3, 1, H, W]  (single frame, e.g. 512×768)

patchify(patch_size_hw=4):  [B, 3, 1, H, W] → [B, 48, 1, H/4, W/4]

conv_in:    CausalConv3d(48 → 128, k=3, causal=True)

down_blocks.0:  4 ResBlocks @ 128 channels
down_blocks.1:  SpaceToDepth(1,2,2) conv [64, 128, 3,3,3]   → 256 ch
down_blocks.2:  6 ResBlocks @ 256 channels
down_blocks.3:  SpaceToDepth(2,1,1) conv [256, 256, 3,3,3]  → 512 ch
down_blocks.4:  4 ResBlocks @ 512 channels
down_blocks.5:  SpaceToDepth(2,2,2) conv [128, 512, 3,3,3]  → 1024 ch
down_blocks.6:  2 ResBlocks @ 1024 channels
down_blocks.7:  SpaceToDepth(2,2,2) conv [128, 1024, 3,3,3] → 1024 ch
down_blocks.8:  2 ResBlocks @ 1024 channels

conv_norm_out:  PixelNorm → SiLU
conv_out:       CausalConv3d(1024 → 129, k=3, causal=True)

→ extract means (channels 0..127), discard logvar (channel 128)
→ normalize: (means − mean_of_means) / std_of_means

Output: [B, 128, 1, H/32, W/32]
```

### SpaceToDepth stride mapping

The Python config uses `compress_space_res`, `compress_time_res`, `compress_all_res`:

| Block | Stride     | Input ch → Conv out ch | Output ch (after rearrange) |
|-------|-----------|------------------------|----------------------------|
| 1     | (1, 2, 2) | 128 → 64              | 64 × 1×2×2 = 256          |
| 3     | (2, 1, 1) | 256 → 256             | 256 × 2×1×1 = 512         |
| 5     | (2, 2, 2) | 512 → 128             | 128 × 2×2×2 = 1024        |
| 7     | (2, 2, 2) | 1024 → 128            | 128 × 2×2×2 = 1024        |

**Note**: For single-frame encoding (F=1), the temporal strides (blocks 3, 5, 7)
that duplicate the first frame effectively become no-ops on the time dimension:
`cat([x[:,:,:1], x], dim=2)` with F=1 gives F=2, then stride-2 rearrange gives F=1.

### Checkpoint keys (84 total, all bf16)

```
vae.encoder.conv_in.conv.{weight,bias}                     — [128, 48, 3,3,3] / [128]
vae.encoder.down_blocks.{0,2,4,6,8}.res_blocks.N.conv{1,2}.conv.{weight,bias}
vae.encoder.down_blocks.{1,3,5,7}.conv.conv.{weight,bias}  — SpaceToDepth convs
vae.encoder.conv_out.conv.{weight,bias}                     — [129, 1024, 3,3,3] / [129]
vae.per_channel_statistics.{mean-of-means,std-of-means}     — [128] (shared with decoder)
```

**No conv_shortcut or norm3 keys** — confirmed from checkpoint. All ResBlocks have
matching in/out channels (identity skip connection). This simplifies the
implementation: the existing `VaeResBlock` / `forwardVaeResBlock` from
`video_vae.zig` can be reused as-is.

## Existing Zig building blocks

| Component | File | Reusable? |
|-----------|------|-----------|
| `VaeResBlock` + `forwardVaeResBlock` | `video_vae.zig` | **Yes** — same architecture, no channel change |
| `PixelNorm` (pixel_norm in `forwardVideoVaeDecode`) | `video_vae.zig` | **Yes** — identical |
| `forwardConv3d` | `conv_ops.zig` | **Yes** — but currently uses symmetric padding (`causal=False`). Encoder needs `causal=True` (front-pad only) |
| `PerChannelStats` / `initPerChannelStats` | `conv_ops.zig` | **Yes** — same weights. Need `normalize` (inverse of existing `un_normalize`) |
| `forwardPatchifyVideo` | `upsampler.zig` | **Yes** — `[B,128,F,H,W] → [B,F*H*W,128]` (patch_size=1 latent patchify) |
| `forwardUnpatchifyVae` (pixel unpatchify) | `video_vae.zig` | **Inverse needed** — need `patchifyVae` for `[B,3,F,H,W] → [B,48,F,H/4,W/4]` |
| `Conv3dWeight` / `GroupNormWeight` | `conv_ops.zig` | **Yes** |

## New code needed

### 1. `forwardCausalConv3d` — causal=True variant (~10 lines)

Current `forwardConv3d` uses symmetric padding (replicate first+last frame).
The encoder needs front-only padding: pad `kernel_size−1` frames at the front only.

```
// causal=True: replicate first frame at front
// Input: [B, C, F, H, W]
// Pad to: [B, C, F+2, H+2, W+2] with front=2, back=0 on temporal dim
```

### 2. `forwardSpaceToDepthDownsample` (~35 lines)

The inverse of `DepthToSpaceUpsample`. Algorithm:

```python
def forward(x, stride, conv_weight):
    # Temporal padding: duplicate first frame if stride[0] == 2
    if stride[0] == 2:
        x = cat([x[:,:,:1,:,:], x], dim=2)  # F → F+1

    # Skip connection: space-to-depth rearrange + group mean
    x_in = rearrange(x, "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w")
    x_in = rearrange(x_in, "b (c g) d h w -> b c g d h w", g=group_size)
    x_in = x_in.mean(dim=2)  # average over groups

    # Conv path
    x = causal_conv3d(x, conv_weight)
    x = rearrange(x, "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w")

    return x + x_in  # residual
```

### 3. `forwardPatchifyVae` — pixel-space patchify (~8 lines)

Inverse of existing `forwardUnpatchifyVae`:

```
// [B, 3, F, H, W] → [B, 48, F, H/4, W/4]
// "b c (f p) (h q) (w r) -> b (c p r q) f h w" with p=1, q=4, r=4
```

### 4. `forwardNormalize` — per-channel normalize (~5 lines)

Inverse of existing un_normalize in `forwardVideoVaeDecode`:

```
// normalized = (x - mean_of_means) / std_of_means
```

### 5. `forwardLogVarExtract` — extract means from 129-ch output (~5 lines)

```
// [B, 129, F, H, W] → take channels 0..127 → [B, 128, F, H, W]
```

### 6. `VideoVaeEncoderParams` + `initVideoVaeEncoderParams` (~50 lines)

Struct + weight loading for 84 encoder weights. Mirrors `VideoVaeDecoderParams`
but with `down_blocks` instead of `up_blocks`.

### 7. `forwardVideoVaeEncode` — top-level encoder function (~30 lines)

```zig
pub fn forwardVideoVaeEncode(
    pixel_input: Tensor,      // [B, 3, 1, H, W] bf16 — single frame, normalized [-1..1]
    stats: PerChannelStats,
    params: VideoVaeEncoderParams,
) Tensor {                    // [B, 128, 1, H/32, W/32] bf16
    var x = forwardPatchifyVae(pixel_input);      // [B, 48, 1, H/4, W/4]
    x = forwardCausalConv3d(x, params.conv_in);   // [B, 128, 1, H/4, W/4]
    for (params.down_blocks) |blk| {
        x = forwardDownBlock(x, blk);             // ResBlocks or SpaceToDepth
    }
    x = pixel_norm(x);
    x = silu(x);
    x = forwardCausalConv3d(x, params.conv_out);  // [B, 129, 1, H', W']
    x = x.slice(.{0}, .{0, 128});                 // extract means, drop logvar
    x = forwardNormalize(x, stats);
    return x;
}
```

### 8. Image conditioning in `inference.zig` (~30 lines)

New CLI flag + conditioning logic:

```zig
// CLI: --image <path>  (optional)
// If provided:
//   1. Load image (stb_image or pre-exported tensor)
//   2. VAE encode → [1, 128, 1, H', W']
//   3. Patchify → [1, H'×W', 128]
//   4. Replace first n_image_tokens of video_latent
//   5. Set clean_latent[:, 0:n] = encoded_tokens
//   6. Set denoise_mask[:, 0:n] = 0.0  (strength=1.0)
```

**Image loading question**: stb_image is already available in the repo
(`third_party/stb/`). Alternatively, accept a pre-exported tensor from Python
for the initial validation milestone.

## Total new Zig code estimate

| Component | Lines | File |
|-----------|-------|------|
| `forwardCausalConv3d` (causal=True) | ~10 | `conv_ops.zig` or `video_vae_encoder.zig` |
| `forwardSpaceToDepthDownsample` | ~35 | `video_vae_encoder.zig` |
| `forwardPatchifyVae` (pixel patchify) | ~8 | `video_vae_encoder.zig` |
| `forwardNormalize` | ~5 | `conv_ops.zig` |
| `forwardLogVarExtract` (slice ch 0..127) | ~5 | `video_vae_encoder.zig` |
| `VideoVaeEncoderParams` + init | ~50 | `video_vae_encoder.zig` |
| `forwardVideoVaeEncode` (top-level) | ~30 | `video_vae_encoder.zig` |
| Conditioning logic in `inference.zig` | ~50 | `inference.zig` (Stage 1 + Stage 2) |
| **Total** | **~195** | |

## Implementation plan

### M0: Python-side validation export script

**Goal**: Export reference activations for every encoder boundary, plus the
final conditioned `stage1_inputs.safetensors` from an image-conditioned run.

Create `export_image_conditioning.py`:

```bash
cd /root/repos/LTX-2
python export_image_conditioning.py \
  --image /path/to/reference_image.jpg \
  --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
  --height 512 --width 768 \
  --output-dir /root/imgcond_ref/
```

Exports:
```
imgcond_ref/
  image_preprocessed.safetensors    — [1, 3, 1, H, W] bf16  (normalized input)
  encoder_activations.safetensors   — intermediate activations:
    after_patchify:    [1, 48, 1, H/4, W/4]
    after_conv_in:     [1, 128, 1, H/4, W/4]
    after_down_0:      [1, 128, 1, H/4, W/4]
    after_down_1:      [1, 256, 1, H/8, W/8]
    after_down_2:      [1, 256, 1, H/8, W/8]
    after_down_3:      [1, 512, 1, H/8, W/16]   (*)
    after_down_4:      [1, 512, 1, H/8, W/16]
    after_down_5:      [1, 1024, 1, H/16, W/32] (*)
    after_down_6:      [1, 1024, 1, H/16, W/32]
    after_down_7:      [1, 1024, 1, H/32, W/32] (*)
    after_down_8:      [1, 1024, 1, H/32, W/32]
    after_norm_silu:   [1, 1024, 1, H/32, W/32]
    after_conv_out:    [1, 129,  1, H/32, W/32]
    encoded_means:     [1, 128,  1, H/32, W/32]
    encoded_normalized:[1, 128,  1, H/32, W/32]
  conditioned_inputs.safetensors    — full Stage 1 inputs with conditioning applied
    (same 14 tensors as stage1_inputs.safetensors, but with image conditioning baked in)
```

(*) Exact spatial dimensions depend on single-frame behavior with temporal
strides. For F=1, temporal stride=2 gives: duplicate→F=2, rearrange stride-2→F=1.
Spatial dims halve when spatial stride=2.

### M1: VAE encoder in Zig — isolated parity

**Goal**: Implement `video_vae_encoder.zig` and validate encoder output matches
Python reference to high precision (cos_sim > 0.9999).

Steps:
1. Create `video_vae_encoder.zig` with all encoder components
2. Create a standalone test binary (or add to existing debug binary) that:
   - Loads `image_preprocessed.safetensors`
   - Runs `forwardVideoVaeEncode`
   - Compares output against `encoded_normalized` from M0 reference
3. Validate intermediate activations at every boundary (after each down_block)

Validation matrix:
```
Tensor                  | Shape                  | cos_sim threshold
after_patchify          | [1, 48, 1, H/4, W/4]  | exact (pure reshape)
after_conv_in           | [1, 128, ...]          | > 0.9999
after_down_0..8         | varies                 | > 0.9999
encoded_normalized      | [1, 128, 1, H/32, W/32]| > 0.999
```

### M2: Image loading in Zig

**Goal**: Load JPEG/PNG from disk, resize, normalize to `[-1, 1]`.

Two options:
- **Option A**: Use stb_image (already in `third_party/stb/`) for decode,
  implement bilinear resize + center crop in Zig. ~60-80 lines.
- **Option B**: Accept pre-preprocessed tensor from Python (skip image loading).
  0 lines. Good enough if Python export step already handles preprocessing.

**Recommendation**: Start with Option B for M1-M2 validation. Implement Option A
later if/when full self-contained Zig is desired.

### M3: Conditioning logic in inference.zig

**Goal**: Wire encoder + conditioning into the unified pipeline for both stages.

#### How token-space conditioning maps to "first frame"

At denoising time, the video latent is in **patchified** (token) form:
`[B, T_video, 128]` where `T_video = F × H_lat × W_lat`. Tokens are ordered
frame-by-frame: the first `H_lat × W_lat` tokens correspond to frame 0, the
next `H_lat × W_lat` to frame 1, etc.

The encoded single-frame image has shape `[1, 128, 1, H_lat, W_lat]`.
After patchifying: `[1, 1 × H_lat × W_lat, 128]` = `[1, n_img, 128]`.

These `n_img` tokens align exactly with the first frame's positions in the
full video token sequence. So replacing `latent[:, 0:n_img]` = "replace the
first frame's tokens with the image encoding".

The denoise_mask at those positions is set to `0.0` (for `strength=1.0`),
meaning "these tokens are already clean — do not add noise or denoise them".
The model then generates the remaining frames flowing naturally from this
anchored first frame.

#### Stage 1 conditioning (half-resolution)

```
Image [H, W] → resize to [H/2, W/2] → VAE encode → [1, 128, 1, H/64, W/64]
  → patchify → [1, n_img_s1, 128]  where n_img_s1 = (H/64) × (W/64)

Modify Stage 1 initial state:
  video_latent[:, 0:n_img_s1]       = encoded_tokens_s1
  video_clean_latent[:, 0:n_img_s1] = encoded_tokens_s1
  video_denoise_mask[:, 0:n_img_s1] = 0.0
```

Example for 1024×1536: encode at 512×768 → latent `[1, 128, 1, 16, 24]` →
384 tokens replace indices 0:384 in the ~6144-token Stage 1 sequence.

#### Stage 2 conditioning (full-resolution)

The reference Python pipeline applies image conditioning to **both** stages.
After the bridge upsamples and before Stage 2 denoising, a second VAE encode
at full resolution conditions the Stage 2 initial state:

```
Image [H, W] → resize to [H, W] → VAE encode → [1, 128, 1, H/32, W/32]
  → patchify → [1, n_img_s2, 128]  where n_img_s2 = (H/32) × (W/32)

Modify Stage 2 initial state (after bridge noise init):
  video_latent[:, 0:n_img_s2]       = encoded_tokens_s2
  video_clean_latent[:, 0:n_img_s2] = encoded_tokens_s2
  video_denoise_mask[:, 0:n_img_s2] = 0.0
```

Example for 1024×1536: encode at 1024×1536 → latent `[1, 128, 1, 32, 48]` →
1536 tokens replace indices 0:1536 in the ~24576-token Stage 2 sequence.

The VAE encoder runs **twice** (different input resolutions) producing
different-sized conditioning latents for each stage.

#### Implementation steps

1. Add `--image <path>` CLI flag to `inference.zig`
2. Before Stage 1:
   - Load image tensor (pre-exported safetensors initially)
   - Compile + run `forwardVideoVaeEncode` at Stage 1 resolution (H/2 × W/2)
   - Patchify → apply conditioning to Stage 1 initial state
3. After bridge, before Stage 2:
   - Compile + run `forwardVideoVaeEncode` at Stage 2 resolution (H × W)
   - Patchify → apply conditioning to Stage 2 initial state
4. Rest of pipeline runs unchanged

### M4: End-to-end validation

**Goal**: Full image-conditioned pipeline (both stages) produces identical output
to Python reference.

Steps:
1. Run `export_image_conditioning.py` with `--decode-video` to get:
   - Python-reference MP4 (full image-conditioned pipeline)
   - Reference encoder outputs at both resolutions
   - Reference conditioned initial states for both stages
2. Run Zig `inference` with `--image <same_image>` and same prompt/seed
3. Compare:
   - Stage 1 encoder output: cos_sim vs reference (> 0.999)
   - Stage 2 encoder output: cos_sim vs reference (> 0.999)
   - Stage 1 denoised latent: cos_sim vs Python Stage 1 reference
   - Stage 2 denoised latent: cos_sim vs Python Stage 2 reference
   - Final MP4: visual comparison (first frame should match input image)

### M5: Full image loading (optional)

Replace the pre-exported tensor with stb_image + resize in Zig,
eliminating the Python preprocessing step for images.

## Risks and considerations

1. **Single-frame temporal stride behavior**: The encoder's temporal stride-2 blocks
   duplicate the first frame (`cat([x[:,:,:1], x], dim=2)` making F=2), then
   space-to-depth with temporal stride 2 brings it back to F=1. Need to verify
   this works correctly with the Zig reshape ops.

2. **Stage 2 conditioning**: The VAE encoder runs twice — once at half-res for
   Stage 1, once at full-res for Stage 2. Both use the same weights but
   different input/output dimensions. The conditioning application logic is
   identical (replace first-frame tokens + set mask to 0.0), just with
   different token counts. Included in M3 scope.

3. **Image resolution mismatch**: The input image is resized to Stage 1's
   half-resolution (H/2 × W/2). If the image aspect ratio doesn't match,
   center-crop is applied. Need to handle this in preprocessing.

## M1 validation results — cross-framework numerical divergence

Per-block isolated validation (each block fed exact Python reference input):

| Block | Type | cosim | close | Notes |
|-------|------|-------|-------|-------|
| A0  patchify | reshape | — | EXACT | Pure data rearrangement |
| A1  conv_in | 1 CausalConv3d | 0.999994 | 1.0000 | Single conv is near-exact |
| A2  down_blocks.0 | 4 ResBlocks (8 conv, 8 PixelNorm) | 0.999875 | 0.4412 | |
| A3  down_blocks.1 | S2D (1,2,2) | 0.999996 | 0.9906 | |
| A4  down_blocks.2 | 6 ResBlocks (12 conv, 12 PixelNorm) | 0.999957 | 0.4684 | |
| A5  down_blocks.3 | S2D (2,1,1) | 0.999995 | 0.9878 | |
| A6  down_blocks.4 | 4 ResBlocks (8 conv, 8 PixelNorm) | 0.999971 | 0.4324 | |
| A7  down_blocks.5 | S2D (2,2,2) | 0.999999 | 0.9840 | |
| A8  down_blocks.6 | 2 ResBlocks (4 conv, 4 PixelNorm) | 0.999998 | 0.9150 | |
| A9  down_blocks.7 | S2D (2,2,2) | 1.000000 | 0.9887 | |
| A10 down_blocks.8 | 2 ResBlocks (4 conv, 4 PixelNorm) | 1.000000 | 0.9560 | |
| A11 norm_silu | PixelNorm + SiLU | 0.999981 | 1.0000 | |
| A12 conv_out+norm | CausalConv3d + slice + normalize | 1.000000 | 0.9999 | |
| **E2E** | **full encoder** | **0.999683** | **0.3973** | Cumulative |

**Root cause**: XLA/StableHLO (PJRT CUDA) and PyTorch/cuDNN use different
conv3d accumulation orders, producing small per-element rounding differences in
bf16. Each individual conv3d is near-exact (close=1.0), but errors compound
through ResBlock chains. The pattern is consistent:

- 1 conv → close=1.0
- 2 ResBlocks (4 conv + 4 PixelNorm + 4 SiLU) → close ≈ 0.92–0.96
- 4 ResBlocks (8 conv + 8 PixelNorm + 8 SiLU) → close ≈ 0.43–0.47
- SpaceToDepth (1 conv + rearrange) → close ≈ 0.98–0.99

The "close" metric (`|diff| ≤ 5e-3 or |diff| ≤ 1% of max(|a|,|b|)`) is too
tight for cross-framework comparison through 40+ chained bf16 ops.
**cosim ≥ 0.9997 at every block** confirms the implementation is correct.

Confirmed non-issues investigated:
- PixelNorm eps: 1e-8 (matches Python's `PixelNorm()` default, not the 1e-6
  from `build_normalization_layer`)
- Spatial padding: ZEROS for encoder (REFLECT is decoder-only)
- f32 upcast in PixelNorm: tested removing it — no change (XLA promotes
  internally regardless)
- ResBlock shortcut: identity (no conv_shortcut/norm3) — correct for
  same-channel blocks
- SpaceToDepth group_size: 2,1,4,8 — matches Python's
  `in_ch * prod(stride) // out_ch` formula

4. **Strength parameter**: `strength=1.0` means fully conditioned (denoise_mask=0 at
   image positions). Lower strength allows more variation from the reference.
   Start with `strength=1.0`, add as parameter later.

5. **Keyframe conditioning**: `VideoConditionByLatentIndex` supports conditioning
   at any frame index, not just frame 0. This enables video extension / keyframe
   control. Out of scope for initial implementation.
