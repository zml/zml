# Image Conditioning — Implementation Plan

Enable image-to-video generation by implementing the VAE encoder in Zig and
wiring the conditioning logic into the inference pipeline.

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
| Conditioning logic in `inference.zig` | ~30 | `inference.zig` |
| **Total** | **~175** | |

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

**Goal**: Wire encoder + conditioning into the unified pipeline.

Steps:
1. Add `--image <path>` CLI flag to `inference.zig`
2. If provided:
   - Load image tensor (pre-exported safetensors for now)
   - Compile + run `forwardVideoVaeEncode`
   - Patchify encoded latent: `[1, 128, 1, H', W']` → `[1, n_tokens, 128]`
   - Modify the loaded Stage 1 inputs:
     - `video_latent[:, 0:n_tokens] = encoded_tokens`
     - `video_clean_latent[:, 0:n_tokens] = encoded_tokens`
     - `video_denoise_mask[:, 0:n_tokens] = 0.0`
3. Rest of pipeline runs unchanged

### M4: End-to-end validation

**Goal**: Full image-conditioned pipeline produces identical output to Python reference.

Steps:
1. Run `export_image_conditioning.py` with `--decode-video` to get Python-reference MP4
2. Run Zig `inference` with `--image <same_image>` and same prompt/seed
3. Compare:
   - Encoder output: cos_sim vs `encoded_normalized` (> 0.999)
   - Stage 1 output: cos_sim vs Python Stage 1 reference
   - Final MP4: visual comparison

### M5: Full image loading (optional)

Replace the pre-exported tensor with stb_image + resize in Zig,
eliminating the Python preprocessing step for images.

## Risks and considerations

1. **Single-frame temporal stride behavior**: The encoder's temporal stride-2 blocks
   duplicate the first frame (`cat([x[:,:,:1], x], dim=2)` making F=2), then
   space-to-depth with temporal stride 2 brings it back to F=1. Need to verify
   this works correctly with the Zig reshape ops.

2. **Stage 2 conditioning**: In the full Python pipeline, image conditioning is
   applied to both Stage 1 and Stage 2 (with upscaled resolution for Stage 2).
   The bridge phase would need to carry the conditioning through, or Stage 2
   would need its own encode at 2× resolution. **Start with Stage 1 only**.

3. **Image resolution mismatch**: The input image is resized to Stage 1's
   half-resolution (H/2 × W/2). If the image aspect ratio doesn't match,
   center-crop is applied. Need to handle this in preprocessing.

4. **Strength parameter**: `strength=1.0` means fully conditioned (denoise_mask=0 at
   image positions). Lower strength allows more variation from the reference.
   Start with `strength=1.0`, add as parameter later.

5. **Keyframe conditioning**: `VideoConditionByLatentIndex` supports conditioning
   at any frame index, not just frame 0. This enables video extension / keyframe
   control. Out of scope for initial implementation.
