# LTX-2.3 VAE Decode — Design Document

## Overview

This document describes the VAE decode step for the LTX-2.3 video generation pipeline.
The goal is to replace the current Python-based decode (`decode_latents.py`) with a native
Zig/ZML implementation, making the pipeline: **Python (prompt encode) → Zig (denoise + VAE decode) → output frames**.

Three sub-models are involved:

| Component            | Input                          | Output                          | ~Params  |
|----------------------|--------------------------------|---------------------------------|----------|
| **Video VAE Decoder**| `[B, 128, F', H', W']` bf16   | `[B, 3, F, H, W]` bf16→u8     | ~407 M   |
| **Audio VAE Decoder**| `[B, 8, T, 16]` bf16          | `[B, 2, T_spec, 64]` bf16     | ~8 M     |
| **Vocoder + BWE**    | mel spectrogram                | stereo waveform @ 48 kHz       | ~60 M    |

**Phased plan:** implement the **video VAE decoder first** (Phase 1), then audio+vocoder
(Phase 2). This document focuses on Phase 1.

---

## 1. Video VAE Decoder Architecture (from checkpoint)

### 1.1 Actual Layer Structure

Derived from the weight keys in `ltx-2.3-22b-distilled.safetensors`:

```
Input: [B, 128, F', H', W']                              (latent)
  │
  ├─ per_channel_statistics.un_normalize                   per-channel denorm
  │  x = x * std_of_means + mean_of_means                [128] each, bf16
  │
  ├─ conv_in                                               CausalConv3d(128 → 1024, k=3, causal=False)
  │  weight [1024, 128, 3, 3, 3]  bias [1024]
  │  → [B, 1024, F', H', W']
  │
  ├─ up_blocks.0  UNetMidBlock3D — 2 ResnetBlock3D @ 1024ch
  │  res_blocks.{0,1}: conv1 [1024,1024,3,3,3] + conv2 [1024,1024,3,3,3]
  │  → [B, 1024, F', H', W']
  │
  ├─ up_blocks.1  DepthToSpaceUpsample(stride=(2,2,2))
  │  conv [4096, 1024, 3,3,3]  → rearrange → remove first frame
  │  → [B, 512, 2F'-1, 2H', 2W']
  │
  ├─ up_blocks.2  UNetMidBlock3D — 2 ResnetBlock3D @ 512ch
  │  res_blocks.{0,1}: conv1 [512,512,3,3,3] + conv2 [512,512,3,3,3]
  │  → [B, 512, 2F'-1, 2H', 2W']
  │
  ├─ up_blocks.3  DepthToSpaceUpsample(stride=(2,2,2))
  │  conv [4096, 512, 3,3,3]  → rearrange → remove first frame
  │  → [B, 512, 4F'-3, 4H', 4W']
  │
  ├─ up_blocks.4  UNetMidBlock3D — 4 ResnetBlock3D @ 512ch
  │  res_blocks.{0..3}: conv1 [512,512,3,3,3] + conv2 [512,512,3,3,3]
  │  → [B, 512, 4F'-3, 4H', 4W']
  │
  ├─ up_blocks.5  DepthToSpaceUpsample(stride=(2,1,1))
  │  conv [512, 512, 3,3,3]  → rearrange → remove first frame
  │  → [B, 256, 8F'-7, 4H', 4W']
  │
  ├─ up_blocks.6  UNetMidBlock3D — 6 ResnetBlock3D @ 256ch
  │  res_blocks.{0..5}: conv1 [256,256,3,3,3] + conv2 [256,256,3,3,3]
  │  → [B, 256, 8F'-7, 4H', 4W']
  │
  ├─ up_blocks.7  DepthToSpaceUpsample(stride=(1,2,2))
  │  conv [512, 256, 3,3,3]  → rearrange (no frame removal: stride[0]=1)
  │  → [B, 128, 8F'-7, 8H', 8W']
  │
  ├─ up_blocks.8  UNetMidBlock3D — 4 ResnetBlock3D @ 128ch
  │  res_blocks.{0..3}: conv1 [128,128,3,3,3] + conv2 [128,128,3,3,3]
  │  → [B, 128, 8F'-7, 8H', 8W']
  │
  ├─ conv_norm_out  PixelNorm (no parameters)
  │  x = x / sqrt(mean(x², dim=C) + 1e-8)
  │
  ├─ conv_act       SiLU
  │
  ├─ conv_out       CausalConv3d(128 → 48, k=3, causal=False)
  │  weight [48, 128, 3, 3, 3]  bias [48]
  │  → [B, 48, 8F'-7, 8H', 8W']
  │
  └─ unpatchify(patch_size_hw=4, patch_size_t=1)
     "b (c p r q) f h w -> b c (f p) (h q) (w r)" with p=1, q=4, r=4
     → [B, 3, 8F'-7, 32H', 32W']
```

**Output size formula:** For latent `(F', H', W')`, decoded video is `(8(F'-1)+1, 32H', 32W')`.
Example: `(8, 64, 64)` → `(57, 2048, 2048)`.

### 1.2 Checkpoint Keys (vae.decoder.*)

Total: 42 video decoder weight tensors + 2 per-channel statistics.

```
vae.per_channel_statistics.mean-of-means              [128]        bf16
vae.per_channel_statistics.std-of-means               [128]        bf16
vae.decoder.conv_in.conv.{weight,bias}                [1024,128,3,3,3] / [1024]
vae.decoder.up_blocks.0.res_blocks.{0,1}.conv{1,2}.conv.{weight,bias}   @1024ch
vae.decoder.up_blocks.1.conv.conv.{weight,bias}       [4096,1024,3,3,3] / [4096]
vae.decoder.up_blocks.2.res_blocks.{0,1}.conv{1,2}.conv.{weight,bias}   @512ch
vae.decoder.up_blocks.3.conv.conv.{weight,bias}       [4096,512,3,3,3]  / [4096]
vae.decoder.up_blocks.4.res_blocks.{0..3}.conv{1,2}.conv.{weight,bias}  @512ch
vae.decoder.up_blocks.5.conv.conv.{weight,bias}       [512,512,3,3,3]   / [512]
vae.decoder.up_blocks.6.res_blocks.{0..5}.conv{1,2}.conv.{weight,bias}  @256ch
vae.decoder.up_blocks.7.conv.conv.{weight,bias}       [512,256,3,3,3]   / [512]
vae.decoder.up_blocks.8.res_blocks.{0..3}.conv{1,2}.conv.{weight,bias}  @128ch
vae.decoder.conv_out.conv.{weight,bias}               [48,128,3,3,3]  / [48]
```

**Important:** No timestep conditioning weights in checkpoint (`timestep_conditioning=False`
for this model). No noise injection weights. No norm weights (PixelNorm is parameter-free).
All ResBlocks have `in_channels == out_channels` (no `conv_shortcut` / `nin_shortcut`).

### 1.3 Key Layer Operations

#### CausalConv3d (causal=False mode)

For kernel_size=3 with `causal=False`:
1. Pad temporal: replicate first frame 1× at start, last frame 1× at end
2. Spatial: reflect padding, 1 pixel each side (handled by nn.Conv3d `padding_mode='reflect'`)
3. Conv3d with `padding=(0, 1, 1)` — temporal padding is 0 (handled by frame replication)
4. Output: same spatial/temporal dims as input

```
pad_start = (kernel_size - 1) // 2 = 1 frame
pad_end   = (kernel_size - 1) // 2 = 1 frame
```

**ZML implementation:** Use `Tensor.pad()` for frame replication, then `Tensor.conv3d()`
with padding on H,W only.

#### DepthToSpaceUpsample

For stride `(p1, p2, p3)`:
1. Apply CausalConv3d (pads temporally, then conv)
2. Rearrange: `(B, C*p1*p2*p3, F, H, W) → (B, C, F*p1, H*p2, W*p3)`
3. If `p1 == 2`: remove first frame → output temporal = `2*F - 1`

The rearrange is a reshape + transpose (depth-to-space / pixel-shuffle in 3D):
```
[B, C*p1*p2*p3, F, H, W]
  → reshape [B, C, p1, p2, p3, F, H, W]
  → transpose to [B, C, F, p1, H, p2, W, p3]
  → reshape [B, C, F*p1, H*p2, W*p3]
```

**Channel progression through d2s blocks:**

| Block | Stride | Conv out_ch | After d2s ch | Feature ch |
|-------|--------|-------------|-------------|------------|
| up_blocks.1 | (2,2,2) | 4096 = 8×512 | 512 | 512 (was 1024) |
| up_blocks.3 | (2,2,2) | 4096 = 8×512 | 512 | 512 (unchanged) |
| up_blocks.5 | (2,1,1) | 512 = 2×256  | 256 | 256 |
| up_blocks.7 | (1,2,2) | 512 = 4×128  | 128 | 128 |

#### ResnetBlock3D (simplified, no timestep / noise)

```
residual = x
x = PixelNorm(x)           # RMS norm over channels, dim=1
x = SiLU(x)
x = CausalConv3d(x)        # conv1 (same channels)
x = PixelNorm(x)
x = SiLU(x)
x = CausalConv3d(x)        # conv2 (same channels)
return x + residual
```

No `conv_shortcut` needed (channels don't change). No `norm3` (Identity).

#### PixelNorm

```
y = x / sqrt(mean(x², dim=channels, keepdim=True) + 1e-8)
```

Per-location RMS normalization over the channel dimension. No learnable parameters.

#### Unpatchify

```
"b (c p r q) f h w -> b c (f p) (h q) (w r)" with p=1, q=4, r=4
```

For [B, 48, F, H, W] with 48 = 3×1×4×4:
→ reshape [B, 3, 1, 4, 4, F, H, W]
→ transpose to [B, 3, F, 1, H, 4, W, 4]
→ reshape [B, 3, F, 4H, 4W]

### 1.4 Shape Trace Example

For Stage 2 output with `f_lat=8, h_lat=64, w_lat=64`:

| Layer | Shape | Size (bf16) |
|-------|-------|-------------|
| Input | [1, 128, 8, 64, 64] | 64 MB |
| conv_in | [1, 1024, 8, 64, 64] | 512 MB |
| up_blocks.0 (res×2) | [1, 1024, 8, 64, 64] | 512 MB |
| up_blocks.1 (d2s 2,2,2) | [1, 512, 15, 128, 128] | 240 MB |
| up_blocks.2 (res×2) | [1, 512, 15, 128, 128] | 240 MB |
| up_blocks.3 (d2s 2,2,2) | [1, 512, 29, 256, 256] | 1.9 GB |
| up_blocks.4 (res×4) | [1, 512, 29, 256, 256] | 1.9 GB |
| up_blocks.5 (d2s 2,1,1) | [1, 256, 57, 256, 256] | 1.9 GB |
| up_blocks.6 (res×6) | [1, 256, 57, 256, 256] | 1.9 GB |
| up_blocks.7 (d2s 1,2,2) | [1, 128, 57, 512, 512] | 3.8 GB |
| up_blocks.8 (res×4) | [1, 128, 57, 512, 512] | 3.8 GB |
| conv_out | [1, 48, 57, 512, 512] | 1.4 GB |
| unpatchify | [1, 3, 57, 2048, 2048] | 1.4 GB |

**Peak activation memory:** ~3.8 GB for the later stages. Tiling is needed for
large resolutions.

### 1.5 Memory Budget

| Component | Size (bf16) |
|-----------|-------------|
| Decoder weights | ~814 MB |
| Largest activation ([1,128,57,512,512]) | ~3.8 GB |
| Total peak (weights + 2 activations) | ~8.4 GB |

The Python reference uses spatial/temporal tiling to bound activation memory.
For the initial Zig implementation, we can start without tiling (works for smaller
latents), then add tiling as a second step.

---

## 2. Post-Processing

After the video VAE decoder produces `[B, 3, F, H, W]` in bf16 (range ≈ [-1, 1]):

```
frames_u8 = clamp((x + 1.0) / 2.0, 0.0, 1.0) * 255.0   → cast to u8
frames_u8 = rearrange(frames[0], "c f h w -> f h w c")    → [F, H, W, 3] NHWC
```

This can be done on the host after downloading the decoded tensor.

---

## 3. Zig Implementation Plan

### 3.1 Param Structs (in model.zig)

```zig
// Reuse existing Conv3dWeight from Section 10

pub const VaeResBlock = struct {
    conv1: Conv3dWeight,  // checkpoint: conv1.conv.{weight,bias}
    conv2: Conv3dWeight,  // checkpoint: conv2.conv.{weight,bias}
};

pub const VaeDepthToSpaceBlock = struct {
    conv: Conv3dWeight,   // checkpoint: conv.conv.{weight,bias}
};

pub const VideoVaeDecoderParams = struct {
    conv_in: Conv3dWeight,

    // up_blocks.0: 2 ResBlocks @ 1024
    up0_res0: VaeResBlock,
    up0_res1: VaeResBlock,

    // up_blocks.1: DepthToSpace (2,2,2)
    up1: VaeDepthToSpaceBlock,

    // up_blocks.2: 2 ResBlocks @ 512
    up2_res0: VaeResBlock,
    up2_res1: VaeResBlock,

    // up_blocks.3: DepthToSpace (2,2,2)
    up3: VaeDepthToSpaceBlock,

    // up_blocks.4: 4 ResBlocks @ 512
    up4_res0: VaeResBlock,
    up4_res1: VaeResBlock,
    up4_res2: VaeResBlock,
    up4_res3: VaeResBlock,

    // up_blocks.5: DepthToSpace (2,1,1)
    up5: VaeDepthToSpaceBlock,

    // up_blocks.6: 6 ResBlocks @ 256
    up6_res0: VaeResBlock,
    up6_res1: VaeResBlock,
    up6_res2: VaeResBlock,
    up6_res3: VaeResBlock,
    up6_res4: VaeResBlock,
    up6_res5: VaeResBlock,

    // up_blocks.7: DepthToSpace (1,2,2)
    up7: VaeDepthToSpaceBlock,

    // up_blocks.8: 4 ResBlocks @ 128
    up8_res0: VaeResBlock,
    up8_res1: VaeResBlock,
    up8_res2: VaeResBlock,
    up8_res3: VaeResBlock,

    conv_out: Conv3dWeight,
};
```

### 3.2 Weight Loading

```zig
pub fn initVideoVaeDecoderParams(store: zml.io.TensorStore.View) VideoVaeDecoderParams {
    const dec = store.withPrefix("vae").withPrefix("decoder");
    return .{
        .conv_in = initConv3d(dec.withPrefix("conv_in").withPrefix("conv")),
        .up0_res0 = initVaeResBlock(dec.withPrefix("up_blocks").withLayer(0)
                        .withPrefix("res_blocks").withLayer(0)),
        // ... etc
        .conv_out = initConv3d(dec.withPrefix("conv_out").withPrefix("conv")),
    };
}
```

Note the extra `.withPrefix("conv")` nesting — checkpoint keys are
`vae.decoder.conv_in.conv.weight`, not `vae.decoder.conv_in.weight`.

### 3.3 Forward Functions

```zig
/// CausalConv3d forward (causal=False): pad first/last frame, reflect-pad spatial, conv3d.
fn forwardCausalConv3dNonCausal(input: Tensor, w: Conv3dWeight) Tensor {
    // 1. Temporal padding: replicate first and last frame
    const first = input.slice1d(2, 0, 1);  // dim=2 (temporal), [0:1]
    const last = input.slice1d(2, input.dim(2) - 1, input.dim(2));
    const padded = first.concat(2, input).concat(2, last); // [B,C,F+2,H,W]

    // 2. Conv3d with spatial padding only (reflect)
    const conv_out = padded.conv3d(w.weight, .{
        .padding = &.{ 0, 0, 1, 1, 1, 1 }, // no temporal, pad H and W
    });
    // 3. Add bias
    return conv_out.add(w.bias.reshape(...).broad(conv_out.shape()));
}

/// PixelNorm: x / sqrt(mean(x², dim=1) + eps)
fn forwardPixelNorm(x: Tensor) Tensor {
    const x_f32 = x.convert(.f32);
    const x_sq = x_f32.mul(x_f32);
    const mean_sq = x_sq.mean(1);  // mean over channels
    const rms = mean_sq.addConstant(1e-8).sqrt();
    return x_f32.div(rms.broadcastLeft(x_f32.shape())).convert(x.dtype());
}

/// VAE ResBlock: PixelNorm → SiLU → Conv → PixelNorm → SiLU → Conv + residual
fn forwardVaeResBlock(x: Tensor, rb: VaeResBlock) Tensor {
    var h = forwardPixelNorm(x);
    h = h.silu();
    h = forwardCausalConv3dNonCausal(h, rb.conv1);
    h = forwardPixelNorm(h);
    h = h.silu();
    h = forwardCausalConv3dNonCausal(h, rb.conv2);
    return h.add(x);  // residual
}

/// DepthToSpace 3D: conv → rearrange → optionally remove first frame
fn forwardDepthToSpace(
    x: Tensor,
    w: VaeDepthToSpaceBlock,
    comptime stride: [3]i64,
) Tensor {
    var h = forwardCausalConv3dNonCausal(x, w.conv);
    // Rearrange: [B, C*p1*p2*p3, F, H, W] → [B, C, F*p1, H*p2, W*p3]
    const B = h.dim(0);
    const C_total = h.dim(1);
    const F = h.dim(2);
    const H = h.dim(3);
    const W = h.dim(4);
    const p1 = stride[0]; const p2 = stride[1]; const p3 = stride[2];
    const C = @divExact(C_total, p1 * p2 * p3);

    h = h.reshape(.{ B, C, p1, p2, p3, F, H, W });
    h = h.transpose(.{ 0, 1, 5, 2, 6, 3, 7, 4 }); // [B,C,F,p1,H,p2,W,p3]
    h = h.reshape(.{ B, C, F * p1, H * p2, W * p3 });

    // Remove first frame if temporal upsample
    if (p1 == 2) {
        h = h.slice1d(2, 1, h.dim(2)); // remove frame 0
    }
    return h;
}

/// Full video VAE decoder forward
pub fn forwardVideoVaeDecode(
    latent: Tensor,
    stats: PerChannelStats,
    params: VideoVaeDecoderParams,
) Tensor {
    // 1. Denormalize
    const stats_shape = latent.shape().set(0,1).set(2,1).set(3,1).set(4,1);
    var x = latent
        .mul(stats.std_of_means.reshape(stats_shape).broad(latent.shape()))
        .add(stats.mean_of_means.reshape(stats_shape).broad(latent.shape()));

    // 2. conv_in
    x = forwardCausalConv3dNonCausal(x, params.conv_in);

    // 3. up_blocks.0: 2 ResBlocks @ 1024
    x = forwardVaeResBlock(x, params.up0_res0);
    x = forwardVaeResBlock(x, params.up0_res1);

    // 4. up_blocks.1: DepthToSpace (2,2,2)
    x = forwardDepthToSpace(x, params.up1, .{2, 2, 2});

    // 5. up_blocks.2: 2 ResBlocks @ 512
    x = forwardVaeResBlock(x, params.up2_res0);
    x = forwardVaeResBlock(x, params.up2_res1);

    // 6. up_blocks.3: DepthToSpace (2,2,2)
    x = forwardDepthToSpace(x, params.up3, .{2, 2, 2});

    // ... up_blocks.4-8 follow same pattern ...

    // N-2. PixelNorm → SiLU
    x = forwardPixelNorm(x);
    x = x.silu();

    // N-1. conv_out
    x = forwardCausalConv3dNonCausal(x, params.conv_out);

    // N. Unpatchify: [B, 48, F, H, W] → [B, 3, F, 4H, 4W]
    x = forwardUnpatchifyVae(x);

    return x;
}
```

### 3.4 Compilation & Integration (inference.zig)

After Stage 2 produces `s2.v_latent` (patchified `[1, T_v2, 128]` bf16):

```zig
// Phase 4: Video VAE Decode
std.log.info("=== Phase 4: Video VAE Decode ===", .{});

// 4a. Unpatchify latent: [1, T_v2, 128] → [1, 128, F', H', W']
const v_latent_5d = forwardUnpatchifyVideo(s2.v_latent, target_5d_shape);

// 4b. Compile VAE decoder
var vae_params = model.initVideoVaeDecoderParams(ckpt_store.view());
var vae_bufs = try zml.io.load(
    model.VideoVaeDecoderParams,
    &vae_params, allocator, io, platform,
    .{ .store = &ckpt_store, .parallelism = 4 },
);

var vae_exe = try platform.compileFn(
    allocator, io,
    model.forwardVideoVaeDecode,
    .{
        zml.Tensor.fromShape(v_latent_5d.shape()),
        per_channel_stats_shape,
        vae_params,
    },
    .{},
);

// 4c. Execute
var vae_args = try vae_exe.args(allocator);
vae_args.set(.{ v_latent_5d, per_channel_stats_bufs, vae_bufs });
vae_exe.call(vae_args, &vae_results);
const decoded_video = vae_results.get(zml.Buffer);  // [1, 3, F, H, W]

// 4d. Post-process to uint8 frames and write
// download tensor, (x+1)/2 * 255, cast to u8, write as raw frames
```

---

## 4. Audio VAE Decoder Architecture (Phase 2)

### 4.1 Architecture

The audio VAE decoder is a 2D convolutional network (causal on the height/frequency axis):

```
Input: [B, 8, T, 16]  (8 latent channels, 16 frequency bins after unpatchify)
  ├─ per_channel_statistics.un_normalize         [128] mean/std
  ├─ conv_in: Conv2d(8 → 512, k=3)
  ├─ mid.block_1: ResBlock(512)     conv1 + conv2
  ├─ mid.block_2: ResBlock(512)     conv1 + conv2
  ├─ up.2: 3× ResBlock(512) + upsample Conv2d(512, k=3, stride=2)
  ├─ up.1: 3× ResBlock(256, first has nin_shortcut 512→256) + upsample Conv2d(256)
  ├─ up.0: 3× ResBlock(128, first has nin_shortcut 256→128)
  ├─ conv_out: Conv2d(128 → 2, k=3)
  └→ [B, 2, T_out, F_out]  (mel spectrogram, stereo)
```

### 4.2 Vocoder (Mel → Waveform)

BigVGAN-style with SnakeBeta activations:
- conv_pre (1D) → 5× TransposedConv1d (rates [6,5,2,2,2] = 240× total) →
  AMPBlock1 residual blocks → conv_post
- Bandwidth extension (BWE): runs vocoder output through a second generator
  to upsample 24kHz → 48kHz

The vocoder has ~470 weight tensors. This is substantial and should be a
separate implementation phase.

---

## 5. Implementation Milestones

### M1: Video VAE Decoder + Raw Frame Output ✅
- [x] Add `VideoVaeDecoderParams` and `initVideoVaeDecoderParams` to model.zig
- [x] Implement `forwardCausalConv3dNonCausal`, `forwardPixelNorm`, `forwardVaeResBlock`,
      `forwardDepthToSpace`, `forwardUnpatchifyVae`, `forwardVideoVaeDecode`
- [x] Add Phase 4 to inference.zig: compile VAE decoder, execute
- [x] Post-process decoded video to uint8 NHWC frames on host
- [x] Write raw frame data to `frames.bin` (flat `[F, H, W, 3]` uint8)
- [x] Validate: compare decoded frames against Python reference — **PSNR 56.4 dB**

### M2: ffmpeg Pipe for MP4 Output ✅
- [x] Spawn ffmpeg as a child process from Zig (`std.process.spawn`)
- [x] Pipe decoded uint8 frames to ffmpeg stdin (frames.bin kept for debugging)
- [ ] Add audio track once M4 is complete (pipe WAV to ffmpeg alongside video)
- [x] Single command produces `output.mp4` directly (121 frames → 2 MB MP4)

### M3: Audio VAE Decoder ✅
- [x] Add `AudioVaeDecoderParams`, `AudioVaeResBlock`, `AudioVaeUpsample`, `AudioPerChannelStats` to model.zig
- [x] Implement `forwardCausalConv2dHeight`, `forwardCausalConv2d1x1`, `forwardAudioVaeResBlock`,
      `forwardPixelNorm2d`, `forwardNearest2x`, `forwardAudioVaeUpsample`,
      `forwardUnpatchifyAudio`, `forwardAudioDenormalize`, `forwardAudioVaeDecode`
- [x] Add Phase 5 to inference.zig: `runAudioVaeDecode`
- [x] Standalone validation driver: `audio_vae_decode.zig` + BUILD target
- [x] Python reference export: `e2e/export_audio_vae_activations.py`
- [x] PSNR comparison: `e2e/compare_audio_vae.py`
- [x] Validate against Python reference — **PSNR 56.75 dB**

### M4: Vocoder + BWE 🔧 (in progress)

**Architecture**: Two-stage pipeline, split into two compiled functions to stay under MLIR's 1024-arg limit:
- **Stage 1 — Main Vocoder** (`forwardMainVocoder`, 668 MLIR args): mel [1,2,8,64] bf16 → waveform [1,2,1280] f32 at 16kHz
- **Stage 2 — BWE Pipeline** (`forwardBWEPipeline`, 560 MLIR args): waveform_16k [1,2,1280] f32 → waveform_48k [1,2,3840] f32

**Implementation completed:**
- [x] `MainVocoderParams` (667 tensors) + `initMainVocoderParams` in model.zig
- [x] `BWEPipelineParams` (559 tensors) + `initBWEPipelineParams` in model.zig
- [x] BigVGAN vocoder: `forwardVocoderGeneric` with SnakeBeta activations, AMPBlock1 residual blocks
- [x] Transposed convolution: `forwardVocConvTranspose1d` (via dilated conv1d with explicit kernel flip)
- [x] Anti-aliased activation: `forwardActivation1d` → `forwardUpSample1d` / `forwardDownSample1d`
- [x] BWE mel computation: `forwardSTFT`, `forwardMelProjection`, `forwardComputeMel`
- [x] BWE sinc resampler: `forwardSincResample3x` (3× upsample with Kaiser window)
- [x] `zml/mem.zig`: added `.array` case to `bufferizeInner` for `[18]AMPBlock1Params` / `[15]AMPBlock1Params`
- [x] Standalone validation driver: `vocoder_decode.zig` + BUILD target
- [x] Python reference export: `e2e/export_vocoder_activations.py`, `e2e/export_vocoder_stages.py`
- [x] PSNR comparison: `e2e/compare_vocoder.py`

**Bug found and fixed — `window_reversal` in ZML:**
- ZML's `window_reversal = true` in `conv1d` does NOT flip the kernel as expected
- Workaround: explicit `.reverse(.{2})` on kernel before conv1d (no `window_reversal`)
- Applied to: `forwardVocConvTranspose1d`, `forwardUpSample1d`, `forwardSincResample3x`
- See [zml_window_reversal_bug.md](zml_window_reversal_bug.md) for details

**Validation status:**
- [x] Stage 1 (Main Vocoder): **PSNR 63.74 dB** ✅ (max err 0.004, mean err 0.001)
- [ ] Stage 2 (BWE Pipeline): **PSNR 19.30 dB** ❌ (max err 0.97, mean err 0.14)
- [ ] Write WAV output
- [ ] End-to-end audio pipeline integration

**Next: Debug BWE Stage 2** — see [m4_vocoder_bwe_investigation.md](m4_vocoder_bwe_investigation.md) for full investigation plan. Most likely suspect: `forwardComputeMel` (STFT + mel projection), which is unique to BWE and untested by Stage 1.

### M5: Tiling (Memory Optimization)
- [ ] Implement spatial/temporal tiling for video VAE decode
- [ ] Enable decode of large resolutions (>512×512) within GPU memory

---

## 6. Key Differences from Initial Analysis

The checkpoint reveals several differences from the simplified Python config description:

1. **9 up_blocks (0-8)**, not 4 — the architecture interleaves ResBlock groups and
   DepthToSpaceUpsample blocks
2. **up_blocks.3 has no channel reduction** — conv [4096, 512, 3,3,3] produces
   4096/8=512 channels (same as input), unlike the other d2s blocks
3. **No timestep conditioning** in this checkpoint — no `timestep_scale_multiplier`,
   `last_time_embedder`, or `last_scale_shift_table` keys
4. **No noise injection** — no `per_channel_scale` weights in any ResBlock
5. **Extra `.conv.` in key path** — keys are `conv_in.conv.weight` not `conv_in.weight`
6. **Variable ResBlock counts** — 2, 2, 4, 6, 4 ResBlocks across the 5 groups

---

## 7. Validation Strategy

### 7.1 Python Activation Export Script

Create `examples/ltx/e2e/export_vae_activations.py` following the same pattern as
`export_mixed_pipeline.py`. The script:

1. Loads the VAE decoder from the distilled checkpoint
2. Loads a video latent (either from a previous Zig run's `video_latent.bin`, or
   generates a small random latent for unit testing)
3. Runs the decoder with hooks to capture intermediates
4. Saves all activations as safetensors

**Captured checkpoints:**

| Key | Shape | After |
|-----|-------|-------|
| `input_latent` | [1, 128, F', H', W'] | unpatchify (5D input to decoder) |
| `after_denorm` | [1, 128, F', H', W'] | per_channel_statistics.un_normalize |
| `after_conv_in` | [1, 1024, F', H', W'] | conv_in |
| `after_up0` | [1, 1024, F', H', W'] | up_blocks.0 (2 ResBlocks) |
| `after_up1` | [1, 512, ...] | up_blocks.1 (DepthToSpace 2,2,2) |
| `after_up2` | [1, 512, ...] | up_blocks.2 (2 ResBlocks) |
| `after_up3` | [1, 512, ...] | up_blocks.3 (DepthToSpace 2,2,2) |
| `after_up4` | [1, 512, ...] | up_blocks.4 (4 ResBlocks) |
| `after_up5` | [1, 256, ...] | up_blocks.5 (DepthToSpace 2,1,1) |
| `after_up6` | [1, 256, ...] | up_blocks.6 (6 ResBlocks) |
| `after_up7` | [1, 128, ...] | up_blocks.7 (DepthToSpace 1,2,2) |
| `after_up8` | [1, 128, ...] | up_blocks.8 (4 ResBlocks) |
| `after_norm_silu` | [1, 128, ...] | conv_norm_out + SiLU |
| `after_conv_out` | [1, 48, ...] | conv_out |
| `output` | [1, 3, F, H, W] | unpatchify (final decoded video) |

**Output:**
```
{out}/ref/vae_activations.safetensors  — all intermediates
```

### 7.2 Validation Approach

**Bottom-up** (recommended order):

1. **PixelNorm**: small random tensor, compare against `x / sqrt(mean(x², dim=1) + eps)` in Python
2. **CausalConv3d (non-causal)**: load `conv_in` weights, feed `after_denorm`, compare against `after_conv_in`
3. **VaeResBlock**: load `up_blocks.0.res_blocks.0` weights, feed `after_conv_in`, compare after first ResBlock
4. **DepthToSpace**: load `up_blocks.1` weights, feed `after_up0`, compare against `after_up1`
5. **Full decoder**: feed `input_latent`, compare `output`

**Error thresholds:**
- Per-layer max abs error < 0.01 (bf16 precision)
- Final output PSNR > 40 dB
- Use the same `save_safetensors` / load pattern as the existing denoising validation

### 7.3 Test Latent Size

For fast iteration, use a **small latent** (e.g., `F'=2, H'=4, W'=4`) which decoded
produces `[1, 3, 9, 128, 128]` — small enough to run without tiling. The full-size
decode (F'=8, H'=64, W'=64) can be validated once the small case passes.

---

## 8. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Reflect padding not available in ZML conv3d | High | Use explicit pad + zero-pad conv, or replicate-pad (close enough) |
| Memory for large latents | Medium | Implement tiling in M5; start with small test cases |
| PixelNorm precision (bf16 squaring) | Medium | Compute in f32 as shown in plan |
| 3D conv performance on CUDA | Low | XLA/PJRT handles this; may need tuning later |

---

## 9. M1 Implementation Results

### 9.1 Status: VALIDATED

M1 (Video VAE Decoder + raw frames output) is complete and validated at multiple scales.

### 9.2 Files Created / Modified

| File | Description |
|------|-------------|
| `examples/ltx/model.zig` (Section 11) | ~250 lines: `VideoVaeDecoderParams`, `VaeResBlock`, `VaeDepthToSpaceBlock`, init functions, all forward ops, `forwardVideoVaeDecode` (pub) |
| `examples/ltx/inference.zig` (Phase 4) | `runVideoVaeDecode` — unpatchify → load weights → compile → run → post-process → write `frames.bin` |
| `examples/ltx/vae_decode.zig` | Standalone validation driver: loads input latent from safetensors, runs decoder, writes `decoded_video.bin` (bf16) + `frames.bin` (u8 RGB) |
| `examples/ltx/BUILD.bazel` | Added `vae_decode` target, included in `build_test` |
| `examples/ltx/e2e/export_vae_activations.py` | Python reference: exports all intermediates as safetensors. Supports `--small-test`, `--medium-test`, `--output-only` flags |
| `examples/ltx/e2e/compare_vae_outputs.py` | Compares Zig bf16 output against Python reference. Reports max_abs, mean_abs, PSNR(±1), PSNR(u8) |
| `examples/ltx/e2e/diagnose_vae_intermediates.py` | Step-by-step Python replay with different padding/norm variants for debugging |
| `examples/ltx/e2e/decode_latents.py` | Fixed to accept `--meta pipeline_meta.json` when safetensors metadata is absent |

### 9.3 Zig Implementation Details

**New types** in `model.zig`:
- `VaeResBlock` — two `Conv3dWeight` (conv1, conv2)
- `VaeDepthToSpaceBlock` — one `Conv3dWeight` (conv)
- `VideoVaeDecoderParams` — 22 ResBlocks + 4 DepthToSpace blocks + conv_in/conv_out

**Forward functions:**
- `forwardCausalConv3dNonCausal` — temporal replicate-pad (duplicate first/last frame) + conv3d with zero spatial padding
- `forwardPixelNorm` — f32 computation: `x / sqrt(mean(x², dim=channels) + 1e-8)`
- `forwardVaeResBlock` — PixelNorm → SiLU → Conv → PixelNorm → SiLU → Conv + residual
- `forwardDepthToSpace(x, w, comptime stride)` — conv + reshape + transpose + reshape + optional first-frame removal
- `forwardUnpatchifyVae` — 48→3 channels via reshape + transpose (c=3, p=1, q=4, r=4)
- `forwardVideoVaeDecode` (pub) — full forward: denorm → conv_in → 9 up_blocks → PixelNorm → SiLU → conv_out → unpatchify

**Weight loading:**
- Checkpoint keys have double `.conv.` nesting: `vae.decoder.conv_in.conv.weight`
- DepthToSpace blocks have triple nesting: `vae.decoder.up_blocks.1.conv.conv.weight` (addressed with `.withPrefix("conv")`)
- Reuses `Conv3dWeight`, `PerChannelStats`, and `initPerChannelStats` from Section 10

### 9.4 Bugs Found and Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| **Unpatchify transpose** | einops `(c p r q)` maps q→H, r→W. Transpose `.{0,1,5,2,6,3,7,4}` swapped r↔q | Changed to `.{0,1,5,2,6,4,7,3}` |
| **DepthToSpace key paths** | Checkpoint keys have `up_blocks.1.conv.conv.weight` (double `.conv.`) | Added `.withPrefix("conv")` to all 4 D2S init calls |
| **Spatial padding confusion** | Initially added reflect padding; Python CausalConv3d actually uses Conv3d built-in zero-padding for spatial dims | Reverted to `conv3d(.padding = &.{0,0,1,1,1,1})` |

### 9.5 Validation Matrix

| Test | Input Shape | Output Shape | PSNR (u8) | Status |
|------|------------|-------------|-----------|--------|
| Small (random, seed=42) | `[1, 128, 2, 4, 4]` | `[1, 3, 9, 128, 128]` | **56.4 dB** | PASS |
| Medium (random, seed=42) | `[1, 128, 8, 16, 16]` | `[1, 3, 57, 512, 512]` | **55.4 dB** | PASS |
| Full pipeline latent | `[1, 128, 16, 32, 48]` | `[1, 3, 121, 1024, 1536]` | visual | PASS |

- **Threshold:** 40 dB PSNR on u8 scale
- **Full-size numerical comparison** not possible: Python decoder OOMs on 80GB GPU at this resolution (PyTorch Conv3d workspace exceeds available memory)
- **Full-size visual check:** Zig-decoded `output.mp4` shows coherent 1024×1536 video at 121 frames, matching expected content ("someone walking by the beach at sunset")
- **Cross-decode validation:** Zig-produced latents decoded by Python decoders (video + audio VAE + vocoder) produce correct MP4 with sound (`output_final_decode_python.mp4`)

### 9.6 Diagnostic Findings

- **PixelNorm as error dampener:** Accumulated bf16 error through 22 ResBlocks grows to max_abs=6.0 at up_block 8, but PixelNorm + SiLU before conv_out normalizes it back (64.7 dB after norm+silu)
- **Zero-pad vs reflect-pad:** Python CausalConv3d with `causal=False` uses `Conv3d(padding=(1,1,1))` which is zero-padding. PyTorch `F.pad(mode='reflect')` for 5D tensors requires `pad_size=6` (not 4), so spatial reflect-pad is not trivially applicable
- **PixelNorm precision:** Both f32 eps=1e-8 (Zig) and native bf16 eps=1e-6 (Python) produce acceptable results, with f32 being slightly less accurate at the per-layer level (50.3 vs 54.0 dB) but both converge after PixelNorm normalization

### 9.7 Remote Testing Commands

```bash
# Phase A: Python reference (small test)
cd /root/repos/LTX-2
uv run /root/repos/zml/examples/ltx/e2e/export_vae_activations.py \
    --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    --small-test --output-dir /root/e2e_demo/vae_ref_small/

# Phase B: Zig decoder
cd /root/repos/zml
bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:vae_decode -- \
    /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    /root/e2e_demo/vae_ref_small/vae_activations.safetensors \
    /root/e2e_demo/vae_zig_small/

# Phase C: Compare
cd /root/repos/LTX-2
uv run /root/repos/zml/examples/ltx/e2e/compare_vae_outputs.py \
    --ref /root/e2e_demo/vae_ref_small/vae_activations.safetensors \
    --zig /root/e2e_demo/vae_zig_small/decoded_video.bin

# Phase D: Full pipeline end-to-end (video only, no audio)
cd /root/e2e_demo/vae_zig_full/
ffmpeg -f rawvideo -pix_fmt rgb24 -s 1536x1024 -r 24 \
    -i frames.bin -c:v libx264 -pix_fmt yuv420p output.mp4

# Phase E: Decode Zig latents with Python (video + audio → MP4)
cd /root/repos/LTX-2
uv run /root/repos/zml/examples/ltx/e2e/decode_latents.py \
    --meta /root/e2e_demo/unified_pipeline/pipeline_meta.json \
    --video-latent /root/e2e_demo/unified_pipeline/unified/video_latent.bin \
    --audio-latent /root/e2e_demo/unified_pipeline/unified/audio_latent.bin \
    --output /root/e2e_demo/unified_pipeline/unified/output_final.mp4
```

### 9.8 Next Steps

- [x] M1: Video VAE Decoder + raw frames output
- [x] M2: ffmpeg pipe for MP4 output (write frames directly to ffmpeg stdin)
- [x] M3: Audio VAE Decoder
- [ ] M4: Vocoder + bandwidth extension
- [ ] M5: Tiling for large resolutions

---

## 10. M3 Implementation Results (Audio VAE Decoder)

### 10.1 Status: VALIDATED

M3 (Audio VAE Decoder) is complete and validated at three scales (small, medium, full pipeline latent).

### 10.2 Files Created / Modified

| File | Description |
|------|-------------|
| `examples/ltx/model.zig` (Section 12) | ~280 lines: `AudioVaeDecoderParams`, `AudioVaeResBlock`, `AudioVaeUpsample`, `AudioPerChannelStats`, init functions, all forward ops, `forwardAudioVaeDecode` (pub) |
| `examples/ltx/inference.zig` (Phase 5) | `runAudioVaeDecode` — unpatchify → load weights → compile → run → write `audio_mel.bin` |
| `examples/ltx/audio_vae_decode.zig` | Standalone validation driver: loads 4D input latent from safetensors, runs decoder, writes `decoded_audio.bin` (bf16) |
| `examples/ltx/BUILD.bazel` | Added `audio_vae_decode` target, included in `build_test` |
| `examples/ltx/e2e/export_audio_vae_activations.py` | Python reference: exports input, decoded output, and per-channel stats as safetensors. Supports `--small-test`, `--medium-test`, `--audio-latent`, `--seed` flags |
| `examples/ltx/e2e/compare_audio_vae.py` | Compares Zig bf16 output against Python reference. Reports per-channel PSNR, max/mean abs error |

### 10.3 Architecture Summary

The audio VAE is a 2D convolutional decoder with **HEIGHT causality** (causal padding on the time axis, symmetric on frequency):

```
Input: [B, 8, T, 16]    (8 latent channels, T time steps, 16 frequency bins)
  ├─ Denormalize (patchify → denorm with [128] stats → unpatchify)
  ├─ conv_in: CausalConv2d(8 → 512, k=3)   padding=(2,0,1,1)
  ├─ mid.block_1: ResBlock(512)   PixelNorm → SiLU → Conv → PixelNorm → SiLU → Conv + residual
  ├─ mid.block_2: ResBlock(512)
  ├─ up.2: 3× ResBlock(512) + Upsample(nearest 2× + Conv + drop first row)
  ├─ up.1: 3× ResBlock(first: 512→256 with nin_shortcut, then 256) + Upsample
  ├─ up.0: 3× ResBlock(first: 256→128 with nin_shortcut, then 128)
  ├─ PixelNorm → SiLU
  ├─ conv_out: CausalConv2d(128 → 2, k=3)
  └→ [B, 2, T_out, 64]   (mel spectrogram, stereo)
```

**Output size formula:** T_out = 4×T - 3 (two 2× upsamples with drop-first-row).
Example: T=126 → T_out=501.

### 10.4 Key Design Decisions

| Decision | Rationale |
|----------|----------|
| **HEIGHT causality** (not DEPTH) | Audio VAE uses 2D convolutions; the "time" axis is dim 2 (height in NCHW), causal padding = `(k-1, 0)` on height, symmetric `(1, 1)` on width |
| **PixelNorm2d** (parameter-free) | Same as video VAE — no GroupNorm, no learnable scale/bias. Computed in f32 for precision |
| **Nearest-neighbor upsample** | Implemented as reshape + broadcast (no interpolation kernel): `[B,C,H,1,W,1]` → broadcast to `[B,C,H,2,W,2]` → reshape |
| **Upsample drop-first-row** | After nearest 2× + CausalConv2d, drop row 0 (`slice1d(2, .{.start=1})`) to maintain causal alignment |
| **nin_shortcut** (1×1 conv) | Used only in up.0.block.0 (256→128) and up.1.block.0 (512→256) where channels change |
| **Separate per-channel stats** | Audio uses `audio_vae.per_channel_statistics.*` (not `vae.per_channel_statistics.*`) |
| **Denormalize in patchified space** | Patchify `[B,8,T,16]` → `[B,T,128]`, mul/add with `[128]` stats, then unpatchify back |

### 10.5 Checkpoint Key Mapping

Total: 56 audio decoder weight tensors + 2 per-channel statistics.

```
audio_vae.per_channel_statistics.mean-of-means          [128]   bf16
audio_vae.per_channel_statistics.std-of-means           [128]   bf16
audio_vae.decoder.conv_in.conv.{weight,bias}            [512,8,3,3] / [512]
audio_vae.decoder.mid.block_{1,2}.conv{1,2}.conv.{weight,bias}    @512ch
audio_vae.decoder.up.2.block.{0,1,2}.conv{1,2}.conv.{weight,bias} @512ch
audio_vae.decoder.up.2.upsample.conv.conv.{weight,bias}           @512ch
audio_vae.decoder.up.1.block.0.{conv1,conv2,nin_shortcut}.conv.{weight,bias}  512→256
audio_vae.decoder.up.1.block.{1,2}.conv{1,2}.conv.{weight,bias}   @256ch
audio_vae.decoder.up.1.upsample.conv.conv.{weight,bias}           @256ch
audio_vae.decoder.up.0.block.0.{conv1,conv2,nin_shortcut}.conv.{weight,bias}  256→128
audio_vae.decoder.up.0.block.{1,2}.conv{1,2}.conv.{weight,bias}   @128ch
audio_vae.decoder.conv_out.conv.{weight,bias}           [2,128,3,3] / [2]
```

**Note:** Upsample keys have double `.conv.conv.` nesting: `up.2.upsample.conv.conv.weight`
(the outer `conv` is the CausalConv2d wrapper, the inner is the nn.Conv2d).

### 10.6 Bugs Found and Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| **`reduceSum` doesn't exist** | ZML has `.sum(axis)` (keeps dims, returns `[B,1,H,W]`), not `.reduceSum()` | Changed to `.sum(1).divByConst(c)` |
| **`std` variable shadows import** | Named local variable `std` in `forwardAudioDenormalize` | Renamed to `std_val` |
| **Upsample weight paths** | Checkpoint has `upsample.conv.conv.weight` (double `.conv.`); init only added one | Added `.withPrefix("conv")` to upsample init calls |
| **`slice1d` API** | Takes `(axis, Slice)` not `(axis, start, end)` | Changed to `.slice1d(2, .{ .start = 1 })` |
| **Non-contiguous tensor save** | `rearrange` in export script produces non-contiguous tensor | Added `.contiguous()` before `save_file` |
| **NumPy bf16 unsupported** | `safetensors` returns bf16 which numpy doesn't understand | Read raw bytes from file + manual bf16→f32 conversion |

### 10.7 Validation Matrix

| Test | Input Shape | Output Shape | PSNR | Ch0 | Ch1 | Max Err | Status |
|------|------------|-------------|------|-----|-----|---------|--------|
| Small (random, seed=42) | `[1, 8, 4, 16]` | `[1, 2, 13, 64]` | **55.63 dB** | 54.38 | 56.47 | 0.0625 | PASS |
| Medium (random, seed=42) | `[1, 8, 32, 16]` | `[1, 2, 125, 64]` | **56.95 dB** | 55.62 | 57.99 | 0.0625 | PASS |
| Full pipeline latent | `[1, 8, 126, 16]` | `[1, 2, 501, 64]` | **56.75 dB** | 55.61 | 58.30 | 0.0625 | PASS |

- **Threshold:** 40 dB PSNR
- **Max absolute error** is exactly 0.0625 across all sizes — one bf16 ULP at the relevant magnitude range
- **Channel 1** consistently ~2 dB better than Channel 0
- **Output range:** approximately `[-10, -0.5]` (log-scale mel spectrogram values, all negative)

### 10.8 Remote Testing Commands

```bash
# --- Small test ---
cd /root/repos/LTX-2
uv run /root/repos/zml/examples/ltx/e2e/export_audio_vae_activations.py \
    --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    --small-test --output-dir /root/e2e_demo/audio_vae_ref_small/

cd /root/repos/zml
mkdir -p /root/e2e_demo/audio_vae_zig_small
bazel-bin/examples/ltx/audio_vae_decode \
    /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    /root/e2e_demo/audio_vae_ref_small/audio_vae_activations.safetensors \
    /root/e2e_demo/audio_vae_zig_small/

cd /root/repos/LTX-2
uv run /root/repos/zml/examples/ltx/e2e/compare_audio_vae.py \
    --ref /root/e2e_demo/audio_vae_ref_small/audio_vae_activations.safetensors \
    --zig /root/e2e_demo/audio_vae_zig_small/decoded_audio.bin

# --- Medium test ---
uv run /root/repos/zml/examples/ltx/e2e/export_audio_vae_activations.py \
    --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    --medium-test --output-dir /root/e2e_demo/audio_vae_ref_medium/

mkdir -p /root/e2e_demo/audio_vae_zig_medium
bazel-bin/examples/ltx/audio_vae_decode \
    /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    /root/e2e_demo/audio_vae_ref_medium/audio_vae_activations.safetensors \
    /root/e2e_demo/audio_vae_zig_medium/

uv run /root/repos/zml/examples/ltx/e2e/compare_audio_vae.py \
    --ref /root/e2e_demo/audio_vae_ref_medium/audio_vae_activations.safetensors \
    --zig /root/e2e_demo/audio_vae_zig_medium/decoded_audio.bin

# --- Full pipeline latent ---
uv run /root/repos/zml/examples/ltx/e2e/export_audio_vae_activations.py \
    --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    --audio-latent /root/e2e_demo/unified_pipeline/unified/audio_latent.bin \
    --meta /root/e2e_demo/unified_pipeline/pipeline_meta.json \
    --output-dir /root/e2e_demo/audio_vae_ref/

mkdir -p /root/e2e_demo/audio_vae_zig_out
bazel-bin/examples/ltx/audio_vae_decode \
    /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    /root/e2e_demo/audio_vae_ref/audio_vae_activations.safetensors \
    /root/e2e_demo/audio_vae_zig_out/

uv run /root/repos/zml/examples/ltx/e2e/compare_audio_vae.py \
    --ref /root/e2e_demo/audio_vae_ref/audio_vae_activations.safetensors \
    --zig /root/e2e_demo/audio_vae_zig_out/decoded_audio.bin
```
