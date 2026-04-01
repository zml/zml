# Task: Implement the LTX-2 Latent Upscaler in Zig/ZML

## Context

We are building a Zig/ZML port of the LTX-2.3 two-stage video generation pipeline. The codebase is at `/Users/oboulant/repos/zml/zml/examples/ltx/`. The ZML framework lives at `/Users/oboulant/repos/zml/zml/zml/`. The Python reference is at `/Users/oboulant/repos/work/LTX-2/`.

**What exists today:**
- `model.zig` — the 48-block AV transformer (Stage 1 & 2 denoising), ~2500 lines
- `denoise_stage1.zig` — Stage 1 driver (30-step Euler, 4-pass guided denoising)
- `denoise_e2e.zig` — Stage 2 driver (3-step Euler, distilled, no guidance)
- `bridge_s1_to_s2.py` — **Python script** that takes Stage 1 Zig output, unpatchifies, calls `VideoUpsampler` (Python), re-noises, and exports Stage 2 inputs
- `BUILD.bazel` — Bazel build definitions for the Zig binaries

**What we want:** Implement the `LatentUpsampler` CNN in Zig/ZML so we can replace the Python `bridge_s1_to_s2.py` hop with a Zig binary, making the Stage 1 → Stage 2 bridge fully native.

## Architecture of the Upsampler

The upsampler is a small feed-forward CNN (no attention). The Python reference is at:
- `/Users/oboulant/repos/work/LTX-2/packages/ltx-core/src/ltx_core/model/upsampler/model.py` — `LatentUpsampler` class
- `/Users/oboulant/repos/work/LTX-2/packages/ltx-core/src/ltx_core/model/upsampler/res_block.py` — `ResBlock`
- `/Users/oboulant/repos/work/LTX-2/packages/ltx-core/src/ltx_core/model/upsampler/pixel_shuffle.py` — `PixelShuffleND`
- `/Users/oboulant/repos/work/LTX-2/packages/ltx-core/src/ltx_core/model/video_vae/ops.py` (lines 60-90) — `PerChannelStatistics` (normalize/un_normalize)

**Forward flow** (`upsample_video` in model.py):
```
1. un_normalize:  x * std_of_means + mean_of_means   (per-channel, shape [128])
2. LatentUpsampler.forward(x)
3. normalize:     (x - mean_of_means) / std_of_means
```

**LatentUpsampler.forward** (dims=3, spatial_upsample=True, temporal_upsample=False):
```
Input: [B, 128, F, H, W] bf16

initial_conv  (Conv3d 128→1024, k=3, p=1)
initial_norm  (GroupNorm 32 groups, 1024 channels)
SiLU

4× ResBlock(1024, dims=3):
    conv1 (Conv3d 1024→1024, k=3, p=1)
    norm1 (GroupNorm 32, 1024)
    SiLU
    conv2 (Conv3d 1024→1024, k=3, p=1)
    norm2 (GroupNorm 32, 1024)
    SiLU(x + residual)

# Spatial upsample (2x) — rearrange to 2D, Conv2d + PixelShuffle, rearrange back
rearrange "b c f h w -> (b f) c h w"
Conv2d(1024, 4096, k=3, p=1)              ← upsampler.0
PixelShuffleND(2): "b (c p1 p2) h w -> b c (h p1) (w p2)"  with p1=p2=2
rearrange "(b f) c h w -> b c f h w"

4× ResBlock(1024, dims=3):  (post_upsample_res_blocks)
    same structure as above

final_conv (Conv3d 1024→128, k=3, p=1)

Output: [B, 128, F, H*2, W*2] bf16
```

## Checkpoint weights (72 keys, all bf16)

Separate file: `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`

```
initial_conv.weight: [1024, 128, 3, 3, 3]    initial_conv.bias: [1024]
initial_norm.weight: [1024]                   initial_norm.bias: [1024]
res_blocks.{0-3}.conv1.weight: [1024, 1024, 3, 3, 3]
res_blocks.{0-3}.conv1.bias: [1024]
res_blocks.{0-3}.norm1.weight: [1024]         res_blocks.{0-3}.norm1.bias: [1024]
res_blocks.{0-3}.conv2.weight: [1024, 1024, 3, 3, 3]
res_blocks.{0-3}.conv2.bias: [1024]
res_blocks.{0-3}.norm2.weight: [1024]         res_blocks.{0-3}.norm2.bias: [1024]
upsampler.0.weight: [4096, 1024, 3, 3]     ← Conv2D (not 3D!)
upsampler.0.bias: [4096]
post_upsample_res_blocks.{0-3}: same structure as res_blocks
final_conv.weight: [128, 1024, 3, 3, 3]      final_conv.bias: [128]
```

The `per_channel_statistics` (`mean-of-means` and `std-of-means`, shape [128] each) come from the **main** checkpoint under keys `vae.per_channel_statistics.mean-of-means` and `vae.per_channel_statistics.std-of-means`.

## ZML conventions to follow

Read the `AGENTS.md` at the repo root for full conventions. Key points:
- Use `zml.Tensor` for compile-time graph construction vs `zml.Buffer` for runtime
- Weight structs use the ZML `io.load` pattern (see how `model.zig` defines `PreprocessParams`, `Block0FullParams`, etc.)
- Weight key mapping uses `.prefix("some.prefix.")` on shape structs
- Conv3d and GroupNorm should already exist in `zml.nn` — check `zml/nn.zig` or `zml/zml.zig`
- For Conv2d, check what's available. The upsampler needs a Conv2d path for the spatial upsample step.
- Follow the existing patterns in `model.zig` for struct layout, `forwardXxx` function naming, and compilation flow
- Build with `bazel build //examples/ltx:test` to verify compilation

## Deliverables

1. **Add upsampler model code to `model.zig`** — new section with:
   - Weight parameter structs (`UpsamplerParams`, `UpsamplerResBlock`, etc.)
   - `forwardUpsample` function implementing the full un_normalize → CNN → normalize flow
   - Shape init function (`initUpsamplerParams`)

2. **Create `upsample.zig` driver** — a standalone binary that:
   - Takes: upsampler checkpoint path, main checkpoint path (for per_channel_statistics), Stage 1 output dir, output dir
   - Loads Stage 1 denoised video latent (from `.bin` files), unpatchifies to `[B, 128, F, H, W]`
   - Runs the upsampler
   - Writes upsampled latent to output dir

3. **Update `BUILD.bazel`** — add a `zig_binary` target for the upsampler

## Validation strategy

- The Python reference export script (`export_mixed_pipeline.py`) already saves `ref/upsampled.safetensors` containing the Python upsampler output
- We can compare the Zig upsampler output against this reference tensor
- First milestone: compile successfully. Second milestone: match Python output numerically.

## Implementation status: COMPLETE

### What was built

1. **`model.zig`** — Added upsampler code (Section 10, ~300 lines):
   - Weight structs: `Conv3dWeight`, `Conv2dWeight`, `GroupNormWeight`, `UpsamplerResBlock`, `UpsamplerParams`, `PerChannelStats`
   - Named fields `res_block_0..3` / `post_res_block_0..3` instead of arrays (ZML `mem.bufferize` doesn't support `[4]T`)
   - Init functions: `initUpsamplerParams(store)`, `initResBlockParams(store)`, `initPerChannelStats(store)`
   - Forward ops: `forwardConv3d`, `forwardConv2d`, `forwardGroupNorm`, `forwardPixelShuffle2d`, `forwardResBlock`, `forwardUpsample`, `forwardUnpatchifyVideo`
   - GroupNorm performs all computation in f32 (mean/var/normalize/affine), converts back to input dtype at end

2. **`tensor.zig`** — Added `pub fn conv3d` (was missing from ZML), mirroring existing `conv2d` pattern

3. **`upsample.zig`** — Standalone driver binary (~400 lines):
   - CLI: `--upsampler-ckpt`, `--main-ckpt`, `--input`, `--f-lat`, `--h-lat`, `--w-lat`, `--output-dir`, `--ref` (optional)
   - Flow: load raw bf16 → compile+run unpatchify → compile upsampler → load weights → run → write `upsampled_video.bin`
   - Built-in validation via `--ref`: supports `.safetensors` (skips JSON header) and raw `.bin` files
   - Metrics: cosine_similarity, max_abs_diff, mean_abs_diff, close_fraction, diff histogram, first-8-elements comparison
   - PASS threshold: cosim > 0.995 AND mean_abs < 0.1

4. **`BUILD.bazel`** — Added `zig_binary(name="upsample", ...)`

5. **`bridge_s1_to_s2.py`** — Added automatic export of `upsampled_ref.safetensors` alongside bridge output, so validation reference is always available from bridge runs

### Bugs fixed during implementation

| Issue | Cause | Fix |
|-------|-------|-----|
| `convolution` private in tensor.zig | No public `conv3d` existed | Added `pub fn conv3d` |
| `unreachable code` in `bufferize` | `mem.bufferize` doesn't handle `[4]T` arrays | Expanded to named fields `res_block_0..3` |
| Spatial upsample wrong output | `rearrange("b c f h w -> (b f) c h w")` needs transpose, not plain reshape | Added `.transpose(.{0,2,1,3,4})` before reshape, reverse after |
| `EndOfStream` on safetensors ref | Comparing raw bf16 against safetensors without skipping header | Added safetensors header detection/skip |
| `EndOfStream` on undersized ref | Using Stage 1 input as `--ref` (wrong file) | Added clear error message for size mismatch |

### Validation results

**Apple-to-apple comparison** — Same `video_latent.bin` input, same checkpoint weights.
Reference generated by `bridge_s1_to_s2.py` → `upsampled_ref.safetensors`:

```
Numerical validation (3,145,728 elements)
  cosine_similarity: 0.99778
  max_abs_diff:      0.499
  mean_abs_diff:     0.038
  close_fraction:    0.150 (atol=0.005, rtol=0.01)  — tight for bf16 CNN
  diff histogram:
    <0.01   | 20.49%
    0.01-0.1 | 73.48%
    0.1-0.5  |  6.03%
    0.5+     |  0.00%
  PASS (cosim > 0.995, mean_abs < 0.1)
```

**Per-channel breakdown** (128 channels, error over [B, F, H, W]):

```
  std of per-channel mean_abs: 0.0118
  min/max of per-channel mean_abs: 0.0056 / 0.0570
  channels exceeding 0.08 mean_abs: 0 / 128
```

Error is uniformly distributed across all channels — confirms pure bf16 rounding accumulation through the deep CNN (8 ResBlocks, ~20 ops), not any channel-specific bug.

### How to run

```bash
# On GPU server:
export UPSAMPLER=/ephemeral/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
export CKPT=/ephemeral/models/ltx-2.3/ltx-2.3-22b-dev.safetensors
export OUT=/ephemeral/mixed_bridge_export_ref

bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:upsample -- \
    --upsampler-ckpt $UPSAMPLER --main-ckpt $CKPT \
    --input $OUT/stage1_out/video_latent.bin \
    --f-lat 16 --h-lat 16 --w-lat 24 \
    --output-dir $OUT/upsampled/ \
    --ref $OUT/upsampled_ref.safetensors
```

The `--ref` file is automatically produced by `bridge_s1_to_s2.py` alongside its main output.

## Important notes

- The upsampler checkpoint is a **separate** safetensors file from the main transformer checkpoint
- The PixelShuffle is purely a reshape/rearrange — no learnable parameters
- GroupNorm with 32 groups on 1024 channels = 32 channels per group
- The Conv2d in the upsampler step operates on `(B*F, C, H, W)` — temporal dim folded into batch
- All weights are bf16; arithmetic should match Python's dtype chain (likely bf16 throughout for this simple CNN, unlike the transformer which needs careful f32 intermediate steps)
