# VAE Decode Temporal Tiling — Implementation Plan

## Problem

The video VAE decoder takes latent `[1, 128, F', H', W']` and produces pixel
video `[1, 3, F_px, H_px, W_px]`. At high resolutions (≥1088×1920), intermediate
activations exceed GPU memory because XLA compiles the entire decoder as one
monolithic graph. The 18.96 GiB OOM occurs in the late decoder stages where
tensors reach `[1, 128, 121, 272, 480]` (~3.8 GB each in bf16).

## Solution

Run the decoder on overlapping **temporal chunks** of the latent, then blend
the decoded pixel outputs on the host. Each chunk is small enough to fit in
GPU memory. The decoder function (`forwardVideoVaeDecode`) is unchanged.

**Temporal-only tiling is sufficient** for resolutions up to ~1088×1920.
Spatial tiling would be needed for 4K+ but adds significant complexity and
is out of scope.

## Why temporal is enough

At 1088×1920, peak memory is in decoder stages 6–8 (256ch / 128ch at full
temporal extent). Cutting the frame dimension from F'=16 to ~9 latent frames
halves activation memory from ~15 GB to ~7.5 GB per stage — well within the
80 GB H100 budget.

## Reference implementation

The reference Python pipeline implements tiling in:
- `VideoDecoder.tiled_decode()` in `packages/ltx-core/src/ltx_core/model/video_vae/video_vae.py`
- `TilingConfig` / `TemporalTilingConfig` in `packages/ltx-core/src/ltx_core/model/video_vae/tiling.py`

Default config: spatial tiles 512px / 64px overlap, temporal tiles 64 frames / 24 frames overlap.
In latent space: temporal tile = 64/8 = 8 latent frames, overlap = 24/8 = 3 latent frames.

## Design

### Tiling config

```
TemporalTilingConfig:
    tile_latent_frames: i64 = 9    // 65 pixel frames per tile
    overlap_latent_frames: i64 = 3 // 17 pixel frames overlap
```

- Tile size of 9 gives `8*(9-1)+1 = 65` pixel frames per decode.
- Overlap of 3 gives `8*(3-1)+1 = 17` pixel frames of blending region.
- Stride = tile - overlap = 6 latent frames per step.

### Tile computation

Given latent F', compute tile list `[(start, end), ...]`:

Example with F'=16, tile=9, overlap=3, stride=6:
- Tile 0: latent [0..9]  → pixel [0..65]
- Tile 1: latent [6..15] → pixel [41..105]  (overlap at pixel [41..64])
- Tile 2: latent [12..16] → pixel [89..121] (overlap at pixel [89..104])

The last tile may be shorter than `tile_latent_frames`. It gets **zero-padded**
to tile_latent_frames so we reuse the same compiled exe. The output is then
cropped to the actual frame count.

### Blend mask

Each tile gets a 1D trapezoidal weight mask along the temporal axis:

```
weight[f] =
    1.0                          if in non-overlap interior
    f / ramp_len                 if in left overlap ramp (except first tile)
    (ramp_len - f) / ramp_len   if in right overlap ramp (except last tile)
```

Pixel overlap length = `8 * (overlap_latent - 1) + 1` for interior boundaries.

### Execution flow

```
if F' <= tile_latent_frames:
    # Fast path — single decode, no tiling (current code)
    compile forwardVideoVaeDecode with [1, 128, F', H', W']
    run once, return

# Tiling path
tiles = computeTemporalTiles(F', config)
compile forwardVideoVaeDecode with [1, 128, tile_latent_frames, H', W']

# Host accumulation buffers
pixel_accum = zeros(F_px, H_px, W_px, 3) as f32
weight_accum = zeros(F_px) as f32

for each tile in tiles:
    1. Slice latent on GPU: v_latent_5d.slice1d(dim=2, start..end)
    2. If shorter than tile_latent_frames: pad with zeros on dim 2
    3. Run compiled VAE exe → decoded_tile [1, 3, F_tile_px, H_px, W_px]
    4. Download decoded_tile to host (toSliceAlloc)
    5. Free GPU buffer immediately
    6. Crop padding frames if last tile was padded
    7. Multiply by trapezoidal blend mask, accumulate into pixel_accum
    8. Accumulate mask into weight_accum

# Normalize and convert
result = pixel_accum / weight_accum  (per-frame division)
convert to u8 with (x + 1) / 2 * 255 clamped to [0, 255]
```

### Slicing on GPU

ZML Tensors support `slice1d` which produces a view (no copy). However, the
compiled exe expects a fixed shape `[1, 128, tile_latent_frames, H', W']`.
For the last tile that may be shorter, we need to:

1. Compile a **pad exe** (`forwardPadTemporal`): takes a shorter tensor,
   concatenates zeros along dim 2 to reach tile_latent_frames.
2. Or: compile two VAE exes — one for full tiles, one for the last shorter
   tile. This avoids padding but doubles compilation time.

**Recommended**: pad the last tile. One extra small compiled function is cheaper
than compiling the full VAE twice.

## Files changed

| File | Changes |
|------|---------|
| `video_vae.zig` | Add `TemporalTilingConfig`, `TemporalTile` struct, `computeTemporalTiles()`, `computeBlendMask()`. No changes to `forwardVideoVaeDecode`. |
| `inference.zig` | Rewrite `runVideoVaeDecode` steps 3–4: tiling decision, compile-once, tile loop, host-side blend + u8 conversion. |

## What does NOT change

- `forwardVideoVaeDecode` — stays as a single monolithic function
- Stage 1, Stage 2, Bridge — untouched (operate on patchified tokens, no memory issue)
- Audio VAE, Vocoder — untouched (tiny tensors, no memory issue)
- Resolutions ≤ 1024×1536 with F'≤9 — take the fast path, zero overhead

## Edge cases

| Case | Handling |
|------|----------|
| F' ≤ tile_latent_frames | Skip tiling, single decode (current behavior) |
| Last tile shorter | Pad latent to tile_latent_frames, decode, crop output |
| Single tile needed | Fast path, no blending |
| Pixel frame count | F_px = 8*(F_lat - 1) + 1. After DepthToSpace with temporal stride 2, the first frame is dropped, so the formula accounts for that. |

## Validation

1. **Bit-identical at low res**: Run 1024×1536 (no tiling) — output must match
   current pipeline exactly.
2. **OOM resolved**: Run 1088×1920 — should succeed where it currently OOMs.
3. **Quality check**: Run 1024×1536 with forced tiling (override threshold) — 
   compare visually with non-tiled output. Small numerical differences from
   blending are expected and acceptable.
4. **Frame count correctness**: Verify output has exactly the expected number
   of pixel frames for various num_frames values (121, 97, 57, etc.).
