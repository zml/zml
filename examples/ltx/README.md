# LTX-2.3 Video Generation — Zig/ZML

Zig port of [LTX-2.3 (22B)](https://huggingface.co/Lightricks/LTX-2.3), a two-stage
text-to-video+audio pipeline, running on ZML (MLIR/XLA/PJRT).

## Pipeline

Python runs only the Gemma-3 forward pass. Everything else runs in Zig on GPU:

```
Python                         Zig (single binary)
──────                         ──────────────────────────────────────────
Gemma-3 forward ──→ hidden  ──→ Text embeddings (FeatureExtractor + Connector)
                    states      Positions, masks, clean latents (from geometry)
                                Noise generation (Box-Muller)
                                Stage 1: 30 steps × 4 passes × 48 blocks
                                Bridge: unpatchify → 2× upsample → re-noise
                                Stage 2: 3 steps × 1 pass × 48 blocks (distilled)
                                Video VAE decode → RGB frames
                                Audio VAE decode → mel → vocoder+BWE → 48kHz stereo
                                ffmpeg mux → output.mp4
```

GPU buffers flow between phases directly — no files touch disk unless
`--dump-intermediates` is set.

When `--image` is provided, the first frame is conditioned on a reference image
via VAE encoding + per-token mask blending.

## File Map

| File | Purpose |
|------|---------|
| `inference.zig` | Pipeline orchestrator: CLI, all 7 phases, MP4 mux |
| `model.zig` | Transformer core (48-block), denoising math, guidance, noise gen, RoPE |
| `text_embeddings.zig` | Gemma hidden states → context embeddings (FeatureExtractorV2 + 8-block Connector) |
| `conv_ops.zig` | Shared Conv3d/Conv2d/GroupNorm/PixelShuffle primitives |
| `upsampler.zig` | Bridge CNN: 2× spatial latent upsample + patchify/unpatchify |
| `video_vae.zig` | Video VAE decoder (3D causal conv, DepthToSpace) |
| `video_vae_encoder.zig` | Video VAE encoder (image conditioning) |
| `audio_vae.zig` | Audio VAE decoder (2D causal conv) |
| `vocoder.zig` | BigVGAN vocoder + bandwidth extension (all f32) |
| `image_loading.zig` | stb_image load → resize → center-crop → bf16 normalize |
| `export_pipeline.py` | Gemma hidden-state export; optional full Python reference pipeline |

### External inputs

| File | Contents |
|------|----------|
| `pos_hidden_states.safetensors` | Gemma hidden states for positive prompt |
| `neg_hidden_states.safetensors` | Gemma hidden states for negative prompt |

## Latent geometry

Zig computes all initial-state tensors from `--height`, `--width`,
`--num-frames`, `--fps`:

- `F = (num_frames - 1) / 8 + 1`
- `H = ceil((height / 32) / 2)`, `W = ceil((width / 32) / 2)` (Stage 1)
- `T_v = F × H × W` (video tokens)
- `T_a = round((num_frames / fps) × 25)` (audio tokens; 25 = 16000 / (160 × 4))

| Tensor | Shape | Value |
|--------|-------|-------|
| `video_positions` | `[1, 3, T_v, 2]` bf16 | Pixel-coord grid from `computeVideoPositions` |
| `audio_positions` | `[1, 1, T_a, 2]` f32 | Time intervals from `computeAudioPositions` |
| `video_denoise_mask` | `[1, T_v, 1]` f32 | All 1.0 (modified by image conditioning) |
| `audio_denoise_mask` | `[1, T_a, 1]` f32 | All 1.0 |
| `video_clean_latent` | `[1, T_v, 128]` bf16 | All zeros (modified by image conditioning) |
| `audio_clean_latent` | `[1, T_a, 128]` bf16 | All zeros |

Text embeddings are computed in Zig from Gemma hidden states via
`text_embeddings.zig`.

## Usage

### Prerequisites

- CUDA GPU with sufficient VRAM for the 22B model
- Model weights (e.g. `~/models/ltx-2.3/`)
- Gemma-3 model (e.g. `~/models/gemma-3-12b-it/`)
- Python env with `ltx_core`, `ltx_pipelines` for the Gemma export step

### Step 1: Export Gemma hidden states

```bash
cd /path/to/LTX-2

python examples/ltx/export_pipeline.py \
  --text-only \
  --output-dir $OUT \
  --prompt "Someone walking by the beach at sunset" \
  --negative-prompt "blurry, out of focus" \
  --gemma-root ~/models/gemma-3-12b-it
```

Produces `$OUT/pos_hidden_states.safetensors` and `$OUT/neg_hidden_states.safetensors`.

### Step 2: Run inference

```bash
bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:inference -- \
  --stage1-ckpt ~/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
  --stage2-ckpt ~/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
  --upsampler-ckpt ~/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-hidden-states-pos $OUT/pos_hidden_states.safetensors \
  --gemma-hidden-states-neg $OUT/neg_hidden_states.safetensors \
  --output-dir $OUT/output \
  --height 1024 --width 1536 --num-frames 121 --fps 24 \
  --bf16-attn-stage1 --bf16-attn-stage2
```

Writes `$OUT/output/output.mp4`.

### Input constraints

- `height` and `width` must be divisible by 32 (multiples of 64 recommended)
- `num_frames` must be `8k + 1` (e.g. 121)

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--stage1-ckpt` | (required) | Stage 1 model checkpoint |
| `--stage2-ckpt` | (required) | Stage 2 (distilled) model checkpoint |
| `--upsampler-ckpt` | (required) | Spatial upsampler checkpoint |
| `--gemma-hidden-states-pos` | (required) | Gemma hidden states — positive prompt |
| `--gemma-hidden-states-neg` | (required) | Gemma hidden states — negative prompt |
| `--output-dir` | (required) | Output directory |
| `--height` | `1024` | Output height in pixels |
| `--width` | `1536` | Output width in pixels |
| `--num-frames` | `121` | Frame count (`8k + 1`) |
| `--fps` | `24` | Frame rate |
| `--seed` | `42` | RNG seed |
| `--num-inference-steps` | `30` | Stage 1 denoising steps |
| `--bf16-attn-stage1` | off | bf16 attention in Stage 1 (saves VRAM) |
| `--bf16-attn-stage2` | off | bf16 attention in Stage 2 (recommended — avoids OOM) |
| `--image` | none | Conditioning image path (JPEG/PNG) for image-to-video |
| `--cfg-v` | `3.0` | Video CFG scale |
| `--stg-v` | `1.0` | Video STG scale |
| `--mod-v` | `3.0` | Video modality guidance scale |
| `--rescale-v` | `0.7` | Video guidance rescale |
| `--cfg-a` | `7.0` | Audio CFG scale |
| `--stg-a` | `1.0` | Audio STG scale |
| `--mod-a` | `3.0` | Audio modality guidance scale |
| `--rescale-a` | `0.7` | Audio guidance rescale |
| `--dump-intermediates` | off | Write raw `.bin` snapshots of internal buffers |
| `--profile` | off | Enable XLA profiling |
| `--meta` | none | Legacy: load geometry from `pipeline_meta.json` instead of CLI flags |

## Image Conditioning

Pass `--image <path>` to condition the first frame on a reference image.
Zig handles this entirely: load → VAE encode → patchify → splice into
mask/clean-latent/noise for both Stage 1 and Stage 2.

- **Denoise mask**: 0.0 for first-frame tokens (keep fixed), 1.0 elsewhere
- **Clean latent**: VAE-encoded image at first-frame positions, zeros elsewhere
- **Noise init**: `noised = noise × mask × σ₀ + clean × (1 − mask × σ₀)`

`export_pipeline.py --image` is only needed for Python reference artifacts, not
for normal Zig inference.
