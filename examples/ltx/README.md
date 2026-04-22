# LTX-2 Video Generation — Zig/ZML Implementation

Zig port of the [LTX-2 22B](https://huggingface.co/Lightricks/LTX-2.3) two-stage text-to-video
generation pipeline, running on ZML (MLIR/XLA/PJRT backend).

## Pipeline Overview

The pipeline produces video+audio from a text prompt, with optional image conditioning:

1. **Python** — Gemma forward pass (hidden states export)
2. **Zig** — Everything else on GPU: text embedding post-processing (connector blocks), position/mask/clean-latent construction, noise generation, sigma schedules, optional image VAE encoding + conditioning → Stage 1 denoising → bridge (upsample) → Stage 2 denoising → video VAE decode → audio VAE decode → vocoder + BWE → MP4 mux

When `--image` is provided, the first frame is conditioned on a reference image
via VAE encoding + per-token mask blending.

The `inference` binary runs the full pipeline end-to-end in a single process,
passing GPU buffers between phases without intermediate files.

```
Python (export)                Zig (inference)
───────────────                ──────────────────────────────────────────────────
Gemma forward ──→ hidden    ──→ Text embedding post-processing (connectors)
                  states        Positions + masks + clean latents (from geometry)
                                Noise generation (Box-Muller, seeded RNG)
                                Stage 1 (30 steps × 4 passes)
                                Bridge (upsample 2×, re-noise)
                                Stage 2 (3 steps × 1 pass, distilled sigmas)
                                Video VAE decode → RGB frames
                                Audio VAE decode → vocoder + BWE → waveform
                                ffmpeg mux → output.mp4
```

## File Map

### Zig (compiled to GPU via ZML)
| File | Purpose |
|------|---------|
| `model.zig` | Core transformer: blocks, attention, preprocessing, output projection, denoising step, guidance, noise init, noise generation (`forwardGenerateNoise`), sigma schedule |
| `conv_ops.zig` | Shared convolution types and operations (Conv3d, Conv2d, GroupNorm, PerChannelStats) |
| `upsampler.zig` | Latent spatial upscaler (2×) + video patchify/unpatchify |
| `video_vae.zig` | Video VAE decoder (3D causal convolutions, DepthToSpace) |
| `audio_vae.zig` | Audio VAE decoder (2D causal convolutions, PixelNorm2d) |
| `vocoder.zig` | Vocoder + BWE (BigVGAN, sinc resampler, STFT, mel projection) |
| `video_vae_encoder.zig` | Video VAE encoder (for image conditioning) |
| `image_loading.zig` | JPEG/PNG loading via stb_image, bilinear resize, center crop, bf16 normalize |
| `text_embeddings.zig` | Text embedding post-processing: FeatureExtractorV2 + Embeddings1DConnector (8 transformer blocks, SPLIT RoPE) |
| `inference.zig` | **Unified pipeline**: text embeddings → noise gen → Stage 1 → bridge → Stage 2 → VAE decode → vocoder in one binary |

### Python
| File | Purpose |
|------|---------|
| `export_pipeline.py` | Export Gemma hidden states for Zig inference, or run the full Python reference pipeline for validation |

### Files Zig uses from Python

| File | Contents |
|------|----------|
| `pos_hidden_states.safetensors` | Gemma hidden states for positive prompt (`stacked_hidden_states` + `attention_mask`). Used by Zig |
| `neg_hidden_states.safetensors` | Gemma hidden states for negative prompt. Used by Zig |

### Stage 1 initial state (computed by Zig)

Zig computes all Stage 1 initial state tensors from the pipeline geometry.
The normal path is to pass `--height`, `--width`, `--num-frames`, and `--fps`
directly to the binary:

- `F = ((num_frames - 1) / 8) + 1`
- `H = ceil((height / 32) / 2)` and `W = ceil((width / 32) / 2)` for Stage 1 latents
- `T_v = F × H × W`
- `T_a = round((num_frames / fps) × 25)`

The `25` in `T_a` comes from the audio latent rate: `16000 / (160 × 4) = 25`
tokens per second.

| Tensor | Shape | Zig computation |
|--------|-------|------------------|
| `video_positions` | `[1, 3, T_v, 2]` bf16 | `computeVideoPositions(F, H, W, fps)` |
| `audio_positions` | `[1, 1, T_a, 2]` f32 | `computeAudioPositions(T_a)` |
| `video_denoise_mask` | `[1, T_v, 1]` f32 | All 1.0 (image conditioning modifies after) |
| `audio_denoise_mask` | `[1, T_a, 1]` f32 | All 1.0 |
| `video_clean_latent` | `[1, T_v, 128]` bf16 | All zeros (image conditioning modifies after) |
| `audio_clean_latent` | `[1, T_a, 128]` bf16 | All zeros |

Text embeddings (context vectors) are computed in Zig from the Gemma hidden
states via `text_embeddings.zig`.

## Running the Unified Pipeline

### Prerequisites

- CUDA GPU with enough VRAM for the 22B model
- Model weights at a known path (e.g. `~/models/ltx-2.3/`)
- Gemma 3 model for text encoding (e.g. `~/models/gemma-3-12b-it/`)
- Python env with `ltx_core`, `ltx_pipelines` for the export step

### Step 1: Export Gemma hidden states (Python)

The default workflow is to export only the Gemma hidden states. That is all the
Zig binary needs from Python.

Run the exporter in `--text-only` mode:

```bash
cd /root/repos/LTX-2  # or wherever ltx_core is installed

python examples/ltx/export_pipeline.py \
  --text-only \
  --output-dir $OUT \
  --prompt "Someone walking by the beach at sunset" \
  --negative-prompt "blurry, out of focus" \
  --gemma-root /root/models/gemma-3-12b-it
```

This produces:
- `$OUT/pos_hidden_states.safetensors` — Gemma hidden states (positive prompt)
- `$OUT/neg_hidden_states.safetensors` — Gemma hidden states (negative prompt)

### Step 2: Run inference (Zig)

Build and run the unified binary:

```bash
bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:inference -- \
  --stage1-ckpt /root/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
  --stage2-ckpt /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
  --upsampler-ckpt /root/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-hidden-states-pos $OUT/pos_hidden_states.safetensors \
  --gemma-hidden-states-neg $OUT/neg_hidden_states.safetensors \
  --output-dir $OUT/unified \
  --height 1024 --width 1536 --num-frames 121 --fps 24 \
  --bf16-attn-stage2
```

This runs the full pipeline on GPU and writes `$OUT/unified/output.mp4` directly.

### Input constraints

- `--height` and `--width` must be divisible by 32
- Multiples of 64 are recommended for best Stage 1 to Stage 2 alignment
- `--num-frames` must be of the form `8k+1` such as `121`

**Flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--gemma-hidden-states-pos <path>` | (required) | Gemma hidden states for positive prompt |
| `--gemma-hidden-states-neg <path>` | (required) | Gemma hidden states for negative prompt |
| `--height <int>` | `1024` | Output height in pixels |
| `--width <int>` | `1536` | Output width in pixels |
| `--num-frames <int>` | `121` | Output frame count. Must satisfy `8k+1` |
| `--fps <float>` | `24.0` | Output frame rate |
| `--seed <int>` | 42 | RNG seed for noise generation (all stages) |
| `--num-inference-steps <int>` | `30` | Number of Stage 1 denoising steps |
| `--bf16-attn-stage1` | off | Use bf16 attention in Stage 1 (saves VRAM) |
| `--bf16-attn-stage2` | off | Use bf16 attention in Stage 2 (recommended — avoids OOM after spatial upsample) |
| `--cfg-v <float>` | `3.0` | Video CFG scale for Stage 1 guidance |
| `--stg-v <float>` | `1.0` | Video STG scale for Stage 1 guidance |
| `--mod-v <float>` | `3.0` | Video modality guidance scale for Stage 1 guidance |
| `--rescale-v <float>` | `0.7` | Video guidance rescale for Stage 1 guidance |
| `--cfg-a <float>` | `7.0` | Audio CFG scale for Stage 1 guidance |
| `--stg-a <float>` | `1.0` | Audio STG scale for Stage 1 guidance |
| `--mod-a <float>` | `3.0` | Audio modality guidance scale for Stage 1 guidance |
| `--rescale-a <float>` | `0.7` | Audio guidance rescale for Stage 1 guidance |
| `--dump-intermediates` | off | Write raw internal buffers (`.bin`) such as positions, masks, clean latents, noise tensors, conditioned image tokens, final latents, and audio mel data |
| `--image <path>` | none | Path to conditioning image (JPEG/PNG) for image-to-video |

## Image Conditioning

To generate video conditioned on a reference image, pass `--image` to the
**Zig inference binary**. Zig handles image conditioning entirely on its own:
it loads the image from disk, runs the VAE encoder on GPU for both Stage 1 and
Stage 2 resolutions, and modifies the mask/clean_latent/noised_latent before
denoising begins.

Passing `--image` to `export_pipeline.py` is only needed when you want Python
reference artifacts for validation; it is not part of the normal Zig workflow.

In image-conditioned mode, the denoise mask and clean latent encode the conditioning:
- **Denoise mask**: 0.0 for first-frame tokens (keep clean), 1.0 for the rest (fully denoise)
- **Clean latent**: VAE-encoded image at first-frame token positions, zeros elsewhere
- **Noise init formula**: `noised = noise × mask × σ₀ + clean × (1 - mask × σ₀)`
  — first-frame tokens get the clean image signal, rest get pure noise

## Migration History

Noise generation and sigma schedule computation were migrated from Python to Zig
(Box-Muller via `Tensor.Rng`, logistic sigma schedule in `computeSigmaSchedule()`).

Text embedding post-processing (FeatureExtractorV2 + Embeddings1DConnector with
8 transformer blocks, SPLIT RoPE, gated attention) was migrated from Python to
Zig. Python now only runs the Gemma forward pass; the connector blocks that
transform hidden states into video/audio context embeddings run entirely in Zig.

Stage 1 initial state (positions, denoise masks, clean latents) is now computed
in Zig from the pipeline geometry, eliminating the need for the
`unconditioned_stage1_inputs.safetensors` file. This covers both unconditioned
and image-conditioned paths (image conditioning applies modifications after
the initial state construction).

The Zig binary now exposes Stage 1 guidance scales directly as CLI flags,
instead of requiring them to be sourced from Python-exported metadata.
