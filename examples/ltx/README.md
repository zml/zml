# LTX-2 Video Generation — Zig/ZML Implementation

Zig port of the [LTX-2 22B](https://huggingface.co/Lightricks/LTX-2.3) two-stage text-to-video
generation pipeline, running on ZML (MLIR/XLA/PJRT backend).

## Pipeline Overview

The pipeline produces video+audio from a text prompt:

1. **Python** — Text encoding (Gemma prompt → context embeddings) + noise init + sigma schedule
2. **Zig** — Everything else on GPU: Stage 1 denoising → bridge (upsample) → Stage 2 denoising → video VAE decode → audio VAE decode → vocoder + BWE → MP4 mux

The `inference` binary runs the full pipeline end-to-end in a single process,
passing GPU buffers between phases without intermediate files.

```
Python (export)                Zig (inference)
───────────────                ──────────────────────────────────────────────────
text encoding ──┐              Stage 1 (30 steps × 4 passes)
noise init      ├─→ inputs ──→ Bridge (upsample 2×)
sigma schedule  ┘              Stage 2 (3 steps × 1 pass)
                               Video VAE decode → RGB frames
                               Audio VAE decode → vocoder + BWE → waveform
                               ffmpeg mux → output.mp4
```

## File Map

### Zig (compiled to GPU via ZML)
| File | Purpose |
|------|---------|
| `model.zig` | Core transformer: blocks, attention, preprocessing, output projection, denoising step, guidance, noise init |
| `conv_ops.zig` | Shared convolution types and operations (Conv3d, Conv2d, GroupNorm, PerChannelStats) |
| `upsampler.zig` | Latent spatial upscaler (2×) + video patchify/unpatchify |
| `video_vae.zig` | Video VAE decoder (3D causal convolutions, DepthToSpace) |
| `audio_vae.zig` | Audio VAE decoder (2D causal convolutions, PixelNorm2d) |
| `vocoder.zig` | Vocoder + BWE (BigVGAN, sinc resampler, STFT, mel projection) |
| `inference.zig` | **Unified pipeline**: Stage 1 → bridge → Stage 2 → VAE decode → vocoder in one binary |
| `denoise_stage1.zig` | Standalone Stage 1 driver (for debugging/development) |
| `bridge.zig` | Standalone bridge driver (for debugging/development) |
| `denoise_e2e.zig` | Standalone Stage 2 driver (for debugging/development) |

### Python
| File | Purpose |
|------|---------|
| `export_mixed_pipeline.py` | Export prompt embeddings, noise, sigmas, and pipeline metadata |

## Running the Unified Pipeline
## Running the Unified Pipeline

### Prerequisites

- CUDA GPU with enough VRAM for the 22B model
- Model weights at a known path (e.g. `/root/models/ltx-2.3/`)
- Python env with `ltx_core`, `ltx_pipelines` for the export step

### Step 1: Export inputs (Python)

Run the export script to encode the text prompt and generate noise:

```bash
cd /root/repos/LTX-2  # or wherever ltx_core is installed

python examples/ltx/export_mixed_pipeline.py \
  --output-dir $OUT \
  --prompt "Someone walking by the beach at sunset" \
  --negative-prompt "blurry, out of focus" \
  --seed 42 \
  --height 1024 --width 1536 --num-frames 121 \
  --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
  --stage2-checkpoint /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
  --spatial-upsampler /root/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-path /root/models/gemma-3-12b-it
```

This produces:
- `$OUT/stage1_inputs.safetensors` — context embeddings, initial noise, masks, positions
- `$OUT/stage2_noise.safetensors` — pre-drawn noise for Stage 2
- `$OUT/pipeline_meta.json` — latent dimensions, sigma schedules, guidance parameters

### Step 2: Run inference (Zig)

Build and run the unified binary:

```bash
bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:inference -- \
  --stage1-ckpt /root/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
  --stage2-ckpt /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
  --upsampler-ckpt /root/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --stage1-inputs $OUT/stage1_inputs.safetensors \
  --stage2-noise $OUT/stage2_noise.safetensors \
  --meta $OUT/pipeline_meta.json \
  --output-dir $OUT/unified \
  --bf16-attn-stage2
```

This runs the full pipeline on GPU and writes `$OUT/unified/output.mp4` directly.

**Flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--bf16-attn-stage1` | off | Use bf16 attention in Stage 1 (saves VRAM) |
| `--bf16-attn-stage2` | off | Use bf16 attention in Stage 2 (recommended — avoids OOM after spatial upsample) |
| `--dump-intermediates` | off | Also write intermediate latents for debugging |

## Standalone Drivers

The standalone drivers (`denoise_stage1`, `bridge`, `denoise_e2e`) are kept for
debugging and development. They read/write safetensors files between phases,
making it easy to test a single stage in isolation.
