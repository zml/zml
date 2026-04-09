# LTX-2 Video Generation — Zig/ZML Implementation

Zig port of the [LTX-2 22B](https://huggingface.co/Lightricks/LTX-2.3) two-stage text-to-video
generation pipeline, running on ZML (MLIR/XLA/PJRT backend).

## Pipeline Overview

The pipeline produces video+audio from a text prompt, with optional image conditioning:

1. **Python** — Text encoding (Gemma prompt → context embeddings) + noise init + sigma schedule
2. **Zig** — Everything else on GPU: optional image VAE encoding + conditioning → Stage 1 denoising → bridge (upsample) → Stage 2 denoising → video VAE decode → audio VAE decode → vocoder + BWE → MP4 mux

When `--image` is provided, the first frame is conditioned on a reference image
via VAE encoding + per-token mask blending (see [06_image_conditioning.md](_archive/06_image_conditioning.md)).

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
| `video_vae_encoder.zig` | Video VAE encoder (for image conditioning) |
| `image_loading.zig` | JPEG/PNG loading via stb_image, bilinear resize, center crop, bf16 normalize |
| `inference.zig` | **Unified pipeline**: Stage 1 → bridge → Stage 2 → VAE decode → vocoder in one binary |

### Python
| File | Purpose |
|------|---------|
| `export_pipeline.py` | Export prompt embeddings, noise, sigmas, and pipeline metadata (supports optional `--image` for image conditioning) |

## Running the Unified Pipeline

### Prerequisites

- CUDA GPU with enough VRAM for the 22B model
- Model weights at a known path (e.g. `/root/models/ltx-2.3/`)
- Gemma 3 model for text encoding (e.g. `/root/models/gemma-3-12b-it/`)
- Python env with `ltx_core`, `ltx_pipelines` for the export step

### Step 1: Export inputs (Python)

Run the export script to encode the text prompt and generate noise:

```bash
cd /root/repos/LTX-2  # or wherever ltx_core is installed

python examples/ltx/export_pipeline.py \
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
- `$OUT/unconditioned_stage1_inputs.safetensors` — context embeddings, initial noise, masks, positions
- `$OUT/stage2_noise.safetensors` — pre-drawn noise for Stage 2
- `$OUT/pipeline_meta.json` — latent dimensions, sigma schedules, guidance parameters

To generate with image conditioning, add `--image /path/to/image.jpg`.

### Step 2: Run inference (Zig)

Build and run the unified binary:

```bash
bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:inference -- \
  --stage1-ckpt /root/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
  --stage2-ckpt /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
  --upsampler-ckpt /root/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --stage1-inputs $OUT/unconditioned_stage1_inputs.safetensors \\
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

## Image Conditioning

To generate video conditioned on a reference image, pass `--image` to both the
export script and the Zig inference binary. See [06_image_conditioning.md](_archive/06_image_conditioning.md)
for details on the implementation and per-token AdaLN masking.
