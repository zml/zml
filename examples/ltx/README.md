# LTX-2 Video Generation — Zig/ZML Implementation

Zig port of the [LTX-2 22B](https://huggingface.co/Lightricks/LTX-2.3) two-stage text-to-video
generation pipeline, running on ZML (MLIR/XLA/PJRT backend).

## Pipeline Overview

The pipeline produces video+audio from a text prompt, with optional image conditioning:

1. **Python** — Gemma forward pass (hidden states) + position/mask computation
2. **Zig** — Everything else on GPU: text embedding post-processing (connector blocks), noise generation, sigma schedules, optional image VAE encoding + conditioning → Stage 1 denoising → bridge (upsample) → Stage 2 denoising → video VAE decode → audio VAE decode → vocoder + BWE → MP4 mux

When `--image` is provided, the first frame is conditioned on a reference image
via VAE encoding + per-token mask blending.

The `inference` binary runs the full pipeline end-to-end in a single process,
passing GPU buffers between phases without intermediate files.

```
Python (export)                Zig (inference)
───────────────                ──────────────────────────────────────────────────
Gemma forward ──┐              Text embedding post-processing (connectors)
positions       ├─→ inputs ──→ Noise generation (Box-Muller, seeded RNG)
masks           │              Stage 1 (30 steps × 4 passes)
clean latents   ┘              Bridge (upsample 2×, re-noise)
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
| `export_pipeline.py` | Run full Python reference pipeline; export Gemma hidden states, positions, masks, and metadata for Zig inference |
| `export_gemma_hidden_states.py` | Standalone Gemma export + reference embeddings for validation |

## What `export_pipeline.py` does

The export script runs the **entire** LTX-2.3 pipeline in Python and captures
intermediate states at specific points using callback hooks. It also runs the
Gemma forward pass and saves the raw hidden states as sidecar files — the
text embedding post-processing (connectors) is done in Zig.

### How capture works: the `DiffusionStage.__call__` → `loop` callback

`DiffusionStage.__call__()` (from `ltx_pipelines`) does:
1. Patchify video/audio latent shapes
2. Compute RoPE positions from geometry
3. Create denoise masks (all-ones, or blended with image conditioning)
4. Create clean latents (all-zeros, or with VAE-encoded image tokens)
5. Draw noise via `GaussianNoiser` (torch.randn with generator)
6. Apply noise init: `noised = noise × mask × σ₀ + clean × (1 - mask × σ₀)`
7. **→ Call the `loop` callback** with the initialized `LatentState`
8. The callback captures the state, then calls `euler_denoising_loop()` to denoise

The capture happens **before** any denoising steps run — so what Zig receives
is the initial state (noised latent, mask, clean latent, positions) ready for
the denoising loop.

### Output files

| File | Contents |
|------|----------|
| `pos_hidden_states.safetensors` | Gemma hidden states for positive prompt (`stacked_hidden_states` + `attention_mask`). Used by Zig |
| `neg_hidden_states.safetensors` | Gemma hidden states for negative prompt. Used by Zig |
| `unconditioned_stage1_inputs.safetensors` | 8 tensors (see below) — always produced. Used for input by Zig |
| `conditioned_stage1_inputs.safetensors` | Same 8 tensors but with image conditioning applied (only with `--image`). Only for debug purpose |
| `conditioned_stage2_inputs.safetensors` | Stage 2 captured state + recovered noise (only with `--image`). Only for debug purpose |
| `stage2_noise.safetensors` | Recovered Stage 2 noise (kept for reference; **not used by Zig** — Zig generates its own. Only for debug purpose) |
| `pipeline_meta.json` | Latent geometry, guidance params, generation config. Used for metadata by Zig |
| `ref/stage1_outputs.safetensors` | Stage 1 denoised latents (Python reference). Only for debug purpose |
| `ref/upsampled.safetensors` | Upscaled video latent (Python reference). Only for debug purpose |
| `ref/stage2_outputs.safetensors` | Stage 2 final latents (Python reference). Only for debug purpose |

### Tensors in `unconditioned_stage1_inputs.safetensors`

These are the 8 tensors that Zig loads at runtime:

| Tensor | Shape (example: 1024×1536, 121 frames) | What it is |
|--------|----------------------------------------|------------|
| `video_denoise_mask` | `[1, 12240, 1]` | Per-token denoise weight. Unconditioned: all 1.0. Image: 0.0 for first-frame tokens |
| `audio_denoise_mask` | `[1, 1020, 1]` | Same for audio (always all 1.0) |
| `video_clean_latent` | `[1, 12240, 128]` | Clean signal to blend with noise. Unconditioned: all zeros. Image: VAE-encoded image at frame-0 positions |
| `audio_clean_latent` | `[1, 1020, 128]` | Same for audio (always zeros) |
| `video_positions` | `[1, 12240, 3]` | RoPE position ids `[t, h, w]` — deterministic from latent geometry |
| `audio_positions` | `[1, 1020, 3]` | Same for audio |
| `video_latent` | `[1, 12240, 128]` | Pre-noised video latent (**not used by Zig** — Zig generates its own noise) |
| `audio_latent` | `[1, 1020, 128]` | Pre-noised audio latent (**not used by Zig** — Zig generates its own noise) |

Zig loads 6 of these 8 tensors (skips `video_latent` and `audio_latent` since
noise is generated natively). Text embeddings (context vectors) are computed in
Zig from the Gemma hidden states via `text_embeddings.zig`.

## Running the Unified Pipeline

### Prerequisites

- CUDA GPU with enough VRAM for the 22B model
- Model weights at a known path (e.g. `/root/models/ltx-2.3/`)
- Gemma 3 model for text encoding (e.g. `/root/models/gemma-3-12b-it/`)
- Python env with `ltx_core`, `ltx_pipelines` for the export step

### Step 1: Export inputs (Python)

Run the export script to encode the text prompt and compute positions/masks:

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
  --gemma-root /root/models/gemma-3-12b-it
```

This produces:
- `$OUT/pos_hidden_states.safetensors` — Gemma hidden states (positive prompt)
- `$OUT/neg_hidden_states.safetensors` — Gemma hidden states (negative prompt)
- `$OUT/unconditioned_stage1_inputs.safetensors` — positions, denoise masks, clean latents (8 tensors described above)
- `$OUT/pipeline_meta.json` — latent geometry and guidance parameters

To generate with image conditioning, add `--image /path/to/image.jpg`.

### Step 2: Run inference (Zig)

Build and run the unified binary:

```bash
bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:inference -- \
  --stage1-ckpt /root/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
  --stage2-ckpt /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
  --upsampler-ckpt /root/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --stage1-inputs $OUT/unconditioned_stage1_inputs.safetensors \
  --meta $OUT/pipeline_meta.json \
  --gemma-hidden-states-pos $OUT/pos_hidden_states.safetensors \
  --gemma-hidden-states-neg $OUT/neg_hidden_states.safetensors \
  --output-dir $OUT/unified \
  --bf16-attn-stage2
```

This runs the full pipeline on GPU and writes `$OUT/unified/output.mp4` directly.

**Flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--gemma-hidden-states-pos <path>` | (required) | Gemma hidden states for positive prompt |
| `--gemma-hidden-states-neg <path>` | (required) | Gemma hidden states for negative prompt |
| `--seed <int>` | 42 | RNG seed for noise generation (all stages) |
| `--bf16-attn-stage1` | off | Use bf16 attention in Stage 1 (saves VRAM) |
| `--bf16-attn-stage2` | off | Use bf16 attention in Stage 2 (recommended — avoids OOM after spatial upsample) |
| `--dump-intermediates` | off | Write intermediate latents + noise tensors for debugging |
| `--image <path>` | none | Path to conditioning image (JPEG/PNG) for image-to-video |

## Image Conditioning

To generate video conditioned on a reference image, pass `--image` to the
**Zig inference binary**. The Python export step does not need `--image` — Zig
handles image conditioning entirely on its own: it loads the image from disk,
runs the VAE encoder on GPU, and modifies the mask/clean_latent/noised_latent
before denoising begins.

Passing `--image` to `export_pipeline.py` is only useful for generating
**reference tensors** (`conditioned_stage1_inputs.safetensors`,
`encoder_activations.safetensors`) for validation against the Zig output.

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
