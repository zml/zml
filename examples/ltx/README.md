# LTX-2 Video Generation — Zig/ZML Implementation

Zig port of the [LTX-2 22B](https://huggingface.co/Lightricks/LTX-2.3) two-stage text-to-video
generation pipeline, running on ZML (MLIR/XLA/PJRT backend).

## Pipeline Overview

The pipeline produces video+audio from a text prompt:

1. **Python** — Text encoding (prompt → context embeddings) + noise generation
2. **Zig** — All GPU compute: Stage 1 denoising → bridge (upsample) → Stage 2 denoising
3. **Python** — VAE decode (latents → MP4)

The `inference` binary runs the entire denoising pipeline (Stage 1 + bridge +
Stage 2) in a single process, passing GPU buffers between phases without
intermediate files.

```
Python (export)                Zig (inference)                     Python (decode)
───────────────                ──────────────────────              ───────────────
text encoding ──┐              Stage 1 (30 steps × 4 passes)      VAE decode
noise init      ├─→ inputs ──→ Bridge (upsample 2×)           ──→ video + audio
sigma schedule  ┘              Stage 2 (3 steps × 1 pass)         → output.mp4
```

## File Map

### Zig (compiled to GPU via ZML)
| File | Purpose |
|------|---------|
| `model.zig` | Core model: transformer blocks, preprocessing, output projection, denoising step, guidance combine, noise init (~2500 lines) |
| `inference.zig` | **Unified pipeline**: Stage 1 → bridge → Stage 2 in one binary |
| `denoise_stage1.zig` | Standalone Stage 1 driver (for debugging/development) |
| `bridge.zig` | Standalone bridge driver (for debugging/development) |
| `denoise_e2e.zig` | Standalone Stage 2 driver (for debugging/development) |
| `main.zig` | Utility: safetensors inspector |

### Python
| File | Purpose |
|------|---------|
| `export_mixed_pipeline.py` | Export prompt embeddings, noise, sigmas, and pipeline metadata |
| `e2e/decode_latents.py` | VAE decode Zig-produced latents → MP4 |
| `bridge_s1_to_s2.py` | Legacy: Python bridge (replaced by bridge phase in `inference.zig`) |
| `validate_mixed_pipeline.py` | Boundary validation: cos_sim checks between Python reference and Zig outputs |

### Documentation
| File | Purpose |
|------|---------|
| `04_unified_pipeline.md` | Design doc for the unified pipeline |
| `full_e2e_mixed.md` | Command reference for the legacy multi-binary pipeline |
| `STAGE1_DIVERGENCE_ANALYSIS.md` | Analysis of Stage 1 numerical divergence patterns |

## Running the Unified Pipeline

### Prerequisites

- CUDA GPU with enough VRAM for the 22B model
- Model weights at a known path (e.g. `/root/models/ltx-2.3/`)
- Python env with `ltx_core`, `ltx_pipelines` for export and decode steps

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

This runs all three phases in sequence on GPU and writes:
- `$OUT/unified/video_latent.bin` — final video latent (`[1, T_v2, 128]` bf16)
- `$OUT/unified/audio_latent.bin` — final audio latent (`[1, T_a, 128]` bf16)

**Flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--bf16-attn-stage1` | off | Use bf16 attention in Stage 1 (saves VRAM) |
| `--bf16-attn-stage2` | off | Use bf16 attention in Stage 2 (recommended — avoids OOM after spatial upsample) |
| `--dump-intermediates` | off | Also write `stage1_video_latent.bin` / `stage1_audio_latent.bin` for debugging |

### Step 3: Decode to MP4 (Python)

```bash
python examples/ltx/e2e/decode_latents.py \
  --inputs $OUT/stage1_inputs.safetensors \
  --video-latent $OUT/unified/video_latent.bin \
  --audio-latent $OUT/unified/audio_latent.bin \
  --output $OUT/unified/output.mp4
```

## Legacy: Multi-Binary Pipeline

The standalone drivers (`denoise_stage1`, `bridge`, `denoise_e2e`) are kept for
debugging and development. They write intermediate files between phases. See
[full_e2e_mixed.md](full_e2e_mixed.md) for the multi-binary command reference.
