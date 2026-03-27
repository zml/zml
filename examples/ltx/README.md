# LTX-2 Video Generation — Zig/ZML Implementation

Zig port of the [LTX-2 22B](https://huggingface.co/Lightricks/LTX-Video) two-stage text/image-to-video
generation pipeline, running on ZML (MLIR/XLA/PJRT backend).

## Current Status

**Stage 2 (distilled, 3-step) is fully validated end-to-end.**

The Zig denoiser handles: noise init → preprocessing → 48-block transformer chain →
output projection → Euler denoising step. Combined with Python for upstream (text encoding,
Stage 1, upsample) and downstream (VAE decode), it produces playable video+audio.

| Component | Status | Validation |
|-----------|--------|------------|
| Noise init (`forwardNoiseInit`) | ✅ | cos_sim=1.0, close=1.0 |
| Preprocessing (`forwardPreprocess`) | ✅ | Part of Step 2 |
| 48-block chain (`forwardBlock0Native`) | ✅ | cos_sim=0.9965, close=0.9762 |
| Output projection (`forwardOutputProjection`) | ✅ | Part of Step 2 |
| Denoising step (`forwardDenoisingStep`) | ✅ | Part of Step 3 |
| Full 3-step denoising loop | ✅ | cos_sim=0.978–0.982 |
| E2E demo (noise init → denoise → MP4) | ✅ | Playable video matches Python reference |

## Key Files

### Core implementation
- **`model.zig`** — Core model: transformer blocks, preprocessing, noise init, denoising step (~4800 lines)
- **`denoise_e2e.zig`** — E2E Stage 2 driver: loads inputs, runs noise init + 3-step Euler loop, writes output
- **`check_utils.zig`** — Shared parity comparison utilities (cosine similarity, extended metrics)

### E2E pipeline scripts
- **`e2e/export_stage2_inputs.py`** — Python: text encoding + Stage 1 + upsample + exports safetensors for Zig
- **`e2e/decode_latents.py`** — Python: unpatchify + VAE decode → MP4

### Validation checkers
- **`check_noise_init.zig`** — Validates `forwardNoiseInit` against Python fixture
- **`export_noise_init_fixture.py`** — Exports noise init fixture from trace data

### Configuration
- **`config.zig`** — Model hyperparameters
- **`BUILD.bazel`** — Build targets (`denoise_e2e`, `check_noise_init`, etc.)

## Documentation Index

### Active
| File | Purpose |
|------|---------|
| **[PIPELINE_STATUS.md](PIPELINE_STATUS.md)** | Live tracker — component status, validation metrics, e2e demo results |
| **[STAGE1_IMPLEMENTATION_PLAN.md](STAGE1_IMPLEMENTATION_PLAN.md)** | Plan for Stage 1 (30-step guided denoising with CFG/STG/modality isolation) |

### Historical reference
These document completed work and past debugging. Kept for reference.

| File | Purpose |
|------|---------|
| [VELOCITY_MODEL_PLAN.md](VELOCITY_MODEL_PLAN.md) | Original step-by-step validation plan (Steps 1–3). Superseded by PIPELINE_STATUS. |
| [ADALN_SINGLE_IMPLEMENTATION.md](ADALN_SINGLE_IMPLEMENTATION.md) | AdaLayerNormSingle module implementation tracker |
| [OUTPUT_PROJECTION_IMPLEMENTATION.md](OUTPUT_PROJECTION_IMPLEMENTATION.md) | OutputProjection implementation tracker |
| [AUDIO_PARITY_FINDINGS_2026-03-24.md](AUDIO_PARITY_FINDINGS_2026-03-24.md) | Audio parity analysis and numerical drift diagnosis |
| [transformer_threading_progress.md](transformer_threading_progress.md) | Multi-block validation progress (post-M1–M6) |
| [block0_reverse_engineering_map.md](block0_reverse_engineering_map.md) | Block-0 reverse engineering reference archive |

### Diagnostic infrastructure (historical)
| File | Purpose |
|------|---------|
| [DIAGNOSTIC_QUICKSTART.md](DIAGNOSTIC_QUICKSTART.md) | Quick reference for diagnostic commands |
| [DIAGNOSTIC_PIPELINE.md](DIAGNOSTIC_PIPELINE.md) | Diagnostic pipeline overview |
| [DIAGNOSTIC_CHECKLIST.md](DIAGNOSTIC_CHECKLIST.md) | Diagnostic next-actions checklist |
| [DIAGNOSTIC_COMPLETE.md](DIAGNOSTIC_COMPLETE.md) | Diagnostic completion summary |
| [DIAGNOSTIC_IMPLEMENTATION_SUMMARY.md](DIAGNOSTIC_IMPLEMENTATION_SUMMARY.md) | Diagnostic infrastructure summary |
| [ZIG_DIAGNOSTIC_USAGE.md](ZIG_DIAGNOSTIC_USAGE.md) | Diagnostic tool usage |
| [ZIG_IMPLEMENTATION_COMPLETE.md](ZIG_IMPLEMENTATION_COMPLETE.md) | Zig diagnostic implementation summary |

## Running the E2E Pipeline

### Prerequisites
- GPU server with NVIDIA CUDA
- Model weights at `/root/models/ltx-2.3/`
- Python environment with `ltx-core` and `ltx-pipelines` packages
- ZML repo with Bazel

### Step 1: Export Stage 2 inputs (Python)
```bash
cd /root/repos/LTX-2
uv run python scripts/e2e/export_stage2_inputs.py \
    --prompt "A cat playing piano" \
    --output /root/e2e_demo/stage2_inputs.safetensors
```

### Step 2: Run Zig denoiser (noise init + 3-step Euler loop)
```bash
cd /root/repos/zml
bazel run --config=release --@zml//platforms:cuda=true \
    //examples/ltx:denoise_e2e -- \
    /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    /root/e2e_demo/stage2_inputs.safetensors \
    /root/e2e_demo/
```

### Step 3: Decode to MP4 (Python)
```bash
cd /root/repos/LTX-2
uv run python scripts/e2e/decode_latents.py \
    --inputs /root/e2e_demo/stage2_inputs.safetensors \
    --video-latent /root/e2e_demo/video_latent.bin \
    --audio-latent /root/e2e_demo/audio_latent.bin \
    --output /root/e2e_demo/output.mp4
```
