# LTX-2 Video Generation — Zig/ZML Implementation

Zig port of the [LTX-2 22B](https://huggingface.co/Lightricks/LTX-2.3) two-stage text-to-video
generation pipeline, running on ZML (MLIR/XLA/PJRT backend).

The pipeline produces video+audio from a text prompt using a mixed Python↔Zig architecture:
Python handles text encoding, VAE decode, and pipeline orchestration; Zig handles the
compute-intensive denoising (48-block AV transformer, 4-pass guided Stage 1 + 3-step distilled Stage 2).

## File Map

### Zig (compiled to GPU via ZML)
| File | Purpose |
|------|---------|
| `model.zig` | Core model: transformer blocks, preprocessing, output projection, denoising step, guidance combine, noise init (~2500 lines) |
| `denoise_stage1.zig` | Stage 1 driver: 30-step 4-pass guided denoising (CFG + STG + modality isolation) |
| `denoise_e2e.zig` | Stage 2 driver: 3-step distilled Euler denoising |
| `main.zig` | Utility: safetensors inspector |

### Python (pipeline orchestration)
| File | Purpose |
|------|---------|
| `export_mixed_pipeline.py` | M0: Full Python reference pipeline + Stage 1/2 input export |
| `bridge_s1_to_s2.py` | M2: Bridge Zig Stage 1 output → Stage 2 inputs (unpatchify + upsample + noise) |
| `validate_mixed_pipeline.py` | Boundary validation: cos_sim checks between Python reference and Zig outputs |
| `e2e/export_stage2_inputs.py` | Standalone Stage 2 input exporter |
| `e2e/decode_latents.py` | M4: VAE decode Zig-produced latents → MP4 |

### Documentation
| File | Purpose |
|------|---------|
| `full_e2e_mixed.md` | Full command reference for the mixed pipeline |
| `STAGE1_DIVERGENCE_ANALYSIS.md` | Analysis of Stage 1 numerical divergence patterns |

## Architecture: Mixed Pipeline Flow

```
Python (M0)                    Zig (M1)                     Python (M2)
─────────────────              ───────────────              ─────────────────
text encoding ──┐              Stage 1 denoiser             upsample + noise
noise init      ├─→ inputs ──→ (30 steps × 4 passes) ──→   init for Stage 2
sigma schedule  ┘              per block: 48 blocks          │
                                                             ▼
                               Zig (M3)                     Python (M4)
                               ───────────────              ─────────────────
                               Stage 2 denoiser             VAE decode
                               (3 steps × 1 pass)  ──────→ video + audio
                               per block: 48 blocks         → output.mp4
```

## Running the Full Pipeline

See **[full_e2e_mixed.md](full_e2e_mixed.md)** for the complete command reference.

### Quick overview

```bash
# Build Zig denoisers
bazel build --config=release --@zml//platforms:cuda=true \
  //examples/ltx:denoise_stage1 //examples/ltx:denoise_e2e

# M0: Export inputs from Python
python export_mixed_pipeline.py --prompt "A cat playing piano" ...

# M1: Stage 1 denoising (Zig)
./bazel-bin/examples/ltx/denoise_stage1 <checkpoint> <inputs> <output_dir/>

# M2: Bridge Stage 1 → Stage 2 (Python)
python bridge_s1_to_s2.py --checkpoint <ckpt> --stage1-dir <dir> --output <s2_inputs>

# M3: Stage 2 denoising (Zig)
./bazel-bin/examples/ltx/denoise_e2e <checkpoint> <s2_inputs> <output_dir/> --bf16-attn

# M4: Decode to MP4 (Python)
python e2e/decode_latents.py --inputs <s2_inputs> --video-latent <dir>/video_latent.bin \
  --audio-latent <dir>/audio_latent.bin --output output.mp4
```
