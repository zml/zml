# Mixed E2E Pipeline: Python ↔ Zig

**Target flow:**
```
Python(text enc + noise gen) → Zig(Stage 1) → Python(upsample) → Zig(Stage 2) → Python(VAE decode)
```

---

## What Already Exists

| Component                          | Status | File                                    |
| ---------------------------------- | ------ | --------------------------------------- |
| Stage 1 Zig driver                 | Done   | `denoise_stage1.zig`                    |
| Stage 2 Zig driver                 | Done   | `denoise_e2e.zig`                       |
| Stage 1 input export               | Done   | `export_stage1_inputs.py`               |
| Stage 2 input export (full Python) | Done   | `e2e/export_stage2_inputs.py`           |
| VAE decode                         | Done   | `e2e/decode_latents.py`                 |

## What Was Missing (now resolved)

| Component                        | Needed For                            | Resolution |
| -------------------------------- | ------------------------------------- | ---------- |
| **Reference boundary export**    | Validation baseline at every boundary | `export_mixed_pipeline.py` |
| **Generator state preservation** | Correct Stage 2 noise after Zig Stage 1 | Pre-draw all noise in M0 |
| **Bridge script** (S1→S2)       | Unpatchify → upsample → Stage 2 init | `bridge_s1_to_s2.py` |
| **Stage 1 output metadata**      | Python can read Zig `.bin` output     | Shape from `stage1_inputs.safetensors` |
| **Large-res attention OOM**      | Stage 2 at 1536×1024 (T_v=24576)     | Chunked bf16 SDPA + `--bf16-attn` flag |

---

## Milestones

### M0: Reference Export + Mixed Pipeline Inputs (single Python run)

**Status:** DONE ✅

**Script:** `export_mixed_pipeline.py`

A single script that runs the **full Python pipeline** end-to-end (text enc → Stage 1 denoise → upsample → Stage 2 denoise), saving:
- **Reference tensors** at every boundary (for validation of all later steps)
- **Stage 1 inputs** for the Zig driver (same format as existing `export_stage1_inputs.py`)
- **Pre-drawn Stage 2 noise** (so the bridge script doesn't need to restore generator state)

One run, one generator, all noise draws happen naturally in sequence.

**Outputs:**

| File                             | Contents                                                       |
| -------------------------------- | -------------------------------------------------------------- |
| `mixed/stage1_inputs.safetensors` | Stage 1 noised latents + masks + positions + contexts (14 tensors) |
| `mixed/stage2_noise.safetensors`  | `video_noise_s2` + `audio_noise_s2` (pre-drawn Stage 2 noise) |
| `mixed/pipeline_meta.json`        | seed, resolution, frames, fps, latent shapes, checkpoint paths |
| `mixed/ref/stage1_outputs.safetensors` | `video_latent_denoised`, `audio_latent_denoised` |
| `mixed/ref/upsampled.safetensors` | `upscaled_video_latent` (before patchification) |
| `mixed/ref/stage2_inputs.safetensors` | Full 12-tensor Stage 2 input (same format as existing export) |
| `mixed/ref/stage2_outputs.safetensors` | `video_latent_final`, `audio_latent_final` |

**Generator flow within the single run:**
1. `generator = torch.Generator(device).manual_seed(seed)`
2. `denoise_audio_video()` for Stage 1 → draws video_noise_s1 + audio_noise_s1 internally → captures pre-denoising states → runs 30-step denoising
3. Save Stage 1 denoised outputs (ref)
4. `upsample_video()` → save upsampled latent (ref)
5. `noise_video_state()` / `noise_audio_state()` for Stage 2 → draws video_noise_s2 + audio_noise_s2 → save as `stage2_noise.safetensors`
6. Run Stage 2 denoising → save final outputs (ref)

**Validation (V0):** Self-consistency — the script produces a pure-Python MP4 that we can compare against existing `e2e/export_stage2_inputs.py` + `decode_latents.py` pipeline (same seed/params → same output).

---

### M1: Run Zig Stage 1

**Status:** DONE ✅

**Existing driver:** `denoise_stage1`

```bash
bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:denoise_stage1 -- \
    /root/models/ltx-2.3/ltx-2.3-22b.safetensors \
    /root/mixed/stage1_inputs.safetensors \
    /root/mixed/stage1_out/
```

**Output:** `stage1_out/video_latent.bin` + `stage1_out/audio_latent.bin` (raw bf16 binary)

**Validation (V1):** Compare Zig Stage 1 output against M0 reference. Expected: cos_sim ≈ 0.658 video, 0.797 audio (known iterative divergence — see `STAGE1_DIVERGENCE_ANALYSIS.md`).

---

### M2: Bridge Script (Stage 1 → Stage 2)

**Status:** DONE ✅

**Script:** `bridge_s1_to_s2.py` — the key new piece.

**Inputs:**
- `mixed/stage1_out/video_latent.bin` (Zig Stage 1 output, bf16 `[1, T_v1, 128]`)
- `mixed/stage1_out/audio_latent.bin` (Zig Stage 1 output, bf16 `[1, T_a1, 128]`)
- `mixed/stage1_inputs.safetensors` (metadata for shapes/resolution)
- `mixed/stage2_noise.safetensors` (pre-generated Stage 2 noise)

**Process:**
```
1. Load raw .bin → torch tensor (shape from metadata: T_v1 = F_lat × H_s1 × W_s1)
2. Unpatchify video: [1, T_v1, 128] → reshape [1, F_lat, H_s1, W_s1, 128] → permute [1, 128, F_lat, H_s1, W_s1]
3. upsample_video(latent, video_encoder, upsampler) → [1, 128, F_lat, H, W]
4. Patchify upscaled: [1, 128, F_lat, H, W] → [1, F_lat×H_lat×W_lat, 128] = [1, T_v2, 128]
5. Load pre-drawn Stage 2 noise from stage2_noise.safetensors
6. Create Stage 2 states:
   - video: clean=upscaled_patchified, noise=video_noise_s2, mask(full), sigma=0.909375
   - audio: clean=Stage 1 audio output, noise=audio_noise_s2, mask(full), sigma=0.909375
7. Compute Stage 2 positions from output_shape
8. Export stage2_inputs.safetensors (same 12-tensor format as existing)
```

**Key detail — unpatchification:** Stage 1 half-res shapes:
- F_lat = floor((num_frames - 1) / 8) + 1
- H_s1 = (H/2) / 32, W_s1 = (W/2) / 32
- T_v1 = F_lat × H_s1 × W_s1

After upsample (2× spatial):
- T_v2 = F_lat × (H/32) × (W/32) = 4 × T_v1

**Key detail — patchification:** The bridge needs LTX's exact patchification. Safest: call `ltx_core` functions directly rather than hand-rolling reshape+permute.

**Validation (V2):**
- V2a: Compare upscaled video latent against M0 reference → divergence proportional to Stage 1 divergence
- V2b: Compare Stage 2 noise tensors against M0 reference → **must be bitwise identical** (pre-generated from same generator state)
- V2c: Compare Stage 2 positions/masks → must match (computed from same output_shape)

---

### M3: Run Zig Stage 2

**Status:** DONE ✅

**Existing driver:** `denoise_e2e` (with `--bf16-attn` flag for large resolutions)

```bash
bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:denoise_e2e -- \
    /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
    /root/mixed/stage2_inputs.safetensors \
    /root/mixed/stage2_out/ \
    --bf16-attn
```

**Note:** At 1536×1024 (T_v=24576), the vanilla f32 SDPA OOMs (72 GiB attention matrix).
The `--bf16-attn` flag uses `forwardBlock0NativeBf16Attn` with chunked bf16 SDPA
(1024-query chunks) to keep peak memory manageable. Without this flag, the original
f32 path is used (fine for ≤768p resolutions).

**Validation (V3):** Compare Stage 2 output against M0 reference.

---

### M4: VAE Decode → MP4

**Status:** DONE ✅ — Produces playable MP4 with correct visual output.

**Existing script:** `decode_latents.py`

```bash
cd /root/repos/LTX-2
uv run examples/ltx/e2e/decode_latents.py \
    --inputs /root/mixed/stage2_inputs.safetensors \
    --video-latent /root/mixed/stage2_out/video_latent.bin \
    --audio-latent /root/mixed/stage2_out/audio_latent.bin \
    --output /root/mixed/output.mp4
```

**Validation (V4):** Visual + audio comparison of mixed-pipeline MP4 vs pure-Python MP4.

---

## Validation Matrix Summary

| Check | What | Expected | Actual | Status |
| ----- | ---- | -------- | ------ | ------ |
| V0 | M0 self-consistency vs existing pipeline | Bitwise identical | Not run (low priority) | — |
| V1 | Stage 1 output: Zig vs Python | cos_sim ≈ 0.65/0.80 | video=0.657, audio=0.807 | **PASS** ✅ |
| V2a | Upscaled latent: mixed vs ref | Proportional to V1 | video=0.616, audio=0.807 | **PASS** ✅ |
| V2b | Stage 2 noise: mixed vs ref | **Bitwise identical** | **Bitwise identical** | **PASS** ✅ |
| V2c | Stage 2 positions/masks/contexts: mixed vs ref | **Bitwise identical** | **All 6 tensors bitwise identical** | **PASS** ✅ |
| V3 | Stage 2 output: Zig vs Python | cos_sim ≈ 0.66/0.74 | video=0.663, audio=0.742 | **PASS** ✅ |
| V4 | Final MP4: mixed vs pure-Python | Visually reasonable | Playable sunset video | **PASS** ✅ |

**Script:** `validate_mixed_pipeline.py --mixed-dir /root/mixed/`

**Key observations:**
- All bitwise checks (V2b, V2c) pass — Python→Zig→Python data handoff is exact.
- V1 divergence (cos_sim ~0.66/0.81) is the known Stage 1 iterative divergence.
- V2a video (0.616) is slightly below V1 video (0.657) — the upsampler amplifies small differences.
- V3 divergence compounds V1 through Stage 2's A/V cross-attention coupling.
- Audio cos_sim drops from 0.807 (V1) → 0.742 (V3) due to video divergence bleeding through cross-attn.

---

## Implementation Order

1. **M0** — single Python run: reference baseline + Stage 1 inputs + Stage 2 noise
2. **M2** — the bridge script is the most novel piece and the highest-risk
3. **M1 + M3 + M4** — existing drivers, just wire up the commands
4. **Run script** — `run_mixed_pipeline.sh` chaining everything with validation prints

---

## Resolved Questions

1. **Patchify/unpatchify in bridge:** ✅ Solved — `bridge_s1_to_s2.py` uses reshape+permute for unpatchify and calls `ltx_core` VideoLatentTools for Stage 2 state creation.
2. **Stage 2 conditionings:** ✅ Solved — for text-to-video (no image conditioning), all-ones denoise masks and zero clean latent for conditioned frames. The bridge calls `create_initial_state()` directly.
3. **Audio pass-through:** ✅ Confirmed — audio token count preserved through Stage 1→Stage 2 (T_a=126 throughout).
4. **Stage 1 base model path:** ✅ Confirmed — `/root/models/ltx-2.3/ltx-2.3-22b.safetensors`.
5. **Large resolution OOM:** ✅ Solved — added `--bf16-attn` flag to `denoise_e2e.zig` with chunked bf16 SDPA (1024-query chunks). Vanilla f32 SDPA OOMs at 72 GiB (24576² × 32 heads × 4B); chunked bf16 peaks at ~1.5 GiB per chunk.

---

## Key Files Reference

| File | Purpose |
| ---- | ------- |
| `denoise_stage1.zig` | Zig Stage 1 driver (30 steps × 4-pass guidance) |
| `denoise_e2e.zig` | Zig Stage 2 driver (noise init + 3-step Euler, `--bf16-attn` for large res) |
| `model.zig` | Core model (~4800 lines, includes `sdpaNoF32Upcast` chunked attention) |
| `export_mixed_pipeline.py` | M0: full Python ref run + Stage 1 inputs + Stage 2 noise |
| `bridge_s1_to_s2.py` | M2: Zig .bin → unpatchify → upsample → Stage 2 inputs |
| `validate_mixed_pipeline.py` | Boundary validation (V1–V3) |
| `run_mixed_pipeline.sh` | End-to-end chain: M0 → M1 → M2 → M3 → M4 |
| `export_stage1_inputs.py` | Standalone Stage 1 input export |
| `e2e/export_stage2_inputs.py` | Standalone full-Python Stage 2 export |
| `e2e/decode_latents.py` | VAE decode to MP4 |
| `STAGE1_DIVERGENCE_ANALYSIS.md` | Stage 1 divergence analysis |
| Python reference pipeline | `/Users/oboulant/repos/work/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py` |

## Run Parameters (validated config)

- **Resolution:** 1536×1024, 121 frames @ 24 fps
- **Prompt:** "A beautiful sunset over the ocean"
- **Seed:** 10
- **Stage 1:** 30 steps, base checkpoint (`ltx-2.3-22b.safetensors`), 4-pass guidance (cond/CFG/STG/isolated)
- **Stage 2:** 3 steps (distilled), distilled checkpoint (`ltx-2.3-22b-distilled.safetensors`), `--bf16-attn`
- **Sigmas (Stage 2):** [0.909375, 0.725, 0.421875, 0.0]

## Server Info

- GPU server: `root@dev-oboulant`
- Models: `/root/models/ltx-2.3/`
- Zig/ZML: `/root/repos/zml/`
- Python/LTX-2: `/root/repos/LTX-2/`

---

## Full Command Reference

All commands below run on the GPU server. Adjust `OUT`, `PROMPT`, and `SEED` for different runs.

```bash
# ============================================================
# Full mixed pipeline: "Someone walking by the beach at sunset"
# ============================================================
export OUT=/root/mixed_beach
export CKPT=/root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors
export UPSAMPLER=/root/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
export GEMMA=/root/models/gemma-3-12b-it
export PROMPT="Someone walking by the beach at sunset"
export SEED=42

mkdir -p $OUT/stage1_out $OUT/stage2_out

# ---- M0: Python — text encode + noise init + reference trace ----
cd /root/repos/LTX-2
uv run /root/repos/zml/examples/ltx/export_mixed_pipeline.py \
    --output-dir $OUT \
    --prompt "$PROMPT" \
    --seed $SEED \
    --checkpoint $CKPT \
    --spatial-upsampler $UPSAMPLER \
    --gemma-root $GEMMA
    --decode-video

echo "M0 done" && ls -lh $OUT/stage1_inputs.safetensors $OUT/stage2_noise.safetensors $OUT/pipeline_meta.json

# ---- M1: Zig Stage 1 (30 steps, 4-pass guidance) ----
cd /root/repos/zml
ulimit -s unlimited && bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:denoise_stage1 -- \
    $CKPT \
    $OUT/stage1_inputs.safetensors \
    $OUT/stage1_out/

echo "M1 done" && ls -lh $OUT/stage1_out/

# ---- M2: Python — bridge (unpatchify + upsample + Stage 2 noise) ----
cd /root/repos/LTX-2
uv run /root/repos/zml/examples/ltx/bridge_s1_to_s2.py \
    --stage1-video $OUT/stage1_out/video_latent.bin \
    --stage1-audio $OUT/stage1_out/audio_latent.bin \
    --stage2-noise $OUT/stage2_noise.safetensors \
    --meta $OUT/pipeline_meta.json \
    --stage1-inputs $OUT/stage1_inputs.safetensors \
    --output $OUT/stage2_inputs.safetensors \
    --checkpoint $CKPT \
    --spatial-upsampler $UPSAMPLER

echo "M2 done" && ls -lh $OUT/stage2_inputs.safetensors

# ---- M3: Zig Stage 2 (3 steps, distilled, --bf16-attn needed for 1536x1024) ----
cd /root/repos/zml
ulimit -s unlimited && bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:denoise_e2e -- \
    $CKPT \
    $OUT/stage2_inputs.safetensors \
    $OUT/stage2_out/ \
    --bf16-attn

echo "M3 done" && ls -lh $OUT/stage2_out/

# ---- M4: Python — VAE decode → MP4 ----
cd /root/repos/LTX-2
uv run /root/repos/zml/examples/ltx/e2e/decode_latents.py \
    --inputs $OUT/stage2_inputs.safetensors \
    --video-latent $OUT/stage2_out/video_latent.bin \
    --audio-latent $OUT/stage2_out/audio_latent.bin \
    --output $OUT/output.mp4 \
    --checkpoint $CKPT

echo "===== PIPELINE COMPLETE ====="
echo "Output: $OUT/output.mp4"
```
