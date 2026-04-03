#!/usr/bin/env bash
# run_mixed_pipeline.sh — End-to-end mixed LTX-2 pipeline
#
# Python(text enc) → Zig(Stage 1) → Python(upsample) → Zig(Stage 2) → Python(VAE decode)
#
# Usage:
#   bash examples/ltx/run_mixed_pipeline.sh [output_dir]
#
# Prerequisites:
#   - Built Zig targets: denoise_stage1, denoise_e2e
#   - Python env with ltx_core, ltx_pipelines available (uv)
#   - Model weights at /root/models/ltx-2.3/
#
# Default paths (edit for your setup):

set -euo pipefail

OUTPUT_DIR="${1:-/root/mixed}"
SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"

# Model paths
BASE_CKPT="/root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors"
DISTILLED_CKPT="/root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors"
SPATIAL_UPSAMPLER="/root/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA_ROOT="/root/models/gemma-3-12b-it"

# Generation params
PROMPT="A beautiful sunset over the ocean"
SEED=10

echo "============================================================"
echo "Mixed LTX-2 Pipeline"
echo "============================================================"
echo "Output: ${OUTPUT_DIR}"
echo "Prompt: ${PROMPT}"
echo "Seed:   ${SEED}"
echo ""

mkdir -p "${OUTPUT_DIR}/stage1_out" "${OUTPUT_DIR}/stage2_out" "${OUTPUT_DIR}/ref"

# ============================================================
# M0: Export reference + mixed pipeline inputs
# ============================================================
echo ""
echo "===== M0: Export reference + Stage 1 inputs + Stage 2 noise ====="
echo ""

cd /root/repos/LTX-2
uv run "${SCRIPTS_DIR}/export_mixed_pipeline.py" \
    --output-dir "${OUTPUT_DIR}" \
    --prompt "${PROMPT}" \
    --seed "${SEED}" \
    --checkpoint "${BASE_CKPT}" \
    --spatial-upsampler "${SPATIAL_UPSAMPLER}" \
    --gemma-root "${GEMMA_ROOT}"

echo ""
echo "M0 complete. Outputs:"
ls -lh "${OUTPUT_DIR}"/stage1_inputs.safetensors \
       "${OUTPUT_DIR}"/stage2_noise.safetensors \
       "${OUTPUT_DIR}"/pipeline_meta.json
ls -lh "${OUTPUT_DIR}"/ref/

# ============================================================
# M1: Run Zig Stage 1
# ============================================================
echo ""
echo "===== M1: Zig Stage 1 denoising ====="
echo ""

cd /root/repos/zml
bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:denoise_stage1 -- \
    "${BASE_CKPT}" \
    "${OUTPUT_DIR}/stage1_inputs.safetensors" \
    "${OUTPUT_DIR}/stage1_out/"

echo ""
echo "M1 complete. Outputs:"
ls -lh "${OUTPUT_DIR}/stage1_out/"

# ============================================================
# M2: Bridge (upsample + Stage 2 noise init)
# ============================================================
echo ""
echo "===== M2: Bridge Stage 1 → Stage 2 ====="
echo ""

cd /root/repos/LTX-2
uv run "${SCRIPTS_DIR}/bridge_s1_to_s2.py" \
    --stage1-video "${OUTPUT_DIR}/stage1_out/video_latent.bin" \
    --stage1-audio "${OUTPUT_DIR}/stage1_out/audio_latent.bin" \
    --stage2-noise "${OUTPUT_DIR}/stage2_noise.safetensors" \
    --meta "${OUTPUT_DIR}/pipeline_meta.json" \
    --stage1-inputs "${OUTPUT_DIR}/stage1_inputs.safetensors" \
    --output "${OUTPUT_DIR}/stage2_inputs.safetensors" \
    --checkpoint "${BASE_CKPT}" \
    --spatial-upsampler "${SPATIAL_UPSAMPLER}" \
    --gemma-root "${GEMMA_ROOT}"

echo ""
echo "M2 complete. Output:"
ls -lh "${OUTPUT_DIR}/stage2_inputs.safetensors"

# ============================================================
# M3: Run Zig Stage 2
# ============================================================
echo ""
echo "===== M3: Zig Stage 2 denoising ====="
echo ""

cd /root/repos/zml
bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:denoise_e2e -- \
    "${DISTILLED_CKPT}" \
    "${OUTPUT_DIR}/stage2_inputs.safetensors" \
    "${OUTPUT_DIR}/stage2_out/"

echo ""
echo "M3 complete. Outputs:"
ls -lh "${OUTPUT_DIR}/stage2_out/"

# ============================================================
# M4: VAE decode → MP4
# ============================================================
echo ""
echo "===== M4: VAE decode → MP4 ====="
echo ""

cd /root/repos/LTX-2
uv run "${SCRIPTS_DIR}/e2e/decode_latents.py" \
    --inputs "${OUTPUT_DIR}/stage2_inputs.safetensors" \
    --video-latent "${OUTPUT_DIR}/stage2_out/video_latent.bin" \
    --audio-latent "${OUTPUT_DIR}/stage2_out/audio_latent.bin" \
    --output "${OUTPUT_DIR}/output.mp4" \
    --checkpoint "${DISTILLED_CKPT}" \
    --spatial-upsampler "${SPATIAL_UPSAMPLER}" \
    --gemma-root "${GEMMA_ROOT}"

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Output video: ${OUTPUT_DIR}/output.mp4"
echo ""
echo "Validation:"
echo "  V1: Compare ${OUTPUT_DIR}/stage1_out/ vs ${OUTPUT_DIR}/ref/stage1_outputs.safetensors"
echo "  V2: Compare ${OUTPUT_DIR}/stage2_inputs.safetensors vs ${OUTPUT_DIR}/ref/stage2_inputs.safetensors"
echo "  V3: Compare ${OUTPUT_DIR}/stage2_out/ vs ${OUTPUT_DIR}/ref/stage2_outputs.safetensors"
echo "  V4: Visual comparison of ${OUTPUT_DIR}/output.mp4"
