#!/usr/bin/env bash
# run_native_t3_slice.sh
#
# Run one T3 contiguous 8-block slice parity check.
#
# Usage (remote):
#   bash /root/repos/LTX-2/scripts/run_native_t3_slice.sh <start> <end> <token_limit> <lora_strength>
# Example:
#   bash /root/repos/LTX-2/scripts/run_native_t3_slice.sh 0 7 128 0.0

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <start_block> <end_block> <token_limit> <lora_strength>"
  exit 1
fi

START="$1"
END="$2"
TOKEN_LIMIT="$3"
LORA="$4"

COUNT=$((END - START + 1))
if [[ "$COUNT" -ne 8 ]]; then
  echo "This checker currently supports exactly 8 contiguous blocks. Got ${COUNT} for [${START}, ${END}]."
  exit 1
fi

SCRIPTS_DIR="/root/repos/LTX-2/scripts"
TRACE_DIR="/root/repos/LTX-2/trace_run"
ZML_ROOT="/root/repos/zml"

BLOCK_RX=$(seq "${START}" "${END}" | paste -sd'|' -)
INCLUDE_RX="^velocity_model\\.transformer_blocks\\.(${BLOCK_RX})(\\.|$)"
PASS_LABEL="t3_slice_${START}_${END}_lora${LORA}"
TRACE_PT="${TRACE_DIR}/acts_stage2_transformer_step_000_${PASS_LABEL}_t${TOKEN_LIMIT}.pt"
CKPT="${TRACE_DIR}/stage2_blocks_${START}_${END}_lora${LORA}_merged.safetensors"
FIXTURE="${TRACE_DIR}/block_slice_native_${START}_${END}_lora${LORA}_t${TOKEN_LIMIT}.safetensors"

echo "========================================================================"
echo "T3 native slice [${START}, ${END}] token=${TOKEN_LIMIT} lora=${LORA}"
echo "========================================================================"

echo "--- Step 1: replay slice trace ---"
cd /root/repos/LTX-2
uv run python "${SCRIPTS_DIR}/replay_stage2_transformer_step.py" \
  --pass-label "${PASS_LABEL}" \
  --capture-inputs \
  --capture-kwargs \
  --all-modules \
  --max-capture-gib 8.0 \
  --distilled-lora-strength "${LORA}" \
  --token-limit "${TOKEN_LIMIT}" \
  --include "${INCLUDE_RX}"

echo "--- Step 2: export slice checkpoint (reindexed 0..7) ---"
uv run python "${SCRIPTS_DIR}/export_stage2_block_slice_checkpoint.py" \
  --start-block "${START}" \
  --end-block "${END}" \
  --distilled-lora-strength "${LORA}" \
  --output "${CKPT}"

echo "--- Step 3: export native slice fixture ---"
uv run python "${SCRIPTS_DIR}/export_block_slice_native_fixture.py" \
  "${TRACE_PT}" \
  "${FIXTURE}" \
  --start-block "${START}" \
  --end-block "${END}"

echo "--- Step 4: run native slice checker ---"
cd "${ZML_ROOT}"
ulimit -s unlimited
bazel run //examples/ltx:block_slice_native_check --@zml//platforms:cuda=true -- "${CKPT}" "${FIXTURE}"

echo "T3 slice parity PASS for [${START}, ${END}] token=${TOKEN_LIMIT} lora=${LORA}"
