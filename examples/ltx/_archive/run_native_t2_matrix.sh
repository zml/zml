#!/usr/bin/env bash
# run_native_t2_matrix.sh
#
# Run the remaining T2 validation matrix for block0_native_check:
#   - LoRA 0.0: t256, t512
#   - LoRA 0.5: t128, t256, t512
#
# Already validated (PASS): LoRA 0.0 / t128
#
# Usage (from /root/repos/LTX-2 on the remote machine):
#   bash /root/repos/zml/examples/ltx/run_native_t2_matrix.sh
#
# Paths are set for the standard remote layout:
#   scripts:   /root/repos/LTX-2/scripts/
#   traces:    /root/repos/LTX-2/trace_run/
#   zml root:  /root/repos/zml

set -uo pipefail

SCRIPTS_DIR="/root/repos/LTX-2/scripts"
TRACE_DIR="/root/repos/LTX-2/trace_run"
ZML_ROOT="/root/repos/zml"
CHECKER_TARGET="//examples/ltx:block0_native_check"
CUDA_FLAG="--@zml//platforms:cuda=true"

# Ordered list: "lora_str token_limit"
MATRIX=(
    "0.0 256"
    "0.0 512"
    "0.5 128"
    "0.5 256"
    "0.5 512"
)

declare -A RESULTS

# ─── helpers ─────────────────────────────────────────────────────────────────
# Return 0 if a safetensors file exists AND contains all required SST keys.
_ckpt_has_sst_keys() {
    local ckpt="$1"
    [[ -f "${ckpt}" ]] || return 1
    python3 - "${ckpt}" <<'EOF'
import sys
from safetensors import safe_open
path = sys.argv[1]
required = {
    "velocity_model.transformer_blocks.0.scale_shift_table",
    "velocity_model.transformer_blocks.0.audio_scale_shift_table",
    "velocity_model.transformer_blocks.0.scale_shift_table_a2v_ca_video",
    "velocity_model.transformer_blocks.0.scale_shift_table_a2v_ca_audio",
    "velocity_model.transformer_blocks.0.prompt_scale_shift_table",
    "velocity_model.transformer_blocks.0.audio_prompt_scale_shift_table",
}
with safe_open(path, framework="pt", device="cpu") as f:
    keys = set(f.keys())
missing = required - keys
if missing:
    print(f"[stale] missing keys: {sorted(missing)}", flush=True)
    sys.exit(1)
sys.exit(0)
EOF
}

# ─── Step 0: export LoRA 0.5 merged checkpoint (once) ───────────────────────
CKPT_LORA05="${TRACE_DIR}/stage2_block0_lora0.5_merged.safetensors"
if _ckpt_has_sst_keys "${CKPT_LORA05}"; then
    echo "[skip] LoRA 0.5 checkpoint exists and has all SST keys: ${CKPT_LORA05}"
else
    [[ -f "${CKPT_LORA05}" ]] && echo "[stale] re-exporting LoRA 0.5 checkpoint (missing SST keys)"
    echo "========================================================================"
    echo "Exporting LoRA 0.5 merged checkpoint..."
    echo "========================================================================"
    cd /root/repos/LTX-2
    uv run python "${SCRIPTS_DIR}/export_stage2_block0_checkpoint.py" \
        --distilled-lora-strength 0.5 \
        --output "${CKPT_LORA05}"
    if [[ $? -ne 0 ]]; then
        echo "ERROR: LoRA 0.5 checkpoint export failed — aborting."
        exit 1
    fi
    echo "LoRA 0.5 checkpoint written: ${CKPT_LORA05}"
    # Invalidate any fixtures derived from the stale checkpoint.
    for tl in 128 256 512; do
        stale="${TRACE_DIR}/block0_native_lora0.5_t${tl}.safetensors"
        [[ -f "${stale}" ]] && { rm "${stale}"; echo "[cleanup] removed stale fixture: ${stale}"; }
    done
fi

# ─── Main loop ───────────────────────────────────────────────────────────────
for entry in "${MATRIX[@]}"; do
    lora_str=$(echo "${entry}" | awk '{print $1}')
    token_limit=$(echo "${entry}" | awk '{print $2}')
    combo_key="lora${lora_str}_t${token_limit}"

    # Normalise lora tag: "0.0" → "lora0.0", "0.5" → "lora0.5"
    PASS_LABEL="m6_native_lora${lora_str}"
    TRACE_PT="${TRACE_DIR}/acts_stage2_transformer_step_000_${PASS_LABEL}_t${token_limit}.pt"
    FIXTURE_ST="${TRACE_DIR}/block0_native_lora${lora_str}_t${token_limit}.safetensors"

    if [[ "${lora_str}" == "0.0" ]]; then
        CKPT="${TRACE_DIR}/stage2_block0_lora0.0_merged.safetensors"
    else
        CKPT="${TRACE_DIR}/stage2_block0_lora0.5_merged.safetensors"
    fi

    echo ""
    echo "========================================================================"
    echo " combo: ${combo_key}"
    echo "========================================================================"

    # ── Step 1: replay ────────────────────────────────────────────────────────
    if [[ -f "${TRACE_PT}" ]]; then
        echo "[skip] replay trace exists: ${TRACE_PT}"
    else
        echo "--- Step 1: replay (lora=${lora_str}, token_limit=${token_limit}) ---"
        cd /root/repos/LTX-2
        uv run python "${SCRIPTS_DIR}/replay_stage2_transformer_step.py" \
            --pass-label "${PASS_LABEL}" \
            --capture-inputs \
            --capture-kwargs \
            --all-modules \
            --max-capture-gib 8.0 \
            --distilled-lora-strength "${lora_str}" \
            --token-limit "${token_limit}" \
            --include '^velocity_model\.transformer_blocks\.0(\.|$)'
        if [[ $? -ne 0 ]]; then
            echo "ERROR: replay failed for ${combo_key}"
            RESULTS["${combo_key}"]="FAIL(replay)"
            continue
        fi
    fi

    # ── Step 2: export fixture ────────────────────────────────────────────────
    if [[ -f "${FIXTURE_ST}" ]]; then
        echo "[skip] fixture exists: ${FIXTURE_ST}"
    else
        echo "--- Step 2: export fixture ---"
        cd /root/repos/LTX-2
        uv run python "${SCRIPTS_DIR}/export_block0_native_fixture.py" \
            "${TRACE_PT}" \
            "${FIXTURE_ST}"
        if [[ $? -ne 0 ]]; then
            echo "ERROR: fixture export failed for ${combo_key}"
            RESULTS["${combo_key}"]="FAIL(export)"
            continue
        fi
    fi

    # ── Step 3: parity check ──────────────────────────────────────────────────
    echo "--- Step 3: parity check ---"
    cd "${ZML_ROOT}"
    if bazel run "${CHECKER_TARGET}" "${CUDA_FLAG}" -- "${CKPT}" "${FIXTURE_ST}"; then
        RESULTS["${combo_key}"]="PASS"
        echo "=== ${combo_key}: PASS ==="
    else
        RESULTS["${combo_key}"]="FAIL(check)"
        echo "=== ${combo_key}: FAIL ==="
    fi
done

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "T2 NATIVE MATRIX SUMMARY"
echo "========================================================================"
all_pass=1
for entry in "${MATRIX[@]}"; do
    lora_str=$(echo "${entry}" | awk '{print $1}')
    token_limit=$(echo "${entry}" | awk '{print $2}')
    combo_key="lora${lora_str}_t${token_limit}"
    result="${RESULTS[${combo_key}]:-NOT_RUN}"
    echo "  ${combo_key}: ${result}"
    if [[ "${result}" != "PASS" ]]; then
        all_pass=0
    fi
done
echo "========================================================================"
if [[ "${all_pass}" -eq 1 ]]; then
    echo "ALL PASS — T2 matrix complete."
    exit 0
else
    echo "Some combinations FAILED — see above."
    exit 1
fi
