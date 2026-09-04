#!/usr/bin/env bash
# Compile MiniMax-H3 Super once, then serve.
# Tensor-parallel degree is 1/2/4/8 from the visible GPU count. Leftover
# devices are dropped from CUDA_VISIBLE_DEVICES so they are not opened.
#
#   ./examples/minimax_h3/serve.sh
#   CUDA_VISIBLE_DEVICES=2,3 ./examples/minimax_h3/serve.sh
#   H3_SKUS=5s PORT=8081 ./examples/minimax_h3/serve.sh
#   MODEL=/var/models/MiniMaxAI/MiniMax-H3 ./examples/minimax_h3/serve.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8080}"
DEVICES="${DEVICES:-0}"
MODEL="${MODEL:-hf://MiniMaxAI/MiniMax-H3}"

if [ -z "${SOL_ATTN_LIB:-}" ] && [ -f "$ROOT/output/sol-attn/libh3_sol_attn_sm100.so" ]; then
  export SOL_ATTN_LIB="$ROOT/output/sol-attn/libh3_sol_attn_sm100.so"
fi
export ZML_AUTOTUNE_CACHE_DIR="${ZML_AUTOTUNE_CACHE_DIR:-$ROOT/output/xla-cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$ZML_AUTOTUNE_CACHE_DIR/cuda}"
mkdir -p "${ZML_AUTOTUNE_CACHE_DIR}" "${CUDA_CACHE_PATH}"

visible="${CUDA_VISIBLE_DEVICES:-all}"
echo "port=${PORT}  CUDA_VISIBLE_DEVICES=${visible}  H3_SKUS=${H3_SKUS:-all}"
echo "  model=${MODEL}"
echo "  http://127.0.0.1:${PORT}"
host_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [ -n "${host_ip}" ]; then
  echo "  http://${host_ip}:${PORT}"
fi
echo

extra=()
if [ "${DEVICES}" != "0" ]; then
  extra+=(--devices="${DEVICES}")
fi

exec bazel run -c opt //examples/minimax_h3 --@zml//platforms:cuda=true -- \
  --model="${MODEL}" --port="${PORT}" "${extra[@]}" "$@"
