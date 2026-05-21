#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  launch_server.sh baseline [-- sglang launch args...]
  launch_server.sh dflash [-- sglang launch args...]

Environment overrides:
  TARGET_MODEL=/var/models/meta-llama/Llama-3.1-8B-Instruct/
  DFLASH_MODEL=/var/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat/
  CUDA_VISIBLE_DEVICES=0
  HOST=0.0.0.0
  PORT=30000
  TP=1
  CONTEXT_LENGTH=2048
  MEM_FRACTION_STATIC=0.7
  CUDA_GRAPH_MAX_BS=8
  DTYPE=float16
  LOG_LEVEL=warning

Optional DFlash overrides, left unset by default so SGLang can infer from
the draft config:
  SPECULATIVE_NUM_DRAFT_TOKENS=9
  SPECULATIVE_DFLASH_BLOCK_SIZE=9
  SPECULATIVE_DFLASH_DRAFT_WINDOW_SIZE=128

The Python binary is resolved as ./.venv/bin/python next to this script unless
PYTHON is set explicitly.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

mode="$1"
shift

case "${mode}" in
  -h | --help | help)
    usage
    exit 0
    ;;
esac

if [[ "${1:-}" == "--" ]]; then
  shift
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-${script_dir}/.venv/bin/python}"

if [[ ! -x "${python_bin}" ]]; then
  echo "error: Python binary not found or not executable: ${python_bin}" >&2
  echo "Run setup first:" >&2
  echo "  ${script_dir}/setup.sh" >&2
  exit 1
fi

target_model="${TARGET_MODEL:-/var/models/meta-llama/Llama-3.1-8B-Instruct/}"
dflash_model="${DFLASH_MODEL:-/var/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat/}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
host="${HOST:-0.0.0.0}"
port="${PORT:-30000}"
tp="${TP:-1}"
context_length="${CONTEXT_LENGTH:-2048}"
mem_fraction_static="${MEM_FRACTION_STATIC:-0.7}"
cuda_graph_max_bs="${CUDA_GRAPH_MAX_BS:-8}"
dtype="${DTYPE:-float16}"
log_level="${LOG_LEVEL:-warning}"

server_args=(
  "${python_bin}" -m sglang.launch_server
  --model-path "${target_model}"
  --host "${host}"
  --port "${port}"
  --tp-size "${tp}"
  --context-length "${context_length}"
  --mem-fraction-static "${mem_fraction_static}"
  --cuda-graph-max-bs "${cuda_graph_max_bs}"
  --dtype "${dtype}"
  --log-level "${log_level}"
)

case "${mode}" in
  baseline | target | no-dflash)
    ;;
  dflash)
    server_args+=(
      --speculative-algorithm DFLASH
      --speculative-draft-model-path "${dflash_model}"
    )
    if [[ -n "${SPECULATIVE_NUM_DRAFT_TOKENS:-}" ]]; then
      server_args+=(--speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}")
    fi
    if [[ -n "${SPECULATIVE_DFLASH_BLOCK_SIZE:-}" ]]; then
      server_args+=(--speculative-dflash-block-size "${SPECULATIVE_DFLASH_BLOCK_SIZE}")
    fi
    if [[ -n "${SPECULATIVE_DFLASH_DRAFT_WINDOW_SIZE:-}" ]]; then
      server_args+=(--speculative-dflash-draft-window-size "${SPECULATIVE_DFLASH_DRAFT_WINDOW_SIZE}")
    fi
    ;;
  *)
    echo "error: expected mode 'baseline' or 'dflash', got '${mode}'" >&2
    usage >&2
    exit 2
    ;;
esac

server_args+=("$@")

echo "Starting SGLang server (${mode}) on CUDA_VISIBLE_DEVICES=${cuda_visible_devices}" >&2
printf 'Command:' >&2
printf ' %q' "${server_args[@]}" >&2
printf '\n' >&2

CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" exec "${server_args[@]}"
