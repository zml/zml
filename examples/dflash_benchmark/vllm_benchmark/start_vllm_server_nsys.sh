#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  start_vllm_server_nsys.sh baseline [-- vllm serve args...]
  start_vllm_server_nsys.sh dflash [-- vllm serve args...]

Environment overrides:
  TARGET_MODEL=/var/models/meta-llama/Llama-3.1-8B-Instruct/
  DFLASH_MODEL=/var/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat/
  CUDA_VISIBLE_DEVICES=1
  HOST=0.0.0.0
  PORT=8000
  TENSOR_PARALLEL_SIZE=1
  GPU_MEMORY_UTILIZATION=0.9
  MAX_MODEL_LEN=2048
  MAX_NUM_BATCHED_TOKENS=4096
  MAX_NUM_SEQS=16
  NUM_SPECULATIVE_TOKENS=10

The vLLM executable is resolved as ./.venv/bin/vllm next to this script.
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
vllm_bin="${script_dir}/.venv/bin/vllm"

if [[ ! -x "${vllm_bin}" ]]; then
  echo "error: vLLM binary not found or not executable: ${vllm_bin}" >&2
  echo "Create the venv from the repo root with:" >&2
  echo "  uv venv examples/dflash_benchmark/vllm_benchmark/.venv" >&2
  echo "  uv pip install --python examples/dflash_benchmark/vllm_benchmark/.venv/bin/python -r examples/dflash_benchmark/vllm_benchmark/requirements.txt" >&2
  exit 1
fi

target_model="${TARGET_MODEL:-/var/models/meta-llama/Llama-3.1-8B-Instruct/}"
dflash_model="${DFLASH_MODEL:-/var/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat/}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-1}"
host="${HOST:-0.0.0.0}"
port="${PORT:-8000}"
tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-1}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.9}"
max_model_len="${MAX_MODEL_LEN:-2048}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-4096}"
max_num_seqs="${MAX_NUM_SEQS:-16}"
num_speculative_tokens="${NUM_SPECULATIVE_TOKENS:-10}"

serve_args=(
  "${vllm_bin}" serve "${target_model}"
  --host "${host}"
  --port "${port}"
  --tensor-parallel-size "${tensor_parallel_size}"
  --gpu-memory-utilization "${gpu_memory_utilization}"
  --max-model-len "${max_model_len}"
  --max-num-batched-tokens "${max_num_batched_tokens}"
  --max-num-seqs "${max_num_seqs}"
)

case "${mode}" in
  baseline | target | no-dflash)
    ;;
  dflash)
    serve_args+=(
      --speculative-config
      "{\"method\":\"dflash\",\"model\":\"${dflash_model}\",\"num_speculative_tokens\":${num_speculative_tokens}}"
    )
    ;;
  *)
    echo "error: expected mode 'baseline' or 'dflash', got '${mode}'" >&2
    usage >&2
    exit 2
    ;;
esac

serve_args+=("$@")

echo "Starting vLLM server under nsys (${mode}) on CUDA_VISIBLE_DEVICES=${cuda_visible_devices}" >&2
printf 'Command:' >&2
printf ' %q' "${serve_args[@]}" >&2
printf '\n' >&2

CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" sudo -E nsys launch \
  --session-new=session \
  -t cuda,nvtx,cublas,cusparse,cudnn \
  --inherit-environment=true \
  --cuda-memory-usage=true \
  --cuda-graph-trace=node \
  "${serve_args[@]}"
