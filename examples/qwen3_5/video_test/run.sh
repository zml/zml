#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Example Usage:
  run.sh --model "/var/models/Qwen/Qwen3.5-0.8B/" --video "sample.mp4" --prompt "What is happening in this video?" --fps 2.0
NB: video is expected to be in data folder
USAGE
}

model=""
video=""
prompt=""
fps="2.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      model="${2:-}"
      shift 2
      ;;
    --video)
      video="${2:-}"
      shift 2
      ;;
    --prompt)
      prompt="${2:-}"
      shift 2
      ;;
    --fps)
      fps="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${model}" || -z "${video}" || -z "${prompt}" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
work_dir="${script_dir}"

venv_dir="${work_dir}/.venv"
requirements_file="${work_dir}/requirements.txt"
python_bin="${venv_dir}/bin/python"
pip_bin="${venv_dir}/bin/pip"

if [[ ! -d "${venv_dir}" ]]; then
  python3 -m venv "${venv_dir}"
fi

"${python_bin}" -m pip install --quiet --upgrade pip
"${pip_bin}" install --quiet -r "${requirements_file}"

video_file="${work_dir}/data/${video}"
video_name="$(basename "${video_file}")"
suffix="${video_name%.*}"
video_inputs_file="${work_dir}/data/video_input_${suffix}.safetensors"

echo "PYTHON: Preprocessing video..."
"${python_bin}" "${work_dir}/generate_video_inputs.py" \
  --model "${model}" \
  --video "${video_file}" \
  --out "${video_inputs_file}" \
  --fps "${fps}"

echo -e "\nZML: Running model..."
(
  cd "${repo_root}"
  bazel run //examples/qwen3_5:video_test --@zml//platforms:cuda=true -- --model="${model}" --prompt="${prompt}" --pixel-values-file="${video_inputs_file}"
)
