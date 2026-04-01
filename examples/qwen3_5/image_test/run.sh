#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Example Usage:
  run.sh --model "/var/models/Qwen/Qwen3.5-0.8B/" --image "tigre.jpg" --prompt "What is in this picture?"
NB: image is expected to be in data folder
EOF
}

model=""
image=""
prompt=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      model="${2:-}"
      shift 2
      ;;
    --image)
      image="${2:-}"
      shift 2
      ;;
    --prompt)
      prompt="${2:-}"
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

if [[ -z "${model}" || -z "${image}" || -z "${prompt}" ]]; then
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

image_file="${work_dir}/data/${image}"
image_name="$(basename "${image_file}")"
suffix="${image_name%.*}"
image_inputs_file="${work_dir}/data/image_input_${suffix}.safetensors"

echo "PYTHON: Preprocessing image..."
"${python_bin}" "${work_dir}/generate_image_inputs.py" \
  --model "${model}" \
  --image "${image_file}" \
  --out "${image_inputs_file}"


echo -e "\nZML: Running model..."
(
  cd "${repo_root}"
  bazel run //examples/qwen3_5:image_test --@zml//platforms:cuda=true -- --model="${model}" --prompt="${prompt}" --pixel-values-file="${image_inputs_file}"
)
