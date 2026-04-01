#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Runs the Qwen 3.5 model with any number of input media files.
Example usage:
  run.sh --model "/var/models/Qwen/Qwen3.5-0.8B/" --image "cat.jpg" --image "chart.png" --video "clip.mp4" --prompt "Describe all media"
Notes:
- Media files are expected in media_test/data.
- You can pass any number of --image/--video flags.
- Lower --fps reduces sampled video frames.
- Lower --max-pixels reduces image and video frame resolution while preserving aspect ratio.

Good source of video samples: https://www.pexels.com/videos/
USAGE
}

model=""
prompt=""
fps="2.0"
max_pixels="262144"
images=()
videos=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      model="${2:-}"
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
    --max-pixels)
      max_pixels="${2:-}"
      shift 2
      ;;
    --image)
      images+=("${2:-}")
      shift 2
      ;;
    --video)
      videos+=("${2:-}")
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

if [[ -z "${model}" || -z "${prompt}" ]]; then
  echo "Missing required --model or --prompt." >&2
  usage
  exit 1
fi

if [[ ${#images[@]} -eq 0 && ${#videos[@]} -eq 0 ]]; then
  echo "Pass at least one --image or --video." >&2
  usage
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"
work_dir="${script_dir}"

for image in "${images[@]}"; do
  if [[ ! -f "${work_dir}/data/${image}" ]]; then
    echo "Image not found: ${work_dir}/data/${image}" >&2
    exit 1
  fi
done
for video in "${videos[@]}"; do
  if [[ ! -f "${work_dir}/data/${video}" ]]; then
    echo "Video not found: ${work_dir}/data/${video}" >&2
    exit 1
  fi
done

venv_dir="${work_dir}/.venv"
requirements_file="${work_dir}/requirements.txt"
python_bin="${venv_dir}/bin/python"
pip_bin="${venv_dir}/bin/pip"

if [[ ! -d "${venv_dir}" ]]; then
  python3 -m venv "${venv_dir}"
fi

"${python_bin}" -m pip install --quiet --upgrade pip
"${pip_bin}" install --quiet -r "${requirements_file}"

media_inputs_file="${work_dir}/data/media_input.safetensors"

echo "PYTHON: Preprocessing media..."
python_args=(
  "${work_dir}/generate_media_inputs.py"
  --model "${model}"
  --out "${media_inputs_file}"
  --fps "${fps}"
  --max-pixels "${max_pixels}"
)
for image in "${images[@]}"; do
  python_args+=(--image "${work_dir}/data/${image}")
done
for video in "${videos[@]}"; do
  python_args+=(--video "${work_dir}/data/${video}")
done

"${python_bin}" "${python_args[@]}"

echo -e "\nZML: Running model..."
(
  cd "${repo_root}"
  bazel run //examples/qwen3_5:media_test --@zml//platforms:cuda=true -- \
    --model="${model}" \
    --prompt="${prompt}" \
    --media-input-file="${media_inputs_file}"
)
