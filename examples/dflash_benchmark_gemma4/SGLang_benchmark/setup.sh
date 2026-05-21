#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="${VENV_DIR:-${script_dir}/.venv}"
python_bin="${PYTHON:-python3}"

uv_extra_args=(--prerelease=allow)
if [[ -n "${UV_PIP_EXTRA_ARGS:-}" ]]; then
  read -r -a uv_extra_args <<<"${UV_PIP_EXTRA_ARGS}"
fi

pip_extra_args=(--pre)
if [[ -n "${PIP_EXTRA_ARGS:-}" ]]; then
  read -r -a pip_extra_args <<<"${PIP_EXTRA_ARGS}"
fi

if command -v uv >/dev/null 2>&1; then
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    uv venv "${venv_dir}" --python "${python_bin}"
  fi
  uv pip install --python "${venv_dir}/bin/python" -r "${script_dir}/requirements.txt" "${uv_extra_args[@]}"
else
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    "${python_bin}" -m venv "${venv_dir}"
  fi
  "${venv_dir}/bin/python" -m pip install --upgrade pip
  "${venv_dir}/bin/python" -m pip install -r "${script_dir}/requirements.txt" "${pip_extra_args[@]}"
fi

cat <<EOF
SGLang Gemma4 benchmark venv ready:
  ${venv_dir}

Easy 10-sample runs:
  ${script_dir}/run_benchmark.py baseline --samples 10
  ${script_dir}/run_benchmark.py dflash --samples 10
EOF
