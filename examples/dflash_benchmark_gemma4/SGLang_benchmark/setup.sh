#!/usr/bin/env bash
set -euo pipefail

workspace="${SGLANG_TESTS_DIR:-${HOME}/sglang_tests}"
venv_dir="${VENV_DIR:-${workspace}/.venv}"
python_bin="${PYTHON:-python3}"
sglang_pr_url="${SGLANG_PR_URL:-git+https://github.com/sgl-project/sglang.git@refs/pull/23000/head#subdirectory=python}"
protoc_shim="${workspace}/bin/protoc"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: setup requires uv on PATH" >&2
  exit 1
fi

mkdir -p "${workspace}"

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  uv venv "${venv_dir}" --python "${python_bin}"
fi

install_sglang() {
  uv pip install --python "${venv_dir}/bin/python" "${sglang_pr_url}"
}

if ! install_sglang; then
  cat >&2 <<EOF
Initial SGLang PR install failed. Installing the known gh200 source-build helpers:
  - user-local Rust toolchain via rustup, if cargo is missing
  - grpcio-tools in ${venv_dir}
  - ${protoc_shim} shim that runs python -m grpc_tools.protoc
EOF

  if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  fi

  uv pip install --python "${venv_dir}/bin/python" grpcio-tools
  mkdir -p "${workspace}/bin"
  cat >"${protoc_shim}" <<EOF
#!/usr/bin/env bash
exec "${venv_dir}/bin/python" -m grpc_tools.protoc "\$@"
EOF
  chmod +x "${protoc_shim}"

  PATH="${HOME}/.cargo/bin:${workspace}/bin:${PATH}" PROTOC="${protoc_shim}" install_sglang
fi

cat <<EOF
SGLang PR benchmark workspace ready:
  workspace: ${workspace}
  venv:      ${venv_dir}

Use this environment when starting SGLang manually:
  export PATH="${HOME}/.cargo/bin:${workspace}/bin:\${PATH}"
  export PROTOC="${protoc_shim}"

Request benchmarks are in:
  $(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EOF
