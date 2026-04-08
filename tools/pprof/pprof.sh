#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bootstrap_rlocation=""
bootstrap_path="${script_dir}/../profile/bootstrap.sh"
remaining_args=()

for arg in "$@"; do
  case "${arg}" in
    --bootstrap-rlocation=*)
      bootstrap_rlocation="${arg#--bootstrap-rlocation=}"
      ;;
    *)
      remaining_args+=("${arg}")
      ;;
  esac
done
set -- "${remaining_args[@]}"

if [[ ! -f "${bootstrap_path}" ]]; then
  if [[ -n "${RUNFILES_DIR:-}" && -f "${RUNFILES_DIR}/${bootstrap_rlocation}" ]]; then
    bootstrap_path="${RUNFILES_DIR}/${bootstrap_rlocation}"
  elif [[ -f "$0.runfiles/${bootstrap_rlocation}" ]]; then
    bootstrap_path="$0.runfiles/${bootstrap_rlocation}"
  elif [[ -f "${RUNFILES_MANIFEST_FILE:-}" ]]; then
    bootstrap_path="$(
      grep -m1 "^${bootstrap_rlocation} " "${RUNFILES_MANIFEST_FILE}" | cut -d ' ' -f 2-
    )"
  elif [[ -f "$0.runfiles_manifest" ]]; then
    bootstrap_path="$(
      grep -m1 "^${bootstrap_rlocation} " "$0.runfiles_manifest" | cut -d ' ' -f 2-
    )"
  elif [[ -f "$0.runfiles/MANIFEST" ]]; then
    bootstrap_path="$(
      grep -m1 "^${bootstrap_rlocation} " "$0.runfiles/MANIFEST" | cut -d ' ' -f 2-
    )"
  fi
fi

if [[ -z "${bootstrap_path}" || ! -f "${bootstrap_path}" ]]; then
  echo "Unable to locate tools/profile/bootstrap.sh" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${bootstrap_path}"

zml_profile_init_runfiles
zml_profile_enter_workspace

readonly pprof_cache_root_default="$(zml_profile_default_cache_root pprof)"
readonly pprof_cache_root="${ZML_PPROF_CACHE_DIR:-${pprof_cache_root_default}}"
readonly go_version="go1.24.9"

usage() {
  cat >&2 <<EOF_USAGE
usage: $0 --source-license-rlocation=<rlocation> [pprof args...]

The first run downloads a pinned official Go toolchain and builds google/pprof
from the Bazel-fetched pinned upstream source tree.
Override the cache location with ZML_PPROF_CACHE_DIR if needed.
EOF_USAGE
  exit 1
}

detect_go_archive() {
  local os=""
  local arch=""

  os="$(uname -s)"
  arch="$(uname -m)"

  case "${os}/${arch}" in
    Darwin/arm64)
      printf '%s %s\n' \
        "${go_version}.darwin-arm64.tar.gz" \
        "af451b40651d7fb36db1bbbd9c66ddbed28b96d7da48abea50a19f82c6e9d1d6"
      ;;
    Darwin/x86_64)
      printf '%s %s\n' \
        "${go_version}.darwin-amd64.tar.gz" \
        "961aa2ae2b97e428d6d8991367e7c98cb403bac54276b8259aead42a0081591c"
      ;;
    Linux/aarch64|Linux/arm64)
      printf '%s %s\n' \
        "${go_version}.linux-arm64.tar.gz" \
        "9aa1243d51d41e2f93e895c89c0a2daf7166768c4a4c3ac79db81029d295a540"
      ;;
    Linux/x86_64)
      printf '%s %s\n' \
        "${go_version}.linux-amd64.tar.gz" \
        "5b7899591c2dd6e9da1809fde4a2fad842c45d3f6b9deb235ba82216e31e34a6"
      ;;
    *)
      echo "Unsupported platform for bootstrapped Go toolchain: ${os}/${arch}" >&2
      exit 1
      ;;
  esac
}

ensure_go_toolchain() {
  local archive_filename="$1"
  local archive_sha256="$2"
  local archive_url="https://go.dev/dl/${archive_filename}"
  local toolchain_dir="${pprof_cache_root}/toolchains/${archive_filename%.tar.gz}"
  local archive_path="${pprof_cache_root}/toolchains/${archive_filename}"
  local extract_dir="${toolchain_dir}.tmp"
  local go_bin="${toolchain_dir}/go/bin/go"

  mkdir -p "${pprof_cache_root}/toolchains"

  if [[ ! -f "${archive_path}" ]] || [[ "$(zml_profile_sha256_file "${archive_path}")" != "${archive_sha256}" ]]; then
    rm -f "${archive_path}"
    echo "Downloading ${archive_filename}..." >&2
    zml_profile_download_file "${archive_url}" "${archive_path}"
  fi

  if [[ "$(zml_profile_sha256_file "${archive_path}")" != "${archive_sha256}" ]]; then
    echo "Go archive checksum mismatch: ${archive_path}" >&2
    exit 1
  fi

  if [[ ! -x "${go_bin}" ]]; then
    echo "Extracting ${archive_filename}..." >&2
    rm -rf "${extract_dir}" "${toolchain_dir}"
    mkdir -p "${extract_dir}"
    tar -xzf "${archive_path}" -C "${extract_dir}"

    if [[ ! -x "${extract_dir}/go/bin/go" ]]; then
      echo "Unexpected Go archive layout: ${archive_filename}" >&2
      exit 1
    fi

    mv "${extract_dir}" "${toolchain_dir}"
  fi

  printf '%s\n' "${go_bin}"
}

build_pprof() {
  local go_bin="$1"
  local source_root="$2"
  local source_signature="$3"
  local go_identity=""
  local build_id=""
  local build_root=""
  local build_log=""
  local binary_path=""
  local stamp_file=""
  local go_cache_root="${pprof_cache_root}/go-cache"

  go_identity="$(${go_bin} env GOVERSION 2>/dev/null || ${go_bin} version)"
  build_id="$({
    printf '%s\n' "${source_signature}"
    printf '%s\n' "${go_identity}"
  } | zml_profile_sha256_stdin)"
  build_root="${pprof_cache_root}/build-${build_id}"
  build_log="${build_root}/build.log"
  binary_path="${build_root}/pprof"
  stamp_file="${build_root}/source.signature"

  mkdir -p "${build_root}" "${go_cache_root}/mod" "${go_cache_root}/build"

  if [[ ! -x "${binary_path}" ]] || [[ ! -f "${stamp_file}" ]] || \
    [[ "$(cat "${stamp_file}")" != "${build_id}" ]]; then
    echo "Building pprof..." >&2
    rm -f "${binary_path}" "${stamp_file}" "${build_log}"
    if ! (
      cd "${source_root}" && \
        env \
        GOMODCACHE="${go_cache_root}/mod" \
        GOCACHE="${go_cache_root}/build" \
        GOFLAGS="${GOFLAGS:-} -buildvcs=false" \
        "${go_bin}" build -o "${binary_path}" .
    ) >"${build_log}" 2>&1; then
      echo "pprof build failed. See ${build_log}" >&2
      exit 1
    fi
    printf '%s\n' "${build_id}" >"${stamp_file}"
  fi

  printf '%s\n' "${binary_path}"
}

main() {
  local source_license_rlocation=""
  local source_root=""
  local source_signature=""
  local archive_filename=""
  local archive_sha256=""
  local go_bin=""
  local pprof_bin=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --source-license-rlocation=*)
        source_license_rlocation="${1#--source-license-rlocation=}"
        shift
        ;;
      --)
        shift
        break
        ;;
      *)
        break
        ;;
    esac
  done

  if [[ -z "${source_license_rlocation}" ]]; then
    usage
  fi

  mkdir -p "${pprof_cache_root}"
  source_root="$(zml_profile_resolve_source_root "${source_license_rlocation}" pprof)"
  source_signature="$(zml_profile_compute_source_signature "${source_root}")"
  read -r archive_filename archive_sha256 <<<"$(detect_go_archive)"
  go_bin="$(ensure_go_toolchain "${archive_filename}" "${archive_sha256}")"
  pprof_bin="$(build_pprof "${go_bin}" "${source_root}" "${source_signature}")"

  exec "${pprof_bin}" "$@"
}

main "$@"
