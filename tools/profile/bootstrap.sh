#!/usr/bin/env bash

if [[ "${ZML_PROFILE_BOOTSTRAP_SOURCED:-0}" == "1" ]]; then
  return 0
fi
readonly ZML_PROFILE_BOOTSTRAP_SOURCED=1

zml_profile_init_runfiles() {
  if declare -F rlocation >/dev/null 2>&1; then
    return
  fi

  if [[ ! -d "${RUNFILES_DIR:-}" && ! -f "${RUNFILES_MANIFEST_FILE:-}" ]]; then
    if [[ -f "$0.runfiles_manifest" ]]; then
      export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
    elif [[ -f "$0.runfiles/MANIFEST" ]]; then
      export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
    elif [[ -d "$0.runfiles" ]]; then
      export RUNFILES_DIR="$0.runfiles"
    fi
  fi

  if [[ -f "${RUNFILES_DIR:-}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
    # shellcheck disable=SC1091
    source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
  elif [[ -f "${RUNFILES_MANIFEST_FILE:-}" ]]; then
    local runfiles_bash_path=""
    runfiles_bash_path="$(
      grep -m1 '^bazel_tools/tools/bash/runfiles/runfiles.bash ' "${RUNFILES_MANIFEST_FILE}" \
        | cut -d ' ' -f 2-
    )"
    if [[ -n "${runfiles_bash_path}" ]]; then
      # shellcheck disable=SC1090
      source "${runfiles_bash_path}"
    fi
  fi

  if ! declare -F rlocation >/dev/null 2>&1; then
    echo "Unable to initialize Bazel runfiles library" >&2
    exit 1
  fi
}

zml_profile_enter_workspace() {
  if [[ -n "${BUILD_WORKSPACE_DIRECTORY:-}" ]]; then
    cd "${BUILD_WORKSPACE_DIRECTORY}"
  fi
}

zml_profile_default_cache_root() {
  local tool_name="$1"
  printf '%s\n' "${XDG_CACHE_HOME:-${HOME}/.cache}/zml/${tool_name}"
}

zml_profile_sha256_file() {
  local file_path="$1"

  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${file_path}" | awk '{print $1}'
    return
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${file_path}" | awk '{print $1}'
    return
  fi

  echo "Missing sha256 tool (need shasum or sha256sum)" >&2
  exit 1
}

zml_profile_sha256_stdin() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 | awk '{print $1}'
    return
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum | awk '{print $1}'
    return
  fi

  echo "Missing sha256 tool (need shasum or sha256sum)" >&2
  exit 1
}

zml_profile_download_file() {
  local url="$1"
  local out="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 1 -o "${out}" "${url}"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -O "${out}" "${url}"
    return
  fi

  echo "Missing downloader (need curl or wget)" >&2
  exit 1
}

zml_profile_resolve_source_root() {
  local license_rlocation="$1"
  local tool_name="$2"
  local license_path=""

  license_path="$(rlocation "${license_rlocation}")"
  if [[ -z "${license_path}" || ! -f "${license_path}" ]]; then
    echo "Unable to resolve ${tool_name} source root from runfiles: ${license_rlocation}" >&2
    exit 1
  fi

  dirname "${license_path}"
}

zml_profile_compute_source_signature() {
  local source_root="$1"

  (
    cd "${source_root}"
    find . -type f | LC_ALL=C sort | while IFS= read -r relpath; do
      printf '%s\n' "${relpath}"
      zml_profile_sha256_file "${relpath}"
    done
  ) | zml_profile_sha256_stdin
}

zml_profile_parallelism() {
  if command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.ncpu 2>/dev/null && return
  fi
  if command -v nproc >/dev/null 2>&1; then
    nproc && return
  fi
  echo 4
}

zml_profile_configure_generator() {
  if command -v ninja >/dev/null 2>&1; then
    printf '%s\n' "Ninja"
  fi
}
