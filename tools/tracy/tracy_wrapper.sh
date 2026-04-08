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

readonly cache_root_default="$(zml_profile_default_cache_root tracy)"
readonly cache_root="${ZML_TRACY_CACHE_DIR:-${cache_root_default}}"

usage() {
  cat >&2 <<EOF_USAGE
usage: $0 --tool=<tracy-profiler|tracy-capture|tracy-capture-daemon> \
    --source-license-rlocation=<rlocation> [args...]

The first run builds Tracy from the Bazel-fetched pinned upstream source tree.
Override the cache location with ZML_TRACY_CACHE_DIR if needed.
EOF_USAGE
  exit 1
}

ensure_tools() {
  if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake is required to build Tracy" >&2
    exit 1
  fi
}

build_with_cmake() {
  local build_root="$1"
  local source_dir="$2"
  local target_name="$3"
  local stamp_file="${build_root}/source.signature"
  local source_signature="$4"
  shift 4

  local configure_log="${build_root}/configure.log"
  local build_log="${build_root}/build.log"
  local -a extra_args=("$@")
  local generator=""
  local -a configure_cmd=()

  mkdir -p "${build_root}"
  generator="$(zml_profile_configure_generator || true)"

  configure_cmd=(
    cmake
    -S "${source_dir}"
    -B "${build_root}"
    -DCMAKE_BUILD_TYPE=Release
    -DNO_ISA_EXTENSIONS=ON
    -DNO_LTO=ON
    -DNO_CCACHE=ON
  )
  if [[ -n "${generator}" ]]; then
    configure_cmd+=(-G "${generator}")
  fi
  configure_cmd+=("${extra_args[@]}")

  if [[ ! -x "${build_root}/${target_name}" ]] || [[ ! -f "${stamp_file}" ]] || \
    [[ "$(cat "${stamp_file}")" != "${source_signature}" ]]; then
    echo "Building ${target_name}..." >&2
    rm -f "${configure_log}" "${build_log}" "${stamp_file}"

    if ! "${configure_cmd[@]}" >"${configure_log}" 2>&1; then
      echo "Tracy configure failed. See ${configure_log}" >&2
      exit 1
    fi

    if ! cmake --build "${build_root}" --config Release --target "${target_name}" \
      --parallel "$(zml_profile_parallelism)" >"${build_log}" 2>&1; then
      echo "Tracy build failed. See ${build_log}" >&2
      exit 1
    fi

    printf '%s\n' "${source_signature}" >"${stamp_file}"
  fi

  printf '%s\n' "${build_root}/${target_name}"
}

ensure_profiler() {
  local source_root="$1"
  local source_signature="$2"
  local build_root="${cache_root}/build-profiler-${source_signature}"

  build_with_cmake \
    "${build_root}" \
    "${source_root}/profiler" \
    "tracy-profiler" \
    "${source_signature}" \
    -DNO_FILESELECTOR=ON \
    -DCMAKE_CXX_FLAGS=-DTRACY_NO_FILESELECTOR \
    -DDOWNLOAD_CAPSTONE=ON \
    -DDOWNLOAD_GLFW=ON \
    -DDOWNLOAD_FREETYPE=ON \
    -DDOWNLOAD_LIBCURL=ON \
    -DDOWNLOAD_PUGIXML=ON
}

ensure_capture_tool() {
  local source_root="$1"
  local source_signature="$2"
  local tool_name="$3"
  local build_root="${cache_root}/build-capture-${source_signature}"

  build_with_cmake \
    "${build_root}" \
    "${source_root}/capture" \
    "${tool_name}" \
    "${source_signature}" \
    -DDOWNLOAD_CAPSTONE=ON \
    -DDOWNLOAD_GLFW=ON \
    -DDOWNLOAD_FREETYPE=ON \
    -DDOWNLOAD_LIBCURL=ON \
    -DDOWNLOAD_PUGIXML=ON
}

main() {
  ensure_tools

  local tool=""
  local source_license_rlocation=""
  local source_root=""
  local source_signature=""
  local tool_path=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --tool=*)
        tool="${1#--tool=}"
        shift
        ;;
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

  if [[ -z "${tool}" || -z "${source_license_rlocation}" ]]; then
    usage
  fi

  mkdir -p "${cache_root}"
  source_root="$(zml_profile_resolve_source_root "${source_license_rlocation}" Tracy)"
  source_signature="$(zml_profile_compute_source_signature "${source_root}")"

  case "${tool}" in
    tracy-profiler)
      tool_path="$(ensure_profiler "${source_root}" "${source_signature}")"
      ;;
    tracy-capture|tracy-capture-daemon)
      tool_path="$(ensure_capture_tool "${source_root}" "${source_signature}" "${tool}")"
      ;;
    *)
      usage
      ;;
  esac

  exec "${tool_path}" "$@"
}

main "$@"
