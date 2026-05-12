#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <program> [args...]" >&2
  exit 1
fi

if [[ -n "${RUNFILES_DIR:-}" ]]; then
  runfiles_root="${RUNFILES_DIR}"
elif [[ -d "$0.runfiles" ]]; then
  runfiles_root="$0.runfiles"
else
  echo "failed to locate Bazel runfiles for sandboxed rocprofv3" >&2
  exit 1
fi

rocprofv3_bin="${runfiles_root}/+rocm_packages+rocprofiler-sdk/bin/rocprofv3"
if [[ ! -x "${rocprofv3_bin}" ]]; then
  echo "sandboxed rocprofv3 not found at ${rocprofv3_bin}" >&2
  exit 1
fi

rocm_root="${runfiles_root}/+rocm_packages+libpjrt_rocm/sandbox"
if [[ ! -d "${rocm_root}" ]]; then
  echo "sandboxed ROCm runtime root not found at ${rocm_root}" >&2
  exit 1
fi

declare -a rocprof_args
if [[ -n "${ZML_ROCPROFV3_ARGS:-}" ]]; then
  rocprof_args=(${ZML_ROCPROFV3_ARGS})
else
  rocprof_args=(
    --sys-trace
    --runtime-trace
    --hip-trace
    --kernel-trace
    --memory-copy-trace
    --hip-runtime-trace
    --hsa-core-trace
    --scratch-memory-trace
    --marker-trace
    --kernel-rename
    --stats
    --output-format
    pftrace
  )
fi

exec env SKIP_PJRT_PROFILER=true "${rocprofv3_bin}" --rocm-root "${rocm_root}" "${rocprof_args[@]}" -- "$@"
