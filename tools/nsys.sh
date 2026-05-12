#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <program> [args...]" >&2
  exit 1
fi

sudo_cmd=()
if [[ "${ZML_PROFILE_NO_SUDO:-0}" != "1" ]]; then
  sudo_cmd=(sudo -E)
fi

declare -a nsys_args
if [[ -n "${ZML_NSYS_ARGS:-}" ]]; then
  nsys_args=(${ZML_NSYS_ARGS})
else
  nsys_args=(
    profile
    -t
    cuda,syscall,nvtx,cublas,cublas-verbose,cusparse,cusparse-verbose,cudnn,osrt
    --inherit-environment=true
    --cuda-memory-usage
    true
    --cuda-event-trace=false
    --backtrace=dwarf
    --cuda-graph-trace=node
  )
fi

exec env SKIP_PJRT_PROFILER=true "${sudo_cmd[@]}" nsys "${nsys_args[@]}" "$@"
