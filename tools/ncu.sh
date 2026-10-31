#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <program> [args...]" >&2
  exit 1
fi

profile_mode=0
for arg in "$@"; do
  if [[ "$arg" == "--mode=profile" ]]; then
    profile_mode=1
  fi
done
if [[ "$profile_mode" != "1" ]]; then
  echo "Use --mode=profile --case=INDEX for capture; run benchmark mode without the ncu wrapper." >&2
  exit 1
fi

ncu_bin="${ZML_NCU_BIN:-ncu}"
if ! command -v "$ncu_bin" >/dev/null 2>&1; then
  echo "Nsight Compute CLI (ncu) is required on the CUDA host." >&2
  exit 1
fi
ncu_help="$("$ncu_bin" --help)"
if [[ "$ncu_help" != *"--nvtx-push-pop-scope"* ]]; then
  echo "This capture requires an ncu version with --nvtx-push-pop-scope process (XLA launches on worker threads)." >&2
  exit 1
fi

report_prefix="${ZML_NCU_OUTPUT:-${BUILD_WORKSPACE_DIRECTORY:-$PWD}/profiling-data/paged_attention_profile-$(date +%Y%m%d-%H%M%S)-$$}"
report_prefix="${report_prefix%.ncu-rep}"
report_file="${report_prefix}.ncu-rep"
metrics_file="${report_prefix}.csv"
if [[ -e "$report_file" || -e "$metrics_file" ]]; then
  echo "Report already exists; choose a new ZML_NCU_OUTPUT prefix: $report_prefix" >&2
  exit 1
fi
mkdir -p "$(dirname "$report_prefix")"

# Process scope includes launches on PJRT worker threads. Only the explicit
# post-warmup range is captured; graph nodes preserve attention/merge separation.
env SKIP_PJRT_PROFILER=true "$ncu_bin" \
  --nvtx \
  --nvtx-push-pop-scope process \
  --nvtx-include 'TSL@paged_attention.profile/' \
  --kernel-name 'regex:paged_attention_cute.*' \
  --graph-profiling node \
  --replay-mode kernel \
  --set full \
  --cache-control "${ZML_NCU_CACHE_CONTROL:-all}" \
  --clock-control none \
  --export "$report_file" \
  "$@"

if [[ ! -s "$report_file" ]]; then
  echo "No profile captured. Run the benchmark with --mode=profile --case=INDEX." >&2
  exit 1
fi
"$ncu_bin" --import "$report_file" --page raw --csv --print-units base > "$metrics_file"
if ! grep -q 'paged_attention_cute' "$metrics_file"; then
  echo "Report contains no paged-attention kernels. Check --mode=profile --case=INDEX and NVTX support." >&2
  exit 1
fi
echo "Nsight Compute report: $report_file" >&2
echo "Per-kernel metrics CSV: $metrics_file" >&2
