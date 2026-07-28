#!/usr/bin/env bash
set -euo pipefail

profile_root="$1"
shift

run_dir="${profile_root}/$(date -u '+%Y%m%dT%H%M%SZ')"
execution_dir="${run_dir}/execution"
compile_dir="${run_dir}/compile"

mkdir -p "${execution_dir}" "${compile_dir}"

neuron_explorer_bin="$(rlocation "libpjrt_neuron/sandbox/bin/neuron-explorer")"

target_executable="$1"
target_runfiles_env=()
if [[ -f "${target_executable}.runfiles_manifest" ]]; then
  target_runfiles_env+=("RUNFILES_MANIFEST_FILE=${target_executable}.runfiles_manifest")
elif [[ -d "${target_executable}.runfiles" ]]; then
  target_runfiles_env+=("RUNFILES_DIR=${target_executable}.runfiles")
else
  echo "failed to locate runfiles for profiled executable: ${target_executable}" >&2
  exit 1
fi

xla_flags="--xla_dump_to=${compile_dir} --xla_dump_hlo_as_text --xla_dump_hlo_as_proto --xla_dump_hlo_pass_re=.*"

set +e
env -u RUNFILES_DIR -u RUNFILES_MANIFEST_FILE -u RUNFILES_REPO_MAPPING \
  "${target_runfiles_env[@]}" \
  NEURON_RT_INSPECT_ENABLE=1 \
  NEURON_RT_INSPECT_SYSTEM_PROFILE=1 \
  NEURON_RT_INSPECT_DEVICE_PROFILE=1 \
  NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1 \
  NEURON_RT_INSPECT_OUTPUT_DIR="${execution_dir}" \
  SKIP_PJRT_PROFILER=true \
  XLA_IR_DEBUG=1 \
  XLA_HLO_DEBUG=1 \
  XLA_FLAGS="${xla_flags}" \
  "${neuron_explorer_bin}" inspect -o "${execution_dir}" "$@"
status=$?
set -e

echo "Neuron profile dump path: ${run_dir}" >&2

exit "${status}"
