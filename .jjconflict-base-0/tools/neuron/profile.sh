#!/usr/bin/env bash
set -euo pipefail

profile_root="$1"
shift

run_dir="${profile_root}/$(date -u '+%Y%m%dT%H%M%SZ')"
execution_dir="${run_dir}/execution"
compile_dir="${run_dir}/compile"

mkdir -p "${execution_dir}" "${compile_dir}"

neuron_explorer_bin="$0.runfiles/+neuron_packages+libpjrt_neuron/sandbox/bin/neuron-explorer"

xla_flags="--xla_dump_to=${compile_dir} --xla_dump_hlo_as_text --xla_dump_hlo_as_proto --xla_dump_hlo_pass_re=.*"

set +e
env \
  NEURON_RT_INSPECT_ENABLE=1 \
  NEURON_RT_INSPECT_SYSTEM_PROFILE=1 \
  NEURON_RT_INSPECT_DEVICE_PROFILE=1 \
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
