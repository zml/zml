#!/usr/bin/env bash
set -euo pipefail

neuron_explorer_bin="$0.runfiles/+neuron_packages+libpjrt_neuron/sandbox/bin/neuron-explorer"

run_dir="$1"
execution_dir="${run_dir}/execution"

case "$(basename "$0")" in
  summary-txt)
    output_format="summary-text"
    output_file="summary.txt"
    ;;
  summary-json)
    output_format="summary-json"
    output_file="summary.json"
    ;;
  summary-perfetto)
    output_format="perfetto"
    output_file="summary.pftrace"
    ;;
esac

if [[ "${output_format}" == "summary-text" || "${output_format}" == "summary-json" ]]; then
  "${neuron_explorer_bin}" view \
    -d "${execution_dir}" \
    --output-format "${output_format}" \
    >"${run_dir}/${output_file}"
else
  "${neuron_explorer_bin}" view \
    -d "${execution_dir}" \
    --output-format "${output_format}" \
    --output-file "${run_dir}/${output_file}"
fi

echo "Neuron summary path: ${run_dir}/${output_file}" >&2
