#!/usr/bin/env bash
set -euo pipefail

data_path="$1"
port="$2"
neuron_explorer_bin="$0.runfiles/+neuron_packages+libpjrt_neuron/sandbox/bin/neuron-explorer"

mkdir -p "${data_path}"

echo "Neuron Explorer server: http://127.0.0.1:${port}"

exec "${neuron_explorer_bin}" view \
  --data-path "${data_path}" \
  --port "${port}"
