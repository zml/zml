#!/usr/bin/env bash

set -euo pipefail

device_selector=${ONEAPI_DEVICE_SELECTOR:-level_zero:0}
read_initial_parallelism=${ZML_LOAD_READ_INITIAL_PARALLELISM:-12}
read_parallelism=${ZML_LOAD_READ_PARALLELISM:-128}
sharding=${ZML_LOAD_SHARDING:-sharded}
model_path=${ZML_BENCH_MODEL_PATH:?Set ZML_BENCH_MODEL_PATH to the local model directory}

load_env=(
    "ONEAPI_DEVICE_SELECTOR=${device_selector}"
    "ZML_LOAD_READ_INITIAL_PARALLELISM=${read_initial_parallelism}"
    "ZML_LOAD_READ_PARALLELISM=${read_parallelism}"
)

for name in ZML_LOAD_FIXED_READ_PARALLELISM; do
    if [[ -v "${name}" ]]; then
        load_env+=("${name}=${!name}")
    fi
done

env "${load_env[@]}" \
    ./bazel.sh run --config=release --@zml//platforms:oneapi=true //examples/io:playground -- load "${model_path}" "${sharding}"
