#!/usr/bin/env bash

set -euo pipefail

device_selector=${ONEAPI_DEVICE_SELECTOR:-level_zero:0}
read_initial_parallelism=${ZML_LOAD_READ_INITIAL_PARALLELISM:-12}
read_parallelism=${ZML_LOAD_READ_PARALLELISM:-128}
dma_initial_parallelism=${ZML_LOAD_DMA_INITIAL_PARALLELISM:-8}
dma_parallelism=${ZML_LOAD_DMA_PARALLELISM:-32}
read_request_max_mib=${ZML_LOAD_READ_REQUEST_MAX_MIB:-128}
dma_block_mib=${ZML_LOAD_DMA_BLOCK_MIB:-2}
max_pinned_mib=${ZML_LOAD_MAX_PINNED_MIB:-2048}
sharding=${ZML_LOAD_SHARDING:-sharded}

load_env=(
    "ONEAPI_DEVICE_SELECTOR=${device_selector}"
    "ZML_LOAD_READ_INITIAL_PARALLELISM=${read_initial_parallelism}"
    "ZML_LOAD_READ_PARALLELISM=${read_parallelism}"
    "ZML_LOAD_DMA_INITIAL_PARALLELISM=${dma_initial_parallelism}"
    "ZML_LOAD_DMA_PARALLELISM=${dma_parallelism}"
    "ZML_LOAD_READ_REQUEST_MAX_MIB=${read_request_max_mib}"
    "ZML_LOAD_DMA_BLOCK_MIB=${dma_block_mib}"
    "ZML_LOAD_MAX_PINNED_MIB=${max_pinned_mib}"
)

# Fixed read/DMA controls and ZML_LOAD_READ_REQUEST_MIB take precedence.
# ZML_LOAD_READ_REQUEST_INITIAL_MIB optionally replaces the source minimum.
for name in ZML_LOAD_FIXED_READ_PARALLELISM ZML_LOAD_FIXED_DMA_PARALLELISM ZML_LOAD_READ_REQUEST_MIB ZML_LOAD_READ_REQUEST_INITIAL_MIB; do
    if [[ -v "${name}" ]]; then
        load_env+=("${name}=${!name}")
    fi
done

env "${load_env[@]}" \
    ./bazel.sh run --config=release --@zml//platforms:oneapi=true //examples/io:playground -- load ~/s3proxy/data/lfm/ "${sharding}"
