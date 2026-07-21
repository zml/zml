#!/usr/bin/env bash

set -euo pipefail

device_selector=${ONEAPI_DEVICE_SELECTOR:-level_zero:0}
read_parallelism=${ZML_LOAD_READ_PARALLELISM:-32}
dma_parallelism=${ZML_LOAD_DMA_PARALLELISM:-32}
read_request_mib=${ZML_LOAD_READ_REQUEST_MIB:-2}
dma_block_mib=${ZML_LOAD_DMA_BLOCK_MIB:-2}
max_pinned_mib=${ZML_LOAD_MAX_PINNED_MIB:-128}
sharding=${ZML_LOAD_SHARDING:-sharded}

ONEAPI_DEVICE_SELECTOR="${device_selector}" \
    ZML_LOAD_READ_PARALLELISM="${read_parallelism}" \
    ZML_LOAD_DMA_PARALLELISM="${dma_parallelism}" \
    ZML_LOAD_READ_REQUEST_MIB="${read_request_mib}" \
    ZML_LOAD_DMA_BLOCK_MIB="${dma_block_mib}" \
    ZML_LOAD_MAX_PINNED_MIB="${max_pinned_mib}" \
    ./bazel.sh run --config=release --@zml//platforms:oneapi=true //examples/io:playground -- load ~/s3proxy/data/lfm/ "${sharding}"
