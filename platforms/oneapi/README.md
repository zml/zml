# EXPERIMENTAL: oneAPI


# About

This directory contains the oneAPI backend for ZML. It is currently in an early stage of development, and only supports a subset of ZML's features. The oneAPI backend is only available on Linux, and requires an Intel GPU.

The shardind is not yet supported, so only models that fit in the GPU memory can be run.

# Getting Started

## Prerequisites

- An Intel GPU
- bazel see [Getting Started](../README.md#getting-started) for installation instructions.

### Linux Only

One GPU is supported on Linux.
ONEAPI_DEVICE_SELECTOR environment variable must be set to select the GPU device.

For example, to select the first GPU:
```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0
```

#### Run 


To run the LLM example with oneAPI and the Llama 3.2 1B model:
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0
bazel run //examples/llm \
    --config=release \
    --@zml//platforms:cpu=false \
    --@zml//platforms:oneapi=true \
    -- \
    --model=hf://meta-llama/Llama-3.2-1B-Instruct \
    --prompt="Tell me a story about a cat in 2 lines"
```

To run the LLM example with oneAPI and a local model:
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0
bazel run //examples/llm \
    --config=release \
    --@zml//platforms:cpu=false \
    --@zml//platforms:oneapi=true \
    -- \
    --model=path/to/your/model/Llama-3.1-8B-Instruct \
    --prompt="Tell me a story about a cat in 2 lines"
```
