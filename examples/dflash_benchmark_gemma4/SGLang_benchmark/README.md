# Gemma4 SGLang Benchmark

This directory contains a simplified SGLang benchmark flow for Gemma4. The
recommended workflow is to start one SGLang server in either baseline or DFlash
mode, then run request-only benchmarks against it.

## Setup

```bash
examples/dflash_benchmark_gemma4/SGLang_benchmark/setup.sh
```

## Start a Server

Baseline server:

```bash
CUDA_VISIBLE_DEVICES=0 \
examples/dflash_benchmark_gemma4/SGLang_benchmark/.venv/bin/python \
  -m sglang.launch_server \
  --model-path /var/models/google/gemma-4-31B-it \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --context-length 2048 \
  --mem-fraction-static 0.7 \
  --cuda-graph-max-bs 8 \
  --dtype float16 \
  --log-level warning
```

DFlash server:

```bash
CUDA_VISIBLE_DEVICES=0 \
examples/dflash_benchmark_gemma4/SGLang_benchmark/.venv/bin/python \
  -m sglang.launch_server \
  --model-path /var/models/google/gemma-4-31B-it \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --context-length 2048 \
  --mem-fraction-static 0.7 \
  --cuda-graph-max-bs 8 \
  --dtype float16 \
  --log-level warning \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /var/models/z-lab/gemma-4-31B-it-DFlash
```

Optional DFlash knobs can be passed directly to `sglang.launch_server`, for
example `--speculative-num-draft-tokens`,
`--speculative-dflash-block-size`, and
`--speculative-dflash-draft-window-size`.

Wait until `http://127.0.0.1:30000/v1/models` responds before running requests.
Stop the server with `Ctrl-C` when finished.

## Send Requests

Math-500:

```bash
examples/dflash_benchmark_gemma4/SGLang_benchmark/request_benchmark.py \
  math500.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --model /var/models/google/gemma-4-31B-it \
  --samples 10
```

Stories:

```bash
examples/dflash_benchmark_gemma4/SGLang_benchmark/request_benchmark.py \
  stories.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --model /var/models/google/gemma-4-31B-it \
  --samples 10
```

Relative dataset names resolve to `examples/dflash_benchmark_gemma4/data` first,
then the shared `examples/dflash_benchmark/data` directory.

## Defaults

```text
target model: /var/models/google/gemma-4-31B-it
dflash model: /var/models/z-lab/gemma-4-31B-it-DFlash
gpu: CUDA_VISIBLE_DEVICES or 0
port: 30000
```

## Start-Run-Stop Mode

`run_benchmark.py` is still available when you want a single command that starts
a server, runs requests, and shuts the server down:

```bash
examples/dflash_benchmark_gemma4/SGLang_benchmark/run_benchmark.py \
  dflash math500.jsonl --samples 10
```
