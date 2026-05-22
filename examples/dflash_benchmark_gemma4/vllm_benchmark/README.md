# Gemma4 vLLM Benchmark

This folder contains a simplified HTTP benchmark for the Gemma4 DFlash example.
The recommended workflow is to start one vLLM server in either baseline or
DFlash mode, then run request-only benchmarks against it.

## Setup

```bash
uv venv examples/dflash_benchmark_gemma4/vllm_benchmark/.venv
uv pip install --python examples/dflash_benchmark_gemma4/vllm_benchmark/.venv/bin/python \
  -r examples/dflash_benchmark_gemma4/vllm_benchmark/requirements.txt
```

## Start a Server

Baseline server:

```bash
CUDA_VISIBLE_DEVICES=1 \
examples/dflash_benchmark_gemma4/vllm_benchmark/.venv/bin/vllm serve \
  /var/models/google/gemma-4-31B-it \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 16
```

DFlash server:

```bash
CUDA_VISIBLE_DEVICES=1 \
examples/dflash_benchmark_gemma4/vllm_benchmark/.venv/bin/vllm serve \
  /var/models/google/gemma-4-31B-it \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 16 \
  --speculative-config '{"method":"dflash","model":"/var/models/z-lab/gemma-4-31B-it-DFlash","num_speculative_tokens":10}'
```

Wait until `http://127.0.0.1:8000/v1/models` responds before running requests.
Stop the server with `Ctrl-C` when finished.

## Send Requests

Math-500:

```bash
examples/dflash_benchmark_gemma4/vllm_benchmark/.venv/bin/python \
  examples/dflash_benchmark_gemma4/vllm_benchmark/request_benchmark.py \
  math500.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --model /var/models/google/gemma-4-31B-it \
  --samples 10
```

Stories:

```bash
examples/dflash_benchmark_gemma4/vllm_benchmark/.venv/bin/python \
  examples/dflash_benchmark_gemma4/vllm_benchmark/request_benchmark.py \
  stories.jsonl \
  --base-url http://127.0.0.1:8000/v1 \
  --model /var/models/google/gemma-4-31B-it \
  --samples 10
```

Relative dataset names resolve to `examples/dflash_benchmark_gemma4/data` first,
then the shared `examples/dflash_benchmark/data` directory.

## Defaults

- target model: `/var/models/google/gemma-4-31B-it`
- DFlash model: `/var/models/z-lab/gemma-4-31B-it-DFlash`
- GPU: `CUDA_VISIBLE_DEVICES=1`
- samples: `10`
- max model length: `4096`
- server: `http://127.0.0.1:8000/v1`

Override model paths in the server command and request command as needed.

## Start-Run-Stop Mode

`run_vllm_benchmark.py` is still available when you want a single command that
starts a server, runs one dataset, and shuts the server down:

```bash
examples/dflash_benchmark_gemma4/vllm_benchmark/.venv/bin/python \
  examples/dflash_benchmark_gemma4/vllm_benchmark/run_vllm_benchmark.py \
  math500.jsonl --mode dflash --samples 10
```
