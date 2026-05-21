# Gemma4 vLLM Benchmark

This folder contains a simplified HTTP benchmark for the Gemma4 DFlash example.
`run_vllm_benchmark.py` starts a vLLM server on the 5090 GPU, waits for
readiness, sends dataset requests, and shuts the server down.

## Setup

```bash
uv venv examples/dflash_benchmark_gemma4/vllm_benchmark/.venv
uv pip install --python examples/dflash_benchmark_gemma4/vllm_benchmark/.venv/bin/python \
  -r examples/dflash_benchmark_gemma4/vllm_benchmark/requirements.txt
```

## Easy 10-sample Runs

Baseline:

```bash
examples/dflash_benchmark_gemma4/vllm_benchmark/.venv/bin/python \
  examples/dflash_benchmark_gemma4/vllm_benchmark/run_vllm_benchmark.py \
  math500.jsonl --mode baseline --samples 10
```

DFlash:

```bash
examples/dflash_benchmark_gemma4/vllm_benchmark/.venv/bin/python \
  examples/dflash_benchmark_gemma4/vllm_benchmark/run_vllm_benchmark.py \
  math500.jsonl --mode dflash --samples 10
```

Use `stories.jsonl` as the positional dataset for the stories dataset. Relative
dataset names resolve to `examples/dflash_benchmark_gemma4/data` first, then the
shared `examples/dflash_benchmark/data` directory.

## Defaults

- target model: `/var/models/google/gemma-4-31B-it`
- DFlash model: `/var/models/z-lab/gemma-4-31B-it-DFlash`
- GPU: `CUDA_VISIBLE_DEVICES=1`
- samples: `10`
- max model length: `4096`
- server: `http://127.0.0.1:8000/v1`

Override model paths with `--target-model` and `--dflash-model`.

## Request-only Mode

If a compatible vLLM server is already running:

```bash
examples/dflash_benchmark_gemma4/vllm_benchmark/.venv/bin/python \
  examples/dflash_benchmark_gemma4/vllm_benchmark/request_benchmark.py \
  stories.jsonl --base-url http://127.0.0.1:8000/v1 --samples 10
```
