# Gemma4 SGLang Benchmark

This directory contains a simplified SGLang benchmark flow for Gemma4. One
script starts the SGLang server, waits for `/v1/models`, runs JSONL dataset
requests, and shuts the server down.

Setup:

```bash
examples/dflash_benchmark_gemma4/SGLang_benchmark/setup.sh
```

Easy 10-sample runs:

```bash
examples/dflash_benchmark_gemma4/SGLang_benchmark/run_benchmark.py baseline --samples 10
examples/dflash_benchmark_gemma4/SGLang_benchmark/run_benchmark.py dflash --samples 10
```

By default both `math500.jsonl` and `stories.jsonl` are read from
`examples/dflash_benchmark_gemma4/data`. Pass either dataset name to run one:

```bash
examples/dflash_benchmark_gemma4/SGLang_benchmark/run_benchmark.py baseline math500.jsonl --samples 10
examples/dflash_benchmark_gemma4/SGLang_benchmark/run_benchmark.py dflash stories.jsonl --samples 10
```

Defaults are:

```text
target model: /var/models/google/gemma-4-31B-it
dflash model: /var/models/z-lab/gemma-4-31B-it-DFlash
gpu: CUDA_VISIBLE_DEVICES or 0
port: 30000
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=1 \
examples/dflash_benchmark_gemma4/SGLang_benchmark/run_benchmark.py dflash \
  --gpu 1 \
  --target-model /var/models/google/gemma-4-31B-it \
  --dflash-model /var/models/z-lab/gemma-4-31B-it-DFlash \
  --samples 10 \
  --server-log /tmp/sglang-gemma4.log
```

`request_benchmark.py` can also benchmark an already-running SGLang server:

```bash
examples/dflash_benchmark_gemma4/SGLang_benchmark/request_benchmark.py \
  --base-url http://127.0.0.1:30000/v1 \
  --model /var/models/google/gemma-4-31B-it \
  --samples 10
```
