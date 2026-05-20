# vLLM DFlash Benchmark

This folder contains a Python/vLLM benchmark that compares:

- a target-model autoregressive baseline
- vLLM speculative decoding with the DFlash/draft model

The output format mirrors the Zig benchmark in `examples/dflash_benchmark`:
per-sample token counts, TPOT, TPS, speedup, tau, exact-match quality, and a
valid draft-token histogram when acceptance data is available.

The primary target host is `9960x-5090x2`.

## Setup

Create an isolated Python environment from the repository root:

```bash
uv venv examples/dflash_benchmark/vllm_benchmark/.venv
uv pip install --python examples/dflash_benchmark/vllm_benchmark/.venv/bin/python \
  -r examples/dflash_benchmark/vllm_benchmark/requirements.txt
```

Run with:

```bash
source examples/dflash_benchmark/vllm_benchmark/.venv/bin/activate
```

## Dataset Paths

The runner expects JSONL inputs and supports the same prompt formats as the Zig
benchmark:

```text
math500         ~/data/MATH-500/test.jsonl
swe_bench_lite ~/data/SWE-bench_Lite/test.jsonl
alpaca          ~/data/alpaca/train.jsonl
mt_bench        ~/data/mt-bench/test.jsonl
generic_jsonl   any JSONL with prompt/text/input
```

Use `examples/dflash_benchmark/python_tools` to convert Alpaca or SWE-bench
Lite parquet downloads to JSONL.

## Example

```bash
CUDA_VISIBLE_DEVICES=1 \
python examples/dflash_benchmark/vllm_benchmark/run_benchmark.py \
  --model /var/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat/ \
  --target-model /var/models/meta-llama/Llama-3.1-8B-Instruct/ \
  --dataset math500 \
  --dataset-path ~/data/MATH-500/test.jsonl \
  --samples 10 \
  --max-model-len 2048 \
  --temperature 0
```

Other useful datasets:

```bash
CUDA_VISIBLE_DEVICES=1 python examples/dflash_benchmark/vllm_benchmark/run_benchmark.py --model /var/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat/ --target-model /var/models/meta-llama/Llama-3.1-8B-Instruct/ --dataset swe_bench_lite --dataset-path ~/data/SWE-bench_Lite/test.jsonl --samples 10 --max-model-len 2048 --temperature 0
CUDA_VISIBLE_DEVICES=1 python examples/dflash_benchmark/vllm_benchmark/run_benchmark.py --model /var/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat/ --target-model /var/models/meta-llama/Llama-3.1-8B-Instruct/ --dataset alpaca --dataset-path ~/data/alpaca/train.jsonl --samples 10 --max-model-len 2048 --temperature 0
CUDA_VISIBLE_DEVICES=1 python examples/dflash_benchmark/vllm_benchmark/run_benchmark.py --model /var/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat/ --target-model /var/models/meta-llama/Llama-3.1-8B-Instruct/ --dataset mt_bench --dataset-path ~/data/mt-bench/test.jsonl --samples 10 --max-model-len 2048 --temperature 0
```

## Notes

- The runner uses vLLM offline `LLM.generate`.
- Baseline and speculative runs are executed in separate vLLM engines so the
  target-only timing is not affected by speculative configuration.
- vLLM public outputs do not currently expose per-step accepted draft-token
  counts in a stable offline API. When those metrics are unavailable, the
  benchmark still reports TPOT/TPS/speedup and exact-match quality, while tau
  and histogram fields remain zero/empty.
- `--max-model-len` defaults to 2048 to keep KV-cache allocation smaller on the
  RTX 5090. Raise it only when prompts plus generated tokens need more context.
- `--num-speculative-tokens` defaults to 10 and is passed through vLLM's
  `speculative_config`.
