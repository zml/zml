# DFlash Benchmark Agent Notes

## Hugging Face Dataset Downloads

Refresh the Python requirements used by the local Hugging Face CLI wrapper before
using it:

```bash
bazel run //tools/hf:requirements.update
```

Download datasets through the Bazel-managed CLI so the same environment is used
locally and on remote benchmark hosts:

```bash
bazel run //tools/hf -- download datasets/HuggingFaceH4/MATH-500 --local-dir ~/data/MATH-500
bazel run //tools/hf -- download datasets/princeton-nlp/SWE-bench_Lite --local-dir ~/data/SWE-bench_Lite
bazel run //tools/hf -- download datasets/tatsu-lab/alpaca --local-dir ~/data/alpaca
bazel run //tools/hf -- download lmsys/mt_bench --local-dir ~/data/mt-bench
```

The benchmark reads dataset files from `--dataset-path`; it does not download
them itself. MATH-500 can be run directly from:

```text
~/data/MATH-500/test.jsonl
```

SWE-bench Lite is distributed as parquet shards. Convert the downloaded parquet
split to normalized JSONL before passing it to this benchmark:

```text
~/data/SWE-bench_Lite/data/test-00000-of-00001.parquet
~/data/SWE-bench_Lite/test.jsonl
```

Alpaca is also downloaded as parquet by the HF dataset repo. Convert the train
parquet shard to normalized JSONL before passing it to this benchmark:

```text
~/data/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
~/data/alpaca/train.jsonl
```

MT-Bench is distributed as JSONL and can be run from:

```text
~/data/mt-bench/raw/question.jsonl
```

## CUDA Smoke Command

On `9960x-5090x2`, the repo checkout is `~/zml` and the model paths are local:

```bash
cd ~/zml
CUDA_VISIBLE_DEVICES=1 bazel run --@zml//platforms:cuda=true //examples/dflash_benchmark:benchmark -- \
  --model=/var/models/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat/ \
  --target-model=/var/models/meta-llama/Llama-3.1-8B-Instruct/ \
  --dataset=math500 \
  --dataset-path=$HOME/data/MATH-500/test.jsonl \
  --samples=1 \
  --temperature=0
```
