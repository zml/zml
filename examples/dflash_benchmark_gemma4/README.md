# Gemma4 DFlash Benchmark

This package benchmarks Gemma4 baseline decoding against DFlash speculative
decoding with the local ZML runner, vLLM, and SGLang.

The commands below are intended for a GH200 machine from `~/zml` on branch
`tristan/dflash`. Gemma4 31B plus the DFlash draft model is too large for the
current 2x32GB RTX 5090 setup for the full ZML benchmark.

## Common Setup

```bash
cd ~/zml
git checkout tristan/dflash
git pull --ff-only origin tristan/dflash

export TARGET_MODEL=/var/models/google/gemma-4-31B-it
export DFLASH_MODEL=/var/models/z-lab/gemma-4-31B-it-DFlash
export OUT_DIR=/tmp/dflash_benchmark_gemma4
mkdir -p "${OUT_DIR}"
```

Build the ZML benchmark:

```bash
bazel build --@zml//platforms:cuda=true //examples/dflash_benchmark_gemma4:test
```

Install HTTP benchmark dependencies when needed:

```bash
uv venv examples/dflash_benchmark_gemma4/vllm_benchmark/.venv
uv pip install --python examples/dflash_benchmark_gemma4/vllm_benchmark/.venv/bin/python \
  -r examples/dflash_benchmark_gemma4/vllm_benchmark/requirements.txt

examples/dflash_benchmark_gemma4/SGLang_benchmark/setup.sh
```

## ZML

`//examples/dflash_benchmark_gemma4:benchmark` runs both the baseline target
model and DFlash in one process.

```bash
bazel run --@zml//platforms:cuda=true \
  //examples/dflash_benchmark_gemma4:benchmark -- \
  --model="${DFLASH_MODEL}" \
  --target-model="${TARGET_MODEL}" \
  --dataset-path=math500.jsonl \
  --dataset=math500 \
  --samples=10 \
  --max-new-tokens=1024 \
  --output-json="${OUT_DIR}/zml_math500.json"

bazel run --@zml//platforms:cuda=true \
  //examples/dflash_benchmark_gemma4:benchmark -- \
  --model="${DFLASH_MODEL}" \
  --target-model="${TARGET_MODEL}" \
  --dataset-path=stories.jsonl \
  --dataset=generic_jsonl \
  --samples=10 \
  --max-new-tokens=1024 \
  --output-json="${OUT_DIR}/zml_stories.json"
```

## vLLM

These commands start a vLLM server, wait for readiness, run requests, and shut
the server down. Run baseline and DFlash separately.

```bash
examples/dflash_benchmark_gemma4/vllm_benchmark/.venv/bin/python \
  examples/dflash_benchmark_gemma4/vllm_benchmark/run_vllm_benchmark.py \
  math500.jsonl \
  --mode baseline \
  --target-model="${TARGET_MODEL}" \
  --dflash-model="${DFLASH_MODEL}" \
  --samples=10 \
  --max-tokens=1024 \
  --output-json="${OUT_DIR}/vllm_baseline_math500.json" \
  --log-file="${OUT_DIR}/vllm_baseline_math500.log"

examples/dflash_benchmark_gemma4/vllm_benchmark/.venv/bin/python \
  examples/dflash_benchmark_gemma4/vllm_benchmark/run_vllm_benchmark.py \
  math500.jsonl \
  --mode dflash \
  --target-model="${TARGET_MODEL}" \
  --dflash-model="${DFLASH_MODEL}" \
  --samples=10 \
  --max-tokens=1024 \
  --output-json="${OUT_DIR}/vllm_dflash_math500.json" \
  --log-file="${OUT_DIR}/vllm_dflash_math500.log"

examples/dflash_benchmark_gemma4/vllm_benchmark/.venv/bin/python \
  examples/dflash_benchmark_gemma4/vllm_benchmark/run_vllm_benchmark.py \
  stories.jsonl \
  --mode baseline \
  --target-model="${TARGET_MODEL}" \
  --dflash-model="${DFLASH_MODEL}" \
  --samples=10 \
  --max-tokens=1024 \
  --output-json="${OUT_DIR}/vllm_baseline_stories.json" \
  --log-file="${OUT_DIR}/vllm_baseline_stories.log"

examples/dflash_benchmark_gemma4/vllm_benchmark/.venv/bin/python \
  examples/dflash_benchmark_gemma4/vllm_benchmark/run_vllm_benchmark.py \
  stories.jsonl \
  --mode dflash \
  --target-model="${TARGET_MODEL}" \
  --dflash-model="${DFLASH_MODEL}" \
  --samples=10 \
  --max-tokens=1024 \
  --output-json="${OUT_DIR}/vllm_dflash_stories.json" \
  --log-file="${OUT_DIR}/vllm_dflash_stories.log"
```

## SGLang

These commands start an SGLang server, wait for readiness, run requests, and
shut the server down. Run baseline and DFlash separately.

```bash
examples/dflash_benchmark_gemma4/SGLang_benchmark/run_benchmark.py \
  baseline math500.jsonl \
  --target-model="${TARGET_MODEL}" \
  --dflash-model="${DFLASH_MODEL}" \
  --samples=10 \
  --max-tokens=1024 \
  --server-log="${OUT_DIR}/sglang_baseline_math500.log"

examples/dflash_benchmark_gemma4/SGLang_benchmark/run_benchmark.py \
  dflash math500.jsonl \
  --target-model="${TARGET_MODEL}" \
  --dflash-model="${DFLASH_MODEL}" \
  --samples=10 \
  --max-tokens=1024 \
  --server-log="${OUT_DIR}/sglang_dflash_math500.log"

examples/dflash_benchmark_gemma4/SGLang_benchmark/run_benchmark.py \
  baseline stories.jsonl \
  --target-model="${TARGET_MODEL}" \
  --dflash-model="${DFLASH_MODEL}" \
  --samples=10 \
  --max-tokens=1024 \
  --server-log="${OUT_DIR}/sglang_baseline_stories.log"

examples/dflash_benchmark_gemma4/SGLang_benchmark/run_benchmark.py \
  dflash stories.jsonl \
  --target-model="${TARGET_MODEL}" \
  --dflash-model="${DFLASH_MODEL}" \
  --samples=10 \
  --max-tokens=1024 \
  --server-log="${OUT_DIR}/sglang_dflash_stories.log"
```
