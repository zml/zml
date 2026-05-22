# Gemma4 SGLang DFlash Benchmark

This directory contains the request benchmark and minimal setup helpers for the
Gemma4 DFlash SGLang PR flow on `gh200`.

## Clean PR Setup

Use a dedicated workspace outside the ZML checkout:

```bash
mkdir -p ~/sglang_tests
uv venv ~/sglang_tests/.venv
uv pip install --python ~/sglang_tests/.venv/bin/python \
  "git+https://github.com/sgl-project/sglang.git@refs/pull/23000/head#subdirectory=python"
```

The helper script does the same install:

```bash
examples/dflash_benchmark_gemma4/SGLang_benchmark/setup.sh
```

If the SGLang source build fails, install the known gh200 build helpers and
rerun the PR install:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
uv pip install --python ~/sglang_tests/.venv/bin/python grpcio-tools
mkdir -p ~/sglang_tests/bin
cat >~/sglang_tests/bin/protoc <<'EOF'
#!/usr/bin/env bash
exec "$HOME/sglang_tests/.venv/bin/python" -m grpc_tools.protoc "$@"
EOF
chmod +x ~/sglang_tests/bin/protoc

PATH="$HOME/.cargo/bin:$HOME/sglang_tests/bin:$PATH" \
PROTOC="$HOME/sglang_tests/bin/protoc" \
uv pip install --python ~/sglang_tests/.venv/bin/python \
  "git+https://github.com/sgl-project/sglang.git@refs/pull/23000/head#subdirectory=python"
```

## Start SGLang

Baseline target model:

```bash
CUDA_VISIBLE_DEVICES=0 \
PATH="$HOME/.cargo/bin:$HOME/sglang_tests/bin:$PATH" \
PROTOC="$HOME/sglang_tests/bin/protoc" \
~/sglang_tests/.venv/bin/python -m sglang.launch_server \
  --model-path /var/models/google/gemma-4-31B-it \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --attention-backend triton \
  --trust-remote-code
```

DFlash target plus draft model:

```bash
CUDA_VISIBLE_DEVICES=0 \
PATH="$HOME/.cargo/bin:$HOME/sglang_tests/bin:$PATH" \
PROTOC="$HOME/sglang_tests/bin/protoc" \
~/sglang_tests/.venv/bin/python -m sglang.launch_server \
  --model-path /var/models/google/gemma-4-31B-it \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --attention-backend triton \
  --trust-remote-code \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /var/models/z-lab/gemma-4-31B-it-DFlash \
  --speculative-num-draft-tokens 16 \
  --speculative-draft-attention-backend fa4
```

Wait until `http://127.0.0.1:30000/v1/models` responds before running requests.

## Request Benchmark

OpenAI-compatible streaming request mode:

```bash
examples/dflash_benchmark_gemma4/SGLang_benchmark/request_benchmark.py \
  math500.jsonl stories.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --model /var/models/google/gemma-4-31B-it \
  --samples 10
```

Verbose `/generate` mode renders the dataset prompt through the model chat
template, sends `input_ids`, and prints `meta_info` fields including
`spec_accept_length`, `spec_accept_rate`, and `spec_accept_histogram` when
SGLang returns them:

```bash
examples/dflash_benchmark_gemma4/SGLang_benchmark/request_benchmark.py \
  math500.jsonl \
  --base-url http://127.0.0.1:30000/v1 \
  --model /var/models/google/gemma-4-31B-it \
  --tokenizer-path /var/models/google/gemma-4-31B-it \
  --samples 10 \
  --verbose
```

Add `--raw-prompt --verbose` only when you intentionally want to bypass chat
template rendering.

Relative dataset names resolve to `examples/dflash_benchmark_gemma4/data` first,
then the shared `examples/dflash_benchmark/data` directory.

## Start-Run-Stop Mode

`run_benchmark.py` starts a server, runs requests, and shuts it down. It uses
`~/sglang_tests/.venv/bin/python` by default when that venv exists:

```bash
examples/dflash_benchmark_gemma4/SGLang_benchmark/run_benchmark.py \
  dflash math500.jsonl --samples 10 --verbose
```
