<!-- Merge cleanup: keep the block regression instructions. The marked archive override paragraph is temporary. -->
# LLM

`//examples/llm` runs interactive or one-shot text generation from a model repository from HuggingFace.

We support the following models, automatically detected from the `model_type` in the `config.json`:

- Llama 3.1
- Qwen 3.5
- LFM 2.5

## Run

To load a model from HuggingFace directly:

```bash
# CPU
bazel run //examples/llm -- --model=hf://meta-llama/Llama-3.1-8B-Instruct
# CUDA
bazel run //examples/llm --@zml//platforms:cuda=true -- --model=hf://meta-llama/Llama-3.1-8B-Instruct
# ROCm
bazel run //examples/llm --@zml//platforms:rocm=true -- --model=hf://meta-llama/Llama-3.1-8B-Instruct
```

From a local directory:

```bash
bazel run //examples/llm --@zml//platforms:cuda=true -- --model=/var/models/meta-llama/Llama-3.1-8B-Instruct/
```

For a single non-interactive prompt:

```bash
bazel run //examples/llm --@zml//platforms:cuda=true -- --model=hf://meta-llama/Llama-3.1-8B-Instruct --prompt="Write a haiku about Zig"
```

## Check one Llama block on an S4000

The hardware regression uses layer 0 of a local BF16 Llama 3.2 3B Instruct checkpoint. Generate the fixture on the CPU machine with PyTorch 2.8.0, Transformers 4.56.2 and safetensors installed.

```bash
python3 examples/llm/models/llama_block_fixture.py /path/to/Llama-3.2-3B-Instruct /tmp/llama-block
```

Build `//examples/llm:llama_block_test` with `--@zml//platforms:musa=true` and select one S4000 with `MUSA_VISIBLE_DEVICES=0`. The test runner takes `--target=musa --fixture=/path/to/llama-block`. It compares hidden states, keys and values after a four-token prefill and four one-token decodes. It also checks that previously written cache positions stay unchanged and unused positions stay zero. The S4000 needs the executable, its Bazel runfiles and the two generated fixture files. Keep the weights outside the source tree.

<!-- Merge cleanup: remove this paragraph once MODULE.bazel pins the published archive URL. -->
While the XLA archive is unpublished, pass `--repo_env=ZML_MUSA_ARCHIVE_URL=file:///absolute/path/to/archive.tar.zst` to Bazel.

## Options

- `--model=<path>`: Required. Model repository to load. This can be a local path or a huggingface/S3 URI such as `hf://...` or `s3://...`.
- `--prompt=<string>`: Optional. Runs a single prompt instead of opening the interactive chat loop.
- `--seqlen=<number>`: Optional. Maximum sequence length. Defaults to `2048`.
- `--backend=<stablehlo|cute|nki|metal>`: Optional. Attention backend. `cute` uses CuTe flash attention on CUDA GPUs with compute capability 8.0 or newer. If omitted, the program auto-selects one for the current platform, including `cute` on supported CUDA GPUs.
