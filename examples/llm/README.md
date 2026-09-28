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

## Options

- `--model=<path>`: Required. Model repository to load. This can be a local path or a huggingface/S3 URI such as `hf://...` or `s3://...`.
- `--prompt=<string>`: Optional. Runs a single prompt instead of opening the interactive chat loop.
- `--seqlen=<number>`: Optional. Maximum sequence length. Defaults to `2048`.
- `--backend=<vanilla|cuda_fa2|cuda_fa3>`: Optional. Attention backend. If omitted, the program auto-selects one for the current platform.

## Furiosa

Build and package the typed TCL plugin using the [Furiosa guide](../../platforms/furiosa/README.md),
then export its SDK environment. With the plugin environment override unset:

```sh
bazel run --override_repository=libzml_furiosa=/home/kevin/furiosa/xla-override \
  --jobs=16 --@zml//platforms:furiosa=true --@zml//platforms:cpu=false \
  //examples/llm -- --model=/var/models/meta-llama/Llama-3.1-8B-Instruct \
  --furiosa-pe-count=8 --backend=vanilla --seqlen=128 --topk=1 \
  --prompt='Count from 1 to 20, separated by commas.'
```
