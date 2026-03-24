# LLM

`//examples/llm` runs interactive or one-shot text generation from a model repository from HuggingFace.

We support the following models, automatically detected from the `model_type` in the `config.json`:

- Llama 3.1
- Qwen 3.5
- LFM2.5

## Run

To load a model from HuggingFace directly:

```bash
# CPU
bazel run //examples/llm -- --model=hf://Qwen/Qwen3.5-9B
# CUDA
bazel run //examples/llm --@zml//platforms:cuda=true -- --model=hf://Qwen/Qwen3.5-9B
# ROCm
bazel run //examples/llm --@zml//platforms:rocm=true -- --model=hf://Qwen/Qwen3.5-9B
```

With a local directory:

```bash
bazel run //examples/llm --@zml//platforms:cuda=true -- --model=/var/models/meta-llama/Llama-3.1-8B-Instruct/
```

For a single non-interactive prompt:

```bash
bazel run //examples/llm --@zml//platforms:cuda=true -- --model=hf://Qwen/Qwen3.5-9B --prompt="Write a haiku about Zig"
```

## Options

- `--model=<path>`: Required. Model repository to load. This can be a local path or a VFS URI such as `hf://...` or `s3://...`.
- `--prompt=<string>`: Optional. Runs a single prompt instead of opening the interactive chat loop.
- `--seqlen=<number>`: Optional. Maximum sequence length. Defaults to `2048`.
- `--backend=<vanilla|cuda_fa2|cuda_fa3>`: Optional. Attention backend. If omitted, the program auto-selects one for the current platform.

## Notes

- The examples above enable CUDA with `--@zml//platforms:cuda=true`. Switch platform flags as needed for your system.
- The binary prints platform details, resolves the repository, loads weights and tokenizer, then starts generation.
