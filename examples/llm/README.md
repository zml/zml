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

## Run Llama on an S4000

Until the SDK 5.1.0 plugin archive is published to the ZML mirror, build the sandbox in the sibling XLA checkout. Copy it to a directory dedicated to this generated override. `rsync -a` preserves compiler and library symlinks, and `--delete` removes files dropped from later sandbox builds.

```bash
# From the ZML checkout
override="$(cd .. && pwd)/xla-override"
(cd ../xla-private && bazel build --config=musa_remote -c opt --remote_download_outputs=toplevel //zml/musa:sandbox)
mkdir -p "$override"
rsync -a --delete --chmod=Du+w ../xla-private/bazel-bin/zml/musa/sandbox/ "$override/"
install -m644 platforms/musa/libzml_musa.BUILD.bazel "$override/BUILD.bazel"
touch "$override/REPO.bazel"

MUSA_VISIBLE_DEVICES=0 bazel run --config=release \
  --//platforms:cpu=false --//platforms:musa=true \
  --override_repository="+musa_packages+libzml_musa=$override" \
  //examples/llm:llm -- --model=/path/to/Llama-3.2-3B-Instruct \
  --backend=stablehlo --topk=1 --seqlen=16 --prompt=Paris
```

Run the inference command on the machine with the card. When building elsewhere, deploy the resulting executable, its repository mapping and SDK runfiles to the S4000 and use the same model arguments there.

### Check one Llama block against the CPU reference

The hardware regression uses layer 0 of a local BF16 Llama 3.2 3B Instruct checkpoint. Generate the fixture on the CPU machine with PyTorch 2.8.0, Transformers 4.56.2 and safetensors installed.

```bash
python3 examples/llm/models/llama_block_fixture.py /path/to/Llama-3.2-3B-Instruct /tmp/llama-block
```

Build `//examples/llm:llama_block_test` with `--@zml//platforms:musa=true` and select one S4000 with `MUSA_VISIBLE_DEVICES=0`. The test runner takes `--target=musa --fixture=/path/to/llama-block`. It compares hidden states, keys and values after a four-token prefill and four one-token decodes. It also checks that previously written cache positions stay unchanged and unused positions stay zero. The S4000 needs the executable, its Bazel runfiles and the two generated fixture files. Keep the weights outside the source tree.

## Options

- `--model=<path>`: Required. Model repository to load. This can be a local path or a huggingface/S3 URI such as `hf://...` or `s3://...`.
- `--prompt=<string>`: Optional. Runs a single prompt instead of opening the interactive chat loop.
- `--seqlen=<number>`: Optional. Maximum sequence length. Defaults to `2048`.
- `--backend=<stablehlo|cute|nki|metal>`: Optional. Attention backend. `cute` uses CuTe flash attention on CUDA GPUs with compute capability 8.0 or newer. If omitted, the program auto-selects one for the current platform, including `cute` on supported CUDA GPUs.
