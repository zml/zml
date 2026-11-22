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
- `--backend=<stablehlo|cute|nki|metal>`: Optional. Attention backend. `cute` uses CuTe flash attention on CUDA GPUs with compute capability 8.0 or newer. If omitted, the program auto-selects one for the current platform, including `cute` on supported CUDA GPUs.

## Tenstorrent

On a host with Blackhole cards, build with `--@zml//platforms:tenstorrent=true`; the example runs tensor-parallel
over every card of the default view (the sequence length must be a multiple of 32). Run anything that opens the
cards one process at a time.

```bash
bazel run //examples/llm --@zml//platforms:tenstorrent=true -- --model=/var/models/meta-llama/Llama-3.1-8B-Instruct --prompt="What is the capital of France?" --topk=1
```

`//examples/llm:tenstorrent_m2` judges the cards against the CPU platform at the token level (the backend's M2):
greedy references with every position's top logits on the CPU, then teacher forcing of those tokens on 1 or 2
cards, then free-running generation with this example's own programs, timed per step.

```bash
M=/var/models/meta-llama/Llama-3.1-8B-Instruct
bazel run //examples/llm:tenstorrent_m2 --@zml//platforms:tenstorrent=true -- --model=$M --mode=reference --platform=cpu --tokens=100 --reference=/tmp/ref_cpu.json
bazel run //examples/llm:tenstorrent_m2 --@zml//platforms:tenstorrent=true -- --model=$M --mode=compare --cards=1 --reference=/tmp/ref_cpu.json
bazel run //examples/llm:tenstorrent_m2 --@zml//platforms:tenstorrent=true -- --model=$M --mode=compare --cards=2 --reference=/tmp/ref_cpu.json
bazel run //examples/llm:tenstorrent_m2 --@zml//platforms:tenstorrent=true -- --model=$M --mode=generate --cards=2 --tokens=100 --runs=3 --reference=/tmp/ref_cpu.json
```

`compare` exits 1 when fewer than 99% of all teacher-forced positions (the prompt's and the decode steps') agree, or
when a disagreement is not a near-tie (the two tokens more than 2 BF16 ulps apart in either run); each disagreement is
printed with both runs' logits for the two tokens and their gap in BF16 ulps. `--platform=cpu --cards=2` runs the
same comparison between the CPU's 2-partition and 1-partition runs, the noise floor of the agreement.
