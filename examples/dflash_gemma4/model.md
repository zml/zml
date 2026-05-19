# DFlash Gemma 4 Runbook

This example lives on branch `tristan/dflash_gemma4` under `examples/dflash_gemma4`.

## Inputs

- `--model`: DFlash draft model repository. It must use the existing DFlash checkpoint layout from `examples/dflash`.
- `--target-model`: Gemma 4 text or top-level Gemma 4 repository. The loader follows `llmd` semantics:
  - top-level Gemma 4 configs read `text_config` and load weights below `model.language_model`;
  - text-only Gemma 4 configs read the repository root and load weights without adding a prefix.
- `--prompt`: raw user text. The executable formats it as a single Gemma chat turn before tokenization.

## Reference Implementations

- DFlash reference: checkout `tristan/dflash` in `~/zml`, then inspect `examples/dflash`.
- Gemma 4 semantics: inspect `~/monorepo/llmd/models/gemma4.zig` and `~/monorepo/llmd/models/gemma4_text.zig`.
- Local spec: `examples/dflash_gemma4/spec.md`.

## Build

From `~/zml`:

```sh
bazel build //examples/dflash_gemma4:dflash_gemma4 //examples/dflash_gemma4:test
```

## Run

```sh
bazel run //examples/dflash_gemma4:dflash_gemma4 -- \
  --model=/path/to/dflash/model \
  --target-model=/path/to/gemma4/model \
  --prompt="Tell me a story about Paris." \
  --max-seq-len=256 \
  --temperature=0
```

The target model path should contain `config.json`, tokenizer files, a supported Gemma chat template, and safetensors weights. The DFlash path should contain its `config.json` and safetensors weights; tokenization is loaded from the target model repository.

## Remote Smoke Test

To test, SSH to the remote machine:

```sh
ssh -A tristan@9960x-5090x2
```

Then run:

```sh
CUDA_VISIBLE_DEVICES=0,1 bazel run --@zml//platforms:cuda=true //examples/dflash_gemma4 -- --model=/var/models/z-lab/gemma-4-31B-it-DFlash --target-model=/var/models/google/gemma-4-31B-it --prompt="Give me a detailed account of the history of the Richelieu-Drouot part of Paris." --max-seq-len=4096 --temperature=0
```
