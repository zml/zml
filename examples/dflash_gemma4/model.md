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
