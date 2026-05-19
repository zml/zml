# DFlash Gemma 4 Runbook

This example lives on branch `tristan/dflash_gemma4` under `examples/dflash_gemma4`.

## Inputs

- `--model`: DFlash draft model repository. It must use the existing DFlash checkpoint layout from `examples/dflash`.
- `--target-model`: Gemma 4 text or top-level Gemma 4 repository. The loader follows `llmd` semantics:
  - top-level Gemma 4 configs read `text_config` and load weights below `model.language_model`;
  - text-only Gemma 4 configs read the repository root and load weights without adding a prefix.
- `--prompt`: raw user text. The executable formats it as a single Gemma 4 chat turn before tokenization.

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

To test on `9960x-5090x2`, SSH with agent forwarding:

```sh
ssh -A tristan@9960x-5090x2
```

From the remote `~/zml` checkout, pull this branch and build with CUDA:

```sh
cd ~/zml
git checkout tristan/dflash_gemma4
git pull --ff-only
bazel build --@zml//platforms:cuda=true //examples/dflash_gemma4
```

Then run the smoke test:

```sh
CUDA_VISIBLE_DEVICES=0,1 bazel run --@zml//platforms:cuda=true //examples/dflash_gemma4 -- --model=/var/models/z-lab/gemma-4-31B-it-DFlash --target-model=/var/models/google/gemma-4-31B-it --prompt="Give me a detailed account of the history of the Richelieu-Drouot part of Paris." --max-seq-len=4096 --temperature=0
```

The branch was tested on `9960x-5090x2` with 2x RTX 5090. The CUDA build succeeded and the executable reached all four compile stages:

- target prefill;
- DFlash prefill drafter;
- DFlash steady-state drafter;
- target verify.

The run then failed while loading weights with a CUDA PJRT out-of-memory error. At that point the GPUs were otherwise free. The relevant model sizes on that machine were:

```text
/var/models/google/gemma-4-31B-it         59G
/var/models/z-lab/gemma-4-31B-it-DFlash  2.9G
```

With Gemma 4 31B, the DFlash draft model, and `--max-seq-len=4096`, 2x32GB is not enough for this current implementation. Use a larger GPU pool, a shorter `--max-seq-len`, or a smaller target model for the next runtime iteration.

The same command succeeded on `gh200-2` via `ssh -A tristan@100.71.245.118`. The hostname `gh200-2` did not resolve from the local machine, but the IP did. On that machine, Bazel initially failed because `/tmp/zig-cache` was not writable; this was fixed with:

```sh
sudo mkdir -p /tmp/zig-cache
sudo chown -R tristan:tristan /tmp/zig-cache
```

After that, the CUDA build and full run completed on one visible `NVIDIA GH200 480GB` device. The run generated 1168 tokens before EOS in 36.299s, for 32.177 tokens/s. DFlash acceptance was very low:

```text
steps: 1159
valid_draft_tokens total: 9
draft_acceptance_rate: 0.001
zero_accept_steps: 1150
```

This confirms the Gemma 4 target path works end-to-end, but the DFlash draft model is not yet useful for acceleration on this checkpoint/config.

## Notes From The First Remote Iteration

- The working SSH command is `ssh -A tristan@9960x-5090x2`; the bare host alias did not resolve locally.
- The target Gemma 4 model path `/var/models/google/gemma-4-31B-it` exists and contains `chat_template.jinja`, `config.json`, tokenizer files, and two safetensors shards.
- The DFlash draft model path `/var/models/z-lab/gemma-4-31B-it-DFlash` must exist separately. A missing draft-model path fails early in `resolveModelRepo`.
- The real Gemma 4 chat template uses `<|turn>...<turn|>` and generation starts with `<|turn>model\n<|channel>thought\n<channel|>`. Older Gemma-style `<start_of_turn>` templates are not correct for this checkpoint.
- For future remote iterations, skip local Bazel builds after small fixes. Commit locally, push, pull on the remote, and use the remote CUDA build as the verification gate.
- If testing on a new multi-GPU machine, first check free memory with `nvidia-smi` and confirm both model paths with `du -sh`.
- If a remote build fails with `AccessDenied` under `/tmp/zig-cache`, fix ownership of that cache directory before retrying.
