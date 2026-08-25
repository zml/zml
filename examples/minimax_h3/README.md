# MiniMax-H3

`//examples/minimax_h3` runs MiniMax-H3 Base: a 33B dense omni-transformer that jointly denoises video and stereo audio latents.

H3 is not an LLM. Detection uses the official Hugging Face layout (`FL2VA/` or `Ref2VA/`), not `examples/llm` `model_type`.

## Variants

| `--variant` | Weights | Conditioning |
| --- | --- | --- |
| `t2va` (default) | `FL2VA/` | text only |
| `fl2va` | `FL2VA/` | `--image` and/or `--last-image` |
| `ref2va` | `Ref2VA/transformer` (required; no FL2VA DiT fallback) | `--refs` images, videos, audio (max 12) |

Not in the open release (hosted APIs only): H3-Context-IR, H3-Regenerate-2K, native sparse attention.

## Canvas

`--tiny`, `--preview`, and `--full` are mutually exclusive. With none set, canvas is **auto**.

| Canvas | Short side | Steps | When |
| --- | --- | --- | --- |
| `--tiny` | 128 | 4 | compile / smoke |
| `--preview` | 352 | 10 | CPU, Metal, and devices under 40 GiB (auto) |
| `--full` | 768 | 30 | official 768p; auto on large accelerators |

`--full` and `--short-side` above 352 are refused below 40 GiB per device.

CPU uses one PJRT device. Accelerators tensor-parallel `.model` across the fastest high-bandwidth axis (TPU/Neuron fold `link_x`/`link_y`/`link_z`/`link` when present). DiT 56 heads and encoder GQA (64/8) require degree 1, 2, 4, or 8; 16/32/64-device meshes keep the first 8.

Weights stream one encoder layer and one DiT block at a time. After denoise the official ViT video decoder and BigVGAN audio decoder write `frame_*.ppm`, `audio.wav`, and `output.mp4` when `ffmpeg` is on PATH.

## Prompt IR

- `--ir=auto` (default): OpenH3-IR when `H3IR_LLM_URL` is set and `h3ir` is on PATH, otherwise official Prompting Guidance fields
- `--ir=h3ir`: [open-h3-ir](https://github.com/ruashots/open-h3-ir) plus a local OpenAI-compatible model
- `--ir=prompt`: official guidance wrap, no LLM
- `--ir=off`: raw string

## Run

Platform flags match every other example (`--@zml//platforms:cuda=true`, `rocm`, `tpu`, `metal`).

```bash
# Host tests (no weights)
bazel run //examples/minimax_h3:h3_tests

# Text-to-video+audio (preview on consumer GPUs)
bazel run --config=release //examples/minimax_h3 --@zml//platforms:cuda=true -- \
  --model=hf://MiniMaxAI/MiniMax-H3 --variant=t2va --preview --ir=prompt \
  --prompt="A cinematic wide shot of waves at dusk." --out=out_t2va

# First / last frame
bazel run --config=release //examples/minimax_h3 --@zml//platforms:cuda=true -- \
  --model=hf://MiniMaxAI/MiniMax-H3 --variant=fl2va --preview \
  --image=first.png --last-image=last.png --prompt="..." --out=out_fl2va

# Image reference (Ref2VA transformer)
bazel run --config=release //examples/minimax_h3 --@zml//platforms:cuda=true -- \
  --model=hf://MiniMaxAI/MiniMax-H3 --variant=ref2va --preview \
  --refs=ref.png --prompt="..." --out=out_ref2va

# Image + audio refs: use --tiny on 24 GB cards
bazel run --config=release //examples/minimax_h3 --@zml//platforms:cuda=true -- \
  --model=hf://MiniMaxAI/MiniMax-H3 --variant=ref2va --tiny \
  --refs=ref.png,bed.wav --prompt="..." --out=out_ref2va_audio

# Official 768p (needs ≥40 GiB per device)
bazel run --config=release //examples/minimax_h3 --@zml//platforms:cuda=true -- \
  --model=hf://MiniMaxAI/MiniMax-H3 --full --ir=prompt --prompt="..."

# VAE probe / decode saved latents
bazel run //examples/minimax_h3 -- --tiny --probe
bazel run //examples/minimax_h3 -- --tiny --decode-only --out=out_t2va
```

Metal, ROCm, and TPU use the same flags plus the matching `--@zml//platforms:...=true`.

`--shared=<dir>` loads official `video_noise.f32`, `audio_noise.f32`, and `prompt_embeds.f32` and skips encoder sampling. Use this to compare DiT output against official dumps of the same inputs.

## Weights

```bash
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" --local-dir MiniMax-H3
# Required for --variant=ref2va
hf download MiniMaxAI/MiniMax-H3 --include "Ref2VA/*" --local-dir MiniMax-H3
```

Encoder, visual VAE, audio VAE, and tokenizer may fall back to `FL2VA/` when a task dir omits them. The Ref2VA DiT will not.

## Architecture

- Encoder: Qwen3-VL-32B text tower, unnormalized hidden state after layer 50
- Visual VAE: f16t4d24, then patchify `1×2×2` (effective spatial 32×)
- Audio VAE: 32 kHz stereo → 40 Hz, 32 latent channels
- Omni transformer: 50 blocks, hidden 5376, 56×128 heads, SwiGLU 14336, 3-D MM-RoPE, per-(timestep, modality) AdaLN
- Dual rectified-flow Euler: video `shift=12`, audio `shift=3`; data-ward velocity `x0 = xt + σ v`
- Attention: full-sequence bidirectional `zml.nn.sdpa` on every target

Checkpoints are mixed precision: patch projections, timestep MLP, and output heads are float32; the block stack is bfloat16.

Layout: `config.zig` hyperparams, `sharding.zig` device mesh, `conditions.zig` FL2VA/Ref2VA prep, `pipeline.zig` compile, `session.zig` encode/denoise, `encode.zig` / `decode.zig` VAE I/O, `dit.zig` / `encoder.zig` / `vision.zig` graphs.

## Hardware

| Target | Physical `.model` binding |
| --- | --- |
| CPU | `bus` (1 PJRT device) |
| CUDA / ROCm | `link` (NVLink), first 8 of 16+ |
| TPU | fastest of `link_x`/`link_y`/`link_z`, folded; first 8 of 16+ |
| Metal | `bus` |
| Neuron | `link` then chip `link_x`/`link_y`/`link_z`, folded |
| oneAPI | `link` if present else `bus`; fold both when both exist |

## Limits

- Official 768p is 1344×768 and needs at least 40 GiB per device.
- Preview + audio `--refs` is refused on devices under 40 GiB. Use `--tiny` for image+audio, or `--preview` with image refs only.
- Audio decode matches official BigVGAN on the same stereo latents.

## Tests

Host tests cover config aliases, MM-RoPE dims, AdaLN table width, dual schedules, packing, patchify, canvas snap-to-32, and tensor-parallel degree for every `Platform.Target`. They do not need weights or a GPU.

```bash
bazel run //examples/minimax_h3:h3_tests
bazel test //examples/minimax_h3:test
```
