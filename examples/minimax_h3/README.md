# MiniMax-H3

`//examples/minimax_h3` runs MiniMax-H3 Base: a 33B dense omni-transformer that jointly denoises video and stereo audio latents.

H3 is not an LLM. Detection uses the official Hugging Face layout (`FL2VA/` or `Ref2VA/`), not `examples/llm` `model_type`.

Supported locally (open weights):

- H3-Base FL2VA: `t2va`, `fl2va`
- H3-Base Ref2VA: `ref2va`

Not in the open release (hosted APIs only): H3-Context-IR, H3-Regenerate-2K, native sparse attention.

Context-IR locally:

- default `--ir=auto`: OpenH3-IR (`h3ir`) when `H3IR_LLM_URL` is set, otherwise official Prompting Guidance fields (`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`)
- `--ir=h3ir` requires [open-h3-ir](https://github.com/ruashots/open-h3-ir) and a local OpenAI-compatible model
- `--ir=prompt` is the official guidance wrap with no LLM
- `--ir=off` sends the raw string

CPU and Metal default to the community Mac preview canvas (short side 352, 10 steps). Official 768p is `--full`. `--tiny` is a compile smoke (128 / 4 steps). CPU uses one PJRT device so Eigen takes every core and VAE weights are not copied four times. Accelerators tensor-parallel `.model` across every high-bandwidth axis (TPU/Neuron fold `link_x`/`link_y`/`link_z`/`link` when present). DiT 56 heads and encoder GQA (64/8) require degree 1, 2, 4, or 8; 16/32/64-device meshes keep the first 8. Compile of independent graphs is concurrent. Visual VAE weights load once, four blocks at a time, then reuse across tiles.

Weights stream one encoder layer and one DiT block at a time so a 64 GB Mac can run without holding the 33B+32B checkpoints resident. After denoise the official ViT video decoder and BigVGAN audio decoder write `frame_*.ppm`, `audio.wav`, and `output.mp4` when `ffmpeg` is on PATH.

## Architecture

- Encoder: Qwen3-VL-32B text tower, unnormalized hidden state after layer 50
- Visual VAE: f16t4d24, then patchify `1×2×2` (effective spatial 32×)
- Audio VAE: 32 kHz stereo → 40 Hz, 32 latent channels
- Omni transformer: 50 blocks, hidden 5376, 56×128 heads, SwiGLU 14336, 3-D MM-RoPE, per-(timestep, modality) AdaLN
- Dual rectified-flow Euler: video `shift=12`, audio `shift=3`; data-ward velocity `x0 = xt + σ v`

Checkpoints are mixed precision: patch projections, timestep MLP, and output heads are float32; the block stack is bfloat16.

## Hardware

Same ZML platform flags as every other example. DiT attention is full-sequence bidirectional `zml.nn.sdpa` (no KV cache). XLA compiles that path on every target. Flash-attn LLM backends are logged only.

| Target | Physical `.model` binding | Attention used by H3 |
| --- | --- | --- |
| CPU | `bus` (1 PJRT device by default) | sdpa |
| CUDA | `link` (NVLink), first 8 of 16+ | sdpa |
| ROCm | `link`, first 8 of 16+ | sdpa |
| TPU | fastest present of `link_x`/`link_y`/`link_z`, folded; first 8 of 16+ | sdpa |
| Metal | `bus` | sdpa |
| Neuron | `link` then chip `link_x`/`link_y`/`link_z`, folded | sdpa |
| oneAPI | `link` if present else `bus`; fold both when both exist | sdpa |

```bash
# Probe each compiled VAE executable once (embed, 1 block, finish, audio)
bazel run //examples/minimax_h3 -- --tiny --probe
# Decode saved latents one visual block / one chunk first
bazel run //examples/minimax_h3 -- --tiny --decode-only --max-vae-blocks=1 --max-vae-chunks=1 --out=minimax_h3_tiny
# Mac CPU (preview canvas)
bazel run //examples/minimax_h3 -- --tiny --prompt="..."
# First/last frame (FL2VA)
bazel run //examples/minimax_h3 -- --tiny --variant=fl2va --image=first.ppm --last-image=last.ppm --prompt="..."
# Reference files (Ref2VA, comma-separated, max 12)
bazel run //examples/minimax_h3 -- --tiny --variant=ref2va --refs=ref.png,clip.mp4,bed.wav --prompt="..."
# Mac Metal
bazel run //examples/minimax_h3 --@zml//platforms:metal=true -- --preview --prompt="..."
# Official 768p (needs a large accelerator)
bazel run //examples/minimax_h3 --@zml//platforms:cuda=true -- --full --model=hf://MiniMaxAI/MiniMax-H3
# ROCm
bazel run //examples/minimax_h3 --@zml//platforms:rocm=true -- --model=hf://MiniMaxAI/MiniMax-H3
# TPU
bazel run //examples/minimax_h3 --@zml//platforms:tpu=true -- --model=hf://MiniMaxAI/MiniMax-H3
```

Scope the download if you only need one task family:

```bash
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" --local-dir MiniMax-H3
bazel run //examples/minimax_h3 --@zml//platforms:cuda=true -- --model=/path/MiniMax-H3 --variant=t2va
```

## Tests

Host tests cover config aliases, MM-RoPE dims, AdaLN table width, dual schedules, packing, patchify, canvas snap-to-32, tensor-parallel degree/strategy for every `Platform.Target`, and every backend mapping. They do not need weights or a GPU.

Layout: `config.zig` hyperparams, `sharding.zig` device mesh, `conditions.zig` FL2VA/Ref2VA prep, `pipeline.zig` compile, `session.zig` encode/denoise, `encode.zig` / `decode.zig` VAE I/O, `dit.zig` / `encoder.zig` / `vision.zig` graphs.

```bash
bazel run //examples/minimax_h3:h3_tests
bazel test //examples/minimax_h3:test
```

End-to-end generation needs the official checkpoint on disk (Hugging Face VFS downloads shards as each layer is read). Peak device memory is one layer plus activations, not the full 33B+32B stack.
