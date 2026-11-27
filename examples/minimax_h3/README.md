# MiniMax-H3

`//examples/minimax_h3` runs text-to-video generation from a MiniMax-H3 model repository.

Prompt in, `video.rgb` + `audio.wav` out. Default 1344×768, 5 s, 24 fps, 30 Euler steps.

## Run

From a local directory:

```bash
# CUDA
bazel run //examples/minimax_h3 --@zml//platforms:cuda=true -- --model=/var/models/MiniMaxAI/MiniMax-H3
# ROCm
bazel run //examples/minimax_h3 --@zml//platforms:rocm=true -- --model=/var/models/MiniMaxAI/MiniMax-H3
```

For a custom prompt:

```bash
bazel run //examples/minimax_h3 --@zml//platforms:cuda=true -- --model=/var/models/MiniMaxAI/MiniMax-H3 --prompt="A cinematic wide shot of waves at dusk."
```

Mux the raw outputs with ffmpeg:

```bash
ffmpeg -y -f rawvideo -pix_fmt rgb24 -s 1344x768 -r 24 -i out/video.rgb \
  -i out/audio.wav -pix_fmt yuv420p -c:v libx264 -c:a aac out/out.mp4
```

## Options

- `--model=<path>`: Required. Local MiniMax-H3 repository to load (`text_encoder/`, `transformer/`, `vae/`, `audio_vae/`).
- `--prompt=<string>`: Optional. Generation prompt. Defaults to a dusk waves shot.
- `--out=<dir>`: Optional. Output directory. Defaults to `out`.
- `--seed=<number>`: Optional. Noise seed. Defaults to `0`.
- `--steps=<number>`: Optional. Sigma points including terminal 0. Defaults to `30`. Must be at least `2`.
- `--width=<pixels>`: Optional. Canvas width, multiple of 32. Defaults to `1344`.
- `--height=<pixels>`: Optional. Canvas height, multiple of 32. Defaults to `768`. Area must be at most `768×1344`.
- `--duration=<seconds>`: Optional. Clip length 5–15. Defaults to `5`. Frame count is snapped to a VAE-legal `17n+5`.

## Run Pi-H3

Pi-H3 supports CUDA/CuTe and Apple Metal with native non-causal attention.
It requires a pruned MiniMax-H3 transformer and a matching DWD adapter.
For 4-step generation on CUDA:

```bash
bazel run //examples/minimax_h3 --@zml//platforms:cuda=true -- \
  --model=/var/models/MiniMax-H3 \
  --variant=pi4 \
  --transformer=/var/models/MiniMax-H3-Pruned/transformer \
  --adapter=/var/models/pi-H3/pi_h3_4step_dwd_adapter \
  --prompt="A cinematic wide shot of waves at dusk."
```

For 2-step generation, use `--variant=pi2` and
`--adapter=/var/models/pi-H3/pi_h3_2step_dwd_adapter`.

The variant fixes the step count; `--steps` is H3-only. Other options, output
files, and the ffmpeg command above are unchanged. Pi-H3 reuses the base model
components, including `vae/config.json`, but does not need video VAE weights.

### Apple Metal

DiT and DWD select the same native attention backend. DWD keeps its original
window/global groups; attention is dense within each gathered group. BF16
attention does not use FP16 or ordinary SDPA as a substitute. The Metal allocator
grows on demand instead of reserving its whole memory budget at startup.

After preparing the checkpoint directories, start with a small canvas:

```bash
bazel run //examples/minimax_h3 \
  --@zml//platforms:metal=true --@zml//platforms:cpu=false --config=release -- \
  --model=/var/models/MiniMax-H3 \
  --transformer=/var/models/MiniMax-H3-Pruned/transformer \
  --adapter=/var/models/pi-H3/pi_h3_4step_dwd_adapter \
  --variant=pi4 --width=384 --height=224 --duration=5 \
  --prompt="A cat sitting in a sunny park." \
  --out="$HOME/pi-h3-results/metal-pi4-smoke"
```

On Metal, denoising loads, executes, and releases one DiT block at a time on
each step. Small AdaLN tables remain resident. This preserves checkpoint dtypes
and computation while trading additional weight transfers for bounded weight
residency. Other platforms retain the load-once strategy. The text encoder and
DWD head also release individual layers after execution.

A real Pi4 smoke run on 2026-11-27 completed on an M4 Max with 48 GiB unified
memory and macOS 26.7.3: 384x224, 124 frames at 24 fps, four evaluations, plus
32 kHz stereo audio. Total wall time was 114.49 seconds; the device allocator
reported 2,829 MiB peak in use and 4,626 MiB peak pool. The run used seed 0 and
prompt `An orange and white cat sitting in a sunny park.` Intermittent red
patches are visible in the output; numerical/visual parity with GB300 has not
been established. These measurements apply to this small canvas only.
