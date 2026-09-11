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
- `--first-frame=<path>`: Optional. First-frame image for image-to-video / first-and-last.
- `--last-frame=<path>`: Optional. Last-frame image.
- `--refs=<paths>`: Optional. Comma-separated reference images, videos (`.mp4`), or audio (`.mp3`/`.wav`). Uses `transformer_ref/` weights.
