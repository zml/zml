# MiniMax-H3

Video + audio. Same modes as Hailuo / the MiniMax video API. Open weights are 768P.

| Mode | Flags |
| --- | --- |
| text-to-video | `--prompt='...'` |
| image-to-video | `--first-frame=still.png` |
| last-frame | `--last-frame=still.png` |
| first-and-last-frame | `--first-frame=a.png --last-frame=b.png` |
| reference-to-video | `--refs=char.png,motion.mp4,voice.wav` |

## Reference limits

`--refs` accepts at most 12 files, with at most 9 images, 3 videos, and 3 audio references.

A video soundtrack counts as one of the 3 audio references. A following wav is a separate audio reference. Audio-only `--refs` are rejected.

`--refs` order: adaptive canvas uses the first visual. Each visual is encoded at its own geometry.

## Geometry

Default canvas: text-to-video is 16:9. Other modes are adaptive from the first visual. `--ratio` overrides the target canvas. `--size=WxH` sets exact pixels (snapped to 32); a snapped `--size` over `--max-pixels` is rejected.

Aspect stays between 1:4 and 4:1. After snap-to-32 the area never exceeds `--max-pixels` (default `768*1344`).

Ref2VA images use official short-edge 2048 (no area cap). Reference videos use their own aspect, never upscaled past the official canvas.

Duration aligns to the VAE `17n+5` grid (5 s → 124 frames). `--duration` is 5-15 s (default 5).

## Scheduler

`--steps=N` (default 30) is N sigma points including terminal zero, so the runtime does N-1 DiT evaluations.

## Run

```bash
# text-to-video (16:9)
bazel run //examples/minimax_h3 --@zml//platforms:cuda=true -- --model=hf://MiniMaxAI/MiniMax-H3

# image-to-video (canvas from the still)
bazel run //examples/minimax_h3 --@zml//platforms:cuda=true -- --model=hf://MiniMaxAI/MiniMax-H3 \
  --first-frame=first.png

# reference-to-video (canvas from the first image or video)
bazel run //examples/minimax_h3 --@zml//platforms:cuda=true -- --model=hf://MiniMaxAI/MiniMax-H3 \
  --refs=char.png,motion.mp4,voice.wav

# preview geometry
bazel run //examples/minimax_h3 --@zml//platforms:cuda=true -- --model=hf://MiniMaxAI/MiniMax-H3 \
  --short-edge=352
```

Local checkpoint: `--model=/var/models/MiniMaxAI/MiniMax-H3`.

## Options

- `--model=<path>`: Required. Local path or `hf://MiniMaxAI/MiniMax-H3`.
- `--prompt=<string>`: Generation prompt.
- `--first-frame`, `--last-frame`, `--refs`
- `--duration=<sec>`: 5-15. Default `5`.
- `--ratio=<spec>`: `adaptive` \| `16:9` \| `9:16` \| `1:1` \| `4:3` \| `3:4` \| `21:9`.
- `--resolution=768P`: Open weights only.
- `--out=<path>`: `.mp4` or directory. Default `output.mp4`.
- `--steps`, `--seed`.

Advanced: `--frames`, `--size=WxH`, `--short-edge`, `--max-pixels`, `--dit`.

CUDA for generation. Tensor parallelism is 1, 2, 4, or 8 devices. `ffmpeg` is required.
