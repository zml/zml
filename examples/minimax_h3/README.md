# MiniMax-H3

Same modes as Hailuo / the MiniMax video API. Open weights are **768P**.

| Mode | Flags |
| --- | --- |
| text-to-video | `--prompt='...'` |
| image-to-video | `--first-frame=still.png` |
| last-frame | `--last-frame=still.png` |
| first-and-last-frame | `--first-frame=a.png --last-frame=b.png` |
| reference-to-video | `--refs=char.png,motion.mp4,voice.wav` |

Default canvas: text-to-video is **16:9**. Other modes are **adaptive** from the first visual (still, last frame, or first non-audio `--refs`). Ref2VA references are still encoded at their own geometry; mixed-size references do not share a padded VAE compile shape. `--ratio=9:16` (and the other Hailuo ratios) override the target canvas. `--size=WxH` sets exact pixels.

`--refs` order matters. A video keeps its own soundtrack. A wav after a video is a separate audio reference.

`--duration` is 5–15 s (default 5). Hosted Hailuo allows 4 s; open weights align to the VAE `17n+5` grid (5 s → 124 frames ≈ 5.2 s). Output defaults to `output.mp4`.

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
```

Local checkpoint: `--model=/var/models/MiniMaxAI/MiniMax-H3`.

## Options

- `--model=<path>`: Required. Local path or `hf://MiniMaxAI/MiniMax-H3`.
- `--prompt=<string>`: What happens in the shot.
- `--first-frame`, `--last-frame`, `--refs`
- `--duration=<sec>`: 5–15. Default `5`.
- `--ratio=<spec>`: `adaptive` \| `16:9` \| `9:16` \| `1:1` \| `4:3` \| `3:4` \| `21:9`.
- `--resolution=768P`: Open weights only. `2K` is hosted API.
- `--out=<path>`: `.mp4` or directory. Default `output.mp4`.
- `--steps`, `--seed`. H3 follows the upstream sigma-grid convention: `N`
  sigma points include terminal zero, so `N` points produce `N-1` DiT evaluations.

Advanced: `--frames`, `--size=WxH`, `--short-edge`, `--max-pixels`, `--dit`.

Not in this example: hosted 2K, Context-IR, Hailuo 4 s. Open weights are 768P, 5–15 s.

## Hardware

CUDA. Tensor parallelism is 1, 2, 4, or 8 devices. `ffmpeg` is required to read image/video size and to mux output.

Full 768P needs **≥80 GiB per device**. That is an early gate before weights load. The memory planner's denoise estimate can look like a 24 GiB card would stream the DiT; that number omits vision encode, compile, and allocator overhead, and the planner runs after visual conditioning. Preview canvases (short side ≤352, e.g. `--short-edge=352`) skip the floor.
