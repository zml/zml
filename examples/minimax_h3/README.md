# MiniMax-H3

Same modes as Hailuo / the MiniMax video API. Open weights are **768P**.

| Mode | Flags |
| --- | --- |
| text-to-video | `--prompt='...'` |
| image-to-video | `--first-frame=still.png` |
| last-frame | `--last-frame=still.png` |
| first-and-last-frame | `--first-frame=a.png --last-frame=b.png` |
| reference-to-video | `--refs=char.png,motion.mp4,voice.wav` |

Default canvas: text-to-video is **16:9**. Other modes are **adaptive** from the first visual (still, last frame, or first non-audio `--refs`). `--ratio=9:16` (and the other Hailuo ratios) override that. `--size=WxH` sets exact pixels.

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
- `--steps`, `--seed`

Advanced: `--frames`, `--size=WxH`, `--short-edge`, `--max-pixels`, `--dit`.
