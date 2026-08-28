# MiniMax-H3

Same modes as Hailuo / the MiniMax video API. Open weights are **768P**.

| Mode | Flags |
| --- | --- |
| text-to-video | `--prompt='...'` |
| image-to-video | `--first-frame=still.png` |
| last-frame | `--last-frame=still.png` |
| first-and-last-frame | `--first-frame=a.png --last-frame=b.png` |
| reference-to-video | `--refs=char.png,motion.mp4,voice.wav` |

## Reference limits

`--refs` accepts at most **12** files, with at most **9** images, **3** videos, and **3** audio references.

A video keeps its own soundtrack when `ffmpeg` reports an audio stream. That soundtrack counts as one of the **3** audio references. A following wav is a separate audio reference. Three audio-bearing videos fill the audio budget; adding another wav is rejected after probing.

`--refs` order matters:

- Adaptive target canvas uses the **first visual** (image or video). Reversing refs changes the target canvas when the first visual changes. That is intentional.
- Each visual reference is encoded at **its own** geometry. Mixed-size refs do not share a zero-padded VAE compile shape.

Audio-only `--refs` are rejected; at least one image or video is required.

## Geometry

Default canvas: text-to-video is **16:9**. Other modes are **adaptive** from the first visual (still, last frame, or first non-audio `--refs`). `--ratio=9:16` (and the other Hailuo ratios) override the target canvas. `--size=WxH` sets exact pixels (snapped to 32); a snapped `--size` that exceeds `--max-pixels` is rejected.

Aspect must stay between **1:4** and **4:1**. After snap-to-32 the area never exceeds `--max-pixels` (default `768*1344`).

Ref2VA images use official short-edge **2048** (no area cap). Reference videos use their own aspect, never upscaled past the official canvas.

Open weights align duration to the VAE `17n+5` grid (5 s → 124 frames ≈ 5.2 s). `--duration` is **5–15** s (default 5). Hosted Hailuo allows 4 s; that is not implemented here.

## Scheduler

`--steps=N` (default 30) is N sigma grid points **including terminal zero**, so the runtime performs **N-1** DiT evaluations. This matches the official H3 scheduler. Do not treat `--steps=30` as 30 transformer forwards.

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
- `--prompt=<string>`: What happens in the shot.
- `--first-frame`, `--last-frame`, `--refs`
- `--duration=<sec>`: 5–15. Default `5`. NaN / ±Inf are rejected.
- `--ratio=<spec>`: `adaptive` \| `16:9` \| `9:16` \| `1:1` \| `4:3` \| `3:4` \| `21:9`.
- `--resolution=768P`: Open weights only. `2K` is hosted API.
- `--out=<path>`: `.mp4` or directory. Default `output.mp4`.
- `--steps`, `--seed`. See scheduler semantics above.

Advanced: `--frames`, `--size=WxH`, `--short-edge`, `--max-pixels`, `--dit`.

Not in this example: hosted 2K, Context-IR, Hailuo 4 s. Open weights are 768P, 5–15 s.

## Hardware and memory

Tested backends: **CUDA** for generation; unit tests also run on **CPU**. ROCm uses vanilla attention (no FA2/FA3). Tensor parallelism is 1, 2, 4, or 8 devices. `ffmpeg` is required to probe/decode media and to mux output.

## Tests

```bash
bazel test //examples/minimax_h3:test
bazel test //zml:test
bazel test //...
bazel test --@zml//platforms:cuda=true --@zml//platforms:cpu=false //...
bazel test --@zml//platforms:rocm=true --@zml//platforms:cpu=false //...
```

Lint (CI): `zig fmt --check` on tracked `*.zig` outside `third_party/`.
