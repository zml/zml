# MiniMax-H3

`//examples/minimax_h3` runs MiniMax-H3 Base: a 33B dense omni-transformer that jointly denoises video and stereo audio.

The checkpoint uses the Hugging Face `FL2VA/` and `Ref2VA/` layout. Pass `--prompt` plus optional `--image` / `--last-image` / `--refs`. A local IR compiler writes the brief; the run writes `output.mp4`.

| Attachments | Weights |
| --- | --- |
| text only | `FL2VA/` |
| `--image` / `--last-image` | `FL2VA/` |
| `--refs` | `Ref2VA/transformer` (required) |

## Run

```bash
# Host tests (no weights)
bazel run //examples/minimax_h3:h3_tests

# Text → video+audio
bazel run --config=release //examples/minimax_h3 --@zml//platforms:cuda=true -- \
  --model=hf://MiniMaxAI/MiniMax-H3 --preview \
  --prompt="A cinematic wide shot of waves at dusk." --out=out_t2va

# First / last frame
bazel run --config=release //examples/minimax_h3 --@zml//platforms:cuda=true -- \
  --model=hf://MiniMaxAI/MiniMax-H3 --tiny \
  --image=first.png --last-image=last.png --prompt="..." --out=out_fl2va

# Image, video, and audio references
bazel run --config=release //examples/minimax_h3 --@zml//platforms:cuda=true -- \
  --model=hf://MiniMaxAI/MiniMax-H3 --tiny \
  --refs=ref.png --prompt="..." --out=out_ref2va
```

## Options

- `--model=<path>`: Local repository or `hf://MiniMaxAI/MiniMax-H3`.
- `--prompt=<string>`: Intent. Compiled to an H3 brief before encode.
- `--image` / `--last-image`: First / last frame.
- `--refs=<paths>`: Comma-separated images, videos, and audio (max 12).
- `--duration=<sec>`: 4–15 seconds. Default `5`.
- `--ratio=<aspect>`: `21:9` `16:9` `4:3` `1:1` `3:4` `9:16`.
- `--tiny` / `--preview` / `--full`: Canvas. Auto is preview below 40 GiB per device and 768p above that.
- `--steps` / `--seed` / `--out`: Denoise steps, RNG seed, output directory or `.mp4` (default `output/`).

`--full` and preview with attachments need at least 40 GiB per device. Use `--tiny` on 24 GiB devices when passing images or refs. Muxing `output.mp4` needs `ffmpeg` on `PATH`. Platform flags: `--@zml//platforms:cuda=true`, `rocm`, `tpu`, `metal`.

```bash
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" --local-dir MiniMax-H3
hf download MiniMaxAI/MiniMax-H3 --include "Ref2VA/*" --local-dir MiniMax-H3
```
