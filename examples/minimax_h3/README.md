# MiniMax-H3

`//examples/minimax_h3` runs MiniMax-H3 Base: a 33B dense omni-transformer that jointly denoises video and stereo audio.

This is not an LLM. Detection uses the Hugging Face layout (`FL2VA/` or `Ref2VA/`), not `examples/llm` `model_type`.

One path: `--prompt` plus optional `--image` / `--last-image` / `--refs`. A local IR compiler writes the brief, then the model writes `output.mp4` with sound.

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

# Image (and optional audio) references
bazel run --config=release //examples/minimax_h3 --@zml//platforms:cuda=true -- \
  --model=hf://MiniMaxAI/MiniMax-H3 --tiny \
  --refs=ref.png --prompt="..." --out=out_ref2va
```

## Options

- `--model=<path>`: Repository. Local directory or `hf://MiniMaxAI/MiniMax-H3`. Prefer a local copy; Hugging Face streaming reloads every DiT block.
- `--prompt=<string>`: Intent. Compiled to an H3 brief before encode.
- `--image` / `--last-image`: First / last frame.
- `--refs=<paths>`: Comma-separated images, videos, audio (max 12).
- `--duration=<sec>`: 4–15 seconds. Default `5`.
- `--ratio=<aspect>`: `21:9` `16:9` `4:3` `1:1` `3:4` `9:16`.
- `--tiny` / `--preview` / `--full`: Canvas. Auto is preview under 40 GiB, 768p on large accelerators. CPU and Metal stay preview.
- `--steps` / `--seed` / `--out`: Denoise steps, RNG seed, output dir or `.mp4` (default `output/`). Frames stay in a temp dir; the run dir gets `output.mp4` and `prompt.txt`.

`--full` and preview + images/refs need at least 40 GiB per device. On 24 GiB cards use `--tiny` for anything with attachments. `ffmpeg` on `PATH` muxes `output.mp4`; without it the run dir keeps `frames/` and `audio.wav`. Platform flags match every other example (`--@zml//platforms:cuda=true`, `rocm`, `tpu`, `metal`).

```bash
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" --local-dir MiniMax-H3
hf download MiniMaxAI/MiniMax-H3 --include "Ref2VA/*" --local-dir MiniMax-H3
```
