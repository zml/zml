# MiniMax-H3

`//examples/minimax_h3` runs MiniMax-H3 Base: a 33B dense omni-transformer that jointly denoises video and stereo audio.

Attachments pick the task. A local draft writes the brief; the run writes `output.mp4`.

| Attachments | Weights |
| --- | --- |
| text only | `FL2VA/` |
| `--image` / `--last-image` | `FL2VA/` |
| `--refs` | `Ref2VA/transformer` (required) |

## Run

```bash
bazel run //examples/minimax_h3:h3_tests

bazel run //examples/minimax_h3 --@zml//platforms:cuda=true -- \
  --model=/var/models/MiniMaxAI/MiniMax-H3

bazel run //examples/minimax_h3 --@zml//platforms:cuda=true -- \
  --model=/var/models/MiniMaxAI/MiniMax-H3 \
  --prompt="A cinematic wide shot of waves at dusk." \
  --duration=10
```

`--image` / `--last-image` select FL2VA. `--refs` selects Ref2VA.

## Options

- `--model=<path>`: Official repository, community bundle, DiT `.safetensors`, or `hf://MiniMaxAI/MiniMax-H3`
- `--dit` / `--encoder` / `--vae-video` / `--vae-audio` / `--tokenizer`: overrides
- `--prompt`, `--image`, `--last-image`, `--refs`
- `--duration` 4–15 (default 5)
- `--ratio` `21:9` `16:9` `4:3` `1:1` `3:4` `9:16`
- `--canvas=auto|tiny|preview|full` — `auto` is 768p at ≥40 GiB/device, else 640×352
- `--seed`, `--out`, `--profile`

Muxing needs `ffmpeg` on `PATH`.

```bash
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*" --local-dir MiniMax-H3
hf download MiniMaxAI/MiniMax-H3 --include "Ref2VA/*" --local-dir MiniMax-H3
```
