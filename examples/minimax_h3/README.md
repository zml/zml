# MiniMax-H3

`//examples/minimax_h3` runs MiniMax-H3 Base: a 33B dense omni-transformer that jointly denoises video and stereo audio.

Attachments pick the open official workflow. The prompt is sent as-is (trailing newlines stripped). The run writes `output.mp4`.

Closed MiniMax pieces (H3-Context-IR, H3-Regenerate-2K / upscaler) are not implemented. Pass an already-structured official prompt if you have one from their API.

| Attachments | Workflow | DiT weights |
| --- | --- | --- |
| text only | t2va | `transformer/` |
| `--image` and/or `--last-image` | fl2va | `transformer/` |
| `--refs` (images, videos, audio+visual) | ref2va | `transformer_ref/` |

Audio-only `--refs` are rejected; official needs at least one image or video.

## Run

```bash
bazel test //examples/minimax_h3:test

bazel run -c opt --@rules_zig//zig/settings:mode=release_fast //examples/minimax_h3 --@zml//platforms:cuda=true -- \
  --model=/var/models/MiniMaxAI/MiniMax-H3

bazel run -c opt --@rules_zig//zig/settings:mode=release_fast //examples/minimax_h3 --@zml//platforms:cuda=true -- \
  --model=/var/models/MiniMaxAI/MiniMax-H3 \
  --prompt="A cinematic wide shot of waves at dusk." \
  --duration=10
```

`--image` / `--last-image` select fl2va. `--refs` selects ref2va.

CPU / Metal / oneAPI default to the preview canvas. CUDA BFC grows with use (`preallocate=false`); it does not reserve 90% of every GPU at start.

## Options

- `--model=<path>`: Official repository or `hf://MiniMaxAI/MiniMax-H3`. Encoder, VAEs, and tokenizer come from here.
- `--dit=<path>`: Swap only the transformer. Leave empty to use `transformer/` (t2va/fl2va) or `transformer_ref/` (ref2va).
- `--prompt`, `--image`, `--last-image`, `--refs`
- `--duration` 5–15 (default 5)
- `--ratio` `21:9` `16:9` `4:3` `1:1` `3:4` `9:16`
- `--canvas=auto|tiny|preview|full` — `auto` is 768p at ≥40 GiB/device, else 640×352
- `--seed`, `--out`

Muxing needs `ffmpeg` on `PATH`.

```bash
hf download MiniMaxAI/MiniMax-H3 --local-dir MiniMax-H3
```
