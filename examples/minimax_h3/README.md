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

Default is official 768p: `--size=1344x768 --steps=30 --duration=5`. CUDA BFC grows with use (`preallocate=false`); it does not reserve 90% of every GPU at start. Tensor-parallel degree follows visible GPU count (1, 2, 4, or 8).

## Options

- `--model=<path>`: Official repository or `hf://MiniMaxAI/MiniMax-H3`. Encoder, VAEs, and tokenizer come from here.
- `--dit=<path>`: Swap only the transformer. Leave empty to use `transformer/` (t2va/fl2va) or `transformer_ref/` (ref2va).
- `--prompt`, `--image`, `--last-image`, `--refs`
- `--duration` 5–15 (default 5)
- `--size=WxH` (default `1344x768`, snap-32, area ≤ 768×1344)
- `--steps` (default 30)
- `--seed`, `--out`

## Layout

`main.zig` handles the CLI. `core/` holds config, checkpoint checks, load policy, requests, sharding, and buffer helpers. Graphs live in `model/` and `vae/`. Prompt assembly and ref sizing live in `conditioning/`. Runtime load, compile, generate, and media I/O live in `runtime/`. Tests sit in `tests/` by subsystem.

Muxing needs `ffmpeg` on `PATH`.

```bash
hf download MiniMaxAI/MiniMax-H3 --local-dir MiniMax-H3
```
