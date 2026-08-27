# MiniMax-H3

`//examples/minimax_h3` generates video and audio from a MiniMax-H3 repository.

Text only is t2va. `--image` / `--last-image` is fl2va. `--refs` is ref2va.

## Run

To load a model from HuggingFace directly:

```bash
# CUDA
bazel run //examples/minimax_h3 --@zml//platforms:cuda=true -- --model=hf://MiniMaxAI/MiniMax-H3
```

From a local directory:

```bash
bazel run //examples/minimax_h3 --@zml//platforms:cuda=true -- --model=/var/models/MiniMaxAI/MiniMax-H3
```

For a first-last-frame or reference run:

```bash
bazel run //examples/minimax_h3 --@zml//platforms:cuda=true -- --model=hf://MiniMaxAI/MiniMax-H3 --image=first.png
```

## Options

- `--model=<path>`: Required. Local path or `hf://MiniMaxAI/MiniMax-H3`.
- `--prompt=<string>`: Optional. Defaults to a short waves-at-dusk line.
- `--image`, `--last-image`, `--refs`
- `--duration=<sec>`: Optional. 5–15. Defaults to `5`.
- `--size=<WxH>`: Optional. Defaults to `1344x768`.
- `--steps=<n>`: Optional. Defaults to `30`.
- `--seed`, `--out`, `--dit`
