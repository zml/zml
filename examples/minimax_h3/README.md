# MiniMax-H3 Super

Compile-once HTTP serve: 4-step H3 draft, GPU handoff, 3-step LTX refine, TAEHV decode.

Weights are fetched through ZML's VFS the same way `//examples/llm` does. Pass a
local directory if you already have the repo on disk; otherwise the defaults are
Hugging Face / HTTPS URIs.

```
examples/minimax_h3/
  main.zig          process entry
  boot.zig          pin GPUs, load weights, compile SKUs
  serve.sh          launcher
    recipe/         SKUs, TP, memory, LoRA, weights
    draft/          Stage 1 — H3 encoder, DiT, TAEH3, audio
    refine/         Stage 2 — LTX, Gemma, Sol-Attn, TAEHV
    serve/          HTTP, page, mux, checkpoints
  taehv_check.zig   TAEHV stitch oracle
```

Tensor-parallel degree is the largest divisor of the attention-head gcd that
fits the visible GPU count (official gcd 8 → 1/2/4/8). Extra GPUs are not
opened.

## Run

```bash
# CUDA, fetch checkpoints automatically (HF token for gated repos)
bazel run -c opt //examples/minimax_h3 --@zml//platforms:cuda=true

# Or the launcher (same defaults)
./examples/minimax_h3/serve.sh

# Local checkout instead of Hugging Face
bazel run -c opt //examples/minimax_h3 --@zml//platforms:cuda=true -- \
  --model=/var/models/MiniMaxAI/MiniMax-H3
```

Then open `http://127.0.0.1:8080`.

```bash
CUDA_VISIBLE_DEVICES=2,3 ./examples/minimax_h3/serve.sh
H3_SKUS=5s ./examples/minimax_h3/serve.sh
```

## Options

- `--model=<path>`: H3 repository. Default `hf://MiniMaxAI/MiniMax-H3`. Local path, `hf://`, or `s3://`.
- `--dit=<path>`: Optional fused Turbo DiT overlay directory.
- `--lora=<path>`: Turbo LoRA if the fused overlay is missing. Default `hf://larryvrh/MiniMax-H3-Turbo-Lora/...`.
- `--taeh3=<path>`: TAEH3 decoder. Default is the public GitHub/HTTPS file; a local `output/` or `/var/models` copy wins if present.
- `--port=<n>`: HTTP port (default 8080).
- `--devices=<n>`: Cap visible GPUs (default: all already visible).
- `--attn=auto|fa2|sdpa`

Stage 2 LTX files default to `hf://Lightricks/LTX-2.5/...` (gated; accept the
license and set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`). Gemma's tokenizer
falls back to `hf://google/gemma-4-12B-it/tokenizer.json`. TAEHV wide weights
are pulled over HTTPS from madebyollin's GitHub release if no local copy exists.

`SOL_ATTN_LIB` is optional. If unset, serve looks for
`output/sol-attn/libh3_sol_attn_sm100.so` and otherwise uses dense attention.

`H3_SKUS=5s,5s-hd` compiles a subset. Unset compiles every row.
