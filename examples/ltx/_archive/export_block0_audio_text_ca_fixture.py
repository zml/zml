"""Export M4-B fixture: block-0 audio text cross-attention residual.

Extracts tensors for M4-B parity checker (block0_audio_text_ca_check):

  Required (attn2-only):
    block0_audio_text_ca.attn2_x    – audio_attn2 query input [B, Tq, 2048]
    block0_audio_text_ca.context    – context input [B, Tk, 2048]
    block0_audio_text_ca.attn2_out  – audio_attn2 output [B, Tq, 2048]

  Optional (residual algebra):
    block0_audio_text_ca.ax_in         – ax before text-ca residual [B, T, 2048]
    block0_audio_text_ca.text_ca_out   – delta from _apply_text_cross_attention [B, T, 2048]
    block0_audio_text_ca.ax_out        – ax_in + text_ca_out [B, T, 2048]

Usage
-----
Step 1: replay (same trace as M4-A is sufficient):

    uv run scripts/replay_stage2_transformer_step.py \\
        --pass-label m4b_t256_l00 \\
        --capture-inputs --capture-kwargs --all-modules \\
        --max-capture-gib 8.0 \\
        --distilled-lora-strength 0.0 \\
        --token-limit 256 \\
        --include '^velocity_model\\.transformer_blocks\\.0'

Step 2: export fixture:

    python scripts/export_block0_audio_text_ca_fixture.py \\
        trace_run/acts_stage2_transformer_step_000_m4b_t256_l00_t256.pt \\
        trace_run/fixtures/block0_audio_text_ca_m4b_t256_l00.safetensors \\
        --token-limit 256

Step 3: run Zig checker:

    bazel run //examples/ltx:block0_audio_text_ca_check -- \\
        /path/to/stage2_checkpoint.safetensors \\
        trace_run/fixtures/block0_audio_text_ca_m4b_t256_l00.safetensors
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


BLOCK0_KEY = "velocity_model.transformer_blocks.0"
_AUDIO_DIM = 2048


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export M4-B block-0 audio text cross-attn fixture")
    p.add_argument("input_pt", type=Path)
    p.add_argument("output_st", type=Path)
    p.add_argument("--block-index", type=int, default=0)
    p.add_argument("--token-limit", type=int, default=None)
    return p.parse_args()


def _as_tensor(item):
    if isinstance(item, torch.Tensor):
        return item.detach().cpu().contiguous()
    if isinstance(item, (list, tuple)) and item and isinstance(item[0], torch.Tensor):
        return item[0].detach().cpu().contiguous()
    return None


def _slice_t(t, limit):
    if t is None or limit is None:
        return t
    if t.ndim == 3:
        return t[:, :limit, :].contiguous()
    if t.ndim == 4:
        return t[:, :, :limit, :limit].contiguous()
    return t


def main() -> None:
    args = parse_args()
    block_idx = args.block_index

    print(f"Loading trace: {args.input_pt}")
    obj = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    acts = obj.get("activations", {})

    block_key = f"velocity_model.transformer_blocks.{block_idx}"
    attn2_key = f"{block_key}.audio_attn2"

    entry = acts.get(attn2_key)
    if not isinstance(entry, dict):
        available = sorted(k for k in acts.keys() if "transformer_blocks" in k)
        raise ValueError(f"Missing audio_attn2 at '{attn2_key}'. Available: {available[:20]}")

    inp = entry.get("input", [])
    out = entry.get("output", [])
    attn2_x = _as_tensor(inp[0]) if inp else None
    attn2_out = _as_tensor(out[0]) if out else None

    # Context from kwargs or aux capture
    kw = acts.get(attn2_key + ".__kwargs__", {})
    context = kw.get("context") if isinstance(kw, dict) and isinstance(kw.get("context"), torch.Tensor) else None
    if context is not None:
        context = context.detach().cpu().contiguous()

    if context is None:
        aux_ctx = acts.get(f"{block_key}.__aux__.audio_text_ca_context")
        if isinstance(aux_ctx, torch.Tensor):
            context = aux_ctx.detach().cpu().contiguous()

    if attn2_x is None or context is None or attn2_out is None:
        raise ValueError("Missing attn2_x / context / attn2_out for M4-B. Re-run replay with --capture-inputs --capture-kwargs.")

    lim = args.token_limit
    attn2_x = _slice_t(attn2_x, lim)
    attn2_out = _slice_t(attn2_out, lim)
    context = _slice_t(context, lim)

    if attn2_x.shape[-1] != _AUDIO_DIM:
        raise ValueError(f"Expected audio dim {_AUDIO_DIM}, got {attn2_x.shape[-1]}. Wrong trace?")

    tensors: dict[str, torch.Tensor] = {
        "block0_audio_text_ca.attn2_x": attn2_x,
        "block0_audio_text_ca.context": context,
        "block0_audio_text_ca.attn2_out": attn2_out,
    }

    # Residual keys from aux hook
    aux_ax_in = acts.get(f"{block_key}.__aux__.audio_text_ca_ax_in")
    aux_ca_out = acts.get(f"{block_key}.__aux__.audio_text_ca_out")

    if isinstance(aux_ax_in, torch.Tensor) and isinstance(aux_ca_out, torch.Tensor):
        ax_in = aux_ax_in.detach().cpu().contiguous()
        ca_out = aux_ca_out.detach().cpu().contiguous().to(ax_in.dtype)
        if tuple(ax_in.shape[-1:]) == (_AUDIO_DIM,) and tuple(ca_out.shape) == tuple(ax_in.shape):
            ax_out = ax_in + ca_out
            ax_in = _slice_t(ax_in, lim)
            ca_out = _slice_t(ca_out, lim)
            ax_out = _slice_t(ax_out, lim)
            tensors["block0_audio_text_ca.ax_in"] = ax_in
            tensors["block0_audio_text_ca.text_ca_out"] = ca_out
            tensors["block0_audio_text_ca.ax_out"] = ax_out
            print(f"Residual keys exported: ax_in={list(ax_in.shape)}, text_ca_out={list(ca_out.shape)}, ax_out={list(ax_out.shape)}")
        else:
            print(f"Residual keys skipped: shape mismatch ax_in={list(ax_in.shape)} ca_out={list(ca_out.shape)}")
    else:
        print("Residual keys skipped: aux audio_text_ca_ax_in or audio_text_ca_out not available.")

    save_file(tensors, str(args.output_st))
    print(f"\nSaved: {args.output_st}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<45}  shape={str(list(v.shape)):<20}  dtype={v.dtype}")


if __name__ == "__main__":
    main()
