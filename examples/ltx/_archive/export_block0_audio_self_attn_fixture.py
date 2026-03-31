"""Export M4-A fixture: block-0 audio self-attn residual.

Extracts tensors for M4-A parity checker (block0_audio_self_attn_check):

  Required (attn1-only test):
    block0_audio_self_attn.norm_ax    – AdaLN-normalised ax fed into audio_attn1 [B, T, 2048]
    block0_audio_self_attn.pe_cos     – cosine PE component
    block0_audio_self_attn.pe_sin     – sine PE component
    block0_audio_self_attn.attn1_out  – reference audio_attn1 output [B, T, 2048]

  Optional (full residual):
    block0_audio_self_attn.ax_in      – ax before self-attn residual [B, T, 2048]
    block0_audio_self_attn.agate_msa  – AdaLN gate [B, T, 2048]
    block0_audio_self_attn.ax_out     – ax after self-attn residual [B, T, 2048]

Usage
-----
Step 1: replay with block0 and audio_attn1 captures:

    uv run scripts/replay_stage2_transformer_step.py \\
        --pass-label m4a_t256_l00 \\
        --capture-inputs --capture-kwargs --all-modules \\
        --max-capture-gib 8.0 \\
        --distilled-lora-strength 0.0 \\
        --token-limit 256 \\
        --include '^velocity_model\\.transformer_blocks\\.0'

Step 2: export fixture:

    python scripts/export_block0_audio_self_attn_fixture.py \\
        trace_run/acts_stage2_transformer_step_000_m4a_t256_l00_t256.pt \\
        trace_run/fixtures/block0_audio_self_attn_m4a_t256_l00.safetensors \\
        --token-limit 256

Step 3: run Zig checker:

    bazel run //examples/ltx:block0_audio_self_attn_check -- \\
        /path/to/stage2_checkpoint.safetensors \\
        trace_run/fixtures/block0_audio_self_attn_m4a_t256_l00.safetensors
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


BLOCK0_KEY = "velocity_model.transformer_blocks.0"
AUDIO_ATTN1_KEY = f"{BLOCK0_KEY}.audio_attn1"

_AUDIO_DIM = 2048


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export M4-A block-0 audio self-attn fixture")
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


def _slice(t, limit):
    if t is None or limit is None:
        return t
    if t.ndim == 4:
        return t[:, :, :limit, :].contiguous()
    if t.ndim == 3:
        return t[:, :limit, :].contiguous()
    if t.ndim == 2:
        return t[:limit, :].contiguous()
    return t


def main() -> None:
    args = parse_args()
    block_idx = args.block_index

    print(f"Loading trace: {args.input_pt}")
    obj = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    acts = obj.get("activations", {})

    block_key = f"velocity_model.transformer_blocks.{block_idx}"
    attn1_key = f"{block_key}.audio_attn1"

    entry = acts.get(attn1_key)
    if not isinstance(entry, dict):
        available = sorted(k for k in acts.keys() if "transformer_blocks" in k)
        raise ValueError(f"Missing audio_attn1 at '{attn1_key}'. Available: {available[:20]}")

    inp = entry.get("input", [])
    out = entry.get("output", [])
    norm_ax = _as_tensor(inp[0]) if inp else None
    attn1_out = _as_tensor(out[0]) if out else None

    if norm_ax is None or attn1_out is None:
        raise ValueError("Missing norm_ax or attn1_out from audio_attn1 entry.")

    # PE from kwargs
    kw = acts.get(attn1_key + ".__kwargs__", {})
    pe_val = kw.get("pe") if isinstance(kw, dict) else None
    pe_cos = pe_sin = None
    if isinstance(pe_val, dict):
        pe_cos = pe_val.get("cos")
        pe_sin = pe_val.get("sin")
        if isinstance(pe_cos, torch.Tensor):
            pe_cos = pe_cos.detach().cpu().contiguous()
        if isinstance(pe_sin, torch.Tensor):
            pe_sin = pe_sin.detach().cpu().contiguous()
    elif isinstance(pe_val, (list, tuple)) and len(pe_val) == 2:
        pe_cos = pe_val[0].detach().cpu().contiguous()
        pe_sin = pe_val[1].detach().cpu().contiguous()

    lim = args.token_limit
    norm_ax = _slice(norm_ax, lim)
    attn1_out = _slice(attn1_out, lim)
    if pe_cos is not None:
        pe_cos = _slice(pe_cos, lim)
    if pe_sin is not None:
        pe_sin = _slice(pe_sin, lim)

    # Validate audio dim
    if norm_ax.shape[-1] != _AUDIO_DIM:
        raise ValueError(f"Expected audio dim {_AUDIO_DIM}, got {norm_ax.shape[-1]}. Wrong trace?")

    tensors: dict[str, torch.Tensor] = {
        "block0_audio_self_attn.norm_ax": norm_ax,
        "block0_audio_self_attn.attn1_out": attn1_out,
    }
    if pe_cos is not None:
        tensors["block0_audio_self_attn.pe_cos"] = pe_cos
    if pe_sin is not None:
        tensors["block0_audio_self_attn.pe_sin"] = pe_sin

    # Optional residual keys from aux hook
    aux_agate = acts.get(f"{block_key}.__aux__.agate_msa")
    ax_in_key = None
    # ax_in is the audio_attn1 input BEFORE AdaLN modulation — look in block container input
    block_entry = acts.get(block_key)
    if block_entry is not None:
        block_inp = block_entry.get("input", [])
        # audio stream x is the second positional tensor in the block container (after video.x)
        # The block is called as forward(video, audio, ...) and flattened inputs include both.
        # We search for a tensor with dim[-1]==2048 that matches [B, T, 2048].
        for candidate in (block_inp if isinstance(block_inp, (list, tuple)) else []):
            if isinstance(candidate, torch.Tensor) and candidate.ndim == 3 and candidate.shape[-1] == _AUDIO_DIM:
                ax_in_key = candidate.detach().cpu().contiguous()
                break

    if isinstance(aux_agate, torch.Tensor) and ax_in_key is not None:
        ax_in = ax_in_key
        agate_msa = aux_agate.detach().cpu().contiguous().to(ax_in.dtype)
        ax_out = ax_in + attn1_out * agate_msa

        ax_in = _slice(ax_in, lim)
        ax_out = _slice(ax_out, lim)

        if tuple(ax_in.shape) == tuple(norm_ax.shape):
            tensors["block0_audio_self_attn.ax_in"] = ax_in
            tensors["block0_audio_self_attn.agate_msa"] = agate_msa
            tensors["block0_audio_self_attn.ax_out"] = ax_out
            print(f"Residual keys exported: ax_in={list(ax_in.shape)}, agate_msa={list(agate_msa.shape)}, ax_out={list(ax_out.shape)}")
        else:
            print(f"Residual keys skipped: shape mismatch ax_in={list(ax_in.shape)} vs norm_ax={list(norm_ax.shape)}")
    else:
        print("Residual keys skipped: aux agate_msa or ax_in not available.")

    save_file(tensors, str(args.output_st))
    print(f"\nSaved: {args.output_st}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<45}  shape={str(list(v.shape)):<20}  dtype={v.dtype}")


if __name__ == "__main__":
    main()
