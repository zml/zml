"""Export M4-C fixture: block-0 audio FF residual.

Extracts tensors for M4-C parity checker (block0_audio_ff_residual_check):

  Required (FF-only):
    block0_audio_ff_residual.ax_scaled   – AdaLN-normalised ax fed into audio_ff [B, T, 2048]
    block0_audio_ff_residual.ff_out      – reference audio_ff output [B, T, 2048]

  Optional (residual algebra):
    block0_audio_ff_residual.agate_mlp   – AdaLN gate [B, T, 2048]
    block0_audio_ff_residual.ax_in       – ax before FF residual [B, T, 2048]
    block0_audio_ff_residual.ax_out      – ax after FF residual [B, T, 2048]

Usage
-----
Step 1: replay (same trace as M4-A/M4-B is sufficient):

    uv run scripts/replay_stage2_transformer_step.py \\
        --pass-label m4c_t256_l00 \\
        --capture-inputs --capture-kwargs --all-modules \\
        --max-capture-gib 8.0 \\
        --distilled-lora-strength 0.0 \\
        --token-limit 256 \\
        --include '^velocity_model\\.transformer_blocks\\.0'

Step 2: export fixture:

    python scripts/export_block0_audio_ff_residual_fixture.py \\
        trace_run/acts_stage2_transformer_step_000_m4c_t256_l00_t256.pt \\
        trace_run/fixtures/block0_audio_ff_residual_m4c_t256_l00.safetensors \\
        --token-limit 256

Step 3: run Zig checker:

    bazel run //examples/ltx:block0_audio_ff_residual_check -- \\
        /path/to/stage2_checkpoint.safetensors \\
        trace_run/fixtures/block0_audio_ff_residual_m4c_t256_l00.safetensors
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


BLOCK0_KEY = "velocity_model.transformer_blocks.0"
_AUDIO_DIM = 2048


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export M4-C block-0 audio FF residual fixture")
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
    return t


def main() -> None:
    args = parse_args()
    block_idx = args.block_index

    print(f"Loading trace: {args.input_pt}")
    obj = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    acts = obj.get("activations", {})

    block_key = f"velocity_model.transformer_blocks.{block_idx}"
    audio_ff_key = f"{block_key}.audio_ff"

    # audio_ff input = ax_scaled (AdaLN-normalised ax)
    ff_entry = acts.get(audio_ff_key)
    if not isinstance(ff_entry, dict):
        available = sorted(k for k in acts.keys() if "transformer_blocks" in k)
        raise ValueError(f"Missing audio_ff at '{audio_ff_key}'. Available: {available[:20]}")

    ff_inp = ff_entry.get("input", [])
    ff_out_list = ff_entry.get("output", [])
    ax_scaled = _as_tensor(ff_inp[0]) if ff_inp else None
    ff_out = _as_tensor(ff_out_list[0]) if ff_out_list else None

    if ax_scaled is None or ff_out is None:
        raise ValueError("Missing ax_scaled or ff_out from audio_ff entry.")

    if ax_scaled.shape[-1] != _AUDIO_DIM:
        raise ValueError(f"Expected audio dim {_AUDIO_DIM}, got {ax_scaled.shape[-1]}. Wrong trace?")

    lim = args.token_limit
    ax_scaled = _slice_t(ax_scaled, lim)
    ff_out = _slice_t(ff_out, lim)

    tensors: dict[str, torch.Tensor] = {
        "block0_audio_ff_residual.ax_scaled": ax_scaled,
        "block0_audio_ff_residual.ff_out": ff_out,
    }

    # Residual keys: agate_mlp from aux; ax_in derived from ax_out - ff_out * agate_mlp
    aux_agate_mlp = acts.get(f"{block_key}.__aux__.agate_mlp")

    # Look for the block output tensor with correct audio shape for ax_out
    ax_out_candidate = None
    block_out_entry = acts.get(block_key)
    if block_out_entry is not None:
        for candidate in (block_out_entry.get("output", []) if isinstance(block_out_entry, dict) else []):
            if isinstance(candidate, torch.Tensor) and candidate.ndim == 3 and candidate.shape[-1] == _AUDIO_DIM:
                ax_out_candidate = candidate.detach().cpu().contiguous()
                break
        # Also check nested structures (TransformerArgs outputs may be tuples)
        if ax_out_candidate is None:
            raw_out = block_out_entry.get("output", None)
            if isinstance(raw_out, (list, tuple)):
                for item in raw_out:
                    if isinstance(item, (list, tuple)):
                        for sub in item:
                            if isinstance(sub, torch.Tensor) and sub.ndim == 3 and sub.shape[-1] == _AUDIO_DIM:
                                ax_out_candidate = sub.detach().cpu().contiguous()
                                break
                    elif isinstance(item, torch.Tensor) and item.ndim == 3 and item.shape[-1] == _AUDIO_DIM:
                        ax_out_candidate = item.detach().cpu().contiguous()
                        break

    if isinstance(aux_agate_mlp, torch.Tensor) and ax_out_candidate is not None:
        agate_mlp = aux_agate_mlp.detach().cpu().contiguous()
        ax_out = ax_out_candidate
        ax_in = ax_out - ff_out * agate_mlp.to(ff_out.dtype)

        ax_in = _slice_t(ax_in, lim)
        ax_out = _slice_t(ax_out, lim)

        tensors["block0_audio_ff_residual.agate_mlp"] = agate_mlp
        tensors["block0_audio_ff_residual.ax_in"] = ax_in
        tensors["block0_audio_ff_residual.ax_out"] = ax_out
        print(f"Residual keys exported: ax_in={list(ax_in.shape)}, agate_mlp={list(agate_mlp.shape)}, ax_out={list(ax_out.shape)}")
    else:
        print("Residual keys skipped: aux agate_mlp or ax_out not available.")
        if aux_agate_mlp is None:
            print("  Hint: aux agate_mlp not captured. Check replay script aux hook.")
        if ax_out_candidate is None:
            print("  Hint: ax_out not found in block container output.")

    save_file(tensors, str(args.output_st))
    print(f"\nSaved: {args.output_st}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<48}  shape={str(list(v.shape)):<20}  dtype={v.dtype}")


if __name__ == "__main__":
    main()
