"""Export M3 fixture: block-0 video FF residual.

Extracts tensors for M3 parity checker (block0_ff_residual_check):

  Required (FF parity):
    block0_ff_residual.vx_scaled - FF input (post AdaLN modulation) [B, T, D]
    block0_ff_residual.ff_out    - FF output [B, T, D]

  Optional (full residual algebra):
    block0_ff_residual.vgate_mlp - AdaLN FF gate [B, 1 or T, D]
    block0_ff_residual.vx_in      - residual base before FF residual [B, T, D]
    block0_ff_residual.vx_out     - residual output after FF residual [B, T, D]

Notes:
- `vx_out` is extracted from block0 output tensors by shape match with FF output.
- When `vx_in` is not directly captured, it is derived as:
    vx_in = vx_out - ff_out * vgate_mlp
  and metadata marks this as derived.
"""

import argparse
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


BLOCK0_KEY = "velocity_model.transformer_blocks.0"
FF_KEY = f"{BLOCK0_KEY}.ff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export M3 block-0 video FF residual fixture from replay .pt to safetensors"
    )
    parser.add_argument("input_pt", type=Path, help="Path to replay .pt file")
    parser.add_argument("output_st", type=Path, help="Output safetensors path")
    parser.add_argument("--block-index", type=int, default=0, help="Transformer block index (default: 0)")
    parser.add_argument("--token-limit", type=int, default=None, help="Optional token prefix length")
    return parser.parse_args()


def _as_tensor(item: Any) -> torch.Tensor | None:
    if isinstance(item, torch.Tensor):
        return item.detach().cpu().contiguous()
    if isinstance(item, (list, tuple)) and item:
        for v in item:
            t = _as_tensor(v)
            if t is not None:
                return t
    if isinstance(item, dict):
        for v in item.values():
            t = _as_tensor(v)
            if t is not None:
                return t
    return None


def _collect_tensors(obj: Any) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    if isinstance(obj, torch.Tensor):
        out.append(obj.detach().cpu().contiguous())
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            out.extend(_collect_tensors(item))
    elif isinstance(obj, dict):
        for item in obj.values():
            out.extend(_collect_tensors(item))
    return out


def _slice_token_prefix(t: torch.Tensor | None, limit: int | None) -> torch.Tensor | None:
    if t is None or limit is None:
        return t
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

    block_key = BLOCK0_KEY if block_idx == 0 else f"velocity_model.transformer_blocks.{block_idx}"
    ff_key = f"{block_key}.ff"

    ff_entry = acts.get(ff_key)
    if not isinstance(ff_entry, dict):
        available = sorted(k for k in acts.keys() if f"transformer_blocks.{block_idx}" in k)
        raise ValueError(
            f"Could not find FF capture at '{ff_key}'.\n"
            "Run replay with --capture-inputs and include block0 container.\n"
            f"Available block keys (sample): {available[:40]}"
        )

    ff_inputs = ff_entry.get("input", [])
    ff_outputs = ff_entry.get("output", [])

    vx_scaled = _as_tensor(ff_inputs[0]) if len(ff_inputs) > 0 else None
    ff_out = _as_tensor(ff_outputs[0]) if len(ff_outputs) > 0 else None

    if vx_scaled is None or ff_out is None:
        raise ValueError("Missing required FF tensors (vx_scaled/ff_out). Re-run replay with --capture-inputs.")

    vx_scaled = _slice_token_prefix(vx_scaled, args.token_limit)
    ff_out = _slice_token_prefix(ff_out, args.token_limit)

    tensors: dict[str, torch.Tensor] = {
        "block0_ff_residual.vx_scaled": vx_scaled,
        "block0_ff_residual.ff_out": ff_out,
    }

    vgate = acts.get(f"{block_key}.__aux__.vgate_mlp")
    if isinstance(vgate, torch.Tensor):
        vgate_t = vgate.detach().cpu().contiguous().to(ff_out.dtype)
        tensors["block0_ff_residual.vgate_mlp"] = vgate_t
    else:
        vgate_t = None

    derived_vx_in = False
    if vgate_t is not None:
        block_entry = acts.get(block_key, {})
        out_tensors = _collect_tensors(block_entry.get("output", [])) if isinstance(block_entry, dict) else []

        vx_out = None
        for t in out_tensors:
            if tuple(t.shape) == tuple(ff_out.shape):
                vx_out = t.to(ff_out.dtype)
                break

        if vx_out is not None:
            vx_out = _slice_token_prefix(vx_out, args.token_limit)
            vx_in = vx_out - ff_out * vgate_t
            vx_in = _slice_token_prefix(vx_in, args.token_limit)
            tensors["block0_ff_residual.vx_in"] = vx_in
            tensors["block0_ff_residual.vx_out"] = vx_out
            derived_vx_in = True
            print(
                "Residual keys exported: "
                f"vx_in={list(vx_in.shape)}, vgate_mlp={list(vgate_t.shape)}, vx_out={list(vx_out.shape)}"
            )
        else:
            print("Could not find block output tensor matching FF shape; residual keys skipped.")
    else:
        print("vgate_mlp missing; residual keys skipped.")

    metadata = {
        "source_pt": str(args.input_pt),
        "block_index": str(block_idx),
        "step_idx": str(obj.get("step_idx", "")),
        "pass_label": str(obj.get("pass_label", "")),
        "has_residual_keys": str("block0_ff_residual.vx_out" in tensors),
        "vx_in_derived": str(derived_vx_in),
    }

    args.output_st.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output_st), metadata=metadata)

    print(f"\nSaved: {args.output_st}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:45s}  shape={list(v.shape)}  dtype={v.dtype}")


if __name__ == "__main__":
    main()
