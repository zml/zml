"""Export M2 fixture: block-0 video text cross-attention residual.

Extracts tensors needed for M2 parity checker (block0_text_ca_check):

  Required (attn2 parity):
    block0_text_ca.attn2_x        - attn2 query input [B, Tq, D]
    block0_text_ca.context        - attn2 context input [B, Tk, D]
    block0_text_ca.attn2_out      - attn2 output [B, Tq, D]

  Optional:
    block0_text_ca.context_mask   - context/cross mask if captured

  Optional (residual algebra):
    block0_text_ca.vx_in          - vx before text-ca residual [B, T, D]
    block0_text_ca.text_ca_out    - delta returned by _apply_text_cross_attention [B, T, D]
    block0_text_ca.vx_out         - vx_in + text_ca_out [B, T, D]

Usage
-----
Step 1: replay with attn2 + block0 captures:

    uv run scripts/replay_stage2_transformer_step.py \
        --pass-label m2_capture \
        --capture-inputs \
        --capture-kwargs \
        --all-modules \
        --max-capture-gib 8.0 \
        --distilled-lora-strength 0.0 \
        --include '^velocity_model\\.transformer_blocks\\.0(\\.attn2)?(\\..*)?$'

Step 2: export fixture:

    python scripts/export_block0_text_ca_fixture.py \
        trace_run/acts_stage2_transformer_step_000_m2_capture.pt \
        fixtures/block0_text_ca.safetensors

Step 3: run Zig checker:

    bazel run //examples/ltx:block0_text_ca_check -- \
        /path/to/stage2_checkpoint.safetensors \
        fixtures/block0_text_ca.safetensors
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


BLOCK0_KEY = "velocity_model.transformer_blocks.0"
ATTN2_KEY = f"{BLOCK0_KEY}.attn2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export M2 block-0 video text cross-attention fixture from replay .pt to safetensors"
    )
    parser.add_argument("input_pt", type=Path, help="Path to replay .pt file")
    parser.add_argument("output_st", type=Path, help="Output safetensors path")
    parser.add_argument(
        "--block-index",
        type=int,
        default=0,
        help="Transformer block index (default: 0)",
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=None,
        help="Optional token prefix length to slice token-major tensors",
    )
    return parser.parse_args()


def _as_tensor(item):
    if isinstance(item, torch.Tensor):
        return item.detach().cpu().contiguous()
    if isinstance(item, (list, tuple)) and item and isinstance(item[0], torch.Tensor):
        return item[0].detach().cpu().contiguous()
    return None


def _slice_token_prefix(t: torch.Tensor | None, limit: int | None) -> torch.Tensor | None:
    if t is None or limit is None:
        return t

    if t.ndim == 3:
        # [B, T, D]
        return t[:, :limit, :].contiguous()

    if t.ndim == 2:
        # [T, D]
        return t[:limit, :].contiguous()

    return t


def _slice_context_mask(mask: torch.Tensor | None, limit: int | None) -> torch.Tensor | None:
    if mask is None or limit is None:
        return mask

    # Typical context mask layout prepared by TransformerArgs preprocessor: [B, 1, Q, K].
    if mask.ndim == 4:
        return mask[:, :, :limit, :limit].contiguous()

    return mask


def main() -> None:
    args = parse_args()
    block_idx = args.block_index

    print(f"Loading trace: {args.input_pt}")
    obj = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    acts = obj.get("activations", {})

    block_key = BLOCK0_KEY if block_idx == 0 else f"velocity_model.transformer_blocks.{block_idx}"
    attn2_key = f"{block_key}.attn2"

    attn2_entry = acts.get(attn2_key)
    if not isinstance(attn2_entry, dict):
        available = sorted(k for k in acts.keys() if f"transformer_blocks.{block_idx}" in k)
        raise ValueError(
            f"Could not find attn2 capture at '{attn2_key}'.\n"
            f"Run replay with --capture-inputs and include regex matching that module.\n"
            f"Available block keys (sample): {available[:40]}"
        )

    # Required attn2 tensors.
    attn2_inputs = attn2_entry.get("input", [])
    attn2_outputs = attn2_entry.get("output", [])
    attn2_x = _as_tensor(attn2_inputs[0]) if len(attn2_inputs) > 0 else None
    attn2_out = _as_tensor(attn2_outputs[0]) if len(attn2_outputs) > 0 else None

    kwargs_key = attn2_key + ".__kwargs__"
    kwargs = acts.get(kwargs_key, {})
    if not isinstance(kwargs, dict):
        kwargs = {}

    context = kwargs.get("context") if isinstance(kwargs.get("context"), torch.Tensor) else None
    context_mask = kwargs.get("mask") if isinstance(kwargs.get("mask"), torch.Tensor) else None

    # Fallback context from auxiliary block-level wrapper capture.
    if context is None:
        aux_ctx = acts.get(f"{block_key}.__aux__.text_ca_context")
        if isinstance(aux_ctx, torch.Tensor):
            context = aux_ctx.detach().cpu().contiguous()

    if context_mask is None:
        aux_mask = acts.get(f"{block_key}.__aux__.text_ca_context_mask")
        if isinstance(aux_mask, torch.Tensor):
            context_mask = aux_mask.detach().cpu().contiguous()

    if attn2_x is None or context is None or attn2_out is None:
        raise ValueError(
            "Missing required M2 tensors. Need attn2_x, context, attn2_out. "
            "Re-run replay with --capture-inputs --capture-kwargs and include block0.attn2."
        )

    # Apply optional token prefix slicing.
    attn2_x = _slice_token_prefix(attn2_x, args.token_limit)
    attn2_out = _slice_token_prefix(attn2_out, args.token_limit)
    context = _slice_token_prefix(context, args.token_limit)
    context_mask = _slice_context_mask(context_mask, args.token_limit)

    tensors: dict[str, torch.Tensor] = {
        "block0_text_ca.attn2_x": attn2_x,
        "block0_text_ca.context": context,
        "block0_text_ca.attn2_out": attn2_out,
    }

    if context_mask is not None:
        tensors["block0_text_ca.context_mask"] = context_mask

    # Optional residual algebra tensors from block-level aux capture.
    aux_vx_in = acts.get(f"{block_key}.__aux__.text_ca_vx_in")
    aux_text_ca_out = acts.get(f"{block_key}.__aux__.text_ca_out")

    if isinstance(aux_vx_in, torch.Tensor) and isinstance(aux_text_ca_out, torch.Tensor):
        vx_in = aux_vx_in.detach().cpu().contiguous()
        text_ca_out = aux_text_ca_out.detach().cpu().contiguous().to(vx_in.dtype)
        expected_shape = attn2_x.shape
        if tuple(vx_in.shape) != tuple(expected_shape) or tuple(text_ca_out.shape) != tuple(expected_shape):
            print(
                "Residual aux tensors do not match video attn2 shape; skipping residual export. "
                f"expected={list(expected_shape)} got vx_in={list(vx_in.shape)} text_ca_out={list(text_ca_out.shape)}"
            )
        else:
            vx_out = vx_in + text_ca_out

            vx_in = _slice_token_prefix(vx_in, args.token_limit)
            text_ca_out = _slice_token_prefix(text_ca_out, args.token_limit)
            vx_out = _slice_token_prefix(vx_out, args.token_limit)

            tensors["block0_text_ca.vx_in"] = vx_in
            tensors["block0_text_ca.text_ca_out"] = text_ca_out
            tensors["block0_text_ca.vx_out"] = vx_out

            print(
                "Residual keys exported: "
                f"vx_in={list(vx_in.shape)}, "
                f"text_ca_out={list(text_ca_out.shape)}, "
                f"vx_out={list(vx_out.shape)}"
            )
    else:
        print(
            "Residual keys not found (block aux capture missing). "
            "Checker will run in attn2-only mode."
        )

    metadata = {
        "source_pt": str(args.input_pt),
        "block_index": str(block_idx),
        "step_idx": str(obj.get("step_idx", "")),
        "pass_label": str(obj.get("pass_label", "")),
        "has_context_mask": str("block0_text_ca.context_mask" in tensors),
        "has_residual_keys": str("block0_text_ca.vx_out" in tensors),
    }

    args.output_st.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output_st), metadata=metadata)

    print(f"\nSaved: {args.output_st}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:45s}  shape={list(v.shape)}  dtype={v.dtype}")


if __name__ == "__main__":
    main()
