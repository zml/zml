"""Export M5-A fixture: block-0 AV cross-attn A->V branch.

Required keys (attn parity):
  block0_av_a2v.x
  block0_av_a2v.context
  block0_av_a2v.pe_cos
  block0_av_a2v.pe_sin
  block0_av_a2v.k_pe_cos
  block0_av_a2v.k_pe_sin
  block0_av_a2v.attn_out

Required keys (gated delta parity):
  block0_av_a2v.gate
  block0_av_a2v.mask
  block0_av_a2v.delta
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export M5-A block0 AV A->V fixture")
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


def _slice_t(t: torch.Tensor | None, limit: int | None) -> torch.Tensor | None:
    if t is None or limit is None:
        return t
    if t.ndim == 4:
        return t[:, :, :limit, :].contiguous()
    if t.ndim == 3:
        return t[:, :limit, :].contiguous()
    if t.ndim == 2:
        return t[:limit, :].contiguous()
    return t


def _extract_pe_pair(value):
    pe_cos = pe_sin = None
    if isinstance(value, dict):
        pe_cos = value.get("cos")
        pe_sin = value.get("sin")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        pe_cos, pe_sin = value[0], value[1]

    if isinstance(pe_cos, torch.Tensor):
        pe_cos = pe_cos.detach().cpu().contiguous()
    else:
        pe_cos = None
    if isinstance(pe_sin, torch.Tensor):
        pe_sin = pe_sin.detach().cpu().contiguous()
    else:
        pe_sin = None
    return pe_cos, pe_sin


def main() -> None:
    args = parse_args()
    print(f"Loading trace: {args.input_pt}")
    obj = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    acts = obj.get("activations", {})

    block_key = f"velocity_model.transformer_blocks.{args.block_index}"
    attn_key = f"{block_key}.audio_to_video_attn"

    entry = acts.get(attn_key)
    if not isinstance(entry, dict):
        raise ValueError(f"Missing A->V attn entry: {attn_key}")

    inp = entry.get("input", [])
    out = entry.get("output", [])
    x = _as_tensor(inp[0]) if inp else None
    attn_out = _as_tensor(out[0]) if out else None

    kw = acts.get(attn_key + ".__kwargs__", {})
    context = kw.get("context") if isinstance(kw, dict) else None
    pe_val = kw.get("pe") if isinstance(kw, dict) else None
    k_pe_val = kw.get("k_pe") if isinstance(kw, dict) else None

    if isinstance(context, torch.Tensor):
        context = context.detach().cpu().contiguous()
    else:
        context = None

    pe_cos, pe_sin = _extract_pe_pair(pe_val)
    k_pe_cos, k_pe_sin = _extract_pe_pair(k_pe_val)

    gate = acts.get(f"{block_key}.__aux__.a2v_gate")
    mask = acts.get(f"{block_key}.__aux__.a2v_mask")
    delta = acts.get(f"{block_key}.__aux__.a2v_delta")

    gate = gate.detach().cpu().contiguous() if isinstance(gate, torch.Tensor) else None
    mask = mask.detach().cpu().contiguous() if isinstance(mask, torch.Tensor) else None
    delta = delta.detach().cpu().contiguous() if isinstance(delta, torch.Tensor) else None

    if delta is None and attn_out is not None and gate is not None and mask is not None:
        delta = (attn_out * gate.to(attn_out.dtype) * mask.to(attn_out.dtype)).contiguous()

    missing = [
        k
        for k, v in {
            "x": x,
            "context": context,
            "pe_cos": pe_cos,
            "pe_sin": pe_sin,
            "k_pe_cos": k_pe_cos,
            "k_pe_sin": k_pe_sin,
            "attn_out": attn_out,
            "gate": gate,
            "mask": mask,
            "delta": delta,
        }.items()
        if v is None
    ]
    if missing:
        raise ValueError(f"Missing required A->V keys: {missing}")

    lim = args.token_limit
    x = _slice_t(x, lim)
    context = _slice_t(context, lim)
    pe_cos = _slice_t(pe_cos, lim)
    pe_sin = _slice_t(pe_sin, lim)
    k_pe_cos = _slice_t(k_pe_cos, lim)
    k_pe_sin = _slice_t(k_pe_sin, lim)
    attn_out = _slice_t(attn_out, lim)
    gate = _slice_t(gate, lim)
    mask = _slice_t(mask, lim)
    delta = _slice_t(delta, lim)

    tensors = {
        "block0_av_a2v.x": x,
        "block0_av_a2v.context": context,
        "block0_av_a2v.pe_cos": pe_cos,
        "block0_av_a2v.pe_sin": pe_sin,
        "block0_av_a2v.k_pe_cos": k_pe_cos,
        "block0_av_a2v.k_pe_sin": k_pe_sin,
        "block0_av_a2v.attn_out": attn_out,
        "block0_av_a2v.gate": gate,
        "block0_av_a2v.mask": mask,
        "block0_av_a2v.delta": delta,
    }

    save_file(tensors, str(args.output_st))
    print(f"\nSaved: {args.output_st}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:<40} shape={list(v.shape)} dtype={v.dtype}")


if __name__ == "__main__":
    main()
