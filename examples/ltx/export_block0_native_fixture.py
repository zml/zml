"""Export native block-0 fixture for inline-AdaLN threading checks.

This fixture targets `model.forwardBlock0NativeVideo` and
`model.forwardBlock0NativeAudio`, where AdaLN values are computed inline from
scale-shift tables rather than passed as precomputed module inputs.

Saved keys:
  block0_native.vx_in
  block0_native.ax_in
  block0_native.video_timesteps
  block0_native.audio_timesteps
    block0_native.v_prompt_timestep
    block0_native.a_prompt_timestep
  block0_native.v_pe_cos
  block0_native.v_pe_sin
  block0_native.a_pe_cos
  block0_native.a_pe_sin
  block0_native.v_text_ctx
  block0_native.a_text_ctx
  block0_native.v_cross_ss_ts
  block0_native.v_cross_gate_ts
  block0_native.a_cross_ss_ts
  block0_native.a_cross_gate_ts
    block0_native.a2v_pe_cos
    block0_native.a2v_pe_sin
    block0_native.a2v_k_pe_cos
    block0_native.a2v_k_pe_sin
    block0_native.v2a_pe_cos
    block0_native.v2a_pe_sin
    block0_native.v2a_k_pe_cos
    block0_native.v2a_k_pe_sin
  block0_native.vx_out
  block0_native.ax_out
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export block0 native fixture")
    p.add_argument("input_pt", type=Path)
    p.add_argument("output_st", type=Path)
    p.add_argument("--block-index", type=int, default=0)
    return p.parse_args()


def _tensor(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().contiguous()
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
        return v[0].detach().cpu().contiguous()
    return None


def _extract_pe_pair(value):
    pe_cos = pe_sin = None
    if isinstance(value, dict):
        pe_cos = value.get("cos")
        pe_sin = value.get("sin")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        pe_cos, pe_sin = value[0], value[1]

    pe_cos = pe_cos.detach().cpu().contiguous() if isinstance(pe_cos, torch.Tensor) else None
    pe_sin = pe_sin.detach().cpu().contiguous() if isinstance(pe_sin, torch.Tensor) else None
    return pe_cos, pe_sin


def main() -> None:
    args = parse_args()
    print(f"Loading trace: {args.input_pt}")
    obj = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    acts = obj.get("activations", {})

    bi = args.block_index
    bk = f"velocity_model.transformer_blocks.{bi}"

    def aux(key: str):
        return _tensor(acts.get(f"{bk}.__aux__.{key}"))

    def mod_kwargs(mod_key: str):
        return acts.get(f"{bk}.{mod_key}.__kwargs__", {}) or {}

    v_pe_cos, v_pe_sin = _extract_pe_pair(mod_kwargs("attn1").get("pe"))
    a_pe_cos, a_pe_sin = _extract_pe_pair(mod_kwargs("audio_attn1").get("pe"))

    # AV cross PE pairs come from each cross-attention module kwargs.
    a2v_kwargs = mod_kwargs("audio_to_video_attn")
    a2v_pe_cos, a2v_pe_sin = _extract_pe_pair(a2v_kwargs.get("pe"))
    a2v_k_pe_cos, a2v_k_pe_sin = _extract_pe_pair(a2v_kwargs.get("k_pe"))

    v2a_kwargs = mod_kwargs("video_to_audio_attn")
    v2a_pe_cos, v2a_pe_sin = _extract_pe_pair(v2a_kwargs.get("pe"))
    v2a_k_pe_cos, v2a_k_pe_sin = _extract_pe_pair(v2a_kwargs.get("k_pe"))

    tensors = {
        "block0_native.vx_in": aux("vx_in"),
        "block0_native.ax_in": aux("ax_in"),
        "block0_native.video_timesteps": aux("video_timesteps"),
        "block0_native.audio_timesteps": aux("audio_timesteps"),
        "block0_native.v_prompt_timestep": aux("v_prompt_timestep"),
        "block0_native.a_prompt_timestep": aux("a_prompt_timestep"),
        "block0_native.v_pe_cos": v_pe_cos,
        "block0_native.v_pe_sin": v_pe_sin,
        "block0_native.a_pe_cos": a_pe_cos,
        "block0_native.a_pe_sin": a_pe_sin,
        "block0_native.v_text_ctx": aux("text_ca_context"),
        "block0_native.a_text_ctx": aux("audio_text_ca_context"),
        "block0_native.v_cross_ss_ts": aux("v_cross_ss_ts"),
        "block0_native.v_cross_gate_ts": aux("v_cross_gate_ts"),
        "block0_native.a_cross_ss_ts": aux("a_cross_ss_ts"),
        "block0_native.a_cross_gate_ts": aux("a_cross_gate_ts"),
        "block0_native.a2v_pe_cos": a2v_pe_cos,
        "block0_native.a2v_pe_sin": a2v_pe_sin,
        "block0_native.a2v_k_pe_cos": a2v_k_pe_cos,
        "block0_native.a2v_k_pe_sin": a2v_k_pe_sin,
        "block0_native.v2a_pe_cos": v2a_pe_cos,
        "block0_native.v2a_pe_sin": v2a_pe_sin,
        "block0_native.v2a_k_pe_cos": v2a_k_pe_cos,
        "block0_native.v2a_k_pe_sin": v2a_k_pe_sin,
        "block0_native.vx_out": aux("vx_out"),
        "block0_native.ax_out": aux("ax_out"),
    }

    missing = [k for k, v in tensors.items() if v is None]
    if missing:
        raise RuntimeError(f"Missing required native fixture tensors: {missing}")

    tensors_final: dict[str, torch.Tensor] = {k: v for k, v in tensors.items() if v is not None}

    args.output_st.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors_final, str(args.output_st))

    print(f"Saved: {args.output_st}")
    max_key = max(len(k) for k in tensors_final)
    for k, v in sorted(tensors_final.items()):
        print(f"  {k:<{max_key + 2}} shape={list(v.shape)} dtype={v.dtype}")


if __name__ == "__main__":
    main()
