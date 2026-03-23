"""Export M6 fixture: full block-0 video + audio stream.

Fixture keys (video stream):
  block0_full.vx_in        – initial video token state [B, T, 4096]
  block0_full.norm_vx      – AdaLN pre-normalised input to attn1 [B, T, 4096]
    block0_full.v_text_x     – exact query input to attn2 module [B, T, 4096]
  block0_full.v_pe_cos     – video self-attn PE cos [B, H, T, HD]
  block0_full.v_pe_sin     – video self-attn PE sin [B, H, T, HD]
  block0_full.vgate_msa    – video self-attn AdaLN gate [B, 1, 4096]
  block0_full.v_text_ctx   – video text cross-attn context [B, T_ctx, 4096]
    block0_full.a2v_x        – exact query input to audio_to_video_attn module [B, T, 4096]
  block0_full.a2v_ctx      – A->V cross-attn audio context [B, T_audio, 2048]
  block0_full.a2v_pe_cos   – A->V video PE cos [B, H, T, HD]
  block0_full.a2v_pe_sin   – A->V video PE sin [B, H, T, HD]
  block0_full.a2v_k_pe_cos – A->V audio K PE cos [B, H, T_audio, HD]
  block0_full.a2v_k_pe_sin – A->V audio K PE sin [B, H, T_audio, HD]
  block0_full.a2v_gate     – A->V gate [B, 1, 4096]
  block0_full.a2v_mask     – A->V mask [B, T, 4096]
  block0_full.vx_scaled    – AdaLN pre-normalised input to FF [B, T, 4096]
  block0_full.vgate_mlp    – video FF AdaLN gate [B, 1, 4096]
  block0_full.vx_out       – expected final video state [B, T, 4096]

Fixture keys (audio stream):
  block0_full.ax_in        – initial audio token state [B, T_audio, 2048]
  block0_full.norm_ax      – AdaLN pre-normalised input to audio_attn1 [B, T_audio, 2048]
    block0_full.a_text_x     – exact query input to audio_attn2 module [B, T_audio, 2048]
  block0_full.a_pe_cos     – audio self-attn PE cos [B, H, T_audio, HD]
  block0_full.a_pe_sin     – audio self-attn PE sin [B, H, T_audio, HD]
  block0_full.agate_msa    – audio self-attn AdaLN gate [B, 1, 2048]
  block0_full.a_text_ctx   – audio text cross-attn context [B, T_ctx, 2048]
    block0_full.v2a_x        – exact query input to video_to_audio_attn module [B, T_audio, 2048]
  block0_full.v2a_ctx      – V->A cross-attn video context [B, T, 4096]
  block0_full.v2a_pe_cos   – V->A audio PE cos [B, H, T_audio, HD]
  block0_full.v2a_pe_sin   – V->A audio PE sin [B, H, T_audio, HD]
  block0_full.v2a_k_pe_cos – V->A video K PE cos [B, H, T, HD]
  block0_full.v2a_k_pe_sin – V->A video K PE sin [B, H, T, HD]
  block0_full.v2a_gate     – V->A gate [B, 1, 2048]
  block0_full.v2a_mask     – V->A mask [B, T_audio, 2048]
  block0_full.ax_scaled    – AdaLN pre-normalised input to audio FF [B, T_audio, 2048]
  block0_full.agate_mlp    – audio FF AdaLN gate [B, 1, 2048]
  block0_full.ax_out       – expected final audio state [B, T_audio, 2048]
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export M6 block0 full stream fixture")
    p.add_argument("input_pt", type=Path)
    p.add_argument("output_st", type=Path)
    p.add_argument("--block-index", type=int, default=0)
    return p.parse_args()


def _as_tensor(item) -> torch.Tensor | None:
    if isinstance(item, torch.Tensor):
        return item.detach().cpu().contiguous()
    if isinstance(item, (list, tuple)) and item and isinstance(item[0], torch.Tensor):
        return item[0].detach().cpu().contiguous()
    return None


def _extract_pe_pair(value) -> tuple[torch.Tensor | None, torch.Tensor | None]:
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

    def aux(key: str) -> torch.Tensor | None:
        v = acts.get(f"{bk}.__aux__.{key}")
        return v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else None

    def mod_input(mod_key: str) -> torch.Tensor | None:
        entry = acts.get(f"{bk}.{mod_key}", {})
        inp = entry.get("input", [])
        return _as_tensor(inp[0]) if inp else None

    def mod_kwargs(mod_key: str) -> dict:
        return acts.get(f"{bk}.{mod_key}.__kwargs__", {}) or {}

    # ── Video stream ──────────────────────────────────────────────────────
    vx_in = aux("vx_in")
    norm_vx = mod_input("attn1")
    v_text_x = mod_input("attn2")
    v_pe_cos, v_pe_sin = _extract_pe_pair(mod_kwargs("attn1").get("pe"))
    vgate_msa = aux("vgate_msa")
    v_text_ctx_raw = aux("text_ca_context")
    a2v_x = mod_input("audio_to_video_attn")
    a2v_ctx_raw = mod_kwargs("audio_to_video_attn").get("context")
    a2v_ctx = _as_tensor(a2v_ctx_raw)
    a2v_pe_cos, a2v_pe_sin = _extract_pe_pair(mod_kwargs("audio_to_video_attn").get("pe"))
    a2v_k_pe_cos, a2v_k_pe_sin = _extract_pe_pair(mod_kwargs("audio_to_video_attn").get("k_pe"))
    a2v_gate = aux("a2v_gate")
    a2v_mask = aux("a2v_mask")
    vx_scaled = mod_input("ff")
    vgate_mlp = aux("vgate_mlp")
    vx_out = aux("vx_out")

    # ── Audio stream ──────────────────────────────────────────────────────
    ax_in = aux("ax_in")
    norm_ax = mod_input("audio_attn1")
    a_text_x = mod_input("audio_attn2")
    a_pe_cos, a_pe_sin = _extract_pe_pair(mod_kwargs("audio_attn1").get("pe"))
    agate_msa = aux("agate_msa")
    a_text_ctx_raw = aux("audio_text_ca_context")
    v2a_x = mod_input("video_to_audio_attn")
    v2a_ctx_raw = mod_kwargs("video_to_audio_attn").get("context")
    v2a_ctx = _as_tensor(v2a_ctx_raw)
    v2a_pe_cos, v2a_pe_sin = _extract_pe_pair(mod_kwargs("video_to_audio_attn").get("pe"))
    v2a_k_pe_cos, v2a_k_pe_sin = _extract_pe_pair(mod_kwargs("video_to_audio_attn").get("k_pe"))
    v2a_gate = aux("v2a_gate")
    v2a_mask = aux("v2a_mask")
    ax_scaled = mod_input("audio_ff")
    agate_mlp = aux("agate_mlp")
    ax_out = aux("ax_out")

    tensors: dict[str, torch.Tensor] = {}

    def add(name: str, t: torch.Tensor | None) -> None:
        if t is not None:
            tensors[name] = t
        else:
            print(f"  WARNING: missing tensor '{name}'")

    # Video
    add("block0_full.vx_in", vx_in)
    add("block0_full.norm_vx", norm_vx)
    add("block0_full.v_text_x", v_text_x)
    add("block0_full.v_pe_cos", v_pe_cos)
    add("block0_full.v_pe_sin", v_pe_sin)
    add("block0_full.vgate_msa", vgate_msa)
    add("block0_full.v_text_ctx", v_text_ctx_raw.detach().cpu().contiguous() if isinstance(v_text_ctx_raw, torch.Tensor) else None)
    add("block0_full.a2v_x", a2v_x)
    add("block0_full.a2v_ctx", a2v_ctx)
    add("block0_full.a2v_pe_cos", a2v_pe_cos)
    add("block0_full.a2v_pe_sin", a2v_pe_sin)
    add("block0_full.a2v_k_pe_cos", a2v_k_pe_cos)
    add("block0_full.a2v_k_pe_sin", a2v_k_pe_sin)
    add("block0_full.a2v_gate", a2v_gate)
    add("block0_full.a2v_mask", a2v_mask)
    add("block0_full.vx_scaled", vx_scaled)
    add("block0_full.vgate_mlp", vgate_mlp)
    add("block0_full.vx_out", vx_out)

    # Audio
    add("block0_full.ax_in", ax_in)
    add("block0_full.norm_ax", norm_ax)
    add("block0_full.a_text_x", a_text_x)
    add("block0_full.a_pe_cos", a_pe_cos)
    add("block0_full.a_pe_sin", a_pe_sin)
    add("block0_full.agate_msa", agate_msa)
    add("block0_full.a_text_ctx", a_text_ctx_raw.detach().cpu().contiguous() if isinstance(a_text_ctx_raw, torch.Tensor) else None)
    add("block0_full.v2a_x", v2a_x)
    add("block0_full.v2a_ctx", v2a_ctx)
    add("block0_full.v2a_pe_cos", v2a_pe_cos)
    add("block0_full.v2a_pe_sin", v2a_pe_sin)
    add("block0_full.v2a_k_pe_cos", v2a_k_pe_cos)
    add("block0_full.v2a_k_pe_sin", v2a_k_pe_sin)
    add("block0_full.v2a_gate", v2a_gate)
    add("block0_full.v2a_mask", v2a_mask)
    add("block0_full.ax_scaled", ax_scaled)
    add("block0_full.agate_mlp", agate_mlp)
    add("block0_full.ax_out", ax_out)

    missing = [k for k, v in tensors.items() if v is None]
    if missing:
        raise RuntimeError(f"Missing required fixture tensors: {missing}")

    args.output_st.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v for k, v in tensors.items()}, str(args.output_st))

    print(f"\nSaved: {args.output_st}")
    max_key = max(len(k) for k in tensors)
    for k, v in sorted(tensors.items()):
        print(f"  {k:<{max_key + 2}} shape={list(v.shape)}  dtype={v.dtype}")


if __name__ == "__main__":
    main()
