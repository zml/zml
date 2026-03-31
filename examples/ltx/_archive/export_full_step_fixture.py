"""Export a full single-step velocity_model fixture.

Replays one stage-2 denoising step and captures everything needed to validate
a full `velocity_model.forward` call in Zig:

  - Post-patchify hidden states: vx_in [B, T_v, 4096], ax_in [B, T_a, 2048]
  - All SharedInputs conditioning tensors (timestep modulations, PE cos/sin,
    text contexts, masks)
  - Sigma value
  - Final outputs: video_out [B, T_v, 128], audio_out [B, T_a, 128]

The SharedInputs tensors are captured from block 0's pre-hook — they are
identical across all 48 blocks (only the per-block scale_shift_table differs,
and that comes from checkpoint weights, not from inputs).

Usage:
  cd /root/repos/LTX-2
  uv run ./scripts/export_full_step_fixture.py \\
      trace_run/acts_stage2_transformer_step_000_full.pt \\
      --step-idx 0

  # With token limit (for memory-constrained runs):
  uv run ./scripts/export_full_step_fixture.py \\
      trace_run/acts_stage2_transformer_step_000_full_t512.pt \\
      --step-idx 0 --token-limit 512
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


TRACE_DIR = Path("trace_run")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export full-step fixture")
    p.add_argument(
        "input_pt",
        type=Path,
        help="Path to replay .pt file (e.g. acts_stage2_transformer_step_000_full.pt)",
    )
    p.add_argument(
        "--step-idx",
        type=int,
        default=0,
        help="Step index (for filename only, the .pt already has a fixed step)",
    )
    p.add_argument(
        "--token-limit",
        type=int,
        default=None,
        help="Token limit suffix for output filename (for reference only)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output safetensors path. Auto-generated if omitted.",
    )
    return p.parse_args()


def _t(v):
    """Detach, move to CPU, contiguous clone."""
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().contiguous().clone()
    return None


def _extract_pe_pair(value):
    """Extract (cos, sin) from a dict or tuple."""
    cos = sin = None
    if isinstance(value, dict):
        cos = value.get("cos")
        sin = value.get("sin")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        cos, sin = value[0], value[1]
    cos = _t(cos) if isinstance(cos, torch.Tensor) else None
    sin = _t(sin) if isinstance(sin, torch.Tensor) else None
    return cos, sin


def main() -> None:
    args = parse_args()

    print(f"Loading: {args.input_pt}")
    obj = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    acts = obj.get("activations", {})

    # Block keys
    block0 = "velocity_model.transformer_blocks.0"
    block47 = "velocity_model.transformer_blocks.47"

    def aux(block_key: str, key: str):
        return _t(acts.get(f"{block_key}.__aux__.{key}"))

    def mod_kwargs(block_key: str, mod_key: str):
        return acts.get(f"{block_key}.{mod_key}.__kwargs__", {}) or {}

    def _kwargs_tensor(kwargs_obj, key):
        v = kwargs_obj.get(key) if isinstance(kwargs_obj, dict) else None
        return _t(v) if isinstance(v, torch.Tensor) else None

    # --- Stream inputs (post-patchify, at block 0 entry) ---
    vx_in = aux(block0, "vx_in")
    ax_in = aux(block0, "ax_in")

    # --- Timestep modulations (from block 0, identical across blocks) ---
    video_timesteps = aux(block0, "video_timesteps")
    audio_timesteps = aux(block0, "audio_timesteps")
    v_prompt_timestep = aux(block0, "v_prompt_timestep")
    a_prompt_timestep = aux(block0, "a_prompt_timestep")
    v_cross_ss_ts = aux(block0, "v_cross_ss_ts")
    v_cross_gate_ts = aux(block0, "v_cross_gate_ts")
    a_cross_ss_ts = aux(block0, "a_cross_ss_ts")
    a_cross_gate_ts = aux(block0, "a_cross_gate_ts")

    # --- PE cos/sin (from block 0 kwargs, identical across blocks) ---
    v_pe_cos, v_pe_sin = _extract_pe_pair(mod_kwargs(block0, "attn1").get("pe"))
    a_pe_cos, a_pe_sin = _extract_pe_pair(mod_kwargs(block0, "audio_attn1").get("pe"))

    a2v_kwargs = mod_kwargs(block0, "audio_to_video_attn")
    a2v_pe_cos, a2v_pe_sin = _extract_pe_pair(a2v_kwargs.get("pe"))
    a2v_k_pe_cos, a2v_k_pe_sin = _extract_pe_pair(a2v_kwargs.get("k_pe"))

    v2a_kwargs = mod_kwargs(block0, "video_to_audio_attn")
    v2a_pe_cos, v2a_pe_sin = _extract_pe_pair(v2a_kwargs.get("pe"))
    v2a_k_pe_cos, v2a_k_pe_sin = _extract_pe_pair(v2a_kwargs.get("k_pe"))

    # --- Text contexts (pre-prompt-modulation, from block 0 aux) ---
    v_text_ctx = aux(block0, "text_ca_context")
    a_text_ctx = aux(block0, "audio_text_ca_context")
    v_text_ctx_mask = _kwargs_tensor(mod_kwargs(block0, "attn2"), "mask")
    if v_text_ctx_mask is None:
        v_text_ctx_mask = aux(block0, "text_ca_context_mask")
    a_text_ctx_mask = _kwargs_tensor(mod_kwargs(block0, "audio_attn2"), "mask")
    if a_text_ctx_mask is None:
        a_text_ctx_mask = aux(block0, "audio_text_ca_context_mask")

    # --- AV cross-attn masks (per-block, stacked) ---
    a2v_mask_blocks = []
    v2a_mask_blocks = []
    for block_idx in range(48):
        bk = f"velocity_model.transformer_blocks.{block_idx}"
        a2v_m = aux(bk, "a2v_mask")
        v2a_m = aux(bk, "v2a_mask")
        if a2v_m is not None and v2a_m is not None:
            a2v_mask_blocks.append(a2v_m)
            v2a_mask_blocks.append(v2a_m)

    a2v_masks = torch.stack(a2v_mask_blocks, dim=0) if len(a2v_mask_blocks) == 48 else None
    v2a_masks = torch.stack(v2a_mask_blocks, dim=0) if len(v2a_mask_blocks) == 48 else None

    # --- Final outputs (denoised video/audio from transformer) ---
    denoised_video = _t(obj.get("denoised_video"))
    denoised_audio = _t(obj.get("denoised_audio"))

    # --- Assemble fixture ---
    tensors = {}

    def add(key, val):
        if val is not None:
            tensors[key] = val
        else:
            print(f"  WARNING: missing {key}")

    # Stream inputs
    add("full_step.vx_in", vx_in)
    add("full_step.ax_in", ax_in)

    # Timestep modulations
    add("full_step.video_timesteps", video_timesteps)
    add("full_step.audio_timesteps", audio_timesteps)
    add("full_step.v_prompt_timestep", v_prompt_timestep)
    add("full_step.a_prompt_timestep", a_prompt_timestep)
    add("full_step.v_cross_ss_ts", v_cross_ss_ts)
    add("full_step.v_cross_gate_ts", v_cross_gate_ts)
    add("full_step.a_cross_ss_ts", a_cross_ss_ts)
    add("full_step.a_cross_gate_ts", a_cross_gate_ts)

    # PE cos/sin
    add("full_step.v_pe_cos", v_pe_cos)
    add("full_step.v_pe_sin", v_pe_sin)
    add("full_step.a_pe_cos", a_pe_cos)
    add("full_step.a_pe_sin", a_pe_sin)
    add("full_step.a2v_pe_cos", a2v_pe_cos)
    add("full_step.a2v_pe_sin", a2v_pe_sin)
    add("full_step.a2v_k_pe_cos", a2v_k_pe_cos)
    add("full_step.a2v_k_pe_sin", a2v_k_pe_sin)
    add("full_step.v2a_pe_cos", v2a_pe_cos)
    add("full_step.v2a_pe_sin", v2a_pe_sin)
    add("full_step.v2a_k_pe_cos", v2a_k_pe_cos)
    add("full_step.v2a_k_pe_sin", v2a_k_pe_sin)

    # Text contexts
    add("full_step.v_text_ctx", v_text_ctx)
    add("full_step.a_text_ctx", a_text_ctx)
    if v_text_ctx_mask is not None:
        tensors["full_step.v_text_ctx_mask"] = v_text_ctx_mask
    if a_text_ctx_mask is not None:
        tensors["full_step.a_text_ctx_mask"] = a_text_ctx_mask

    # AV masks
    if a2v_masks is not None:
        tensors["full_step.a2v_masks"] = a2v_masks
    if v2a_masks is not None:
        tensors["full_step.v2a_masks"] = v2a_masks

    # Expected outputs
    add("full_step.video_out", denoised_video)
    add("full_step.audio_out", denoised_audio)

    # Summary
    print(f"\nCaptured {len(tensors)} tensors:")
    for key in sorted(tensors.keys()):
        t = tensors[key]
        print(f"  {key:50s}  {str(tuple(t.shape)):30s}  {t.dtype}")

    # Save
    token_suffix = f"_t{args.token_limit}" if args.token_limit is not None else ""
    default_out = TRACE_DIR / f"full_step_fixture_step_{args.step_idx:03d}{token_suffix}.safetensors"
    out_path = args.output if args.output is not None else default_out
    TRACE_DIR.mkdir(exist_ok=True)
    save_file(tensors, str(out_path))
    print(f"\nSaved: {out_path}  ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
