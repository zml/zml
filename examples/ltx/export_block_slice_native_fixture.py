"""Export native block-slice fixture for inline-AdaLN threading checks.

The fixture stores:
- stream inputs at slice entry (from start block aux hook)
- shared conditioning tensors (timesteps, text ctx, pe/k_pe)
- stream outputs at slice exit (from end block aux hook)

Keys follow the `block_slice_native.*` namespace.
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export native block-slice fixture")
    p.add_argument("input_pt", type=Path)
    p.add_argument("output_st", type=Path)
    p.add_argument("--start-block", type=int, required=True)
    p.add_argument("--end-block", type=int, required=True)
    return p.parse_args()


def _owned_tensor(v: torch.Tensor) -> torch.Tensor:
    return v.detach().cpu().contiguous().clone()


def _tensor(v):
    if isinstance(v, torch.Tensor):
        return _owned_tensor(v)
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
        return _owned_tensor(v[0])
    return None


def _extract_pe_pair(value):
    pe_cos = pe_sin = None
    if isinstance(value, dict):
        pe_cos = value.get("cos")
        pe_sin = value.get("sin")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        pe_cos, pe_sin = value[0], value[1]

    pe_cos = _owned_tensor(pe_cos) if isinstance(pe_cos, torch.Tensor) else None
    pe_sin = _owned_tensor(pe_sin) if isinstance(pe_sin, torch.Tensor) else None
    return pe_cos, pe_sin


def _kwargs_tensor(kwargs_obj, key):
    v = kwargs_obj.get(key) if isinstance(kwargs_obj, dict) else None
    return _owned_tensor(v) if isinstance(v, torch.Tensor) else None


def main() -> None:
    args = parse_args()
    if args.start_block < 0 or args.end_block < args.start_block:
        raise ValueError(f"Invalid block range: [{args.start_block}, {args.end_block}]")

    print(f"Loading trace: {args.input_pt}")
    obj = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    acts = obj.get("activations", {})

    start = f"velocity_model.transformer_blocks.{args.start_block}"
    end = f"velocity_model.transformer_blocks.{args.end_block}"

    def aux(block_key: str, key: str):
        return _tensor(acts.get(f"{block_key}.__aux__.{key}"))

    def mod_input(block_key: str, mod_key: str):
        entry = acts.get(f"{block_key}.{mod_key}", {})
        inp = entry.get("input", []) if isinstance(entry, dict) else []
        return _tensor(inp[0]) if inp else None

    def mod_output(block_key: str, mod_key: str):
        entry = acts.get(f"{block_key}.{mod_key}", {})
        out = entry.get("output", []) if isinstance(entry, dict) else []
        return _tensor(out[0]) if out else None

    def mod_kwargs(block_key: str, mod_key: str):
        return acts.get(f"{block_key}.{mod_key}.__kwargs__", {}) or {}

    # Shared conditioning is taken from slice entry block.
    v_pe_cos, v_pe_sin = _extract_pe_pair(mod_kwargs(start, "attn1").get("pe"))
    a_pe_cos, a_pe_sin = _extract_pe_pair(mod_kwargs(start, "audio_attn1").get("pe"))

    a2v_kwargs = mod_kwargs(start, "audio_to_video_attn")
    a2v_pe_cos, a2v_pe_sin = _extract_pe_pair(a2v_kwargs.get("pe"))
    a2v_k_pe_cos, a2v_k_pe_sin = _extract_pe_pair(a2v_kwargs.get("k_pe"))

    v2a_kwargs = mod_kwargs(start, "video_to_audio_attn")
    v2a_pe_cos, v2a_pe_sin = _extract_pe_pair(v2a_kwargs.get("pe"))
    v2a_k_pe_cos, v2a_k_pe_sin = _extract_pe_pair(v2a_kwargs.get("k_pe"))

    a2v_mask_blocks: list[torch.Tensor] = []
    v2a_mask_blocks: list[torch.Tensor] = []
    for block_idx in range(args.start_block, args.end_block + 1):
        block_key = f"velocity_model.transformer_blocks.{block_idx}"
        a2v_mask_i = aux(block_key, "a2v_mask")
        v2a_mask_i = aux(block_key, "v2a_mask")
        if a2v_mask_i is not None and v2a_mask_i is not None:
            a2v_mask_blocks.append(a2v_mask_i)
            v2a_mask_blocks.append(v2a_mask_i)

    a2v_masks = torch.stack(a2v_mask_blocks, dim=0) if len(a2v_mask_blocks) == (args.end_block - args.start_block + 1) else None
    v2a_masks = torch.stack(v2a_mask_blocks, dim=0) if len(v2a_mask_blocks) == (args.end_block - args.start_block + 1) else None

    # Native forward applies prompt modulation to raw text context internally,
    # so export the wrapper-level pre-modulation context here.
    v_text_ctx = aux(start, "text_ca_context")
    a_text_ctx = aux(start, "audio_text_ca_context")
    v_text_ctx_mask = _kwargs_tensor(mod_kwargs(start, "attn2"), "mask")
    if v_text_ctx_mask is None:
        v_text_ctx_mask = aux(start, "text_ca_context_mask")
    a_text_ctx_mask = _kwargs_tensor(mod_kwargs(start, "audio_attn2"), "mask")
    if a_text_ctx_mask is None:
        a_text_ctx_mask = aux(start, "audio_text_ca_context_mask")

    tensors = {
        "block_slice_native.vx_in": aux(start, "vx_in"),
        "block_slice_native.ax_in": aux(start, "ax_in"),
        "block_slice_native.video_timesteps": aux(start, "video_timesteps"),
        "block_slice_native.audio_timesteps": aux(start, "audio_timesteps"),
        "block_slice_native.v_prompt_timestep": aux(start, "v_prompt_timestep"),
        "block_slice_native.a_prompt_timestep": aux(start, "a_prompt_timestep"),
        "block_slice_native.v_pe_cos": v_pe_cos,
        "block_slice_native.v_pe_sin": v_pe_sin,
        "block_slice_native.a_pe_cos": a_pe_cos,
        "block_slice_native.a_pe_sin": a_pe_sin,
        "block_slice_native.v_text_ctx": v_text_ctx,
        "block_slice_native.a_text_ctx": a_text_ctx,
        "block_slice_native.v_text_ctx_mask": v_text_ctx_mask,
        "block_slice_native.a_text_ctx_mask": a_text_ctx_mask,
        "block_slice_native.v_cross_ss_ts": aux(start, "v_cross_ss_ts"),
        "block_slice_native.v_cross_gate_ts": aux(start, "v_cross_gate_ts"),
        "block_slice_native.a_cross_ss_ts": aux(start, "a_cross_ss_ts"),
        "block_slice_native.a_cross_gate_ts": aux(start, "a_cross_gate_ts"),
        "block_slice_native.a2v_pe_cos": a2v_pe_cos,
        "block_slice_native.a2v_pe_sin": a2v_pe_sin,
        "block_slice_native.a2v_k_pe_cos": a2v_k_pe_cos,
        "block_slice_native.a2v_k_pe_sin": a2v_k_pe_sin,
        "block_slice_native.a2v_masks": a2v_masks,
        "block_slice_native.v2a_pe_cos": v2a_pe_cos,
        "block_slice_native.v2a_pe_sin": v2a_pe_sin,
        "block_slice_native.v2a_k_pe_cos": v2a_k_pe_cos,
        "block_slice_native.v2a_k_pe_sin": v2a_k_pe_sin,
        "block_slice_native.v2a_masks": v2a_masks,
        "block_slice_native.vx_out": aux(end, "vx_out"),
        "block_slice_native.ax_out": aux(end, "ax_out"),
    }

    for local_idx, block_idx in enumerate(range(args.start_block, args.end_block + 1)):
        block_key = f"velocity_model.transformer_blocks.{block_idx}"
        vx_block_out = aux(block_key, "vx_out")
        vx_block_in = aux(block_key, "vx_in")
        ax_block_out = aux(block_key, "ax_out")
        ax_block_in = aux(block_key, "ax_in")
        norm_ax = mod_input(block_key, "audio_attn1")
        a_text_x = mod_input(block_key, "audio_attn2")
        # For forwardBlock0AudioStream exact-input parity we need the context tensor
        # exactly as seen by audio_attn2 (post prompt AdaLN modulation).
        a_text_ctx = _tensor(mod_kwargs(block_key, "audio_attn2").get("context"))
        audio_attn1_out = mod_output(block_key, "audio_attn1")
        audio_ff_net0_proj_out = mod_output(block_key, "audio_ff.net.0.proj")
        audio_ff_net0_out = mod_output(block_key, "audio_ff.net.0")
        audio_ff_out = mod_output(block_key, "audio_ff")
        audio_text_ca_out = aux(block_key, "audio_text_ca_out")
        v2a_delta = aux(block_key, "v2a_delta")
        v2a_x = mod_input(block_key, "video_to_audio_attn")
        v2a_ctx = _tensor(mod_kwargs(block_key, "video_to_audio_attn").get("context"))
        v2a_gate = aux(block_key, "v2a_gate")
        ax_scaled = mod_input(block_key, "audio_ff")
        agate_msa = aux(block_key, "agate_msa")
        agate_mlp = aux(block_key, "agate_mlp")
        agate_text_ca = aux(block_key, "agate_text_ca")

        if vx_block_out is not None:
            tensors[f"block_slice_native.vx_out_block_{local_idx}"] = vx_block_out
        if vx_block_in is not None:
            tensors[f"block_slice_native.vx_in_block_{local_idx}"] = vx_block_in
        if ax_block_out is not None:
            tensors[f"block_slice_native.ax_out_block_{local_idx}"] = ax_block_out
        if ax_block_in is not None:
            tensors[f"block_slice_native.ax_in_block_{local_idx}"] = ax_block_in
        if norm_ax is not None:
            tensors[f"block_slice_native.norm_ax_block_{local_idx}"] = norm_ax
        if a_text_x is not None:
            tensors[f"block_slice_native.a_text_x_block_{local_idx}"] = a_text_x
        if a_text_ctx is not None:
            tensors[f"block_slice_native.a_text_ctx_block_{local_idx}"] = a_text_ctx
        if v2a_x is not None:
            tensors[f"block_slice_native.v2a_x_block_{local_idx}"] = v2a_x
        if v2a_ctx is not None:
            tensors[f"block_slice_native.v2a_ctx_block_{local_idx}"] = v2a_ctx
        if v2a_gate is not None:
            tensors[f"block_slice_native.v2a_gate_block_{local_idx}"] = v2a_gate
        if ax_scaled is not None:
            tensors[f"block_slice_native.ax_scaled_block_{local_idx}"] = ax_scaled
        if audio_ff_out is not None:
            tensors[f"block_slice_native.audio_ff_out_block_{local_idx}"] = audio_ff_out
        if audio_ff_net0_proj_out is not None:
            tensors[f"block_slice_native.audio_ff_net0_proj_out_block_{local_idx}"] = audio_ff_net0_proj_out
        if audio_ff_net0_out is not None:
            tensors[f"block_slice_native.audio_ff_net0_out_block_{local_idx}"] = audio_ff_net0_out
        if agate_msa is not None:
            tensors[f"block_slice_native.agate_msa_block_{local_idx}"] = agate_msa
        if agate_mlp is not None:
            tensors[f"block_slice_native.agate_mlp_block_{local_idx}"] = agate_mlp
        if agate_text_ca is not None:
            tensors[f"block_slice_native.agate_text_ca_block_{local_idx}"] = agate_text_ca

        # Block-0 only: export reference intermediate audio states to localize where
        # native math diverges from Python.
        if (
            local_idx == 0
            and ax_block_in is not None
            and audio_attn1_out is not None
            and agate_msa is not None
            and audio_text_ca_out is not None
            and v2a_delta is not None
        ):
            h_after_msa = ax_block_in + audio_attn1_out * agate_msa
            h_after_text_ca = h_after_msa + audio_text_ca_out
            h_after_v2a = h_after_text_ca + v2a_delta
            tensors["block_slice_native.ax_after_msa_block_0"] = _owned_tensor(h_after_msa)
            tensors["block_slice_native.ax_after_text_ca_block_0"] = _owned_tensor(h_after_text_ca)
            tensors["block_slice_native.ax_after_v2a_block_0"] = _owned_tensor(h_after_v2a)

            if audio_ff_out is not None and agate_mlp is not None:
                h_after_ff = h_after_v2a + audio_ff_out * agate_mlp
                tensors["block_slice_native.ax_after_ff_block_0"] = _owned_tensor(h_after_ff)
        if local_idx < len(a2v_mask_blocks):
            tensors[f"block_slice_native.a2v_mask_block_{local_idx}"] = a2v_mask_blocks[local_idx]
        if local_idx < len(v2a_mask_blocks):
            tensors[f"block_slice_native.v2a_mask_block_{local_idx}"] = v2a_mask_blocks[local_idx]

    required_keys = {
        "block_slice_native.vx_in",
        "block_slice_native.ax_in",
        "block_slice_native.video_timesteps",
        "block_slice_native.audio_timesteps",
        "block_slice_native.v_prompt_timestep",
        "block_slice_native.a_prompt_timestep",
        "block_slice_native.v_pe_cos",
        "block_slice_native.v_pe_sin",
        "block_slice_native.a_pe_cos",
        "block_slice_native.a_pe_sin",
        "block_slice_native.v_text_ctx",
        "block_slice_native.a_text_ctx",
        "block_slice_native.v_cross_ss_ts",
        "block_slice_native.v_cross_gate_ts",
        "block_slice_native.a_cross_ss_ts",
        "block_slice_native.a_cross_gate_ts",
        "block_slice_native.a2v_pe_cos",
        "block_slice_native.a2v_pe_sin",
        "block_slice_native.a2v_k_pe_cos",
        "block_slice_native.a2v_k_pe_sin",
        "block_slice_native.v2a_pe_cos",
        "block_slice_native.v2a_pe_sin",
        "block_slice_native.v2a_k_pe_cos",
        "block_slice_native.v2a_k_pe_sin",
        "block_slice_native.vx_out",
        "block_slice_native.ax_out",
    }
    missing = [k for k, v in tensors.items() if v is None and k in required_keys]
    if missing:
        raise RuntimeError(f"Missing required native slice fixture tensors: {missing}")

    tensors_final: dict[str, torch.Tensor] = {k: v for k, v in tensors.items() if v is not None}

    args.output_st.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors_final,
        str(args.output_st),
        metadata={
            "scope": "block-slice native parity fixture",
            "start_block": str(args.start_block),
            "end_block": str(args.end_block),
            "block_count": str(args.end_block - args.start_block + 1),
        },
    )

    print(f"Saved: {args.output_st}")
    max_key = max(len(k) for k in tensors_final)
    for k, v in sorted(tensors_final.items()):
        print(f"  {k:<{max_key + 2}} shape={list(v.shape)} dtype={v.dtype}")


if __name__ == "__main__":
    main()
