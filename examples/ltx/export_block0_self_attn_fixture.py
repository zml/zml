"""Export M1 fixture: block-0 video self-attn residual.

Extracts the tensors needed for the M1 parity checker (block0_self_attn_check):

  Required (attn1-only test):
    block0_self_attn.norm_vx   – AdaLN-normalised vx fed into attn1 [B, T, 4096]
    block0_self_attn.pe_cos    – cosine PE component
    block0_self_attn.pe_sin    – sine PE component
    block0_self_attn.attn1_out – reference attn1 output [B, T, 4096]

  Optional (full residual test, requires --checkpoint-path):
    block0_self_attn.vx_in     – vx before self-attn residual [B, T, 4096]
    block0_self_attn.vgate_msa – AdaLN gate [B, 1, 4096]
    block0_self_attn.vx_out    – vx after self-attn residual [B, T, 4096]

Usage
-----
Step 1: run replay capturing block0.attn1 with inputs and kwargs:

    uv run scripts/replay_stage2_transformer_step.py \\
        --pass-label m1_capture \\
        --capture-inputs \\
        --capture-kwargs \\
        --all-modules \\
        --max-capture-gib 8.0 \\
        --distilled-lora-strength 0.0 \\
        --include '^velocity_model\\.transformer_blocks\\.0(\\.attn1)?(\\..*)?$'

    This captures transformer_blocks.0 (for vx_in) and transformer_blocks.0.attn1
    (for norm_vx, pe_cos/pe_sin, attn1_out) together. If the block container capture
    is too large, run two separate passes:
      Pass A: --include '^velocity_model\\.transformer_blocks\\.0$'
              --max-capture-gib 2.0                     (block container only)
      Pass B: --include '^velocity_model\\.transformer_blocks\\.0\\.attn1$'
              --capture-inputs --capture-kwargs          (attn1 only)

Step 2: export fixture:

    # Attn1-only fixture (no vx_in / vgate / vx_out):
    python scripts/export_block0_self_attn_fixture.py \\
        trace_run/acts_stage2_transformer_step_0_m1_capture.pt \\
        fixtures/block0_self_attn.safetensors

    # Full-residual fixture (adds vx_in, vgate_msa, vx_out via AdaLN computation):
    python scripts/export_block0_self_attn_fixture.py \\
        trace_run/acts_stage2_transformer_step_0_m1_capture.pt \\
        fixtures/block0_self_attn.safetensors \\
        --checkpoint-path /path/to/stage2_model.safetensors

Step 3: run Zig checker:

    bazel run //examples/ltx:block0_self_attn_check -- \\
        /path/to/stage2_model.safetensors \\
        fixtures/block0_self_attn.safetensors
"""
import argparse
from pathlib import Path

import torch
from safetensors import safe_open as _safe_open
from safetensors.torch import save_file

from export_activation_fixture import resolve_activation_key


BLOCK0_KEY = "velocity_model.transformer_blocks.0"
ATTN1_KEY = f"{BLOCK0_KEY}.attn1"
VELOCITY_MODEL_KEY = "velocity_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export M1 block-0 video self-attn fixture from replay .pt to safetensors"
    )
    parser.add_argument("input_pt", type=Path, help="Path to replay .pt file")
    parser.add_argument("output_st", type=Path, help="Output safetensors path")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help=(
            "Optional stage-2 checkpoint (.safetensors). When provided the exporter "
            "computes norm_vx, vgate_msa and vx_out using the block-0 scale_shift_table, "
            "enabling the full residual test in the Zig checker."
        ),
    )
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
        help="Optional token prefix length to slice fixture tensors",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Token-wise RMS normalisation identical to ltx_core.utils.rms_norm."""
    # x: [B, T, D]
    norm = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return x / norm


def _slice_token_prefix(t: torch.Tensor, limit: int | None) -> torch.Tensor:
    if limit is None or t is None:
        return t
    if t.ndim == 4:
        # [B, H, T, D] layout
        return t[:, :, :limit, :].contiguous()
    if t.ndim == 3:
        # [B, T, D] layout
        return t[:, :limit, :].contiguous()
    if t.ndim == 2:
        # [T, D] layout
        return t[:limit, :].contiguous()
    return t


def _extract_pe(acts: dict, attn_key: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return (pe_cos, pe_sin) from __kwargs__ capture or None if unavailable."""
    kwargs_key = attn_key + ".__kwargs__"
    kw = acts.get(kwargs_key, {})
    if not isinstance(kw, dict):
        kw = {}

    pe_val = kw.get("pe")

    if isinstance(pe_val, dict):
        # Captured as {"cos": ..., "sin": ...}
        cos = pe_val.get("cos")
        sin = pe_val.get("sin")
        if isinstance(cos, torch.Tensor) and isinstance(sin, torch.Tensor):
            return cos.detach().cpu().contiguous(), sin.detach().cpu().contiguous()

    if isinstance(pe_val, (list, tuple)) and len(pe_val) == 2:
        c, s = pe_val
        if isinstance(c, torch.Tensor) and isinstance(s, torch.Tensor):
            return c.detach().cpu().contiguous(), s.detach().cpu().contiguous()

    if isinstance(pe_val, torch.Tensor):
        if torch.is_complex(pe_val):
            return pe_val.real.contiguous(), pe_val.imag.contiguous()
        # Assume cos and sin are stacked along the last axis.
        half = pe_val.shape[-1] // 2
        return pe_val[..., :half].contiguous(), pe_val[..., half:].contiguous()

    return None, None


def _get_ada_value(scale_shift_table: torch.Tensor, timestep: torch.Tensor, idx: int) -> torch.Tensor:
    """Partial re-implementation of BasicAVTransformerBlock.get_ada_values.

    Upstream logic reshapes timestep using num_ada_params = scale_shift_table.shape[0].
    Therefore timestep width must be exactly N * D where N, D come from scale_shift_table.
    """
    B = timestep.shape[0]
    N = scale_shift_table.shape[0]
    D = scale_shift_table.shape[1]

    expected = N * D
    if timestep.shape[2] != expected:
        raise ValueError(
            f"Incompatible timestep width for get_ada_values: got {timestep.shape[2]}, "
            f"expected N*D={expected} from scale_shift_table shape {list(scale_shift_table.shape)}."
        )
    if idx >= N:
        raise ValueError(f"idx={idx} out of range for N={N} Ada values from scale_shift_table.")

    # [B, 1, N, D]
    ts = timestep.reshape(B, timestep.shape[1], N, D)

    # [B, 1, D] = table[idx] broadcast + ts[:, :, idx, :]
    sst_row = scale_shift_table[idx].to(device=timestep.device, dtype=timestep.dtype)
    return sst_row.unsqueeze(0).unsqueeze(0) + ts[:, :, idx, :]


def _load_scale_shift_table(ckpt_path: Path, block_idx: int) -> torch.Tensor:
    """Load just the scale_shift_table tensor for block_idx (single-tensor read, no full-ckpt load)."""
    candidates = (
        f"model.velocity_model.transformer_blocks.{block_idx}.scale_shift_table",
        f"model.diffusion_model.transformer_blocks.{block_idx}.scale_shift_table",
        f"velocity_model.transformer_blocks.{block_idx}.scale_shift_table",
        f"diffusion_model.transformer_blocks.{block_idx}.scale_shift_table",
    )
    with _safe_open(str(ckpt_path), framework="pt", device="cpu") as f:
        available = set(f.keys())
        for key in candidates:
            if key in available:
                return f.get_tensor(key).cpu()
    raise KeyError(
        f"Could not find scale_shift_table for block {block_idx} in checkpoint. "
        f"Tried: {list(candidates)}"
    )


def _compute_vgate_msa_from_sst(sst: torch.Tensor, video_timesteps: torch.Tensor) -> torch.Tensor:
    """Compute vgate_msa given a pre-loaded scale_shift_table and video timesteps tensor."""
    # vgate_msa is the 3rd AdaLN value (index 2) for the self-attention modulation.
    vgate_msa = _get_ada_value(sst, video_timesteps, idx=2)  # [B, 1, D]
    return vgate_msa.float()


def _find_video_timesteps_from_block_inputs(
    captured_inputs: list,
    expected_nd: int | None = None,
) -> torch.Tensor | None:
    """Try to heuristically locate video.timesteps from flattened block inputs.

    BasicAVTransformerBlock.forward receives (video: TransformerArgs, audio: ..., perturb).
    _flatten_impl recurses into TransformerArgs fields in declaration order:
        x, context, context_mask?, timesteps, embedded_timestep, ...
    where context_mask may be None (skipped by _flatten).

        When expected_nd is given, require exact shape[2] == expected_nd where
        expected_nd = N * D from scale_shift_table. This matches upstream
        get_ada_values reshape contract exactly and avoids silent misuse of
        unrelated tensors such as embedded timestep variants.

        Falls back to loose shape[2] > 4096 heuristic only when expected_nd is None.
    """
    for t in captured_inputs:
        if not isinstance(t, torch.Tensor):
            continue
        if t.ndim == 3 and t.shape[1] == 1:
            if expected_nd is not None:
                if t.shape[2] == expected_nd:
                    return t
            elif t.shape[2] > 4096:
                return t
    return None


def _collect_tensors(obj) -> list[torch.Tensor]:
    """Recursively collect torch tensors from nested capture structures."""
    out: list[torch.Tensor] = []
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            out.extend(_collect_tensors(item))
    elif isinstance(obj, dict):
        for item in obj.values():
            out.extend(_collect_tensors(item))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    block_idx = args.block_index
    token_limit = args.token_limit

    print(f"Loading trace: {args.input_pt}")
    obj = torch.load(args.input_pt, map_location="cpu", weights_only=False)
    acts = obj.get("activations", {})

    block_key = BLOCK0_KEY if block_idx == 0 else f"velocity_model.transformer_blocks.{block_idx}"
    attn_key = f"{block_key}.attn1"
    aux_vgate_key = f"{block_key}.__aux__.vgate_msa"

    # ── Required: norm_vx (attn1 input) and attn1_out ──────────────────────
    norm_vx: torch.Tensor | None = None
    attn1_out: torch.Tensor | None = None

    if attn_key in acts:
        entry = acts[attn_key]
        inp = entry.get("input", [])
        out = entry.get("output", [])
        if inp:
            item = inp[0]
            norm_vx = (item if isinstance(item, torch.Tensor) else item[0]).detach().cpu().contiguous()
        if out:
            item = out[0]
            attn1_out = (item if isinstance(item, torch.Tensor) else item[0]).detach().cpu().contiguous()
    else:
        # Fallback: try to_q input as norm_vx.
        try:
            q_key = resolve_activation_key(acts, attn_key + ".to_q", allow_proj_suffix=False)
            q_entry = acts[q_key]
            q_inp = q_entry.get("input", [])
            if q_inp:
                item = q_inp[0]
                norm_vx = (item if isinstance(item, torch.Tensor) else item[0]).detach().cpu().contiguous()
        except (KeyError, Exception) as e:
            print(f"WARNING: could not find attn1 input via to_q fallback: {e}")

    if norm_vx is None or attn1_out is None:
        available = sorted(k for k in acts.keys() if f"transformer_blocks.{block_idx}" in k)
        raise ValueError(
            f"Could not extract norm_vx or attn1_out.\n"
            f"Run replay with --capture-inputs and --include matching '{attn_key}'.\n"
            f"Available keys: {available[:30]}"
        )

    # ── Required: PE tensors ────────────────────────────────────────────────
    pe_cos, pe_sin = _extract_pe(acts, attn_key)

    if pe_cos is None or pe_sin is None:
        print(
            f"WARNING: PE tensors not found under {attn_key}.__kwargs__. "
            "Re-run replay with --capture-kwargs to enable RoPE validation. "
            "Fixture will be exported without pe_cos/pe_sin (attn1 RoPE parity NOT tested)."
        )

    # ── Apply token_limit ───────────────────────────────────────────────────
    norm_vx = _slice_token_prefix(norm_vx, token_limit)
    attn1_out = _slice_token_prefix(attn1_out, token_limit)
    if pe_cos is not None:
        pe_cos = _slice_token_prefix(pe_cos, token_limit)
    if pe_sin is not None:
        pe_sin = _slice_token_prefix(pe_sin, token_limit)

    # ── Build fixture ───────────────────────────────────────────────────────
    tensors: dict[str, torch.Tensor] = {
        "block0_self_attn.norm_vx": norm_vx,
        "block0_self_attn.attn1_out": attn1_out,
    }
    if pe_cos is not None:
        tensors["block0_self_attn.pe_cos"] = pe_cos
    if pe_sin is not None:
        tensors["block0_self_attn.pe_sin"] = pe_sin

    # ── Optional: full-residual keys (require block container + checkpoint) ─
    vx_in: torch.Tensor | None = None
    video_timesteps: torch.Tensor | None = None
    sst: torch.Tensor | None = None
    expected_nd: int | None = None
    if args.checkpoint_path is not None:
        try:
            sst = _load_scale_shift_table(args.checkpoint_path, block_idx)
            expected_nd = sst.shape[0] * sst.shape[1]
            print(
                f"scale_shift_table: shape={list(sst.shape)} "
                f"(N={sst.shape[0]}, D={sst.shape[1]}, N*D={expected_nd})"
            )
        except Exception as e:
            print(f"WARNING: could not load scale_shift_table from checkpoint: {e}")

    block_entry = acts.get(block_key)
    if block_entry is not None:
        inp = block_entry.get("input", [])
        if inp:
            item = inp[0]
            vx_in = (item if isinstance(item, torch.Tensor) else item[0]).detach().cpu().contiguous()
            # Try to locate video.timesteps from the flattened inputs.
            if isinstance(inp, (list, tuple)):
                all_input_tensors = _collect_tensors(inp)
                video_timesteps = _find_video_timesteps_from_block_inputs(
                    all_input_tensors,
                    expected_nd=expected_nd,
                )

    # Fallback: capture from velocity_model input if block input does not expose full timesteps.
    # This keeps memory lower than tracing all modules and provides the canonical TransformerArgs inputs.
    if video_timesteps is None and expected_nd is not None:
        vm_entry = acts.get(VELOCITY_MODEL_KEY)
        if vm_entry is not None:
            vm_inp = vm_entry.get("input", [])
            if isinstance(vm_inp, (list, tuple)):
                vm_tensors = _collect_tensors(vm_inp)
                video_timesteps = _find_video_timesteps_from_block_inputs(
                    vm_tensors,
                    expected_nd=expected_nd,
                )
                if video_timesteps is not None:
                    print(
                        f"Found video.timesteps candidate from {VELOCITY_MODEL_KEY} input: "
                        f"shape={list(video_timesteps.shape)}"
                    )
    if vx_in is not None:
        print(f"Found vx_in from block container capture: shape={list(vx_in.shape)} dtype={vx_in.dtype}")
    else:
        print(
            f"Block container key '{block_key}' not in trace (or input not captured). "
            "Skipping vx_in / full-residual fixture keys. "
            "Re-run replay with --all-modules and matching include pattern to enable."
        )

    # First choice: use vgate_msa captured directly during replay via block0 aux hook.
    direct_vgate = acts.get(aux_vgate_key)
    if isinstance(direct_vgate, torch.Tensor) and vx_in is not None:
        print(f"Using directly captured vgate_msa from replay: shape={list(direct_vgate.shape)}")
        vgate_msa_native = direct_vgate.detach().cpu().contiguous().to(vx_in.dtype)
        vx_out = vx_in + attn1_out * vgate_msa_native

        vx_in_sliced = _slice_token_prefix(vx_in, token_limit)
        vx_out_sliced = _slice_token_prefix(vx_out, token_limit)

        tensors["block0_self_attn.vx_in"] = vx_in_sliced
        tensors["block0_self_attn.vgate_msa"] = vgate_msa_native
        tensors["block0_self_attn.vx_out"] = vx_out_sliced

        print(
            f"Full-residual keys exported (direct gate): "
            f"vx_in={list(vx_in_sliced.shape)}, "
            f"vgate_msa={list(vgate_msa_native.shape)}, "
            f"vx_out={list(vx_out_sliced.shape)}"
        )

    elif vx_in is not None and sst is not None:
        if video_timesteps is not None:
            print(f"Found video.timesteps candidate: shape={list(video_timesteps.shape)}")
            try:
                vgate_msa = _compute_vgate_msa_from_sst(sst, video_timesteps)
                # vx_out = vx + attn1_out * vgate_msa
                # Compute in bf16 (model native dtype) to match Zig precision, not f32.
                # vgate_msa is f32 from _compute_vgate_msa_from_sst, so cast to match vx_in.dtype.
                vgate_msa_native = vgate_msa.to(vx_in.dtype)
                vx_out = vx_in + attn1_out * vgate_msa_native  # [B, T, D] in bf16

                vx_in_sliced = _slice_token_prefix(vx_in, token_limit)
                vx_out_sliced = _slice_token_prefix(vx_out, token_limit)
                # vgate_msa is [B, 1, D] — token dim is already 1, no token slice needed.

                tensors["block0_self_attn.vx_in"] = vx_in_sliced
                tensors["block0_self_attn.vgate_msa"] = vgate_msa_native
                tensors["block0_self_attn.vx_out"] = vx_out_sliced

                print(
                    f"Full-residual keys exported: "
                    f"vx_in={list(vx_in_sliced.shape)}, "
                    f"vgate_msa={list(vgate_msa.shape)}, "
                    f"vx_out={list(vx_out_sliced.shape)}"
                )
            except Exception as e:
                print(f"WARNING: could not compute vgate_msa from checkpoint: {e}")
                print("Exporting without full-residual keys.")
        else:
            print(
                "Could not auto-detect video.timesteps from captured block inputs. "
                "Full-residual keys will NOT be exported. "
                f"Expected shape [B, 1, N*D] with N*D={expected_nd}. "
                "Use a checkpoint exported from the same replayed stage-2 model and "
                "capture --include '^velocity_model$' with --capture-inputs."
            )
    elif args.checkpoint_path is not None and vx_in is None:
        print("--checkpoint-path provided but vx_in not available; full-residual keys skipped.")

    # ── Save ────────────────────────────────────────────────────────────────
    metadata = {
        "source_pt": str(args.input_pt),
        "block_index": str(block_idx),
        "step_idx": str(obj.get("step_idx", "")),
        "pass_label": str(obj.get("pass_label", "")),
        "has_pe": str(pe_cos is not None),
        "has_residual_keys": str("block0_self_attn.vx_in" in tensors),
    }

    args.output_st.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output_st), metadata=metadata)

    print(f"\nSaved: {args.output_st}")
    for k, v in sorted(tensors.items()):
        print(f"  {k:45s}  shape={list(v.shape)}  dtype={v.dtype}")

    if pe_cos is None:
        print(
            "\n⚠  pe_cos/pe_sin absent. The Zig attn1 checker will run without RoPE "
            "unless these tensors are present. Re-run replay with --capture-kwargs."
        )
    if "block0_self_attn.vx_out" not in tensors:
        print(
            "\nℹ  Full-residual keys absent. Checker will run in attn1-only mode. "
            "To enable: re-capture block container inputs and pass --checkpoint-path."
        )


if __name__ == "__main__":
    main()
