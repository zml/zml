"""Re-trace stage-2 blocks 0-7 with bf16 matmul accumulation enabled.

Avoids loading the full 22B pipeline. Instead:
  1. Reads all input tensors from an existing fixture safetensors file.
  2. Constructs only 8 BasicAVTransformerBlocks with known architecture params.
  3. Loads weights from the already-exported 8-block merged checkpoint.
  4. Runs the forward pass with:
         torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
     which makes PyTorch use bf16 accumulation for bf16 matmuls, matching XLA's
     dot_precision=.fast used by the ZML implementation.
  5. Saves a new fixture in the same format as export_block_slice_native_fixture.py.

Usage:
    uv run python examples/ltx/retrace_block_slice_bf16.py \\
        --input-fixture  /root/repos/LTX-2/trace_run/block_slice_native_0_7_lora0.0_t128_maskrefresh10.safetensors \\
        --checkpoint     /root/repos/LTX-2/trace_run/stage2_blocks_0_7_lora0.0_merged.safetensors \\
        --output-fixture /root/repos/LTX-2/trace_run/block_slice_native_0_7_lora0.0_t128_maskrefresh11.safetensors
"""

# ── Precision flag: must be set BEFORE any CUDA ops ──────────────────────────
import torch
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
# ─────────────────────────────────────────────────────────────────────────────

import argparse
from pathlib import Path

from safetensors.torch import load_file, save_file

from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.attention import AttentionFunction


def _t(d: dict, key: str) -> torch.Tensor | None:
    return d.get(key)


def _req(d: dict, key: str) -> torch.Tensor:
    v = d.get(key)
    if v is None:
        raise KeyError(f"Required key missing from fixture: {key}")
    return v


def _req_to_dev(d: dict, key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _req(d, key).to(device=device, dtype=dtype)


def _prepare_pe_pair(cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return cos, sin


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-trace blocks 0-7 with bf16 matmul accumulation")
    p.add_argument("--input-fixture", type=Path, required=True, help="Existing maskrefreshN.safetensors fixture")
    p.add_argument("--checkpoint", type=Path, required=True, help="stage2_blocks_0_7_lora0.0_merged.safetensors")
    p.add_argument("--output-fixture", type=Path, required=True, help="Output fixture path (maskrefresh11)")
    p.add_argument("--n-blocks", type=int, default=8, help="Number of blocks in the checkpoint (default 8)")
    return p.parse_args()


# ── LTX 2.3 stage-2 transformer block architecture (22B AV model) ─────────────
# These match the config returned by LTXModelConfigurator.from_config for
# ltx-2.3-22b-distilled.safetensors with cross_attention_adaln=True.
_VIDEO_CFG = TransformerConfig(
    dim=4096,
    heads=32,
    d_head=128,
    context_dim=4096,
    apply_gated_attention=True,
    cross_attention_adaln=True,
)
_AUDIO_CFG = TransformerConfig(
    dim=2048,
    heads=32,
    d_head=64,
    context_dim=2048,
    apply_gated_attention=True,
    cross_attention_adaln=True,
)
_NORM_EPS = 1e-6
_ROPE_TYPE = LTXRopeType.SPLIT


def build_blocks(n: int) -> torch.nn.ModuleList:
    """Construct n uninitialised BasicAVTransformerBlocks."""
    blocks = torch.nn.ModuleList([
        BasicAVTransformerBlock(
            idx=i,
            video=_VIDEO_CFG,
            audio=_AUDIO_CFG,
            rope_type=_ROPE_TYPE,
            norm_eps=_NORM_EPS,
            attention_function=AttentionFunction.DEFAULT,
        )
        for i in range(n)
    ])
    return blocks


def load_blocks(checkpoint: Path, n_blocks: int) -> torch.nn.ModuleList:
    """Build architecture, load weights from the 8-block merged checkpoint."""
    print(f"Building {n_blocks} BasicAVTransformerBlock instances...")
    blocks = build_blocks(n_blocks)

    print(f"Loading weights from: {checkpoint}")
    merged_sd = load_file(str(checkpoint), device="cpu")

    # Merged checkpoint keys are: velocity_model.transformer_blocks.{local_idx}.{suffix}
    # ModuleList keys are: {local_idx}.{suffix}
    block_prefix = "velocity_model.transformer_blocks."
    block_sd = {
        k[len(block_prefix):]: v
        for k, v in merged_sd.items()
        if k.startswith(block_prefix)
    }
    missing, unexpected = blocks.load_state_dict(block_sd, strict=False)
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected[:5]}")
    if missing:
        print(f"  WARNING: missing keys: {missing[:5]}")
    print(f"  Loaded {len(block_sd)} tensors.")
    return blocks


class FixturePerturbations:
    def __init__(self, a2v_mask: torch.Tensor | None, v2a_mask: torch.Tensor | None):
        self.a2v_mask = a2v_mask
        self.v2a_mask = v2a_mask

    def mask_like(self, perturbation_type, block: int, values: torch.Tensor) -> torch.Tensor:
        from ltx_core.guidance.perturbations import PerturbationType

        if perturbation_type == PerturbationType.SKIP_A2V_CROSS_ATTN and self.a2v_mask is not None:
            return self.a2v_mask
        if perturbation_type == PerturbationType.SKIP_V2A_CROSS_ATTN and self.v2a_mask is not None:
            return self.v2a_mask

        return torch.ones(
            (values.shape[0],) + (1,) * (values.dim() - 1),
            device=values.device,
            dtype=values.dtype,
        )

    def any_in_batch(self, perturbation_type, block: int) -> bool:
        return False

    def all_in_batch(self, perturbation_type, block: int) -> bool:
        return False


def main() -> None:
    args = parse_args()
    n_blocks = args.n_blocks

    print(f"Loading input fixture: {args.input_fixture}")
    fix = load_file(str(args.input_fixture))
    print(f"  Keys in fixture: {len(fix)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    def to_dev(t: torch.Tensor | None) -> torch.Tensor | None:
        return t.to(device=device, dtype=dtype) if t is not None else None

    # ── Load conditioning tensors (shared across all blocks) ─────────────────
    vx_in        = _req_to_dev(fix, "block_slice_native.vx_in", device, dtype)
    ax_in        = _req_to_dev(fix, "block_slice_native.ax_in", device, dtype)
    video_ts     = _req_to_dev(fix, "block_slice_native.video_timesteps", device, dtype)
    audio_ts     = _req_to_dev(fix, "block_slice_native.audio_timesteps", device, dtype)
    v_prompt_ts  = _req_to_dev(fix, "block_slice_native.v_prompt_timestep", device, dtype)
    a_prompt_ts  = _req_to_dev(fix, "block_slice_native.a_prompt_timestep", device, dtype)
    v_pe_cos     = _req_to_dev(fix, "block_slice_native.v_pe_cos", device, dtype)
    v_pe_sin     = _req_to_dev(fix, "block_slice_native.v_pe_sin", device, dtype)
    a_pe_cos     = _req_to_dev(fix, "block_slice_native.a_pe_cos", device, dtype)
    a_pe_sin     = _req_to_dev(fix, "block_slice_native.a_pe_sin", device, dtype)
    v_text_ctx   = _req_to_dev(fix, "block_slice_native.v_text_ctx", device, dtype)
    a_text_ctx   = _req_to_dev(fix, "block_slice_native.a_text_ctx", device, dtype)
    v_text_mask  = to_dev(_t(fix, "block_slice_native.v_text_ctx_mask"))
    a_text_mask  = to_dev(_t(fix, "block_slice_native.a_text_ctx_mask"))
    v_cross_ss   = _req_to_dev(fix, "block_slice_native.v_cross_ss_ts", device, dtype)
    v_cross_gate = _req_to_dev(fix, "block_slice_native.v_cross_gate_ts", device, dtype)
    a_cross_ss   = _req_to_dev(fix, "block_slice_native.a_cross_ss_ts", device, dtype)
    a_cross_gate = _req_to_dev(fix, "block_slice_native.a_cross_gate_ts", device, dtype)
    a2v_pe_cos   = _req_to_dev(fix, "block_slice_native.a2v_pe_cos", device, dtype)
    a2v_pe_sin   = _req_to_dev(fix, "block_slice_native.a2v_pe_sin", device, dtype)
    a2v_k_pe_cos = _req_to_dev(fix, "block_slice_native.a2v_k_pe_cos", device, dtype)
    a2v_k_pe_sin = _req_to_dev(fix, "block_slice_native.a2v_k_pe_sin", device, dtype)
    v2a_pe_cos   = _req_to_dev(fix, "block_slice_native.v2a_pe_cos", device, dtype)
    v2a_pe_sin   = _req_to_dev(fix, "block_slice_native.v2a_pe_sin", device, dtype)
    v2a_k_pe_cos = _req_to_dev(fix, "block_slice_native.v2a_k_pe_cos", device, dtype)
    v2a_k_pe_sin = _req_to_dev(fix, "block_slice_native.v2a_k_pe_sin", device, dtype)

    # Per-block AV masks (may be None if not present in fixture)
    a2v_masks = [to_dev(_t(fix, f"block_slice_native.a2v_mask_block_{i}")) for i in range(n_blocks)]
    v2a_masks = [to_dev(_t(fix, f"block_slice_native.v2a_mask_block_{i}")) for i in range(n_blocks)]

    # ── Positional embedding tuples expected by LTX Attention ────────────────
    v_pe     = _prepare_pe_pair(v_pe_cos, v_pe_sin)
    a_pe     = _prepare_pe_pair(a_pe_cos, a_pe_sin)
    a2v_pe   = _prepare_pe_pair(a2v_pe_cos, a2v_pe_sin)
    a2v_k_pe = _prepare_pe_pair(a2v_k_pe_cos, a2v_k_pe_sin)
    v2a_pe   = _prepare_pe_pair(v2a_pe_cos, v2a_pe_sin)
    v2a_k_pe = _prepare_pe_pair(v2a_k_pe_cos, v2a_k_pe_sin)

    # ── Load model blocks (no full pipeline) ─────────────────────────────────
    blocks = load_blocks(args.checkpoint, n_blocks)
    for block in blocks:
        block.to(device=device, dtype=dtype)
        block.eval()

    print(f"\nRunning {n_blocks} blocks with bf16 matmul accumulation on {device}...")
    tensors_out: dict[str, torch.Tensor] = {}

    # Copy all original fixture keys (inputs, masks, conditioning) verbatim.
    # We only replace the per-block *output* tensors below.
    for k, v in fix.items():
        tensors_out[k] = v.cpu()

    vx = vx_in.clone()
    ax = ax_in.clone()

    # Per-block AV cross-attention mask sizes may differ per block (stored separately).
    with torch.inference_mode():
        for local_idx in range(n_blocks):
            block = blocks[local_idx]
            a2v_mask_i = a2v_masks[local_idx]
            v2a_mask_i = v2a_masks[local_idx]

            perturbations = FixturePerturbations(a2v_mask_i, v2a_mask_i)

            # Build TransformerArgs for this block
            video_args = TransformerArgs(
                x=vx,
                context=v_text_ctx,
                context_mask=v_text_mask,
                timesteps=video_ts,
                embedded_timestep=video_ts,   # not used in forward
                positional_embeddings=v_pe,
                cross_positional_embeddings=a2v_pe,
                cross_scale_shift_timestep=v_cross_ss,
                cross_gate_timestep=v_cross_gate,
                enabled=True,
                prompt_timestep=v_prompt_ts,
                self_attention_mask=None,
            )
            audio_args = TransformerArgs(
                x=ax,
                context=a_text_ctx,
                context_mask=a_text_mask,
                timesteps=audio_ts,
                embedded_timestep=audio_ts,   # not used in forward
                positional_embeddings=a_pe,
                cross_positional_embeddings=a2v_k_pe,
                cross_scale_shift_timestep=a_cross_ss,
                cross_gate_timestep=a_cross_gate,
                enabled=True,
                prompt_timestep=a_prompt_ts,
                self_attention_mask=None,
            )

            video_out, audio_out = block(
                video=video_args,
                audio=audio_args,
                perturbations=perturbations,
            )

            vx_out = video_out.x.detach().cpu()
            ax_out = audio_out.x.detach().cpu()

            # Overwrite vx/ax for next block
            vx = video_out.x.detach().clone()
            ax = audio_out.x.detach().clone()

            tensors_out[f"block_slice_native.vx_out_block_{local_idx}"] = vx_out
            tensors_out[f"block_slice_native.ax_out_block_{local_idx}"] = ax_out
            if local_idx == n_blocks - 1:
                tensors_out["block_slice_native.vx_out"] = vx_out.clone()
                tensors_out["block_slice_native.ax_out"] = ax_out.clone()

            print(f"  Block {local_idx}: vx_out={tuple(vx_out.shape)}  ax_out={tuple(ax_out.shape)}")

    print(f"\nSaving new fixture: {args.output_fixture}")
    args.output_fixture.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors_out, str(args.output_fixture))
    print("Done.")


if __name__ == "__main__":
    main()
