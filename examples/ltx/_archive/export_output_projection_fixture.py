"""Export output_projection parity fixture.

Loads the full model checkpoint and runs the OutputProjection (norm_out + scale_shift +
proj_out) module on known hidden states x and embedded_timestep values, saving inputs
and outputs to a fixture.

The fixture is compared against the Zig OutputProjection implementation in model.zig.

Saved keys:
  video.x                   — bf16 [B, T, D_v=4096]  — random hidden states
  video.embedded_timestep   — bf16 [B, D_v=4096]      — from adaln_single
  video.output              — bf16 [B, T, D_out=128]  — Python reference output
  audio.x                   — bf16 [B, T, D_a=2048]
  audio.embedded_timestep   — bf16 [B, D_a=2048]
  audio.output              — bf16 [B, T, D_out=128]

Usage:
  uv run python export_output_projection_fixture.py \\
      /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \\
      /root/repos/LTX-2/trace_run/output_projection_fixture.safetensors
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


BATCH_SIZE = 4
SEQ_LEN = 16   # small: enough to exercise T-dim broadcasting


def run_output_projection(
    scale_shift_table: torch.Tensor,   # [2, D] f32
    proj_out_weight: torch.Tensor,      # [128, D] bf16
    proj_out_bias: torch.Tensor,        # [128] bf16
    x: torch.Tensor,                    # [B, T, D] bf16
    embedded_timestep: torch.Tensor,    # [B, D] bf16
) -> torch.Tensor:
    """Python reference for velocity_model._process_output."""
    D = x.shape[-1]
    # Python: embedded_timestep[:, :, None] with embedded_timestep being [B, 1, D]
    # Our embedded_timestep is [B, D]; add seq dim → [B, 1, D], then add None → [B, 1, 1, D]
    emb = embedded_timestep.unsqueeze(1)  # [B, 1, D]

    sst = scale_shift_table[None, None].to(dtype=x.dtype)  # [1, 1, 2, D]
    scale_shift_values = sst + emb[:, :, None]              # [B, 1, 2, D]
    shift = scale_shift_values[:, :, 0]  # [B, 1, D]
    scale = scale_shift_values[:, :, 1]  # [B, 1, D]

    # LayerNorm(eps=1e-6, elementwise_affine=False)
    out = F.layer_norm(x, (D,), eps=1e-6)
    out = out * (1 + scale) + shift
    # Linear
    out = F.linear(out, proj_out_weight, proj_out_bias)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = load_file(str(args.checkpoint))

    prefix = "model.diffusion_model."

    # --- Video ---
    v_sst    = ckpt[prefix + "scale_shift_table"]               # [2, 4096] f32
    v_w      = ckpt[prefix + "proj_out.weight"]                 # [128, 4096] bf16
    v_b      = ckpt[prefix + "proj_out.bias"]                   # [128] bf16
    D_v = v_sst.shape[1]

    torch.manual_seed(42)
    v_x   = torch.randn(BATCH_SIZE, SEQ_LEN, D_v, dtype=torch.bfloat16)
    v_emb = torch.randn(BATCH_SIZE, D_v, dtype=torch.bfloat16)

    v_out = run_output_projection(v_sst, v_w, v_b, v_x, v_emb)

    print(f"Video  output: {v_out.shape}  mean={v_out.float().mean():.4f}  std={v_out.float().std():.4f}")

    # --- Audio ---
    a_sst  = ckpt[prefix + "audio_scale_shift_table"]           # [2, 2048] f32
    a_w    = ckpt[prefix + "audio_proj_out.weight"]             # [128, 2048] bf16
    a_b    = ckpt[prefix + "audio_proj_out.bias"]               # [128] bf16
    D_a = a_sst.shape[1]

    torch.manual_seed(99)
    a_x   = torch.randn(BATCH_SIZE, SEQ_LEN, D_a, dtype=torch.bfloat16)
    a_emb = torch.randn(BATCH_SIZE, D_a, dtype=torch.bfloat16)

    a_out = run_output_projection(a_sst, a_w, a_b, a_x, a_emb)

    print(f"Audio  output: {a_out.shape}  mean={a_out.float().mean():.4f}  std={a_out.float().std():.4f}")

    tensors = {
        "video.x":                 v_x.contiguous(),
        "video.embedded_timestep": v_emb.contiguous(),
        "video.output":            v_out.contiguous(),
        "audio.x":                 a_x.contiguous(),
        "audio.embedded_timestep": a_emb.contiguous(),
        "audio.output":            a_out.contiguous(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output))
    print(f"Done. {len(tensors)} tensors saved to {args.output}")


if __name__ == "__main__":
    main()
