"""Export adaln_single parity fixture.

Loads the full model checkpoint and runs each AdaLayerNormSingle module on
a known sigma value, saving inputs and outputs to a safetensors fixture.

The fixture can then be compared against the Zig implementation of
AdaLayerNormSingle in model.zig.

Saved keys (for each module prefix P in {adaln_single, audio_adaln_single,
prompt_adaln_single, audio_prompt_adaln_single,
av_ca_video_scale_shift_adaln_single, av_ca_audio_scale_shift_adaln_single,
av_ca_a2v_gate_adaln_single, av_ca_v2a_gate_adaln_single}):

  P.sigma_scaled   — f32 [B] — input to AdaLayerNormSingle.forward
  P.modulation     — bf16 [B, N*D] — returned modulation coefficients
  P.embedded_timestep — bf16 [B, D] — returned embedded timestep

Usage:
  uv run python export_adaln_single_fixture.py \\
      /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \\
      /root/repos/LTX-2/trace_run/adaln_single_fixture.safetensors
"""

import argparse
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ADALN_MODULES = [
    # (fixture_prefix, ckpt_prefix_suffix, D, N_coeff)
    ("adaln_single",                            "adaln_single",                            4096, 9),
    ("audio_adaln_single",                      "audio_adaln_single",                      2048, 9),
    ("prompt_adaln_single",                     "prompt_adaln_single",                     4096, 2),
    ("audio_prompt_adaln_single",               "audio_prompt_adaln_single",               2048, 2),
    ("av_ca_video_scale_shift_adaln_single",    "av_ca_video_scale_shift_adaln_single",    4096, 4),
    ("av_ca_audio_scale_shift_adaln_single",    "av_ca_audio_scale_shift_adaln_single",    2048, 4),
    ("av_ca_a2v_gate_adaln_single",             "av_ca_a2v_gate_adaln_single",             4096, 1),
    ("av_ca_v2a_gate_adaln_single",             "av_ca_v2a_gate_adaln_single",             2048, 1),
]

# Sigma values to test (in [0, 1] range, will be scaled by 1000 before feeding into adaln)
SIGMA_VALUES = [0.0, 0.128, 0.5, 1.0]
BATCH_SIZE = len(SIGMA_VALUES)


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int = 256) -> torch.Tensor:
    """Sinusoidal timestep embedding matching Python get_timestep_embedding
    with flip_sin_to_cos=True, downscale_freq_shift=0.
    """
    assert timesteps.ndim == 1
    half_dim = embedding_dim // 2
    exponent = -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32)
    exponent = exponent / (half_dim - 0)  # downscale_freq_shift=0

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]  # [B, 128]

    # cat([sin, cos]) then flip → cat([cos, sin])
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, 256] but sin first
    # flip_sin_to_cos=True → swap halves
    emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)  # [B, 256]: cos first
    return emb


def run_adaln_single(
    weights: dict,
    ckpt_prefix: str,
    sigma_scaled: torch.Tensor,
    hidden_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run AdaLayerNormSingle forward pass manually.
    Returns (modulation [B, N*D], embedded_timestep [B, D]).
    """
    def W(name):
        return weights[f"{ckpt_prefix}.{name}"]

    # 1. Sinusoidal embedding [B, 256] in f32, then cast to hidden_dtype
    t_proj = get_timestep_embedding(sigma_scaled).to(hidden_dtype)

    # 2. TimestepEmbedding: linear_1 → SiLU → linear_2
    h = t_proj @ W("emb.timestep_embedder.linear_1.weight").T
    h = h + W("emb.timestep_embedder.linear_1.bias")
    h = torch.nn.functional.silu(h)
    h = h @ W("emb.timestep_embedder.linear_2.weight").T
    h = h + W("emb.timestep_embedder.linear_2.bias")
    embedded_timestep = h  # [B, D]

    # 3. AdaLN-single: SiLU → linear_out
    h = torch.nn.functional.silu(h)
    h = h @ W("linear.weight").T
    h = h + W("linear.bias")
    modulation = h  # [B, N*D]

    return modulation, embedded_timestep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export adaln_single parity fixture")
    p.add_argument("checkpoint", type=Path, help="Full model safetensors checkpoint")
    p.add_argument("output", type=Path, help="Output fixture safetensors path")
    p.add_argument("--model-prefix", default="model.diffusion_model",
                   help="Prefix to strip from checkpoint keys")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading checkpoint: {args.checkpoint}")
    raw_weights = load_file(str(args.checkpoint), device="cpu")

    # Strip model prefix to match selectTransformerRoot behavior
    prefix = args.model_prefix + "."
    weights = {k[len(prefix):]: v for k, v in raw_weights.items() if k.startswith(prefix)}
    print(f"Stripped prefix '{prefix}': {len(weights)} keys remain")

    hidden_dtype = torch.bfloat16
    sigma_values = torch.tensor(SIGMA_VALUES, dtype=torch.float32)
    # Scale by 1000 (timestep_scale_multiplier default)
    sigma_scaled = sigma_values * 1000.0

    tensors = {}

    for fix_prefix, ckpt_suffix, D, N in ADALN_MODULES:
        print(f"  Running {fix_prefix} (D={D}, N={N})...")
        modulation, embedded_timestep = run_adaln_single(
            weights, ckpt_suffix, sigma_scaled, hidden_dtype
        )
        assert modulation.shape == (BATCH_SIZE, N * D), \
            f"{fix_prefix}: expected modulation shape ({BATCH_SIZE}, {N*D}), got {modulation.shape}"
        assert embedded_timestep.shape == (BATCH_SIZE, D), \
            f"{fix_prefix}: expected embedded_timestep shape ({BATCH_SIZE}, {D}), got {embedded_timestep.shape}"

        tensors[f"{fix_prefix}.sigma_scaled"] = sigma_scaled.clone().contiguous()
        tensors[f"{fix_prefix}.modulation"] = modulation.contiguous()
        tensors[f"{fix_prefix}.embedded_timestep"] = embedded_timestep.contiguous()

        modulation_f32 = modulation.float()
        print(f"    sigma_scaled: {sigma_scaled.tolist()}")
        print(f"    modulation mean={modulation_f32.mean().item():.4f}  "
              f"std={modulation_f32.std().item():.4f}  "
              f"range=[{modulation_f32.min().item():.4f}, {modulation_f32.max().item():.4f}]")

    print(f"\nSaving fixture: {args.output}")
    save_file(tensors, str(args.output))
    print(f"Done. {len(tensors)} tensors saved.")


if __name__ == "__main__":
    main()
