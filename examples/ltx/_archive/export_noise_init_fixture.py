"""Export noise init fixture from existing trace_run data.

Extracts the noise initialization components from the Stage 2 trace,
back-computes the noise tensor from the formula, and exports as safetensors
for Zig validation.

Noise init formula:
  noised = noise * mask * sigma_0 + clean * (1 - mask * sigma_0)

Requires: trace_run/11_stage2_steps.pt, trace_run/07_stage2_sigmas.pt

Usage (on GPU server):
  cd /root/repos/LTX-2
  python scripts/export_noise_init_fixture.py \
      --output /root/fixtures/noise_init_fixture.safetensors
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export noise init fixture from trace data")
    parser.add_argument("--trace-dir", type=Path, default=Path("trace_run"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load trace data
    steps = torch.load(args.trace_dir / "11_stage2_steps.pt", map_location="cpu", weights_only=False)
    sigmas = torch.load(args.trace_dir / "07_stage2_sigmas.pt", map_location="cpu", weights_only=False)

    sigma_0 = sigmas[0].item()
    print(f"sigma_0 = {sigma_0}")

    # Step 0 contains the initial (noised) state passed to the first denoise call
    step0 = steps[0]

    # Extract patchified tensors from step 0
    v_noised = step0["video_latent"]       # bf16 [B, T_v, 128]
    v_clean  = step0["video_clean_latent"] # bf16 [B, T_v, 128]
    v_mask   = step0["video_denoise_mask"] # f32  [B, T_v, 1]
    a_noised = step0["audio_latent"]       # bf16 [B, T_a, 128]
    a_clean  = step0["audio_clean_latent"] # bf16 [B, T_a, 128]
    a_mask   = step0["audio_denoise_mask"] # f32  [B, T_a, 1]

    print(f"video: noised={list(v_noised.shape)} {v_noised.dtype}, "
          f"clean={list(v_clean.shape)} {v_clean.dtype}, "
          f"mask={list(v_mask.shape)} {v_mask.dtype}")
    print(f"audio: noised={list(a_noised.shape)} {a_noised.dtype}, "
          f"clean={list(a_clean.shape)} {a_clean.dtype}, "
          f"mask={list(a_mask.shape)} {a_mask.dtype}")

    # Back-compute noise from formula:
    #   noised = noise * mask * sigma_0 + clean * (1 - mask * sigma_0)
    #   noise = (noised - clean * (1 - mask * sigma_0)) / (mask * sigma_0)
    # Where mask == 0, noise is irrelevant (multiplied by 0), set to 0.

    def recover_noise(noised, clean, mask, sigma_0):
        """Recover noise tensor from the noise init formula."""
        noised_f32 = noised.float()
        clean_f32 = clean.float()
        mask_f32 = mask.float()

        mask_sigma = mask_f32 * sigma_0
        one_minus_mask_sigma = 1.0 - mask_sigma
        clean_term = clean_f32 * one_minus_mask_sigma

        # Where mask > 0, solve for noise
        noise_f32 = torch.zeros_like(noised_f32)
        nonzero = mask_sigma.squeeze(-1) > 0
        if nonzero.any():
            noise_f32[nonzero] = (noised_f32[nonzero] - clean_term[nonzero]) / mask_sigma[nonzero]

        return noise_f32.to(noised.dtype)

    v_noise = recover_noise(v_noised, v_clean, v_mask, sigma_0)
    a_noise = recover_noise(a_noised, a_clean, a_mask, sigma_0)

    # Verify round-trip (should be very close)
    def verify_roundtrip(noise, clean, mask, sigma_0, expected_noised, label):
        mask_sigma = mask.float() * sigma_0
        recomputed = (noise.float() * mask_sigma + clean.float() * (1.0 - mask_sigma)).to(expected_noised.dtype)
        diff = (recomputed.float() - expected_noised.float()).abs()
        print(f"  {label}: max_diff={diff.max().item():.6e}, "
              f"mean_diff={diff.mean().item():.6e}, "
              f"exact_match={torch.equal(recomputed, expected_noised)}")

    print("Round-trip verification:")
    verify_roundtrip(v_noise, v_clean, v_mask, sigma_0, v_noised, "video")
    verify_roundtrip(a_noise, a_clean, a_mask, sigma_0, a_noised, "audio")

    # Export as safetensors
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tensors = {
        "video_clean":    v_clean.contiguous(),
        "audio_clean":    a_clean.contiguous(),
        "video_noise":    v_noise.contiguous(),
        "audio_noise":    a_noise.contiguous(),
        "video_mask":     v_mask.contiguous(),
        "audio_mask":     a_mask.contiguous(),
        "video_expected": v_noised.contiguous(),
        "audio_expected": a_noised.contiguous(),
    }

    metadata = {
        "sigma_0": str(sigma_0),
        "description": "Noise init fixture: noised = noise * mask * sigma_0 + clean * (1 - mask * sigma_0)",
    }

    save_file(tensors, str(args.output), metadata=metadata)
    print(f"\nSaved fixture to {args.output}")
    for k, v in tensors.items():
        print(f"  {k}: {list(v.shape)} {v.dtype}")


if __name__ == "__main__":
    main()
