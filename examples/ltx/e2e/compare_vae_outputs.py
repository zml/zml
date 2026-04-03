"""Compare Zig VAE decoder output against Python reference activations.

Loads the Python-exported vae_activations.safetensors and the Zig-produced
decoded_video.bin, computes per-element error metrics, and reports PSNR.

Usage (on GPU server):
  cd /root/repos/LTX-2
  uv run examples/ltx/e2e/compare_vae_outputs.py \
      --ref /root/e2e_demo/vae_ref/vae_activations.safetensors \
      --zig /root/e2e_demo/vae_zig_out/decoded_video.bin
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare VAE outputs")
    parser.add_argument("--ref", type=Path, required=True,
                        help="Path to vae_activations.safetensors (Python reference)")
    parser.add_argument("--zig", type=Path, required=True,
                        help="Path to decoded_video.bin (Zig output, raw bf16)")
    return parser.parse_args()


def compute_metrics(ref: torch.Tensor, zig: torch.Tensor, name: str) -> dict:
    """Compute comparison metrics between reference and Zig tensors."""
    ref_f32 = ref.float()
    zig_f32 = zig.float()

    abs_diff = (ref_f32 - zig_f32).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    # Relative error (avoid div by zero)
    ref_abs = ref_f32.abs()
    rel_diff = abs_diff / (ref_abs + 1e-8)
    max_rel = rel_diff.max().item()
    mean_rel = rel_diff.mean().item()

    # PSNR (on the [-1, 1] range signal)
    mse = (abs_diff ** 2).mean().item()
    if mse > 0:
        # Signal range is [-1, 1], so peak = 2.0
        psnr = 10 * np.log10(4.0 / mse)  # 2^2 / MSE
    else:
        psnr = float('inf')

    # Also compute PSNR on [0, 255] scale for u8 comparison
    ref_u8 = ((ref_f32 + 1.0) / 2.0).clamp(0, 1) * 255.0
    zig_u8 = ((zig_f32 + 1.0) / 2.0).clamp(0, 1) * 255.0
    mse_u8 = ((ref_u8 - zig_u8) ** 2).mean().item()
    psnr_u8 = 10 * np.log10(255.0**2 / mse_u8) if mse_u8 > 0 else float('inf')

    print(f"  {name}:")
    print(f"    shape:     {list(ref.shape)}")
    print(f"    max_abs:   {max_abs:.6f}")
    print(f"    mean_abs:  {mean_abs:.6f}")
    print(f"    max_rel:   {max_rel:.6f}")
    print(f"    mean_rel:  {mean_rel:.6f}")
    print(f"    PSNR(±1):  {psnr:.1f} dB")
    print(f"    PSNR(u8):  {psnr_u8:.1f} dB")

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "psnr": psnr,
        "psnr_u8": psnr_u8,
    }


def main() -> None:
    args = parse_args()

    # Load Python reference
    print("Loading Python reference...")
    with safe_open(str(args.ref), framework="pt") as f:
        ref_output = f.get_tensor("output")  # [1, 3, F, H, W] bf16

    print(f"  ref output: {list(ref_output.shape)} {ref_output.dtype}")
    F_out = ref_output.shape[2]
    H_out = ref_output.shape[3]
    W_out = ref_output.shape[4]

    # Load Zig output
    print("Loading Zig output...")
    zig_bytes = args.zig.read_bytes()
    expected_size = ref_output.numel() * 2  # bf16 = 2 bytes per element
    if len(zig_bytes) != expected_size:
        print(f"  ERROR: Zig output size {len(zig_bytes)} != expected {expected_size}")
        print(f"  (ref numel={ref_output.numel()}, expected bf16 bytes={expected_size})")
        sys.exit(1)

    zig_output = torch.frombuffer(bytearray(zig_bytes), dtype=torch.bfloat16)
    zig_output = zig_output.reshape(ref_output.shape)
    print(f"  zig output: {list(zig_output.shape)} {zig_output.dtype}")

    # Compare
    print("\n=== Output Comparison ===")
    metrics = compute_metrics(ref_output, zig_output, "decoded_video")

    # Summary
    print("\n=== Summary ===")
    passed = metrics["psnr_u8"] > 40.0
    status = "PASS" if passed else "FAIL"
    print(f"  PSNR (u8 scale): {metrics['psnr_u8']:.1f} dB  (threshold: 40 dB)  [{status}]")
    print(f"  Max abs error:   {metrics['max_abs']:.6f}")

    if not passed:
        print("\n  WARNING: PSNR below threshold. Possible causes:")
        print("    - Reflect padding vs zero padding mismatch")
        print("    - bf16 precision accumulation in deep network")
        print("    - PixelNorm epsilon or precision difference")
        sys.exit(1)
    else:
        print("\n  VAE decode parity validated!")


if __name__ == "__main__":
    main()
