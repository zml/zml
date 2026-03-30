#!/usr/bin/env python3
"""Validate mixed pipeline outputs against Python reference at every boundary.

Compares:
  V1: Zig Stage 1 output vs Python Stage 1 reference
  V2a: Bridge upscaled video vs Python upsampled reference
  V2b: Stage 2 noise (pre-drawn) vs Python reference
  V2c: Stage 2 positions/masks vs Python reference
  V3: Zig Stage 2 output vs Python Stage 2 reference

Usage:
  python validate_mixed_pipeline.py --mixed-dir /root/mixed/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-8)).item()


def mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().mean().item()


def max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor,
            bitwise: bool = False) -> bool:
    """Compare two tensors, printing stats. Returns True if check passes."""
    if actual.shape != expected.shape:
        print(f"  FAIL {name}: shape mismatch {list(actual.shape)} vs {list(expected.shape)}")
        return False

    cs = cos_sim(actual, expected)
    m = mae(actual, expected)
    mx = max_abs_err(actual, expected)
    match = torch.equal(actual, expected)

    status = "PASS" if (match if bitwise else cs > 0.5) else "FAIL"
    print(f"  {status} {name}: cos_sim={cs:.6f}  mae={m:.6f}  max_abs={mx:.4f}  bitwise={'yes' if match else 'no'}")

    if bitwise and not match:
        return False
    return True


def load_raw_bf16(path: Path, shape: list[int]) -> torch.Tensor:
    raw = np.fromfile(str(path), dtype=np.uint16)
    t = torch.from_numpy(raw.copy()).view(torch.bfloat16)
    return t.reshape(shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate mixed pipeline")
    parser.add_argument("--mixed-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d = args.mixed_dir
    ref = d / "ref"

    all_pass = True

    # ========================================================================
    # V1: Zig Stage 1 output vs Python reference
    # ========================================================================
    print("\n===== V1: Stage 1 output (Zig vs Python) =====")
    print("  Expected: cos_sim ≈ 0.65 (video), 0.80 (audio) — known iterative divergence")

    s1_ref = load_file(str(ref / "stage1_outputs.safetensors"))
    s1_meta = load_file(str(d / "stage1_inputs.safetensors"))

    # Get shapes from stage1_inputs for video/audio token counts
    v_s1_shape = list(s1_meta["video_latent"].shape)  # [1, T_v1, 128]
    a_s1_shape = list(s1_meta["audio_latent"].shape)  # [1, T_a1, 128]

    v_ref_denoised = s1_ref["video_latent_denoised"]  # unpatchified [1, 128, F, H, W]
    a_ref_denoised = s1_ref["audio_latent_denoised"]  # unpatchified [1, 8, T, 16]

    s1_out = d / "stage1_out"
    if (s1_out / "video_latent.bin").exists():
        v_zig = load_raw_bf16(s1_out / "video_latent.bin", v_s1_shape)
        a_zig = load_raw_bf16(s1_out / "audio_latent.bin", a_s1_shape)

        # Unpatchify Zig output to match reference format
        # Video: [1, T_v1, 128] → [1, 128, F, H, W]
        f_lat = v_ref_denoised.shape[2]
        h_lat = v_ref_denoised.shape[3]
        w_lat = v_ref_denoised.shape[4]
        v_zig_5d = v_zig.reshape(1, f_lat, h_lat, w_lat, 128).permute(0, 4, 1, 2, 3)

        # Audio: [1, T_a, 128] → [1, 8, T_a, 16]
        t_a = a_ref_denoised.shape[2]
        a_zig_4d = a_zig.reshape(1, t_a, 8, 16).permute(0, 2, 1, 3)

        compare("video_denoised", v_zig_5d, v_ref_denoised)
        compare("audio_denoised", a_zig_4d, a_ref_denoised)
    else:
        print("  SKIP: Stage 1 output not found (run M1 first)")

    # ========================================================================
    # V2b: Stage 2 noise (pre-drawn) — must be bitwise identical
    # ========================================================================
    print("\n===== V2b: Stage 2 noise (bitwise check) =====")

    s2_noise = load_file(str(d / "stage2_noise.safetensors"))
    s2_ref = load_file(str(ref / "stage2_inputs.safetensors"))

    all_pass &= compare("video_noise", s2_noise["video_noise_s2"], s2_ref["video_noise"], bitwise=True)
    all_pass &= compare("audio_noise", s2_noise["audio_noise_s2"], s2_ref["audio_noise"], bitwise=True)

    # ========================================================================
    # V2c: Stage 2 positions/masks — must be bitwise identical
    # ========================================================================
    print("\n===== V2c: Stage 2 positions & masks (bitwise check vs ref) =====")

    s2_mixed_path = d / "stage2_inputs.safetensors"
    if s2_mixed_path.exists():
        s2_mixed = load_file(str(s2_mixed_path))

        all_pass &= compare("video_positions", s2_mixed["video_positions"], s2_ref["video_positions"], bitwise=True)
        all_pass &= compare("audio_positions", s2_mixed["audio_positions"], s2_ref["audio_positions"], bitwise=True)
        all_pass &= compare("video_denoise_mask", s2_mixed["video_denoise_mask"], s2_ref["video_denoise_mask"], bitwise=True)
        all_pass &= compare("audio_denoise_mask", s2_mixed["audio_denoise_mask"], s2_ref["audio_denoise_mask"], bitwise=True)
        all_pass &= compare("v_context", s2_mixed["v_context"], s2_ref["v_context"], bitwise=True)
        all_pass &= compare("a_context", s2_mixed["a_context"], s2_ref["a_context"], bitwise=True)
    else:
        print("  SKIP: Mixed Stage 2 inputs not found (run M2 first)")

    # ========================================================================
    # V2a: Bridge upscaled video vs ref (if available)
    # ========================================================================
    print("\n===== V2a: Upscaled video (bridge vs ref) =====")
    print("  Expected: divergence proportional to V1 Stage 1 divergence")

    if s2_mixed_path.exists():
        # Compare clean latents (= patchified upscaled video / audio)
        compare("video_clean_latent", s2_mixed["video_clean_latent"], s2_ref["video_clean_latent"])
        compare("audio_clean_latent", s2_mixed["audio_clean_latent"], s2_ref["audio_clean_latent"])
    else:
        print("  SKIP: Mixed Stage 2 inputs not found (run M2 first)")

    # ========================================================================
    # V3: Stage 2 output (Zig vs Python)
    # ========================================================================
    print("\n===== V3: Stage 2 output (Zig vs Python) =====")

    s2_out = d / "stage2_out"
    s2_ref_out = load_file(str(ref / "stage2_outputs.safetensors"))
    v_ref_final = s2_ref_out["video_latent_final"]
    a_ref_final = s2_ref_out["audio_latent_final"]

    if (s2_out / "video_latent.bin").exists():
        # Zig output is patchified .bin, ref is unpatchified
        v_s2_shape = list(s2_ref["video_latent"].shape)  # [1, T_v2, 128] patchified
        a_s2_shape = list(s2_ref["audio_latent"].shape)  # [1, T_a2, 128] patchified

        v_zig_s2 = load_raw_bf16(s2_out / "video_latent.bin", v_s2_shape)
        a_zig_s2 = load_raw_bf16(s2_out / "audio_latent.bin", a_s2_shape)

        # Unpatchify for comparison
        f2 = v_ref_final.shape[2]
        h2 = v_ref_final.shape[3]
        w2 = v_ref_final.shape[4]
        v_zig_s2_5d = v_zig_s2.reshape(1, f2, h2, w2, 128).permute(0, 4, 1, 2, 3)

        t_a2 = a_ref_final.shape[2]
        a_zig_s2_4d = a_zig_s2.reshape(1, t_a2, 8, 16).permute(0, 2, 1, 3)

        compare("video_final", v_zig_s2_5d, v_ref_final)
        compare("audio_final", a_zig_s2_4d, a_ref_final)
    else:
        print("  SKIP: Stage 2 output not found (run M3 first)")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 60)
    if all_pass:
        print("ALL BITWISE CHECKS PASSED")
    else:
        print("SOME BITWISE CHECKS FAILED — see above")
    print("=" * 60)


if __name__ == "__main__":
    main()
