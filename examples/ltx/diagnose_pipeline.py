#!/usr/bin/env python3
"""Diagnose garbled video output from the Zig inference pipeline.

Compares Zig intermediate dumps against Python reference tensors.

Usage (on GPU server):
  uv run examples/ltx/diagnose_pipeline.py \
      --zig-dir /root/imgcond_out \
      --ref-dir /root/imgcond_ref
"""
import argparse
import os
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def load_bf16_bin(path: str, shape: list[int]) -> torch.Tensor:
    """Load raw bf16 binary file into a torch tensor."""
    data = Path(path).read_bytes()
    expected = int(np.prod(shape)) * 2
    assert len(data) == expected, f"Expected {expected} bytes, got {len(data)} for shape {shape}"
    arr = np.frombuffer(data, dtype=np.uint16).reshape(shape)
    # Convert bf16 bits to float32
    f32 = np.zeros_like(arr, dtype=np.float32)
    f32_view = f32.view(np.uint32)
    f32_view[:] = arr.astype(np.uint32) << 16
    return torch.from_numpy(f32).to(torch.bfloat16)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm())).item()


def compare_tensors(name: str, zig: torch.Tensor, ref: torch.Tensor):
    print(f"\n  {name}:")
    print(f"    Zig shape: {list(zig.shape)}, dtype: {zig.dtype}")
    print(f"    Ref shape: {list(ref.shape)}, dtype: {ref.dtype}")
    print(f"    Zig range: [{zig.float().min():.4f}, {zig.float().max():.4f}]")
    print(f"    Ref range: [{ref.float().min():.4f}, {ref.float().max():.4f}]")

    if zig.shape != ref.shape:
        print(f"    *** SHAPE MISMATCH ***")
        return

    cos = cosine_sim(zig, ref)
    mae = (zig.float() - ref.float()).abs().mean().item()
    print(f"    Cosine similarity: {cos:.6f}")
    print(f"    MAE: {mae:.6f}")

    if cos < 0.99:
        print(f"    *** LOW COSINE SIMILARITY ***")
    elif cos < 0.999:
        print(f"    (moderate match)")
    else:
        print(f"    (good match)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zig-dir", required=True, help="Dir with Zig --dump-intermediates output")
    parser.add_argument("--ref-dir", required=True, help="Dir with Python reference (imgcond_ref)")
    args = parser.parse_args()

    zig_dir = Path(args.zig_dir)
    ref_dir = Path(args.ref_dir)

    # Load pipeline meta for shapes
    import json
    with open(ref_dir / "pipeline_meta.json") as f:
        meta = json.load(f)
    s1 = meta["stage1"]
    s2 = meta["stage2"]
    F, H1, W1 = s1["f_lat"], s1["h_lat"], s1["w_lat"]
    H2, W2 = s2["h_lat"], s2["w_lat"]
    T_v1 = F * H1 * W1
    T_v2 = F * H2 * W2
    T_a = s1["t_audio"]

    print(f"Pipeline: F={F} H1={H1} W1={W1} H2={H2} W2={W2}")
    print(f"  T_v1={T_v1} T_v2={T_v2} T_a={T_a}")

    # ========================================================================
    # Check 1: Stage 1 output
    # ========================================================================
    print("\n" + "=" * 60)
    print("CHECK 1: Stage 1 denoised output")
    print("=" * 60)

    s1_video_path = zig_dir / "stage1_video_latent.bin"
    s1_ref_path = ref_dir / "ref" / "stage1_outputs.safetensors"

    if s1_video_path.exists():
        # Zig output: [1, T_v1, 128] bf16 (patchified)
        zig_s1 = load_bf16_bin(str(s1_video_path), [1, T_v1, 128])
        # Unpatchify: [1, T, C] → [1, F, H, W, C] → [1, C, F, H, W]
        zig_s1_5d = zig_s1.reshape(1, F, H1, W1, 128).permute(0, 4, 1, 2, 3)

        if s1_ref_path.exists():
            with safe_open(str(s1_ref_path), framework="pt") as f:
                ref_s1 = f.get_tensor("video_latent_denoised")  # [1, 128, F, H, W]
            compare_tensors("video_latent (Stage 1)", zig_s1_5d, ref_s1)
        else:
            print(f"  No Python reference: {s1_ref_path}")
            print(f"  Zig Stage 1 output range: [{zig_s1.float().min():.4f}, {zig_s1.float().max():.4f}]")
    else:
        print(f"  No Zig dump found: {s1_video_path}")
        print(f"  Re-run with --dump-intermediates")

    # Audio Stage 1
    s1_audio_path = zig_dir / "stage1_audio_latent.bin"
    if s1_audio_path.exists() and s1_ref_path.exists():
        zig_a1 = load_bf16_bin(str(s1_audio_path), [1, T_a, 128])
        # Unpatchify audio: [1, T_a, 128] → [1, 8, T_a/8, 16] (audio uses different layout)
        with safe_open(str(s1_ref_path), framework="pt") as f:
            ref_a1 = f.get_tensor("audio_latent_denoised")  # [1, 8, T_a/8?, 16] or similar
        print(f"\n  audio Zig shape: {list(zig_a1.shape)}, ref shape: {list(ref_a1.shape)}")
        # Compare flat
        zig_a1_flat = zig_a1.flatten()
        ref_a1_flat = ref_a1.flatten()
        if zig_a1_flat.shape == ref_a1_flat.shape:
            cos = cosine_sim(zig_a1_flat, ref_a1_flat)
            print(f"  audio cosine (flat): {cos:.6f}")

    # ========================================================================
    # Check 2: Stage 2 output (final denoised latent)
    # ========================================================================
    print("\n" + "=" * 60)
    print("CHECK 2: Stage 2 denoised output")
    print("=" * 60)

    s2_video_path = zig_dir / "video_latent.bin"
    s2_ref_path = ref_dir / "ref" / "stage2_outputs.safetensors"

    if s2_video_path.exists():
        # Zig output: [1, T_v2, 128] bf16 (patchified)
        zig_s2 = load_bf16_bin(str(s2_video_path), [1, T_v2, 128])
        zig_s2_5d = zig_s2.reshape(1, F, H2, W2, 128).permute(0, 4, 1, 2, 3)

        if s2_ref_path.exists():
            with safe_open(str(s2_ref_path), framework="pt") as f:
                ref_s2 = f.get_tensor("video_latent_final")
            compare_tensors("video_latent (Stage 2)", zig_s2_5d, ref_s2)
        else:
            print(f"  No Python reference: {s2_ref_path}")
            print(f"  Zig Stage 2 output range: [{zig_s2.float().min():.4f}, {zig_s2.float().max():.4f}]")
    else:
        print(f"  No Zig dump found: {s2_video_path}")
        print(f"  Re-run with --dump-intermediates")

    # ========================================================================
    # Check 3: Decoded video frames
    # ========================================================================
    print("\n" + "=" * 60)
    print("CHECK 3: Decoded video frames")
    print("=" * 60)

    frames_path = zig_dir / "frames.bin"
    if frames_path.exists():
        # VAE output: 121 frames of 1024x1536, RGB24
        # But we compute from pipeline meta: frames = (F-1)*8 + 1 = 121
        # Size = H_out x W_out (H2*32 x W2*32 for stage2)
        H_out = H2 * 32
        W_out = W2 * 32
        F_out = (F - 1) * 8 + 1
        expected_size = F_out * H_out * W_out * 3
        actual_size = frames_path.stat().st_size
        print(f"  Expected {F_out} frames of {W_out}x{H_out} = {expected_size} bytes")
        print(f"  Actual file size: {actual_size} bytes")

        if actual_size == expected_size:
            frames = np.frombuffer(frames_path.read_bytes(), dtype=np.uint8).reshape(F_out, H_out, W_out, 3)
            # Check first frame
            f0 = frames[0]
            print(f"  Frame 0 mean: {f0.mean():.1f}, std: {f0.std():.1f}, min: {f0.min()}, max: {f0.max()}")
            # Check if frame looks like noise (high-frequency)
            # Compute spatial gradient magnitude
            dy = np.abs(f0[1:, :, :].astype(float) - f0[:-1, :, :].astype(float)).mean()
            dx = np.abs(f0[:, 1:, :].astype(float) - f0[:, :-1, :].astype(float)).mean()
            print(f"  Frame 0 spatial gradient: dx={dx:.2f} dy={dy:.2f}")
            if dx > 50 or dy > 50:
                print(f"  *** HIGH SPATIAL GRADIENT — likely garbled/noisy ***")
            else:
                print(f"  (spatial gradient looks reasonable)")

            # Save first frame as PNG for visual inspection
            try:
                from PIL import Image
                img = Image.fromarray(f0)
                out_path = zig_dir / "frame0.png"
                img.save(str(out_path))
                print(f"  Saved first frame to {out_path}")
            except ImportError:
                print(f"  (install Pillow to save frame0.png)")
        else:
            print(f"  *** SIZE MISMATCH ***")
    else:
        print(f"  No frames dump found: {frames_path}")
        print(f"  Re-run with --dump-intermediates")

    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    print("""
If Stage 1 cosine < 0.99 → Stage 1 denoising diverges from Python.
If Stage 1 cosine >= 0.99 but Stage 2 cosine < 0.99 → Bridge or Stage 2 issue.
If both stages match but frames are garbled → VAE decode or frame conversion bug.
""")


if __name__ == "__main__":
    main()
