#!/usr/bin/env python3
"""Compare ZML vocoder output against Python reference.

Reads:
  - Reference waveform from safetensors (f32)
  - ZML waveform from raw binary (f32)

Reports per-channel and overall PSNR.

Usage:
  uv run python compare_vocoder.py \
      /root/e2e_demo/vocoder_ref/vocoder_activations.safetensors \
      /root/e2e_demo/vocoder_zig_out/waveform.bin
"""

import sys
import struct
import numpy as np


def load_f32_bin(path: str, n_elements: int) -> np.ndarray:
    """Load raw f32 binary."""
    with open(path, "rb") as f:
        raw = f.read()
    expected = n_elements * 4
    actual = len(raw)
    if actual != expected:
        print(f"WARNING: expected {expected} bytes, got {actual}")
        n_elements = actual // 4
    return np.frombuffer(raw[:n_elements * 4], dtype=np.float32)


def load_ref_from_safetensors(path: str):
    """Load reference waveform from safetensors."""
    import safetensors
    with safetensors.safe_open(path, framework="numpy") as f:
        ref = f.get_tensor("ref_waveform")  # [1, 2, T] f32
        shape = list(ref.shape)
    return ref, shape


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    """Compute PSNR between two arrays."""
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    # For waveforms in [-1, 1], peak = 2.0
    peak = 2.0
    return 10 * np.log10(peak ** 2 / mse)


def compare_arrays(name, ref_flat, zml_flat, n_channels=2):
    """Compare two flat arrays and print PSNR + error stats."""
    min_len = min(len(ref_flat), len(zml_flat))
    ref_flat = ref_flat[:min_len].astype(np.float32)
    zml_flat = zml_flat[:min_len].astype(np.float32)

    print(f"\n=== {name} ({min_len} elements) ===")
    overall = psnr(ref_flat, zml_flat)
    print(f"  Overall PSNR: {overall:.2f} dB")

    if n_channels == 2 and min_len >= 2:
        T = min_len // 2
        ref_2d = ref_flat[:T * 2].reshape(2, T)
        zml_2d = zml_flat[:T * 2].reshape(2, T)
        for ch in range(2):
            ch_psnr = psnr(ref_2d[ch], zml_2d[ch])
            print(f"  Channel {ch}: PSNR = {ch_psnr:.2f} dB")

    diff = np.abs(ref_flat - zml_flat)
    print(f"  Max abs error:  {diff.max():.6f}")
    print(f"  Mean abs error: {diff.mean():.6f}")
    print(f"  Ref range:  [{ref_flat.min():.4f}, {ref_flat.max():.4f}]")
    print(f"  ZML range:  [{zml_flat.min():.4f}, {zml_flat.max():.4f}]")


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_vocoder.py <ref_activations.safetensors> <zml_output_dir>")
        print("  zml_output_dir should contain waveform.bin and optionally waveform_16k.bin")
        sys.exit(1)

    ref_path = sys.argv[1]
    zml_path = sys.argv[2]

    print(f"Reference: {ref_path}")
    print(f"ZML:       {zml_path}")

    import safetensors
    import os

    with safetensors.safe_open(ref_path, framework="numpy") as f:
        ref_keys = list(f.keys())
        ref_48k = f.get_tensor("ref_waveform")
        ref_16k = f.get_tensor("ref_waveform_16k") if "ref_waveform_16k" in ref_keys else None

    print(f"Reference 48kHz shape: {list(ref_48k.shape)} dtype: {ref_48k.dtype}")
    if ref_16k is not None:
        print(f"Reference 16kHz shape: {list(ref_16k.shape)} dtype: {ref_16k.dtype}")

    # Determine if zml_path is a directory or a file
    if os.path.isdir(zml_path):
        zml_dir = zml_path
    else:
        zml_dir = os.path.dirname(zml_path)

    # Compare 16kHz intermediate (if both available)
    zml_16k_path = os.path.join(zml_dir, "waveform_16k.bin")
    if ref_16k is not None and os.path.exists(zml_16k_path):
        zml_16k = load_f32_bin(zml_16k_path, ref_16k.size)
        compare_arrays("16kHz waveform (Stage 1: Main Vocoder)", ref_16k.flatten(), zml_16k)

    # Compare 48kHz final
    zml_48k_path = os.path.join(zml_dir, "waveform.bin")
    if os.path.exists(zml_48k_path):
        zml_48k = load_f32_bin(zml_48k_path, ref_48k.size)
        compare_arrays("48kHz waveform (Final)", ref_48k.flatten(), zml_48k)


if __name__ == "__main__":
    main()
