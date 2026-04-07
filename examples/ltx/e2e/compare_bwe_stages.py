#!/usr/bin/env python3
"""Compare BWE stage intermediates: Python reference vs ZML.

Reads:
  - Python reference from bwe_stages.safetensors
  - ZML outputs from raw .bin files in zig output dir

Reports PSNR at each BWE stage to pinpoint where divergence starts.

Usage:
  uv run python compare_bwe_stages.py \
      /root/e2e_demo/vocoder_ref/bwe_stages.safetensors \
      /root/e2e_demo/vocoder_zig_out/
"""

import sys
import os
import numpy as np


def load_f32_bin(path, n_elements):
    """Load raw f32 binary."""
    with open(path, "rb") as f:
        raw = f.read()
    actual_elems = len(raw) // 4
    if actual_elems != n_elements:
        print(f"  WARNING: expected {n_elements} elements, got {actual_elems}")
        n_elements = min(n_elements, actual_elems)
    return np.frombuffer(raw[:n_elements * 4], dtype=np.float32)


def psnr(ref, test):
    """PSNR with peak=max-min of reference (dynamic range)."""
    ref64 = ref.astype(np.float64)
    test64 = test.astype(np.float64)
    mse = np.mean((ref64 - test64) ** 2)
    if mse == 0:
        return float('inf')
    peak = ref64.max() - ref64.min()
    if peak == 0:
        peak = 1.0
    return 10 * np.log10(peak ** 2 / mse)


def psnr_waveform(ref, test):
    """PSNR for waveforms in [-1, 1] range (peak=2)."""
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(4.0 / mse)


def compare(name, ref, zml, is_waveform=False):
    """Compare two arrays and print PSNR + error stats."""
    min_len = min(len(ref), len(zml))
    ref = ref[:min_len].astype(np.float32)
    zml = zml[:min_len].astype(np.float32)

    if is_waveform:
        p = psnr_waveform(ref, zml)
    else:
        p = psnr(ref, zml)
    diff = np.abs(ref - zml)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  PSNR:           {p:.2f} dB")
    print(f"  Max abs error:  {diff.max():.6f}")
    print(f"  Mean abs error: {diff.mean():.6f}")
    print(f"  Ref range:  [{ref.min():.6f}, {ref.max():.6f}]")
    print(f"  ZML range:  [{zml.min():.6f}, {zml.max():.6f}]")
    print(f"  Elements:   {min_len}")
    return p


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_bwe_stages.py <bwe_stages.safetensors> <zig_output_dir>")
        sys.exit(1)

    ref_path = sys.argv[1]
    zig_dir = sys.argv[2]

    print(f"Reference: {ref_path}")
    print(f"ZML dir:   {zig_dir}")

    import safetensors
    with safetensors.safe_open(ref_path, framework="numpy") as f:
        ref_keys = sorted(f.keys())
        ref = {}
        for k in ref_keys:
            ref[k] = f.get_tensor(k)
            print(f"  Loaded ref {k}: shape={list(ref[k].shape)} dtype={ref[k].dtype}")

    # Map: python tensor name → (zig bin filename, is_waveform)
    stage_map = [
        ("bwe_mel_pre_transpose", "debug_bwe_mel.bin", False),
        ("bwe_skip", "debug_bwe_skip.bin", True),
        ("bwe_residual", "debug_bwe_residual.bin", True),
        ("bwe_output", "waveform.bin", True),
    ]

    results = {}
    for ref_key, zig_file, is_waveform in stage_map:
        zig_path = os.path.join(zig_dir, zig_file)
        if ref_key not in ref:
            print(f"\n  SKIP {ref_key}: not in reference")
            continue
        if not os.path.exists(zig_path):
            print(f"\n  SKIP {ref_key}: {zig_file} not found in {zig_dir}")
            continue

        ref_flat = ref[ref_key].flatten().astype(np.float32)
        zml_flat = load_f32_bin(zig_path, ref_flat.size)
        p = compare(f"{ref_key} vs {zig_file}", ref_flat, zml_flat, is_waveform=is_waveform)
        results[ref_key] = p

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for name, p in results.items():
        status = "OK" if p > 40 else ("WARN" if p > 25 else "BAD")
        print(f"  [{status:4s}] {name:30s} {p:8.2f} dB")


if __name__ == "__main__":
    main()
