#!/usr/bin/env python3
"""Compare non-tiled vs tiled VAE decode outputs.

Usage:
  1. Run with --num-frames 65 (non-tiled) to generate /tmp/nontiled_decoded_bf16.bin
  2. Run with --num-frames 481 (tiled) to generate /tmp/tile0_decoded_bf16.bin
  3. python3 examples/ltx/compare_tiles.py
"""
import numpy as np
import sys

def load_bf16(path):
    """Load a raw bf16 binary file as float32 numpy array."""
    raw = np.fromfile(path, dtype=np.uint16)
    # bf16 -> f32: shift left by 16 bits
    f32 = np.frombuffer((raw.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
    return f32

def main():
    nontiled_path = "/tmp/nontiled_decoded_bf16.bin"
    tiled_path = "/tmp/tile0_decoded_bf16.bin"

    try:
        a = load_bf16(nontiled_path)
    except FileNotFoundError:
        print(f"ERROR: {nontiled_path} not found. Run with --num-frames 65 first.")
        sys.exit(1)

    try:
        b = load_bf16(tiled_path)
    except FileNotFoundError:
        print(f"ERROR: {tiled_path} not found. Run with --num-frames 481 first.")
        sys.exit(1)

    print(f"Non-tiled: {len(a)} elements ({len(a)*2} bytes)")
    print(f"Tiled:     {len(b)} elements ({len(b)*2} bytes)")

    print(f"\nNon-tiled stats: mean={a.mean():.6f} std={a.std():.6f} min={a.min():.6f} max={a.max():.6f}")
    print(f"Tiled stats:     mean={b.mean():.6f} std={b.std():.6f} min={b.min():.6f} max={b.max():.6f}")

    # Compare overlapping region
    n = min(len(a), len(b))
    diff = np.abs(a[:n] - b[:n])
    print(f"\nComparison (first {n} elements):")
    print(f"  Max diff:  {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Median diff: {np.median(diff):.6f}")

    # Check for NaN/Inf
    print(f"\nNon-tiled NaN count: {np.isnan(a).sum()}, Inf count: {np.isinf(a).sum()}")
    print(f"Tiled NaN count:     {np.isnan(b).sum()}, Inf count: {np.isinf(b).sum()}")

    # Show first 20 values side by side
    print(f"\nFirst 20 values comparison:")
    print(f"  {'idx':>5}  {'non-tiled':>12}  {'tiled':>12}  {'diff':>12}")
    for i in range(min(20, n)):
        print(f"  {i:5d}  {a[i]:12.6f}  {b[i]:12.6f}  {abs(a[i]-b[i]):12.6f}")

    # If sizes differ, note that
    if len(a) != len(b):
        print(f"\nWARNING: Size mismatch! Non-tiled has {len(a)} elements, tiled has {len(b)} elements.")
        print("This is expected if the non-tiled run used a different frame count than tile 0 produces.")

    # Overall verdict
    if diff.max() < 0.01:
        print("\n✓ PASS: Tile 0 decoded output matches non-tiled reference (max diff < 0.01)")
    elif diff.max() < 0.1:
        print("\n⚠ WARNING: Small differences detected (max diff < 0.1) - may be acceptable bf16 rounding")
    else:
        print("\n✗ FAIL: Large differences detected - tile upload or decode is likely wrong")
        # Show where the largest differences are
        worst_idx = np.argsort(diff)[-10:][::-1]
        print(f"  Top 10 worst indices:")
        for idx in worst_idx:
            print(f"    [{idx}] non-tiled={a[idx]:.6f} tiled={b[idx]:.6f} diff={diff[idx]:.6f}")

if __name__ == "__main__":
    main()
