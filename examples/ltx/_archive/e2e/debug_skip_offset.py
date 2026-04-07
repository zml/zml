#!/usr/bin/env python3
"""Detailed comparison of BWE skip connection — find if there's an offset/shift.

Usage:
  uv run python /root/repos/zml/examples/ltx/e2e/debug_skip_offset.py \
      /root/e2e_demo/vocoder_ref/bwe_stages.safetensors \
      /root/e2e_demo/vocoder_zig_out/debug_bwe_skip.bin
"""

import sys
import numpy as np


def load_f32_bin(path, n_elements):
    with open(path, "rb") as f:
        raw = f.read()
    return np.frombuffer(raw[:n_elements * 4], dtype=np.float32)


def main():
    ref_path = sys.argv[1]
    zig_path = sys.argv[2]

    import safetensors
    with safetensors.safe_open(ref_path, framework="numpy") as f:
        ref = f.get_tensor("bwe_skip").flatten().astype(np.float32)

    zml = load_f32_bin(zig_path, ref.size)

    print(f"ref shape: {ref.shape}, zml shape: {zml.shape}")
    print(f"ref range: [{ref.min():.6f}, {ref.max():.6f}]")
    print(f"zml range: [{zml.min():.6f}, {zml.max():.6f}]")

    # Check direct MSE
    mse_direct = np.mean((ref - zml) ** 2)
    print(f"\nDirect MSE: {mse_direct:.6f}")

    # Reshape to 2 channels
    T = len(ref) // 2
    ref_2d = ref[:T*2].reshape(2, T)
    zml_2d = zml[:T*2].reshape(2, T)

    # Print first/last values for each channel
    for ch in range(2):
        print(f"\nChannel {ch}:")
        print(f"  ref first 20: {ref_2d[ch, :20].tolist()}")
        print(f"  zml first 20: {zml_2d[ch, :20].tolist()}")
        print(f"  ref last  20: {ref_2d[ch, -20:].tolist()}")
        print(f"  zml last  20: {zml_2d[ch, -20:].tolist()}")

    # Try shifted comparisons to detect offset
    print(f"\n=== Offset search (channel 0) ===")
    r = ref_2d[0]
    z = zml_2d[0]
    best_offset = 0
    best_mse = float('inf')
    for offset in range(-20, 21):
        if offset >= 0:
            r_slice = r[offset:]
            z_slice = z[:len(r_slice)]
        else:
            z_slice = z[-offset:]
            r_slice = r[:len(z_slice)]
        min_len = min(len(r_slice), len(z_slice))
        r_slice = r_slice[:min_len]
        z_slice = z_slice[:min_len]
        mse = np.mean((r_slice - z_slice) ** 2)
        marker = " <-- BEST" if mse < best_mse else ""
        if mse < best_mse:
            best_mse = mse
            best_offset = offset
        if abs(offset) <= 5 or mse < best_mse * 1.5:
            print(f"  offset={offset:+3d}: MSE={mse:.8f}{marker}")

    print(f"\nBest offset: {best_offset} (MSE={best_mse:.8f})")

    if best_offset != 0:
        # Show PSNR at best offset
        if best_offset >= 0:
            r_slice = ref_2d[:, best_offset:]
            z_slice = zml_2d[:, :r_slice.shape[1]]
        else:
            z_slice = zml_2d[:, -best_offset:]
            r_slice = ref_2d[:, :z_slice.shape[1]]
        min_len = min(r_slice.shape[1], z_slice.shape[1])
        r_slice = r_slice[:, :min_len]
        z_slice = z_slice[:, :min_len]
        mse = np.mean((r_slice.astype(np.float64) - z_slice.astype(np.float64)) ** 2)
        psnr = 10 * np.log10(4.0 / mse) if mse > 0 else float('inf')
        print(f"PSNR at offset {best_offset}: {psnr:.2f} dB")

    # Also check correlation
    corr = np.corrcoef(ref_2d[0], zml_2d[0])[0, 1]
    print(f"\nPearson correlation (ch0): {corr:.6f}")

    # Check if there's a scale difference
    # Fit zml = a*ref + b
    from numpy.linalg import lstsq
    A = np.vstack([ref_2d[0], np.ones(T)]).T
    result = lstsq(A, zml_2d[0], rcond=None)
    a, b = result[0]
    print(f"Linear fit: zml ≈ {a:.6f} * ref + {b:.6f}")
    fitted_mse = np.mean((zml_2d[0] - (a * ref_2d[0] + b)) ** 2)
    print(f"Fitted MSE: {fitted_mse:.8f}")


if __name__ == "__main__":
    main()
