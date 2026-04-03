"""Compare Zig audio VAE output against Python reference (PSNR).

Usage:
  python compare_audio_vae.py \
      --ref /root/e2e_demo/audio_vae_ref/audio_vae_activations.safetensors \
      --zig /root/e2e_demo/audio_vae_zig_out/decoded_audio.bin
"""

import argparse
import struct
import numpy as np
from safetensors import safe_open


def bf16_bytes_to_f32(data: bytes, shape: tuple) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint16)
    f32 = (raw.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(shape)


def load_bf16_safetensor(path: str, key: str) -> np.ndarray:
    """Load a bf16 tensor from safetensors as f32 numpy array."""
    with safe_open(path, framework="numpy") as f:
        metadata = f.metadata()
    # Use raw file reading to get bf16 bytes
    import json
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
        tensor_info = header[key]
        shape = tuple(tensor_info["shape"])
        offsets = tensor_info["data_offsets"]
        data_start = 8 + header_len + offsets[0]
        data_len = offsets[1] - offsets[0]
        fh.seek(data_start)
        raw = fh.read(data_len)
    return bf16_bytes_to_f32(raw, shape)


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    mse = np.mean((ref - test) ** 2)
    if mse == 0:
        return float("inf")
    max_val = np.max(np.abs(ref))
    return 20 * np.log10(max_val / np.sqrt(mse))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="Python reference safetensors")
    p.add_argument("--zig", required=True, help="Zig decoded_audio.bin (raw bf16)")
    args = p.parse_args()

    # Load Python reference
    ref_f32 = load_bf16_safetensor(args.ref, "decoded_output")
    print(f"Reference shape: {ref_f32.shape}, dtype: f32 (from bf16)")
    print(f"  range: [{ref_f32.min():.4f}, {ref_f32.max():.4f}]")

    # Load Zig output
    zig_bytes = open(args.zig, "rb").read()
    zig_f32 = bf16_bytes_to_f32(zig_bytes, ref_f32.shape)
    print(f"Zig shape: {zig_f32.shape}, dtype: f32 (from bf16)")
    print(f"  range: [{zig_f32.min():.4f}, {zig_f32.max():.4f}]")

    # Compute PSNR
    p_val = psnr(ref_f32, zig_f32)
    print(f"\nPSNR: {p_val:.2f} dB")

    # Per-channel PSNR
    for c in range(ref_f32.shape[1]):
        c_psnr = psnr(ref_f32[0, c], zig_f32[0, c])
        print(f"  Channel {c}: {c_psnr:.2f} dB")

    # Max absolute error
    max_err = np.max(np.abs(ref_f32 - zig_f32))
    mean_err = np.mean(np.abs(ref_f32 - zig_f32))
    print(f"\nMax abs error: {max_err:.6f}")
    print(f"Mean abs error: {mean_err:.6f}")

    if p_val > 40:
        print("\n✓ PASS (PSNR > 40 dB)")
    else:
        print("\n✗ FAIL (PSNR <= 40 dB)")


if __name__ == "__main__":
    main()
