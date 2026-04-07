#!/usr/bin/env python3
"""Isolate sinc resampler: load ZML's 16kHz waveform, run Python resampler, compare.

This eliminates any input differences and tests ONLY the sinc resampler.

Usage:
  uv run python /root/repos/zml/examples/ltx/e2e/isolate_sinc_resampler.py \
      --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
      --zml-16k /root/e2e_demo/vocoder_zig_out/waveform_16k.bin \
      --zml-skip /root/e2e_demo/vocoder_zig_out/debug_bwe_skip.bin \
      --shape 1,2,1280
"""

import argparse
import json
import sys

import torch
import numpy as np
import safetensors


def load_f32_bin(path, shape):
    with open(path, "rb") as f:
        raw = f.read()
    n = 1
    for s in shape:
        n *= s
    arr = np.frombuffer(raw[:n * 4], dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr.copy())


def psnr_waveform(ref, test):
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(4.0 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--zml-16k", required=True, help="ZML 16kHz waveform binary (f32)")
    parser.add_argument("--zml-skip", required=True, help="ZML sinc skip output binary (f32)")
    parser.add_argument("--shape", default="1,2,1280", help="Shape of 16kHz waveform")
    args = parser.parse_args()

    shape = [int(x) for x in args.shape.split(",")]

    sys.path.insert(0, "/root/repos/LTX-2/packages/ltx-core/src")
    sys.path.insert(0, "/root/repos/LTX-2/packages/ltx-pipelines/src")
    from ltx_core.model.audio_vae.vocoder import UpSample1d

    # Load ZML 16kHz waveform
    zml_16k = load_f32_bin(args.zml_16k, shape).cuda()
    print(f"ZML 16kHz waveform: {list(zml_16k.shape)} range=[{zml_16k.min():.6f}, {zml_16k.max():.6f}]")

    # No padding needed (1280 % 80 == 0)
    hop_length = 80
    T = zml_16k.shape[-1]
    remainder = T % hop_length
    x = zml_16k
    if remainder:
        x = torch.nn.functional.pad(x, (0, hop_length - remainder))
        print(f"  Padded: {list(x.shape)}")

    # Build and run Python resampler on ZML input
    ratio = 3
    resampler = UpSample1d(ratio=ratio).cuda().eval()
    print(f"Resampler: kernel_size={resampler.kernel_size}, pad={resampler.pad}, "
          f"pad_left={resampler.pad_left}, pad_right={resampler.pad_right}")
    print(f"  filter shape: {list(resampler.filter.shape)}")

    with torch.no_grad():
        # Step by step to capture intermediates
        _, n_channels, _ = x.shape
        x_padded = torch.nn.functional.pad(x, (resampler.pad, resampler.pad), mode="replicate")
        print(f"  After replicate pad: {list(x_padded.shape)}")

        filt = resampler.filter.expand(n_channels, -1, -1)
        print(f"  Filter expanded: {list(filt.shape)}")

        # Raw conv_transpose1d output (before scale and trim)
        raw_output = torch.nn.functional.conv_transpose1d(
            x_padded, filt, stride=resampler.stride, groups=n_channels
        )
        print(f"  Raw conv_transpose output: {list(raw_output.shape)}")

        # Scale
        scaled = resampler.ratio * raw_output
        print(f"  After scale: {list(scaled.shape)}")

        # Trim
        py_skip = scaled[..., resampler.pad_left:-resampler.pad_right]
        print(f"  After trim: {list(py_skip.shape)}")
        print(f"  Python skip range: [{py_skip.min():.6f}, {py_skip.max():.6f}]")

    py_skip_np = py_skip.cpu().float().numpy().flatten()

    # Load ZML skip
    zml_skip_np = np.frombuffer(
        open(args.zml_skip, "rb").read(), dtype=np.float32
    )
    print(f"\nZML skip: {zml_skip_np.shape} range=[{zml_skip_np.min():.6f}, {zml_skip_np.max():.6f}]")
    print(f"Python skip: {py_skip_np.shape}")

    # Direct comparison
    min_len = min(len(py_skip_np), len(zml_skip_np))
    py = py_skip_np[:min_len]
    zml = zml_skip_np[:min_len]

    p = psnr_waveform(py, zml)
    diff = np.abs(py - zml)
    print(f"\n=== Direct comparison (same input) ===")
    print(f"  PSNR:           {p:.2f} dB")
    print(f"  Max abs error:  {diff.max():.6f}")
    print(f"  Mean abs error: {diff.mean():.6f}")

    # Offset search
    T = min_len // 2
    r = py_skip_np[:T]  # channel 0
    z = zml_skip_np[:T]

    print(f"\n=== Offset search (channel 0, same input) ===")
    best_offset = 0
    best_mse = float('inf')
    for offset in range(-10, 11):
        if offset >= 0:
            r_s = r[offset:]
            z_s = z[:len(r_s)]
        else:
            z_s = z[-offset:]
            r_s = r[:len(z_s)]
        n = min(len(r_s), len(z_s))
        mse = np.mean((r_s[:n].astype(np.float64) - z_s[:n].astype(np.float64)) ** 2)
        marker = " <-- BEST" if mse < best_mse else ""
        if mse < best_mse:
            best_mse = mse
            best_offset = offset
        print(f"  offset={offset:+3d}: MSE={mse:.10f}{marker}")

    print(f"\nBest offset: {best_offset} (MSE={best_mse:.10f})")

    if best_offset != 0:
        if best_offset >= 0:
            r_s = py_skip_np[best_offset:]
            z_s = zml_skip_np[:len(r_s)]
        else:
            z_s = zml_skip_np[-best_offset:]
            r_s = py_skip_np[:len(z_s)]
        n = min(len(r_s), len(z_s))
        mse = np.mean((r_s[:n].astype(np.float64) - z_s[:n].astype(np.float64)) ** 2)
        p = 10 * np.log10(4.0 / mse) if mse > 0 else float('inf')
        print(f"PSNR at offset {best_offset}: {p:.2f} dB")

    # Print first values comparison
    print(f"\n=== First 20 values (channel 0) ===")
    print(f"  Python: {py_skip_np[:20].tolist()}")
    print(f"  ZML:    {zml_skip_np[:20].tolist()}")
    if best_offset == -2:
        print(f"  ZML[2:22]: {zml_skip_np[2:22].tolist()}")


if __name__ == "__main__":
    main()
