#!/usr/bin/env python3
"""Validate Zig-computed Stage 1 initial state against Python reference.

Usage:
    python validate_stage1_state.py \
        --reference /path/to/unconditioned_stage1_inputs.safetensors \
        --zig-dir /path/to/zig_dump_dir/ \
        --meta /path/to/pipeline_meta.json

The --zig-dir should contain files dumped by inference with --dump-intermediates:
    s1_video_positions.bin, s1_audio_positions.bin,
    s1_video_mask.bin, s1_audio_mask.bin,
    s1_video_clean.bin, s1_audio_clean.bin
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def load_bin(path: Path, shape: tuple, dtype: np.dtype) -> np.ndarray:
    raw = path.read_bytes()
    expected = int(np.prod(shape)) * dtype.itemsize
    assert len(raw) == expected, f"{path.name}: got {len(raw)} bytes, expected {expected}"
    return np.frombuffer(raw, dtype=dtype).reshape(shape)


def compare(name: str, ref: np.ndarray, zig: np.ndarray) -> bool:
    if ref.shape != zig.shape:
        print(f"  FAIL {name}: shape mismatch ref={ref.shape} zig={zig.shape}")
        return False

    # Compare as raw bytes for bitwise exactness
    ref_bytes = ref.tobytes()
    zig_bytes = zig.tobytes()
    if ref_bytes == zig_bytes:
        print(f"  OK   {name}: bitwise identical  shape={ref.shape} dtype={ref.dtype}")
        return True

    # Not bitwise identical — report stats
    # Cast to f32 for comparison
    ref_f32 = ref.astype(np.float32)
    zig_f32 = zig.astype(np.float32)
    abs_diff = np.abs(ref_f32 - zig_f32)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()
    n_diff = np.count_nonzero(abs_diff > 0)
    total = ref.size

    print(f"  DIFF {name}: {n_diff}/{total} elements differ  "
          f"max_abs={max_diff:.8f}  mean_abs={mean_diff:.8f}")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="Path to unconditioned_stage1_inputs.safetensors")
    parser.add_argument("--zig-dir", required=True, help="Directory with Zig --dump-intermediates output")
    parser.add_argument("--meta", required=True, help="Path to pipeline_meta.json")
    args = parser.parse_args()

    ref_path = Path(args.reference)
    zig_dir = Path(args.zig_dir)
    meta_path = Path(args.meta)

    # Load pipeline metadata for shapes
    with open(meta_path) as f:
        meta = json.load(f)
    s1 = meta["stage1"]
    f_lat = s1["f_lat"]
    h_lat = s1["h_lat"]
    w_lat = s1["w_lat"]
    t_audio = s1["t_audio"]
    t_video = f_lat * h_lat * w_lat

    print(f"Stage 1 dims: F={f_lat} H={h_lat} W={w_lat}  T_v={t_video}  T_a={t_audio}")

    # Load Python reference from safetensors (using torch for bf16 support)
    ref = load_file(str(ref_path), device="cpu")
    for key, tensor in ref.items():
        print(f"  ref[{key}]: shape={list(tensor.shape)} dtype={tensor.dtype}")

    # Define expected shapes and dtypes for Zig .bin files
    checks = [
        ("video_positions", "s1_video_positions.bin", (1, 3, t_video, 2), np.dtype("uint16")),  # bf16 as raw u16
        ("audio_positions", "s1_audio_positions.bin", (1, 1, t_audio, 2), np.dtype("float32")),
        ("video_denoise_mask", "s1_video_mask.bin", (1, t_video, 1), np.dtype("float32")),
        ("audio_denoise_mask", "s1_audio_mask.bin", (1, t_audio, 1), np.dtype("float32")),
        ("video_clean_latent", "s1_video_clean.bin", (1, t_video, 128), np.dtype("uint16")),  # bf16 as raw u16
        ("audio_clean_latent", "s1_audio_clean.bin", (1, t_audio, 128), np.dtype("uint16")),  # bf16 as raw u16
    ]

    print(f"\n=== Comparing Zig vs Python reference ===")
    all_ok = True
    for ref_key, bin_name, shape, raw_dtype in checks:
        bin_path = zig_dir / bin_name
        if not bin_path.exists():
            print(f"  SKIP {ref_key}: {bin_name} not found in {zig_dir}")
            all_ok = False
            continue

        ref_tensor = ref[ref_key]

        # Compare as raw bytes — works for all dtypes including bf16
        ref_raw = ref_tensor.contiguous().numpy(force=True).tobytes() if ref_tensor.dtype == torch.float32 else ref_tensor.contiguous().view(torch.uint8).numpy().tobytes()
        zig_raw = bin_path.read_bytes()

        if len(ref_raw) != len(zig_raw):
            print(f"  FAIL {ref_key}: byte size mismatch ref={len(ref_raw)} zig={len(zig_raw)}")
            all_ok = False
            continue

        if ref_raw == zig_raw:
            print(f"  OK   {ref_key}: bitwise identical  shape={list(ref_tensor.shape)} dtype={ref_tensor.dtype}")
        else:
            # Count differing values
            ref_arr = np.frombuffer(ref_raw, dtype=np.uint8)
            zig_arr = np.frombuffer(zig_raw, dtype=np.uint8)
            n_diff_bytes = np.count_nonzero(ref_arr != zig_arr)

            # For a more useful diff count, compare at element granularity
            elem_size = ref_tensor.element_size()
            ref_elems = np.frombuffer(ref_raw, dtype=np.dtype(f"u{elem_size}") if elem_size <= 4 else np.uint64)
            zig_elems = np.frombuffer(zig_raw, dtype=ref_elems.dtype)
            n_diff_elems = np.count_nonzero(ref_elems != zig_elems)

            # If f32, show numeric diff and check tolerance
            if ref_tensor.dtype == torch.float32:
                ref_f32 = np.frombuffer(ref_raw, dtype=np.float32)
                zig_f32 = np.frombuffer(zig_raw, dtype=np.float32)
                abs_diff = np.abs(ref_f32 - zig_f32)
                max_ad = abs_diff.max()
                mean_ad = abs_diff.mean()
                if max_ad < 1e-6:
                    print(f"  OK   {ref_key}: f32 near-identical  {n_diff_elems}/{ref_tensor.numel()} differ  "
                          f"max_abs={max_ad:.2e} (<1e-6)")
                else:
                    print(f"  DIFF {ref_key}: {n_diff_elems}/{ref_tensor.numel()} elements differ  "
                          f"max_abs={max_ad:.8f}  mean_abs={mean_ad:.8f}")
                    all_ok = False
            else:
                print(f"  DIFF {ref_key}: {n_diff_elems}/{ref_tensor.numel()} elements differ ({n_diff_bytes} bytes)")
            all_ok = False

    print()
    if all_ok:
        print("ALL CHECKS PASSED — Zig-computed state matches Python reference bitwise.")
    else:
        print("SOME CHECKS FAILED — see above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
