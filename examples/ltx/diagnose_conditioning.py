#!/usr/bin/env python3
"""Diagnose image conditioning: compare Zig's intermediate dumps vs Python reference.

Checks:
  1. Preprocessed image: Zig's stb_image vs Python's load_image_and_preprocess
  2. Encoder output (img_tokens): Zig's VAE encoder vs Python's encoder_activations
  3. Conditioned state: Zig's latent/mask/clean after conditioning vs Python's captured state

Usage:
  uv run examples/ltx/diagnose_conditioning.py \
      --zig-dir /root/imgcond_zig \
      --ref-dir /root/imgcond_ref
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def load_bf16_bin(path: str, shape: list[int]) -> torch.Tensor:
    data = Path(path).read_bytes()
    expected = int(np.prod(shape)) * 2
    assert len(data) == expected, f"Expected {expected} bytes, got {len(data)} for {path} shape {shape}"
    arr = np.frombuffer(data, dtype=np.uint16).reshape(shape)
    f32 = np.zeros_like(arr, dtype=np.float32)
    f32.view(np.uint32)[:] = arr.astype(np.uint32) << 16
    return torch.from_numpy(f32).to(torch.bfloat16)


def load_f32_bin(path: str, shape: list[int]) -> torch.Tensor:
    data = Path(path).read_bytes()
    expected = int(np.prod(shape)) * 4
    assert len(data) == expected, f"Expected {expected} bytes, got {len(data)} for {path} shape {shape}"
    arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr.copy())


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


def compare(name: str, zig: torch.Tensor, ref: torch.Tensor):
    print(f"\n  {name}:")
    print(f"    Zig shape: {list(zig.shape)}, dtype: {zig.dtype}")
    print(f"    Ref shape: {list(ref.shape)}, dtype: {ref.dtype}")
    print(f"    Zig range: [{zig.float().min():.4f}, {zig.float().max():.4f}]")
    print(f"    Ref range: [{ref.float().min():.4f}, {ref.float().max():.4f}]")
    if zig.shape != ref.shape:
        print(f"    *** SHAPE MISMATCH ***")
        # Try flat comparison
        if zig.numel() == ref.numel():
            cos = cosine_sim(zig, ref)
            print(f"    Flat cosine (same numel): {cos:.6f}")
        return
    cos = cosine_sim(zig, ref)
    mae = (zig.float() - ref.float()).abs().mean().item()
    print(f"    Cosine similarity: {cos:.6f}")
    print(f"    MAE: {mae:.6f}")
    if cos < 0.99:
        print(f"    *** LOW COSINE SIMILARITY ***")

    # Check first vs rest
    T = zig.shape[1] if zig.dim() >= 2 else 0
    if T > 384:
        first = zig[:, :384]
        rest = zig[:, 384:]
        ref_first = ref[:, :384]
        ref_rest = ref[:, 384:]
        cos_first = cosine_sim(first, ref_first)
        cos_rest = cosine_sim(rest, ref_rest)
        mae_first = (first.float() - ref_first.float()).abs().mean().item()
        mae_rest = (rest.float() - ref_rest.float()).abs().mean().item()
        print(f"    First 384 tokens: cos={cos_first:.6f} mae={mae_first:.6f}")
        print(f"    Rest {T - 384} tokens: cos={cos_rest:.6f} mae={mae_rest:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zig-dir", required=True)
    parser.add_argument("--ref-dir", required=True)
    args = parser.parse_args()
    zig = Path(args.zig_dir)
    ref = Path(args.ref_dir)

    import json
    with open(ref / "pipeline_meta.json") as f:
        meta = json.load(f)
    s1 = meta["stage1"]
    H1, W1, F = s1["h_lat"], s1["w_lat"], s1["f_lat"]
    T_v1 = F * H1 * W1
    n_img = H1 * W1

    print(f"Pipeline: F={F} H1={H1} W1={W1}, T_v1={T_v1}, n_img={n_img}")

    # ==================================================================
    # CHECK 1: Preprocessed image
    # ==================================================================
    print("\n" + "=" * 60)
    print("CHECK 1: Preprocessed image (Zig stb_image vs Python)")
    print("=" * 60)

    img_bin = zig / "s1_image_preprocessed.bin"
    if img_bin.exists():
        zig_img = load_bf16_bin(str(img_bin), [1, 3, 1, H1 * 32, W1 * 32])
        with safe_open(str(ref / "image_preprocessed.safetensors"), framework="pt") as f:
            ref_img = f.get_tensor("image_s1")
        compare("image_preprocessed", zig_img, ref_img)
    else:
        print(f"  No dump: {img_bin} (re-run with --dump-intermediates)")

    # ==================================================================
    # CHECK 2: Encoder output (img_tokens)
    # ==================================================================
    print("\n" + "=" * 60)
    print("CHECK 2: Image tokens (Zig VAE encoder output)")
    print("=" * 60)

    tokens_bin = zig / "s1_img_tokens.bin"
    if tokens_bin.exists():
        zig_tokens = load_bf16_bin(str(tokens_bin), [1, n_img, 128])

        # Python reference: patchify the encoded_normalized
        with safe_open(str(ref / "encoder_activations.safetensors"), framework="pt") as f:
            ref_encoded = f.get_tensor("s1/encoded_normalized")  # [1, 128, 1, H1, W1]
        # Patchify: [1, 128, 1, H, W] → [1, 128, H*W] → [1, H*W, 128]
        ref_tokens = ref_encoded.flatten(2).transpose(1, 2)  # [1, H1*W1, 128]

        compare("img_tokens", zig_tokens, ref_tokens)
    else:
        print(f"  No dump: {tokens_bin}")

    # ==================================================================
    # CHECK 3: Conditioned latent state
    # ==================================================================
    print("\n" + "=" * 60)
    print("CHECK 3: Conditioned latent (after applyConditioning)")
    print("=" * 60)

    latent_bin = zig / "s1_conditioned_latent.bin"
    if latent_bin.exists():
        zig_latent = load_bf16_bin(str(latent_bin), [1, T_v1, 128])

        with safe_open(str(ref / "conditioned_stage1_inputs.safetensors"), framework="pt") as f:
            ref_latent = f.get_tensor("video_latent")
        compare("conditioned_latent", zig_latent, ref_latent)
    else:
        print(f"  No dump: {latent_bin}")

    # ==================================================================
    # CHECK 4: Conditioned clean_latent
    # ==================================================================
    print("\n" + "=" * 60)
    print("CHECK 4: Conditioned clean_latent")
    print("=" * 60)

    clean_bin = zig / "s1_conditioned_clean.bin"
    if clean_bin.exists():
        zig_clean = load_bf16_bin(str(clean_bin), [1, T_v1, 128])

        with safe_open(str(ref / "conditioned_stage1_inputs.safetensors"), framework="pt") as f:
            ref_clean = f.get_tensor("video_clean_latent")
        compare("conditioned_clean_latent", zig_clean, ref_clean)
    else:
        print(f"  No dump: {clean_bin}")

    # ==================================================================
    # CHECK 5: Conditioned denoise_mask
    # ==================================================================
    print("\n" + "=" * 60)
    print("CHECK 5: Conditioned denoise_mask")
    print("=" * 60)

    mask_bin = zig / "s1_conditioned_mask.bin"
    if mask_bin.exists():
        zig_mask = load_f32_bin(str(mask_bin), [1, T_v1, 1])

        with safe_open(str(ref / "conditioned_stage1_inputs.safetensors"), framework="pt") as f:
            ref_mask = f.get_tensor("video_denoise_mask")

        compare("conditioned_mask", zig_mask, ref_mask)

        # Extra check: verify mask structure
        mask_first = zig_mask[0, :n_img, 0]
        mask_rest = zig_mask[0, n_img:, 0]
        print(f"    Zig mask[:n_img] mean={mask_first.mean():.4f} (expect 0.0)")
        print(f"    Zig mask[n_img:] mean={mask_rest.mean():.4f} (expect 1.0)")
        print(f"    Zig mask unique values: {torch.unique(zig_mask).tolist()}")
    else:
        print(f"  No dump: {mask_bin}")

    # ==================================================================
    # SUMMARY
    # ==================================================================
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    print("""
If CHECK 1 (image) cosine < 0.99 → Image loading/resize differs from Python.
If CHECK 2 (tokens) cosine < 0.99 → VAE encoder output differs.
  → If CHECK 1 is good but CHECK 2 bad → encoder weights or graph issue.
  → If CHECK 1 is bad → image_loading.zig resize/normalize bug.
If CHECK 3 (latent first 384) != CHECK 2 → Conditioning injection bug.
If CHECK 3 (latent rest) cosine < 0.999 → Noise initialization differs.
If CHECK 5 (mask) doesn't have 0.0 for first n_img tokens → Mask bug.
""")


if __name__ == "__main__":
    main()
