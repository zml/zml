#!/usr/bin/env python3
"""Compare Zig Gemma encoder output against Python reference.

Usage:
    python compare_gemma_outputs.py \
        --zig /tmp/zig_pos_hidden_states.bin \
        --ref /home/ubuntu/outputs/gemma_reference/pos_hidden_states.safetensors
"""

import argparse

import numpy as np
import torch
from safetensors.torch import load_file


def load_zig_output(path: str) -> np.ndarray:
    """Load raw bf16 binary as float32 numpy array with shape [1024, 3840, 49]."""
    raw = np.fromfile(path, dtype=np.uint16)  # bf16 stored as uint16
    # Convert bf16 to float32: shift left by 16 bits
    f32 = (raw.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(1024, 3840, 49)


def load_reference(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Load Python reference safetensors, return as (float32 [1024, 3840, 49], attention_mask or None).

    Supports both formats:
      - generate_gemma_reference.py: key "hidden_states" [1, 1024, 3840, 49]
      - export_pipeline.py --text-only: key "stacked_hidden_states" [1, 1024, 3840, 49]
    """
    tensors = load_file(path)
    if "stacked_hidden_states" in tensors:
        hs = tensors["stacked_hidden_states"]
    elif "hidden_states" in tensors:
        hs = tensors["hidden_states"]
    else:
        raise KeyError(f"Expected 'stacked_hidden_states' or 'hidden_states', got {list(tensors.keys())}")
    f32 = hs.float().numpy()  # convert bf16 → f32 via torch
    # Remove batch dim if present
    if f32.ndim == 4:
        f32 = f32[0]
    # Load attention mask if present
    mask = None
    if "attention_mask" in tensors:
        mask = tensors["attention_mask"].numpy().flatten()  # [S] with 0=pad, 1=real
    return f32, mask


def compare(zig: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None):
    """Print detailed comparison statistics."""
    assert zig.shape == ref.shape, f"Shape mismatch: zig={zig.shape}, ref={ref.shape}"
    print(f"Shape: {zig.shape}")
    print(f"Zig  range: [{zig.min():.6f}, {zig.max():.6f}], mean={zig.mean():.6f}")
    print(f"Ref  range: [{ref.min():.6f}, {ref.max():.6f}], mean={ref.mean():.6f}")

    diff = np.abs(zig - ref)
    print(f"\n--- Absolute Error ---")
    print(f"Max:    {diff.max():.6e}")
    print(f"Mean:   {diff.mean():.6e}")
    print(f"Median: {np.median(diff):.6e}")
    print(f"99th %%: {np.percentile(diff, 99):.6e}")

    # Relative error (avoid div by zero)
    denom = np.maximum(np.abs(ref), 1e-8)
    rel = diff / denom
    print(f"\n--- Relative Error ---")
    print(f"Max:    {rel.max():.6e}")
    print(f"Mean:   {rel.mean():.6e}")
    print(f"Median: {np.median(rel):.6e}")

    # Exact match ratio
    exact = np.sum(zig == ref) / zig.size * 100
    print(f"\n--- Exact Match ---")
    print(f"Exact bf16 match: {exact:.2f}%")

    # Per-layer cosine similarity
    print(f"\n--- Per-layer Cosine Similarity ---")
    for layer_idx in range(zig.shape[-1]):
        z = zig[:, :, layer_idx].flatten()
        r = ref[:, :, layer_idx].flatten()
        cos = np.dot(z, r) / (np.linalg.norm(z) * np.linalg.norm(r) + 1e-12)
        layer_diff = np.abs(z - r)
        label = "emb" if layer_idx == 0 else f"L{layer_idx}"
        print(f"  {label:>4s}: cos={cos:.8f}  max_abs_err={layer_diff.max():.4e}  mean_abs_err={layer_diff.mean():.4e}")

    # L48 diagnostics
    if zig.shape[-1] >= 49:
        print(f"\n--- L48 Diagnostics ---")
        z48 = zig[:, :, 48]
        r48 = ref[:, :, 48]
        print(f"  Zig  L48: min={z48.min():.4f}, max={z48.max():.4f}, mean={z48.mean():.4f}, std={z48.std():.4f}")
        print(f"  Ref  L48: min={r48.min():.4f}, max={r48.max():.4f}, mean={r48.mean():.4f}, std={r48.std():.4f}")
        print(f"  Zig  L48 has NaN: {np.any(np.isnan(z48))}, Inf: {np.any(np.isinf(z48))}")
        print(f"  Ref  L48 has NaN: {np.any(np.isnan(r48))}, Inf: {np.any(np.isinf(r48))}")

        # Check L2 norms: if ref L48 is normed but zig L48 is raw, norms will differ
        z47 = zig[:, :, 47]
        r47 = ref[:, :, 47]
        print(f"\n  L47 norms: zig={np.linalg.norm(z47):.4f}, ref={np.linalg.norm(r47):.4f}, ratio={np.linalg.norm(z47)/np.linalg.norm(r47):.6f}")
        print(f"  L48 norms: zig={np.linalg.norm(z48):.4f}, ref={np.linalg.norm(r48):.4f}, ratio={np.linalg.norm(z48)/np.linalg.norm(r48):.6f}")

        # Check if error is concentrated in padding vs real tokens
        # Use attention mask if available, otherwise fall back to heuristic
        if mask is not None:
            pad_positions = mask == 0
            real_positions = mask == 1
            num_pad = int(pad_positions.sum())
            num_real = int(real_positions.sum())
        else:
            # Fallback: assume left-padded, detect boundary from L0 (embedding)
            num_real = 7  # default for "A beautiful sunset over the ocean"
            num_pad = zig.shape[0] - num_real
            pad_positions = np.zeros(zig.shape[0], dtype=bool)
            pad_positions[:num_pad] = True
            real_positions = ~pad_positions

        pad_err = np.abs(z48[pad_positions, :] - r48[pad_positions, :])
        real_err = np.abs(z48[real_positions, :] - r48[real_positions, :])
        print(f"\n  Padding positions ({num_pad} tokens): mean_err={pad_err.mean():.4e}, max_err={pad_err.max():.4e}")
        print(f"  Real token positions ({num_real} tokens): mean_err={real_err.mean():.4e}, max_err={real_err.max():.4e}")

        # Show where the worst errors are
        flat_diff = np.abs(z48 - r48).flatten()
        worst_indices = np.argsort(flat_diff)[-10:][::-1]
        print(f"\n  Top 10 worst element errors in L48:")
        for idx in worst_indices:
            s_idx = idx // 3840
            d_idx = idx % 3840
            print(f"    [{s_idx:4d}, {d_idx:4d}]: zig={z48[s_idx, d_idx]:12.4f}  ref={r48[s_idx, d_idx]:12.4f}  diff={flat_diff[idx]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compare Zig vs Python Gemma outputs")
    parser.add_argument("--zig", required=True, help="Path to Zig raw bf16 binary output")
    parser.add_argument("--ref", required=True, help="Path to Python reference safetensors")
    args = parser.parse_args()

    print(f"Loading Zig output: {args.zig}")
    zig = load_zig_output(args.zig)

    print(f"Loading reference:  {args.ref}")
    ref, mask = load_reference(args.ref)

    num_real = int(mask.sum()) if mask is not None else "unknown"
    print(f"Attention mask: {num_real} real tokens out of {ref.shape[0]}")

    print()
    compare(zig, ref, mask)


if __name__ == "__main__":
    main()
