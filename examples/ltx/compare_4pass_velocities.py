#!/usr/bin/env python3
"""Compare the 4-pass velocity dumps at step 0 to diagnose guidance issues.

Loads:
  step0_video_vel.bin        (cond)
  step0_neg_video_vel.bin    (neg/CFG)
  step0_ptb_video_vel.bin    (ptb/STG)
  step0_iso_video_vel.bin    (iso/modality)
  step0_guided_video_vel.bin (guider combine output)
  step0_pre_out_vx.bin            (patchified latent after pass 1)
  step0_pre_out_vx_after_pass2.bin (patchified latent after pass 2)
  + audio equivalents

Usage:
  python compare_4pass_velocities.py --zig-dir /path/to/stage1_out
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch


def load_bin_bf16(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    n = len(raw) // 2
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return torch.from_numpy(u16.copy()).view(torch.bfloat16).float()


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().double()
    b_flat = b.flatten().double()
    return (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm() + 1e-12)).item()


def summarize_pair(name_a: str, name_b: str, a: torch.Tensor, b: torch.Tensor):
    diff = (a - b).abs()
    cs = cos_sim(a, b)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    bit_identical = torch.equal(a.half(), b.half())  # fp16 comparison
    raw_identical = torch.equal(
        a.to(torch.bfloat16).view(torch.int16),
        b.to(torch.bfloat16).view(torch.int16),
    )
    print(f"  {name_a} vs {name_b}:")
    print(f"    cos_sim     = {cs:.6f}")
    print(f"    max_abs_err = {max_abs:.6f}")
    print(f"    mean_abs    = {mean_abs:.6f}")
    print(f"    bit-identical (bf16) = {raw_identical}")
    if cs > 0.9999:
        print(f"    --> NEARLY IDENTICAL (guidance term ≈ 0)")
    elif cs > 0.99:
        print(f"    --> SIMILAR (small guidance delta)")
    else:
        print(f"    --> DIFFERENT (guidance working)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zig-dir", required=True, help="Directory with step0 dumps")
    args = parser.parse_args()
    d = Path(args.zig_dir)

    for modality in ["video", "audio"]:
        print(f"\n{'='*60}")
        print(f" {modality.upper()} velocities at step 0")
        print(f"{'='*60}")

        cond_path = d / f"step0_{modality}_vel.bin"
        neg_path = d / f"step0_neg_{modality}_vel.bin"
        ptb_path = d / f"step0_ptb_{modality}_vel.bin"
        iso_path = d / f"step0_iso_{modality}_vel.bin"
        guided_path = d / f"step0_guided_{modality}_vel.bin"

        missing = [p for p in [cond_path, neg_path, ptb_path, iso_path, guided_path] if not p.exists()]
        if missing:
            print(f"  Missing files: {[str(m) for m in missing]}")
            continue

        cond = load_bin_bf16(cond_path)
        neg = load_bin_bf16(neg_path)
        ptb = load_bin_bf16(ptb_path)
        iso = load_bin_bf16(iso_path)
        guided = load_bin_bf16(guided_path)

        print(f"\n  cond   range: [{cond.min():.4f}, {cond.max():.4f}]  std={cond.std():.4f}")
        print(f"  neg    range: [{neg.min():.4f}, {neg.max():.4f}]  std={neg.std():.4f}")
        print(f"  ptb    range: [{ptb.min():.4f}, {ptb.max():.4f}]  std={ptb.std():.4f}")
        print(f"  iso    range: [{iso.min():.4f}, {iso.max():.4f}]  std={iso.std():.4f}")
        print(f"  guided range: [{guided.min():.4f}, {guided.max():.4f}]  std={guided.std():.4f}")
        print()

        # Pairwise comparisons — these are the guidance deltas
        summarize_pair("cond", "neg", cond, neg)
        summarize_pair("cond", "ptb", cond, ptb)
        summarize_pair("cond", "iso", cond, iso)
        summarize_pair("cond", "guided", cond, guided)
        print()

    # Check pre_out.vx donation
    vx_path1 = d / "step0_pre_out_vx.bin"
    vx_path2 = d / "step0_pre_out_vx_after_pass2.bin"
    if vx_path1.exists() and vx_path2.exists():
        print(f"\n{'='*60}")
        print(f" BUFFER DONATION CHECK: pre_out.vx before/after Pass 2")
        print(f"{'='*60}")
        vx1 = load_bin_bf16(vx_path1)
        vx2 = load_bin_bf16(vx_path2)
        summarize_pair("pre_out.vx(after_pass1)", "pre_out.vx(after_pass2)", vx1, vx2)
        if torch.equal(
            vx1.to(torch.bfloat16).view(torch.int16),
            vx2.to(torch.bfloat16).view(torch.int16),
        ):
            print("  --> pre_out.vx is STABLE (no buffer donation issue)")
        else:
            print("  --> *** pre_out.vx CHANGED! Buffer donation/corruption detected! ***")


if __name__ == "__main__":
    main()
