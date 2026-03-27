#!/usr/bin/env python3
"""Compare Zig outputs against Python reference (safetensors).

Supports two modes:
  1. Step-0 velocity comparison (default)
  2. Full 30-step final latent comparison (--full)

Reports both cosine similarity AND expectClose metrics (matching zml.testing.expectClose):
  close_fraction = fraction of elements where |a-b| <= atol + rtol * max(|a|, |b|)

Usage:
    # Step-0 only:
    python compare_step0_velocity.py \
        --zig-dir /root/e2e_demo/stage1_out \
        --ref /root/e2e_demo/stage1_step0_reference.safetensors

    # Full 30-step:
    python compare_step0_velocity.py --full \
        --zig-dir /root/e2e_demo/stage1_out \
        --ref /root/e2e_demo/stage1_step0_reference.safetensors
"""
import argparse
from pathlib import Path

import torch
import numpy as np
from safetensors.torch import load_file


def load_bf16_bin(path, shape):
    """Load raw bf16 binary file into a torch tensor."""
    raw = np.fromfile(path, dtype=np.uint16)
    t = torch.from_numpy(raw.copy()).view(torch.bfloat16).reshape(shape)
    return t


def expect_close(a, b, atol=1e-3, rtol=1e-2):
    """Compute expectClose metrics matching zml.testing.expectClose semantics."""
    af, bf = a.float().flatten(), b.float().flatten()

    abs_err = (af - bf).abs()
    scale = torch.max(af.abs(), bf.abs())
    tolerance = atol + rtol * scale
    close_mask = abs_err <= tolerance
    close_fraction = close_mask.float().mean().item()

    sorted_err, _ = abs_err.sort()
    n = len(sorted_err)

    def pct(frac):
        idx = min(int(round((n - 1) * frac)), n - 1)
        return sorted_err[idx].item()

    return {
        "close_fraction": close_fraction,
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "rmse": abs_err.pow(2).mean().sqrt().item(),
        "p50": pct(0.5),
        "p90": pct(0.9),
        "p99": pct(0.99),
        "p999": pct(0.999),
    }


def compare(name, zig, ref, atol=1e-3, rtol=1e-2, min_close=0.999):
    """Full comparison: cosine similarity + expectClose metrics."""
    zf, rf = zig.float().flatten(), ref.float().flatten()
    cos = torch.nn.functional.cosine_similarity(zf, rf, dim=0).item()
    ec = expect_close(zig, ref, atol=atol, rtol=rtol)

    cos_ok = cos >= 0.999
    close_ok = ec["close_fraction"] >= min_close
    status = "PASS" if (cos_ok and close_ok) else "FAIL"

    print(f"  {name}:")
    print(f"    cos_sim        = {cos:.6f}  [{'ok' if cos_ok else 'FAIL'}]")
    print(f"    close_fraction = {ec['close_fraction']:.6f}  [{'ok' if close_ok else 'FAIL'}]  "
          f"(atol={atol}, rtol={rtol}, min={min_close})")
    print(f"    max_abs_err    = {ec['max_abs_err']:.6f}")
    print(f"    mean_abs_err   = {ec['mean_abs_err']:.6f}")
    print(f"    rmse           = {ec['rmse']:.6f}")
    print(f"    p50={ec['p50']:.6f}  p90={ec['p90']:.6f}  "
          f"p99={ec['p99']:.6f}  p999={ec['p999']:.6f}")
    print(f"    zig  range     = [{zig.float().min():.4f}, {zig.float().max():.4f}]")
    print(f"    ref  range     = [{ref.float().min():.4f}, {ref.float().max():.4f}]")
    print(f"    --> {status}")
    return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zig-dir", required=True,
                   help="Dir with step0_video_vel.bin / step0_audio_vel.bin / video_latent.bin / audio_latent.bin")
    p.add_argument("--ref", required=True,
                   help="Path to stage1_step0_reference.safetensors")
    p.add_argument("--full", action="store_true",
                   help="Also compare final latents after 30 steps")
    p.add_argument("--atol", type=float, default=0.2,
                   help="Absolute tolerance (default: 0.2, matching Step 1 validation)")
    p.add_argument("--rtol", type=float, default=0.01,
                   help="Relative tolerance (default: 0.01)")
    p.add_argument("--min-close", type=float, default=0.999,
                   help="Minimum close fraction (default: 0.999)")
    args = p.parse_args()

    zig_dir = Path(args.zig_dir)
    ref = load_file(args.ref)
    all_statuses = []

    # ── Step-0 velocity comparison ──
    if "step0_video_vel" in ref:
        v_shape = list(ref["step0_video_vel"].shape)
        a_shape = list(ref["step0_audio_vel"].shape)
        print(f"Step-0 velocity shapes: video={v_shape}, audio={a_shape}")

        zig_v = load_bf16_bin(zig_dir / "step0_video_vel.bin", v_shape)
        zig_a = load_bf16_bin(zig_dir / "step0_audio_vel.bin", a_shape)

        print(f"\n── Step-0 Raw Velocity: Zig vs Python ──")
        print(f"   expectClose(atol={args.atol}, rtol={args.rtol}, min_close={args.min_close})")
        print()
        all_statuses.append(compare("video_vel", zig_v, ref["step0_video_vel"],
                           atol=args.atol, rtol=args.rtol, min_close=args.min_close))
        print()
        all_statuses.append(compare("audio_vel", zig_a, ref["step0_audio_vel"],
                           atol=args.atol, rtol=args.rtol, min_close=args.min_close))
    else:
        print("(No step-0 velocity in reference — skipping)")

    # ── Final latent comparison (30-step) ──
    if args.full:
        if "final_video_latent" not in ref:
            print("\nERROR: --full requested but reference has no final_video_latent.")
            print("       Re-run export with --full to capture 30-step reference.")
            return

        # ── Intermediate step comparison (error growth curve) ──
        for step_label in (1, 5, 15):
            vkey = f"step{step_label}_video_latent"
            akey = f"step{step_label}_audio_latent"
            vbin = zig_dir / f"step{step_label}_video_latent.bin"
            if vkey in ref and vbin.exists():
                v_shape = list(ref[vkey].shape)
                a_shape = list(ref[akey].shape)
                print(f"\n\n── Step-{step_label} Latent: Zig vs Python ──")
                print(f"   Shapes: video={v_shape}, audio={a_shape}")
                zig_sv = load_bf16_bin(vbin, v_shape)
                zig_sa = load_bf16_bin(zig_dir / f"step{step_label}_audio_latent.bin", a_shape)
                print()
                compare(f"step{step_label}_video", zig_sv, ref[vkey],
                        atol=args.atol, rtol=args.rtol, min_close=args.min_close)
                print()
                compare(f"step{step_label}_audio", zig_sa, ref[akey],
                        atol=args.atol, rtol=args.rtol, min_close=args.min_close)

        fv_shape = list(ref["final_video_latent"].shape)
        fa_shape = list(ref["final_audio_latent"].shape)
        print(f"\n\n── Final Latent (30-step): Zig vs Python ──")
        print(f"   Shapes: video={fv_shape}, audio={fa_shape}")
        print(f"   expectClose(atol={args.atol}, rtol={args.rtol}, min_close={args.min_close})")

        zig_fv = load_bf16_bin(zig_dir / "video_latent.bin", fv_shape)
        zig_fa = load_bf16_bin(zig_dir / "audio_latent.bin", fa_shape)

        print()
        all_statuses.append(compare("final_video_latent", zig_fv, ref["final_video_latent"],
                           atol=args.atol, rtol=args.rtol, min_close=args.min_close))
        print()
        all_statuses.append(compare("final_audio_latent", zig_fa, ref["final_audio_latent"],
                           atol=args.atol, rtol=args.rtol, min_close=args.min_close))

    # ── Overall ──
    all_pass = all(s == "PASS" for s in all_statuses)
    print(f"\n── Overall: {'ALL PASS' if all_pass else 'FAIL'} ──")


if __name__ == "__main__":
    main()
