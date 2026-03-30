#!/usr/bin/env python3
"""Compare Zig step-0 per-pass outputs vs Python per-pass reference.

Reads:
  - Zig .bin files from --zig-dir (raw bf16/f32 binary dumps)
  - Python reference from --ref (safetensors)

Usage:
    python compare_step0_perpass.py \\
        --zig-dir /root/e2e_demo/stage1_out \\
        --ref /root/e2e_demo/step0_perpass_reference.safetensors
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def load_bin(path: str, shape: list[int], dtype=torch.bfloat16) -> torch.Tensor:
    """Load raw binary dump as torch tensor."""
    data = Path(path).read_bytes()
    if dtype == torch.bfloat16:
        np_dtype = np.uint16  # bf16 stored as raw uint16
        t = torch.from_numpy(np.frombuffer(data, dtype=np_dtype).copy()).view(torch.bfloat16)
    elif dtype == torch.float32:
        t = torch.from_numpy(np.frombuffer(data, dtype=np.float32).copy())
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return t.reshape(shape)


def compare(label: str, zig: torch.Tensor, ref: torch.Tensor,
            atol=0.2, rtol=0.01, min_close=0.999):
    """Compare two tensors and print diagnostics."""
    z = zig.float().flatten()
    r = ref.float().flatten()
    diff = (z - r).abs()

    cos = torch.nn.functional.cosine_similarity(z, r, dim=0).item()
    close = ((diff <= atol + rtol * r.abs()).sum().item()) / z.numel()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rmse = diff.pow(2).mean().sqrt().item()

    cos_ok = cos >= min_close
    close_ok = close >= min_close
    passed = cos_ok and close_ok

    status = "PASS" if passed else "FAIL"

    print(f"\n  {label}:")
    cos_tag = "[ok]" if cos_ok else "[FAIL]"
    close_tag = "[ok]" if close_ok else "[FAIL]"
    print(f"    cos_sim        = {cos:.6f}  {cos_tag}")
    print(f"    close_fraction = {close:.6f}  {close_tag}  (atol={atol}, rtol={rtol}, min={min_close})")
    print(f"    max_abs_err    = {max_err:.6f}")
    print(f"    mean_abs_err   = {mean_err:.6f}")
    print(f"    rmse           = {rmse:.6f}")
    p50 = diff.quantile(0.50).item()
    p90 = diff.quantile(0.90).item()
    p99 = diff.quantile(0.99).item()
    p999 = diff.quantile(0.999).item()
    print(f"    p50={p50:.6f}  p90={p90:.6f}  p99={p99:.6f}  p999={p999:.6f}")
    print(f"    zig  range     = [{z.min():.4f}, {z.max():.4f}]")
    print(f"    ref  range     = [{r.min():.4f}, {r.max():.4f}]")
    print(f"    --> {status}")

    return passed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zig-dir", required=True, help="Directory with Zig .bin dumps")
    p.add_argument("--ref", required=True, help="Python per-pass reference safetensors")
    args = p.parse_args()

    zig_dir = Path(args.zig_dir)
    ref = load_file(args.ref)

    print(f"Reference keys: {sorted(ref.keys())}")
    print(f"Zig dir: {zig_dir}")
    print(f"Zig files: {sorted(f.name for f in zig_dir.glob('step0_*.bin'))}")

    # Shapes
    V_SHAPE = [1, 6144, 128]
    A_SHAPE = [1, 126, 128]

    all_pass = True

    # ── Per-pass raw velocities ───────────────────────────────────────────
    # Zig dumps raw velocity before to_denoised conversion.
    # Python reference has step0_{pass}_{mod}_vel
    pass_map = {
        "cond": ("step0_video_vel.bin", "step0_audio_vel.bin"),
        "neg":  ("step0_neg_video_vel.bin", "step0_neg_audio_vel.bin"),
        "ptb":  ("step0_ptb_video_vel.bin", "step0_ptb_audio_vel.bin"),
        "iso":  ("step0_iso_video_vel.bin", "step0_iso_audio_vel.bin"),
    }

    for pass_name, (zig_v_file, zig_a_file) in pass_map.items():
        ref_v_key = f"step0_{pass_name}_video_vel"
        ref_a_key = f"step0_{pass_name}_audio_vel"

        zig_v_path = zig_dir / zig_v_file
        zig_a_path = zig_dir / zig_a_file

        print(f"\n── Pass: {pass_name.upper()} (raw velocity) ──")

        if zig_v_path.exists() and ref_v_key in ref:
            zig_v = load_bin(str(zig_v_path), V_SHAPE)
            ok = compare(f"{pass_name}_video_vel", zig_v, ref[ref_v_key])
            all_pass = all_pass and ok
        else:
            print(f"  SKIP video: zig={zig_v_path.exists()}, ref={ref_v_key in ref}")

        if zig_a_path.exists() and ref_a_key in ref:
            zig_a = load_bin(str(zig_a_path), A_SHAPE)
            ok = compare(f"{pass_name}_audio_vel", zig_a, ref[ref_a_key])
            all_pass = all_pass and ok
        else:
            print(f"  SKIP audio: zig={zig_a_path.exists()}, ref={ref_a_key in ref}")

    # ── Guided x0 (after guider combine) ──────────────────────────────────
    print(f"\n── Guided x0 (after guider combine) ──")

    zig_guided_v = zig_dir / "step0_guided_video_vel.bin"
    zig_guided_a = zig_dir / "step0_guided_audio_vel.bin"

    if zig_guided_v.exists() and "step0_video_guided_x0" in ref:
        zig_v = load_bin(str(zig_guided_v), V_SHAPE)
        ok = compare("guided_video_x0", zig_v, ref["step0_video_guided_x0"])
        all_pass = all_pass and ok

    if zig_guided_a.exists() and "step0_audio_guided_x0" in ref:
        zig_a = load_bin(str(zig_guided_a), A_SHAPE)
        ok = compare("guided_audio_x0", zig_a, ref["step0_audio_guided_x0"])
        all_pass = all_pass and ok

    # ── Step-1 latent (after Euler) ───────────────────────────────────────
    print(f"\n── Step-1 latent (after Euler) ──")

    zig_step1_v = zig_dir / "step1_video_latent.bin"
    zig_step1_a = zig_dir / "step1_audio_latent.bin"

    if zig_step1_v.exists() and "step1_video_latent" in ref:
        zig_v = load_bin(str(zig_step1_v), V_SHAPE)
        ok = compare("step1_video_latent", zig_v, ref["step1_video_latent"])
        all_pass = all_pass and ok

    if zig_step1_a.exists() and "step1_audio_latent" in ref:
        zig_a = load_bin(str(zig_step1_a), A_SHAPE)
        ok = compare("step1_audio_latent", zig_a, ref["step1_audio_latent"])
        all_pass = all_pass and ok

    # ── Error budget breakdown ────────────────────────────────────────────
    print(f"\n── Error Budget Breakdown (video, mean_abs_err) ──")
    print(f"  This shows where the per-step error originates.\n")

    # Compute per-pass contribution to total
    budget = []
    for pass_name, (zig_v_file, _) in pass_map.items():
        ref_key = f"step0_{pass_name}_video_vel"
        zig_path = zig_dir / zig_v_file
        if zig_path.exists() and ref_key in ref:
            zig_v = load_bin(str(zig_path), V_SHAPE)
            mae = (zig_v.float() - ref[ref_key].float()).abs().mean().item()
            budget.append((pass_name, mae))

    if zig_guided_v.exists() and "step0_video_guided_x0" in ref:
        zig_v = load_bin(str(zig_guided_v), V_SHAPE)
        mae = (zig_v.float() - ref["step0_video_guided_x0"].float()).abs().mean().item()
        budget.append(("guided_x0", mae))

    if zig_step1_v.exists() and "step1_video_latent" in ref:
        zig_v = load_bin(str(zig_step1_v), V_SHAPE)
        mae = (zig_v.float() - ref["step1_video_latent"].float()).abs().mean().item()
        budget.append(("step1_lat", mae))

    for name, mae in budget:
        bar = "█" * int(mae * 500)
        print(f"  {name:12s}  mae={mae:.6f}  {bar}")

    # ── Overall ───────────────────────────────────────────────────────────
    print(f"\n── Overall: {'PASS' if all_pass else 'FAIL'} ──")


if __name__ == "__main__":
    main()
