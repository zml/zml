#!/usr/bin/env python3
"""Analyze per-step error from the reset test.

Compares Zig's output latent at each step (starting from Python's input)
with Python's output latent at the same step.

This isolates the per-step error: since both start from the same latent,
any difference is purely from XLA vs CUDA backend + numerical pipeline
differences, NOT from accumulation.

Modes:
  --mode=reset   Compare per-step Zig outputs (reset) vs Python
  --mode=freerun Compare Zig free-running outputs vs Python (accumulation curve)
  --mode=both    Show both (default)

Usage:
    python compare_reset_test.py \\
        --zig-dir /root/e2e_demo/stage1_out_reset \\
        --ref /root/e2e_demo/all_step_latents.safetensors \\
        [--zig-freerun-dir /root/e2e_demo/stage1_out] \\
        [--mode both]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def load_bin(path: str, shape: list[int], dtype=torch.bfloat16) -> torch.Tensor:
    data = Path(path).read_bytes()
    if dtype == torch.bfloat16:
        t = torch.from_numpy(np.frombuffer(data, dtype=np.uint16).copy()).view(torch.bfloat16)
    elif dtype == torch.float32:
        t = torch.from_numpy(np.frombuffer(data, dtype=np.float32).copy())
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return t.reshape(shape)


def metrics(zig: torch.Tensor, ref: torch.Tensor):
    z = zig.float().flatten()
    r = ref.float().flatten()
    diff = (z - r).abs()
    cos = torch.nn.functional.cosine_similarity(z, r, dim=0).item()
    mae = diff.mean().item()
    rmse = diff.pow(2).mean().sqrt().item()
    maxe = diff.max().item()
    return cos, mae, rmse, maxe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zig-dir", required=True,
                   help="Dir with reset-mode Zig outputs (step{i}_video_latent.bin)")
    p.add_argument("--ref", required=True,
                   help="Python all_step_latents.safetensors")
    p.add_argument("--zig-freerun-dir", default=None,
                   help="Dir with free-running Zig outputs (for accumulation curve)")
    p.add_argument("--mode", default="both", choices=["reset", "freerun", "both"])
    args = p.parse_args()

    zig_dir = Path(args.zig_dir)
    ref = load_file(args.ref)

    V_SHAPE = [1, 6144, 128]
    A_SHAPE = [1, 126, 128]

    num_steps = 30

    # ── Reset test: per-step isolated error ───────────────────────────────
    if args.mode in ("reset", "both"):
        print("=" * 70)
        print("  RESET TEST: Per-step isolated error")
        print("  (Zig starts from Python's latent at each step)")
        print("=" * 70)
        print(f"\n{'Step':>5s}  {'cos_sim_v':>10s}  {'mae_v':>10s}  {'rmse_v':>10s}  "
              f"{'cos_sim_a':>10s}  {'mae_a':>10s}  {'rmse_a':>10s}")
        print("-" * 75)

        reset_v_maes = []
        reset_a_maes = []

        for step_idx in range(num_steps):
            out_step = step_idx + 1
            zig_v_path = zig_dir / f"step{out_step}_video_latent.bin"
            zig_a_path = zig_dir / f"step{out_step}_audio_latent.bin"
            ref_v_key = f"v_lat_{out_step}"
            ref_a_key = f"a_lat_{out_step}"

            if not zig_v_path.exists() or ref_v_key not in ref:
                continue

            zig_v = load_bin(str(zig_v_path), V_SHAPE)
            zig_a = load_bin(str(zig_a_path), A_SHAPE)
            ref_v = ref[ref_v_key]
            ref_a = ref[ref_a_key]

            cos_v, mae_v, rmse_v, _ = metrics(zig_v, ref_v)
            cos_a, mae_a, rmse_a, _ = metrics(zig_a, ref_a)

            reset_v_maes.append(mae_v)
            reset_a_maes.append(mae_a)

            print(f"{out_step:5d}  {cos_v:10.6f}  {mae_v:10.6f}  {rmse_v:10.6f}  "
                  f"{cos_a:10.6f}  {mae_a:10.6f}  {rmse_a:10.6f}")

        if reset_v_maes:
            avg_v = sum(reset_v_maes) / len(reset_v_maes)
            avg_a = sum(reset_a_maes) / len(reset_a_maes)
            std_v = (sum((x - avg_v)**2 for x in reset_v_maes) / len(reset_v_maes)) ** 0.5
            std_a = (sum((x - avg_a)**2 for x in reset_a_maes) / len(reset_a_maes)) ** 0.5
            print("-" * 75)
            print(f"{'AVG':>5s}  {'':>10s}  {avg_v:10.6f}  {'':>10s}  "
                  f"{'':>10s}  {avg_a:10.6f}")
            print(f"{'STD':>5s}  {'':>10s}  {std_v:10.6f}  {'':>10s}  "
                  f"{'':>10s}  {std_a:10.6f}")
            print(f"\nIf STD is small relative to AVG, the per-step error is constant")
            print(f"→ proves divergence is from accumulation, not a growing bug.")

    # ── Free-run test: accumulation curve ─────────────────────────────────
    freerun_dir = Path(args.zig_freerun_dir) if args.zig_freerun_dir else zig_dir
    if args.mode in ("freerun", "both"):
        print("\n" + "=" * 70)
        print("  FREE-RUN TEST: Accumulated error growth")
        print("  (Zig uses its own latents, no reset)")
        print("=" * 70)
        print(f"\n{'Step':>5s}  {'cos_sim_v':>10s}  {'mae_v':>10s}  {'rmse_v':>10s}  "
              f"{'cos_sim_a':>10s}  {'mae_a':>10s}  {'rmse_a':>10s}")
        print("-" * 75)

        for step_idx in range(num_steps):
            out_step = step_idx + 1
            zig_v_path = freerun_dir / f"step{out_step}_video_latent.bin"
            zig_a_path = freerun_dir / f"step{out_step}_audio_latent.bin"
            ref_v_key = f"v_lat_{out_step}"
            ref_a_key = f"a_lat_{out_step}"

            if not zig_v_path.exists() or ref_v_key not in ref:
                continue

            zig_v = load_bin(str(zig_v_path), V_SHAPE)
            zig_a = load_bin(str(zig_a_path), A_SHAPE)
            ref_v = ref[ref_v_key]
            ref_a = ref[ref_a_key]

            cos_v, mae_v, rmse_v, _ = metrics(zig_v, ref_v)
            cos_a, mae_a, rmse_a, _ = metrics(zig_a, ref_a)

            print(f"{out_step:5d}  {cos_v:10.6f}  {mae_v:10.6f}  {rmse_v:10.6f}  "
                  f"{cos_a:10.6f}  {mae_a:10.6f}  {rmse_a:10.6f}")

    # ── Summary ───────────────────────────────────────────────────────────
    if args.mode in ("reset", "both") and reset_v_maes:
        print("\n" + "=" * 70)
        print("  CONCLUSION")
        print("=" * 70)
        ratio = std_v / avg_v if avg_v > 0 else 0
        if ratio < 0.3:
            print(f"\n  Per-step error is CONSTANT (CoV={ratio:.2f} < 0.3)")
            print(f"  → The 30-step divergence is purely from iterative accumulation.")
            print(f"  → No systematic Zig bug. Backend numerical differences compound")
            print(f"    through the nonlinear transformer + guidance amplification.")
        else:
            print(f"\n  Per-step error VARIES (CoV={ratio:.2f} >= 0.3)")
            print(f"  → There may be a step-dependent or state-dependent issue.")
            print(f"  → Investigate steps with anomalously high error.")


if __name__ == "__main__":
    main()
