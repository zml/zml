#!/usr/bin/env python3
"""Replay Zig's step-0 computation offline and compare with Zig's step-1 latent.

This script manually computes:
  1. vel→x0 conversion for all 4 passes
  2. Guider combine on x0 predictions
  3. Euler step from guided x0
Then compares step-1 latent with what Zig dumped.

This isolates whether the per-step computation is correct.

Usage:
  python verify_step1_offline.py \
      --zig-dir /path/to/stage1_out \
      --inputs /path/to/stage1_inputs.safetensors
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def load_bin_bf16(path: Path, shape=None) -> torch.Tensor:
    raw = path.read_bytes()
    n = len(raw) // 2
    u16 = np.frombuffer(raw, dtype=np.uint16)
    t = torch.from_numpy(u16.copy()).view(torch.bfloat16)
    if shape:
        t = t.reshape(shape)
    return t


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten().double()
    b_flat = b.flatten().double()
    return (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm() + 1e-12)).item()


def summarize(name, t):
    f = t.float()
    print(f"  {name}: range=[{f.min():.4f}, {f.max():.4f}]  std={f.std():.4f}  mean={f.mean():.4f}")


def compare(name, a, b):
    diff = (a.float() - b.float()).abs()
    cs = cos_sim(a, b)
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"  {name}:")
    print(f"    cos_sim     = {cs:.6f}")
    print(f"    max_abs_err = {max_abs:.6f}")
    print(f"    mean_abs    = {mean_abs:.6f}")
    print(f"    a range = [{a.float().min():.4f}, {a.float().max():.4f}]")
    print(f"    b range = [{b.float().min():.4f}, {b.float().max():.4f}]")
    if cs > 0.9999:
        print(f"    --> MATCH")
    elif cs > 0.99:
        print(f"    --> CLOSE")
    else:
        print(f"    --> DIVERGE")


def guider_combine_single(cond, neg, ptb, iso, cfg, stg, mod, rescale):
    """Exact replica of model.zig guiderCombineSingle."""
    cond_f = cond.float()
    neg_f = neg.float()
    ptb_f = ptb.float()
    iso_f = iso.float()

    cfg_term = (cfg - 1.0) * (cond_f - neg_f)
    stg_term = stg * (cond_f - ptb_f)
    mod_term = (mod - 1.0) * (cond_f - iso_f)
    pred = cond_f + cfg_term + stg_term + mod_term

    # std-rescale
    cond_std = cond_f.std()
    pred_std = pred.std()
    eps = 1e-8
    ratio = cond_std / (pred_std + eps)
    factor = rescale * ratio + (1.0 - rescale)
    pred = pred * factor

    return pred.to(torch.bfloat16)


def to_denoised(sample, velocity, mask, sigma):
    """Exact replica of model.zig forwardToDenoised."""
    sample_f = sample.float()
    vel_f = velocity.float()
    mask_f = mask.float()
    timesteps = mask_f * sigma
    return (sample_f - vel_f * timesteps).to(torch.bfloat16)


def denoising_step_from_x0(sample, guided_x0, mask, clean, sigma, sigma_next):
    """Exact replica of model.zig forwardDenoisingStepFromX0."""
    sample_f = sample.float()
    clean_f = clean.float()
    mask_f = mask.float()
    one_minus_mask = 1.0 - mask_f

    # post_process_latent
    blended_f = guided_x0.float() * mask_f + clean_f * one_minus_mask
    blended = blended_f.to(torch.bfloat16)

    # Euler step
    dt = sigma_next - sigma
    euler_vel_f = (sample_f - blended.float()) / sigma
    euler_vel = euler_vel_f.to(torch.bfloat16)
    next_f = sample_f + euler_vel.float() * dt
    next_latent = next_f.to(torch.bfloat16)

    return next_latent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zig-dir", required=True)
    parser.add_argument("--inputs", required=True)
    args = parser.parse_args()
    d = Path(args.zig_dir)

    # Load inputs
    inputs = load_file(args.inputs)
    v_sample = inputs["video_latent"]   # [1, 6144, 128] bf16
    a_sample = inputs["audio_latent"]   # [1, 126, 128] bf16
    v_clean = inputs["video_clean_latent"]     # [1, 6144, 128] bf16
    a_clean = inputs["audio_clean_latent"]     # [1, 126, 128] bf16
    v_mask = inputs["video_denoise_mask"]  # [1, 6144, 1] bf16
    a_mask = inputs["audio_denoise_mask"]  # [1, 126, 1] bf16

    print("=== Input shapes ===")
    print(f"  v_sample: {list(v_sample.shape)} {v_sample.dtype}")
    print(f"  v_clean:  {list(v_clean.shape)} {v_clean.dtype}")
    print(f"  v_mask:   {list(v_mask.shape)} {v_mask.dtype}")
    print(f"  v_mask unique values: {v_mask.float().unique().tolist()}")
    print(f"  a_sample: {list(a_sample.shape)} {a_sample.dtype}")
    print(f"  a_clean:  {list(a_clean.shape)} {a_clean.dtype}")
    print(f"  a_mask:   {list(a_mask.shape)} {a_mask.dtype}")
    print(f"  a_mask unique values: {a_mask.float().unique().tolist()}")

    # Sigma schedule step 0
    sigma = 1.0
    sigma_next = 0.9949573278427124  # from the log output

    print(f"\n  sigma={sigma}, sigma_next={sigma_next}")
    summarize("v_sample", v_sample)
    summarize("a_sample", a_sample)

    # Load Zig velocity dumps
    v_shape = list(v_sample.shape)
    a_shape = list(a_sample.shape)

    cond_v_vel = load_bin_bf16(d / "step0_video_vel.bin", v_shape)
    neg_v_vel = load_bin_bf16(d / "step0_neg_video_vel.bin", v_shape)
    ptb_v_vel = load_bin_bf16(d / "step0_ptb_video_vel.bin", v_shape)
    iso_v_vel = load_bin_bf16(d / "step0_iso_video_vel.bin", v_shape)

    cond_a_vel = load_bin_bf16(d / "step0_audio_vel.bin", a_shape)
    neg_a_vel = load_bin_bf16(d / "step0_neg_audio_vel.bin", a_shape)
    ptb_a_vel = load_bin_bf16(d / "step0_ptb_audio_vel.bin", a_shape)
    iso_a_vel = load_bin_bf16(d / "step0_iso_audio_vel.bin", a_shape)

    print("\n=== Step 1: vel→x0 conversion ===")
    cond_v_x0 = to_denoised(v_sample, cond_v_vel, v_mask, sigma)
    neg_v_x0 = to_denoised(v_sample, neg_v_vel, v_mask, sigma)
    ptb_v_x0 = to_denoised(v_sample, ptb_v_vel, v_mask, sigma)
    iso_v_x0 = to_denoised(v_sample, iso_v_vel, v_mask, sigma)

    cond_a_x0 = to_denoised(a_sample, cond_a_vel, a_mask, sigma)
    neg_a_x0 = to_denoised(a_sample, neg_a_vel, a_mask, sigma)
    ptb_a_x0 = to_denoised(a_sample, ptb_a_vel, a_mask, sigma)
    iso_a_x0 = to_denoised(a_sample, iso_a_vel, a_mask, sigma)

    print("  Video x0 predictions:")
    summarize("    cond_x0", cond_v_x0)
    summarize("    neg_x0 ", neg_v_x0)
    summarize("    ptb_x0 ", ptb_v_x0)
    summarize("    iso_x0 ", iso_v_x0)
    print("  Audio x0 predictions:")
    summarize("    cond_x0", cond_a_x0)
    summarize("    neg_x0 ", neg_a_x0)
    summarize("    ptb_x0 ", ptb_a_x0)
    summarize("    iso_x0 ", iso_a_x0)

    print("\n=== Step 2: Guider combine (on x0) ===")
    # Video: cfg=3.0, stg=1.0, mod=3.0, rescale=0.7
    guided_v_x0 = guider_combine_single(cond_v_x0, neg_v_x0, ptb_v_x0, iso_v_x0,
                                          cfg=3.0, stg=1.0, mod=3.0, rescale=0.7)
    # Audio: cfg=7.0, stg=1.0, mod=3.0, rescale=0.7
    guided_a_x0 = guider_combine_single(cond_a_x0, neg_a_x0, ptb_a_x0, iso_a_x0,
                                          cfg=7.0, stg=1.0, mod=3.0, rescale=0.7)

    summarize("  guided_v_x0 (computed)", guided_v_x0)
    summarize("  guided_a_x0 (computed)", guided_a_x0)

    # Compare with Zig's dumped guided x0
    zig_guided_v = load_bin_bf16(d / "step0_guided_video_vel.bin", v_shape)
    zig_guided_a = load_bin_bf16(d / "step0_guided_audio_vel.bin", a_shape)
    summarize("  guided_v_x0 (zig dump)", zig_guided_v)
    summarize("  guided_a_x0 (zig dump)", zig_guided_a)

    print("\n  Zig guided vs offline computed:")
    compare("video guided_x0", zig_guided_v, guided_v_x0)
    compare("audio guided_x0", zig_guided_a, guided_a_x0)

    print("\n=== Step 3: Euler step from guided x0 ===")
    step1_v = denoising_step_from_x0(v_sample, guided_v_x0, v_mask, v_clean, sigma, sigma_next)
    step1_a = denoising_step_from_x0(a_sample, guided_a_x0, a_mask, a_clean, sigma, sigma_next)

    summarize("  step1_v (computed)", step1_v)
    summarize("  step1_a (computed)", step1_a)

    # Compare with Zig's dumped step-1 latent (if available)
    step1_v_path = d / "step1_video_latent.bin"
    step1_a_path = d / "step1_audio_latent.bin"
    if step1_v_path.exists():
        zig_step1_v = load_bin_bf16(step1_v_path, v_shape)
        zig_step1_a = load_bin_bf16(step1_a_path, a_shape)
        summarize("  step1_v (zig dump)", zig_step1_v)
        summarize("  step1_a (zig dump)", zig_step1_a)
        print("\n  Zig step-1 vs offline computed:")
        compare("video step-1", zig_step1_v, step1_v)
        compare("audio step-1", zig_step1_a, step1_a)
    else:
        print("\n  [no step-1 latent dump found — re-run Zig with updated code]")

    # Check step-5 and step-15 latent ranges if available
    for step_name, step_n in [("step5", 5), ("step15", 15), ("final", 30)]:
        v_path = d / f"{step_name}_video_latent.bin"
        a_path = d / f"{step_name}_audio_latent.bin"
        if step_name == "final":
            v_path = d / "video_latent.bin"
            a_path = d / "audio_latent.bin"
        if v_path.exists():
            v_lat = load_bin_bf16(v_path, v_shape)
            a_lat = load_bin_bf16(a_path, a_shape)
            print(f"\n  --- Step {step_n} latent ranges ---")
            summarize(f"  video_latent (step {step_n})", v_lat)
            summarize(f"  audio_latent (step {step_n})", a_lat)


if __name__ == "__main__":
    main()
