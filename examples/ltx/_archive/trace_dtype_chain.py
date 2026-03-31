#!/usr/bin/env python3
"""Trace the EXACT runtime dtype of every intermediate operation in the
LTX-2 denoising pipeline.  No model needed — uses synthetic bf16 tensors
of realistic shape and prints dtype after every sub-expression.

This replays the exact same Python source code from:
  - MultiModalGuider.calculate()     (guiders.py)
  - post_process_latent()            (helpers.py)
  - to_velocity() / to_denoised()    (utils.py)
  - EulerDiffusionStep.step()        (diffusion_steps.py)

Run: python trace_dtype_chain.py
"""
import torch

SEP = "─" * 70


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def p(label: str, t):
    """Print label, dtype, and type info for a value."""
    if isinstance(t, torch.Tensor):
        print(f"  {label:.<50s} dtype={t.dtype}  shape={list(t.shape)}")
    else:
        print(f"  {label:.<50s} type={type(t).__name__}  value={t}")


# ---------------------------------------------------------------------------
# Synthetic inputs (realistic shapes, bf16)
# ---------------------------------------------------------------------------
B, T, D = 1, 6144, 128
torch.manual_seed(42)

cond        = torch.randn(B, T, D).to(torch.bfloat16)
uncond_text = torch.randn(B, T, D).to(torch.bfloat16)
uncond_ptb  = torch.randn(B, T, D).to(torch.bfloat16)
uncond_mod  = torch.randn(B, T, D).to(torch.bfloat16)

sample       = torch.randn(B, T, D).to(torch.bfloat16)
denoise_mask = torch.ones(B, T, 1).to(torch.float32)   # f32! (from real pipeline capture)
clean_latent = torch.randn(B, T, D).to(torch.bfloat16)

sigma_val      = 1.0       # Python float
sigma_next_val = 0.994957  # Python float
sigmas = torch.tensor([sigma_val, sigma_next_val, 0.0], dtype=torch.float32)

# Guidance params (LTX-2.3 defaults)
cfg_scale = 3.0        # Python float
stg_scale = 1.0        # Python float
modality_scale = 3.0   # Python float
rescale_scale = 0.7    # Python float

# =========================================================================
# 1. MultiModalGuider.calculate()  — from guiders.py:258
# =========================================================================
section("1. MultiModalGuider.calculate()")
print("  Input dtypes:")
p("cond", cond)
p("uncond_text", uncond_text)
p("uncond_ptb", uncond_ptb)
p("uncond_mod", uncond_mod)
p("cfg_scale", cfg_scale)
p("stg_scale", stg_scale)
p("modality_scale", modality_scale)
p("rescale_scale", rescale_scale)

print("\n  Step-by-step:")

# Exactly as Python: pred = cond + (cfg-1)*(cond-neg) + stg*(cond-ptb) + (mod-1)*(cond-iso)
_cfg_m1 = cfg_scale - 1          # Python float arithmetic
p("(cfg_scale - 1)", _cfg_m1)

_cond_minus_neg = cond - uncond_text
p("cond - uncond_text", _cond_minus_neg)

_cfg_term = _cfg_m1 * _cond_minus_neg
p("(cfg_scale - 1) * (cond - uncond_text)", _cfg_term)

_cond_minus_ptb = cond - uncond_ptb
p("cond - uncond_ptb", _cond_minus_ptb)

_stg_term = stg_scale * _cond_minus_ptb
p("stg_scale * (cond - uncond_ptb)", _stg_term)

_mod_m1 = modality_scale - 1
p("(modality_scale - 1)", _mod_m1)

_cond_minus_mod = cond - uncond_mod
p("cond - uncond_mod", _cond_minus_mod)

_mod_term = _mod_m1 * _cond_minus_mod
p("(modality_scale - 1) * (cond - uncond_mod)", _mod_term)

_sum1 = cond + _cfg_term
p("cond + cfg_term", _sum1)

_sum2 = _sum1 + _stg_term
p("+ stg_term", _sum2)

pred = _sum2 + _mod_term
p("+ mod_term  (= pred)", pred)

print("\n  Rescaling (rescale_scale=0.7):")

_cond_std = cond.std()
p("cond.std()", _cond_std)

_pred_std = pred.std()
p("pred.std()", _pred_std)

_ratio = _cond_std / _pred_std
p("cond.std() / pred.std()", _ratio)

_factor_scaled = rescale_scale * _ratio
p("rescale_scale * ratio", _factor_scaled)

_one_minus_rs = 1 - rescale_scale
p("(1 - rescale_scale)", _one_minus_rs)

factor = _factor_scaled + _one_minus_rs
p("factor = rs*ratio + (1-rs)", factor)

pred_rescaled = pred * factor
p("pred * factor", pred_rescaled)

print(f"\n  ★ FINAL guided output dtype: {pred_rescaled.dtype}")


# =========================================================================
# 2. post_process_latent()  — from helpers.py:282
# =========================================================================
section("2. post_process_latent(denoised, mask, clean)")

denoised = pred_rescaled  # guided output from above (bf16)
p("denoised (input)", denoised)
p("denoise_mask", denoise_mask)
p("clean_latent", clean_latent)

print("\n  Step-by-step:")

_d_times_m = denoised * denoise_mask
p("denoised * denoise_mask", _d_times_m)

_clean_f = clean_latent.float()
p("clean.float()", _clean_f)

_one_minus_mask = 1 - denoise_mask
p("(1 - denoise_mask)", _one_minus_mask)

_clean_term = _clean_f * _one_minus_mask
p("clean.float() * (1 - mask)", _clean_term)

_blend_sum = _d_times_m + _clean_term
p("denoised*mask + clean_f*(1-mask)", _blend_sum)

blended = _blend_sum.to(denoised.dtype)
p(".to(denoised.dtype)", blended)

print(f"\n  ★ FINAL post_process dtype: {blended.dtype}")


# =========================================================================
# 3. to_velocity()  — from utils.py:21
# =========================================================================
section("3. to_velocity(sample, sigma, denoised_sample)")

sigma_tensor = sigmas[0]  # f32 tensor scalar
p("sample", sample)
p("denoised_sample (blended)", blended)
p("sigma (from sigmas tensor)", sigma_tensor)

print("\n  Step-by-step:")

# Python code: sigma = sigma.to(calc_dtype).item()
_sigma_f32 = sigma_tensor.to(torch.float32)
p("sigma.to(f32)", _sigma_f32)
_sigma_item = _sigma_f32.item()
p("sigma.to(f32).item()", _sigma_item)

_sample_f32 = sample.to(torch.float32)
p("sample.to(f32)", _sample_f32)

_denoised_f32 = blended.to(torch.float32)
p("denoised.to(f32)", _denoised_f32)

_diff = _sample_f32 - _denoised_f32
p("sample_f32 - denoised_f32", _diff)

_div = _diff / _sigma_item
p("diff / sigma_item", _div)

velocity = _div.to(sample.dtype)
p(".to(sample.dtype)", velocity)

print(f"\n  ★ FINAL to_velocity dtype: {velocity.dtype}")


# =========================================================================
# 4. EulerDiffusionStep.step()  — from diffusion_steps.py:17
# =========================================================================
section("4. EulerDiffusionStep.step(sample, denoised, sigmas, step_idx=0)")

step_index = 0
_sigma = sigmas[step_index]
_sigma_next = sigmas[step_index + 1]
_dt = _sigma_next - _sigma

p("sigma = sigmas[0]", _sigma)
p("sigma_next = sigmas[1]", _sigma_next)
p("dt = sigma_next - sigma", _dt)

print("\n  Inside step(): calls to_velocity then Euler update")
print("  (to_velocity already traced above, reusing 'velocity')")
p("velocity (from to_velocity)", velocity)

_sample_f32_euler = sample.to(torch.float32)
p("sample.to(f32)", _sample_f32_euler)

_vel_f32 = velocity.to(torch.float32)
p("velocity.to(f32)", _vel_f32)

_vel_times_dt = _vel_f32 * _dt
p("velocity_f32 * dt", _vel_times_dt)

_euler_sum = _sample_f32_euler + _vel_times_dt
p("sample_f32 + vel_f32 * dt", _euler_sum)

next_latent = _euler_sum.to(sample.dtype)
p(".to(sample.dtype)", next_latent)

print(f"\n  ★ FINAL Euler step dtype: {next_latent.dtype}")


# =========================================================================
# 5. to_denoised() — from utils.py:39 (used in X0Model)
# =========================================================================
section("5. to_denoised(sample, velocity, sigma)  [X0Model path]")

# In X0Model, sigma is actually video.timesteps = denoise_mask * sigma
_timesteps = denoise_mask * sigma_tensor
p("timesteps = denoise_mask * sigma", _timesteps)
p("  denoise_mask dtype", denoise_mask)
p("  sigma dtype", sigma_tensor)

print("\n  Inside to_denoised():")

# sigma is a tensor here, so: sigma = sigma.to(calc_dtype)
_sigma_cast = _timesteps.to(torch.float32)
p("sigma.to(calc_dtype=f32)", _sigma_cast)

_samp_f32 = sample.to(torch.float32)
p("sample.to(f32)", _samp_f32)

_vel_f32_2 = velocity.to(torch.float32)
p("velocity.to(f32)", _vel_f32_2)

_vel_times_sigma = _vel_f32_2 * _sigma_cast
p("velocity_f32 * sigma_f32", _vel_times_sigma)

_x0_f32 = _samp_f32 - _vel_times_sigma
p("sample_f32 - vel*sigma", _x0_f32)

x0 = _x0_f32.to(sample.dtype)
p(".to(sample.dtype)", x0)

print(f"\n  ★ FINAL to_denoised dtype: {x0.dtype}")


# =========================================================================
# Summary
# =========================================================================
section("SUMMARY: Actual runtime dtypes")
print("""
  MultiModalGuider.calculate():
    - All guidance arithmetic:           OPERATES IN bf16
    - std() computation:                 OPERATES IN bf16
    - Rescale factor:                    bf16
    - Output:                            bf16

  post_process_latent():      [mask is f32!]
    - denoised * mask:                   bf16 * f32 = f32  (PyTorch promotes)
    - clean.float():                     f32
    - (1 - mask):                        f32  (1.0 - f32 = f32)
    - clean_f32 * (1-mask):              f32 * f32 = f32
    - sum:                               f32 + f32 = f32
    - .to(denoised.dtype):               bf16

  to_velocity():
    - sigma:                             Python float (via .item())
    - arithmetic:                        f32
    - output .to(sample.dtype):          bf16  ← ROUNDTRIP!

  EulerDiffusionStep.step():
    - velocity from to_velocity:         bf16  ← quantized
    - velocity.to(f32):                  f32   ← roundtripped
    - sample.to(f32) + vel_f32 * dt:     f32
    - .to(sample.dtype):                 bf16

  to_denoised() [X0Model]:        [mask is f32!]
    - timesteps = mask * sigma:          f32 * f32 = f32  (NO bf16 quantization!)
    - arithmetic:                        f32
    - output:                            bf16
""")
