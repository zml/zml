#!/usr/bin/env python3
"""Quick analysis: does per-step error scale with |dt| (sigma step size)?

Usage: python analyze_reset_scaling.py
"""
import math

# Reproduce the sigma schedule from model.zig
num_steps = 30
max_shift = 2.05
base_shift = 0.95
terminal = 0.1
num_tokens = 4096

BASE_SHIFT_ANCHOR = 1024.0
MAX_SHIFT_ANCHOR = 4096.0

# linspace(1.0, 0.0, 31)
sigmas = [1.0 - i / num_steps for i in range(num_steps + 1)]

# sigma_shift
mm = (max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR)
b = base_shift - mm * BASE_SHIFT_ANCHOR
sigma_shift = num_tokens * mm + b
exp_shift = math.exp(sigma_shift)

# logistic shift
for i in range(num_steps + 1):
    if sigmas[i] != 0.0:
        sigmas[i] = exp_shift / (exp_shift + (1.0 / sigmas[i] - 1.0))

# stretch to terminal
last_nz = max(i for i in range(num_steps + 1) if sigmas[i] != 0.0)
one_minus_last = 1.0 - sigmas[last_nz]
scale_factor = one_minus_last / (1.0 - terminal)
for i in range(num_steps + 1):
    if sigmas[i] != 0.0:
        sigmas[i] = 1.0 - (1.0 - sigmas[i]) / scale_factor

# Observed per-step MAE from reset test
reset_mae_v = [
    0.000293, 0.001116, 0.001448, 0.001481, 0.003472,
    0.003086, 0.004115, 0.002551, 0.002487, 0.002504,
    0.002599, 0.002929, 0.003065, 0.003498, 0.003938,
    0.004238, 0.004947, 0.005353, 0.005917, 0.006883,
    0.007861, 0.009292, 0.010763, 0.012508, 0.014989,
    0.018100, 0.022703, 0.026599, 0.032591, 0.018537,
]

print(f"{'Step':>5s}  {'sigma':>8s}  {'sigma_n':>8s}  {'|dt|':>8s}  "
      f"{'mae_v':>10s}  {'mae/|dt|':>10s}  {'sigma':>8s}")
print("-" * 75)

ratios = []
for i in range(num_steps):
    sigma = sigmas[i]
    sigma_next = sigmas[i + 1]
    dt = abs(sigma_next - sigma)
    mae = reset_mae_v[i]
    ratio = mae / dt if dt > 1e-8 else float('inf')
    ratios.append(ratio)
    print(f"{i+1:5d}  {sigma:8.4f}  {sigma_next:8.4f}  {dt:8.6f}  "
          f"{mae:10.6f}  {ratio:10.4f}  {sigma:8.4f}")

avg_ratio = sum(ratios) / len(ratios)
std_ratio = (sum((r - avg_ratio)**2 for r in ratios) / len(ratios)) ** 0.5
cov_ratio = std_ratio / avg_ratio

print("-" * 75)
print(f"  mae/|dt| ratio:  avg={avg_ratio:.4f}  std={std_ratio:.4f}  CoV={cov_ratio:.2f}")
print()
if cov_ratio < 0.3:
    print("  ✓ Per-step error scales linearly with |dt|.")
    print("    This is expected: larger Euler steps amplify constant velocity error.")
else:
    print("  Per-step error does NOT scale linearly with |dt|.")
    print("  There may be additional step-dependent factors.")
