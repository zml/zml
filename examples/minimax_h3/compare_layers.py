#!/usr/bin/env python3
"""Host-side parity with official Diffusers MiniMax-H3 (stdlib only, no weights)."""

from __future__ import annotations

import math

ROPE_THETA = 10000.0
ROPE_FREQ_DIM = 16
MODALITY = 3


def official_inv_freq(rope_freq_dim: int = ROPE_FREQ_DIM, theta: float = ROPE_THETA) -> list[float]:
    return [1.0 / (theta ** (i / (2 * rope_freq_dim))) for i in range(0, 2 * rope_freq_dim, 2)]


def official_rotate_half(x: list[float]) -> list[float]:
    half = len(x) // 2
    return [-v for v in x[half:]] + x[:half]


def official_shift_sigma(sigma: float, shift: float) -> float:
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def official_sigmas(steps: int, shift: float) -> list[float]:
    if steps < 2:
        raise ValueError("steps")
    raw = [official_shift_sigma(1.0 - i / (steps - 1), shift) for i in range(steps)]
    unique = [raw[0]]
    for sigma in raw[1:]:
        if sigma != unique[-1]:
            unique.append(sigma)
    if unique[-1] != 0.0:
        unique.append(0.0)
    return unique


def official_spatial_grid(dim: int, patch: int, sqrt_area: float) -> list[float]:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    count = dim // patch
    return [(left + i * (ratio / count)) * 32.0 for i in range(count)]


def official_adaln_index(timestep_index: int, token_tag: int) -> int:
    return timestep_index * MODALITY + token_tag


def official_vit_coords(dim: int) -> list[float]:
    return [2.0 * ((i + 0.5) / dim) - 1.0 for i in range(dim)]


def official_snake(x: float, alpha: float) -> float:
    s = math.sin(alpha * x)
    return x + (1.0 / (alpha + 1e-9)) * (s * s)


def official_timestep_emb(t: float, dim: int = 256) -> list[float]:
    half = dim // 2
    freqs = [math.exp(-math.log(10000.0) * i / half) for i in range(half)]
    return [math.cos(t * f) for f in freqs] + [math.sin(t * f) for f in freqs]


def main() -> None:
    inv = official_inv_freq()
    assert abs(inv[0] - 1.0) < 1e-6
    assert inv[-1] < inv[0]
    assert official_rotate_half([1, 2, 3, 4]) == [-3, -4, 1, 2]
    sig = official_sigmas(8, 12.0)
    assert sig[0] == 1.0
    assert sig[-1] == 0.0
    assert abs(official_shift_sigma(0.5, 12.0) - 12.0 / 13.0) < 1e-6
    axis = official_spatial_grid(8, 2, 8.0)
    assert axis == [0.0, 8.0, 16.0, 24.0]
    assert official_adaln_index(1, 1) == 4
    emb = official_timestep_emb(0.0)
    assert abs(emb[0] - 1.0) < 1e-5
    assert abs(emb[128] - 0.0) < 1e-5
    coords = official_vit_coords(4)
    assert abs(coords[0] + 0.75) < 1e-6
    assert abs(coords[-1] - 0.75) < 1e-6
    assert official_snake(0.0, 1.0) == 0.0
    print("official python layer checks: all passed")
    print(f"inv_freq[0]={inv[0]} inv_freq[-1]={inv[-1]}")
    print(f"sigmas_8={sig}")
    print(f"spatial_8={axis}")


if __name__ == "__main__":
    main()
