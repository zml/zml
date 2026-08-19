#!/usr/bin/env python3
"""Independent NumPy oracles used to validate Kimi K3 golden fixtures.

These routines intentionally do not import Torch, FLA, Transformers, or the
Moonshot implementation.  They are small semantic references, not inference
implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    result = np.empty_like(value)
    positive = value >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def l2_norm(value: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return value / np.sqrt(np.sum(value * value, axis=-1, keepdims=True) + eps)


def rms_norm(value: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Moonshot KimiRMSNorm: FP32 reduction, cast, then direct weight multiply."""
    value = np.asarray(value, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    normalized = value / np.sqrt(
        np.mean(value * value, axis=-1, keepdims=True) + np.float32(eps)
    )
    return normalized * weight


def situ_glu(
    gate: np.ndarray,
    up: np.ndarray,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> np.ndarray:
    """Moonshot SiTU-GLU, including the configured linear-path cap."""
    gate = np.asarray(gate, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    situ = np.float32(beta) * np.tanh(gate / np.float32(beta)) * sigmoid(gate)
    linear = np.float32(linear_beta) * np.tanh(up / np.float32(linear_beta))
    return situ * linear


def softmax(value: np.ndarray) -> np.ndarray:
    """Stable FP32 softmax over the final dimension."""
    value = np.asarray(value, dtype=np.float32)
    shifted = value - np.max(value, axis=-1, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / np.sum(numerator, axis=-1, keepdims=True)


def topk_descending(value: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Stable test-only top-k presentation; production set alignment is ID-based."""
    value = np.asarray(value, dtype=np.float32)
    ids = np.argsort(-value, axis=-1, kind="stable")[..., :k]
    return np.take_along_axis(value, ids, axis=-1), ids.astype(np.int32)


def causal_depthwise_conv1d(
    value: np.ndarray, kernel: np.ndarray
) -> np.ndarray:
    """Causal depthwise convolution for [batch, sequence, channel] inputs.

    ``kernel`` has layout [channel, 1, kernel] and is applied in the same
    correlation order as Torch/ZML conv1d (the newest sample meets kernel[-1]).
    """
    value = np.asarray(value, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    if kernel.ndim != 3 or kernel.shape[0] != value.shape[-1] or kernel.shape[1] != 1:
        raise ValueError("depthwise kernel must have shape [channel, 1, kernel]")
    width = kernel.shape[-1]
    padded = np.pad(value, ((0, 0), (width - 1, 0), (0, 0)))
    output = np.empty_like(value, dtype=np.float32)
    for token in range(value.shape[1]):
        window = padded[:, token : token + width, :]
        output[:, token, :] = np.einsum(
            "bkc,ck->bc", window, kernel[:, 0, :], optimize=True
        )
    return output


def causal_conv_tail(value: np.ndarray, left_context: int) -> np.ndarray:
    """Return the zero-left-padded convolution history used by decode."""
    value = np.asarray(value)
    if left_context < 0:
        raise ValueError("left_context must be non-negative")
    if left_context == 0:
        return value[:, :0, :].copy()
    tail = value[:, -min(value.shape[1], left_context) :, :]
    if tail.shape[1] == left_context:
        return tail.copy()
    return np.pad(tail, ((0, 0), (left_context - tail.shape[1], 0), (0, 0)))


def mla_nope_join(content: np.ndarray, extra: np.ndarray) -> np.ndarray:
    """Join K3 MLA content and unrotated extra dimensions (NoPE is required)."""
    content = np.asarray(content, dtype=np.float32)
    extra = np.asarray(extra, dtype=np.float32)
    if content.shape[:-1] != extra.shape[:-1]:
        raise ValueError("MLA NoPE components must share all non-head dimensions")
    return np.concatenate((content, extra), axis=-1)


def mla_scale(scores: np.ndarray, q_head_dim: int = 192) -> np.ndarray:
    """Apply Moonshot's q_head_dim**-0.5 MLA attention scaling."""
    return np.asarray(scores, dtype=np.float32) * np.float32(q_head_dim**-0.5)


def kda_log_alpha(
    raw_decay: np.ndarray,
    a_log: np.ndarray,
    dt_bias: np.ndarray,
    lower_bound: float,
) -> np.ndarray:
    """FLA safe-gate transform used by the pinned Moonshot KDA call."""
    raw_decay = np.asarray(raw_decay, dtype=np.float32)
    a_log = np.asarray(a_log, dtype=np.float32)
    dt_bias = np.asarray(dt_bias, dtype=np.float32)
    return np.float32(lower_bound) * sigmoid(
        np.exp(a_log)[None, None, :, None]
        * (raw_decay + dt_bias.reshape(1, 1, *raw_decay.shape[-2:]))
    )


def kda_scan(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    log_alpha: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Token-wise KDA with state layout [batch, head, value, key]."""
    q = l2_norm(q)
    k = l2_norm(k)
    v = np.asarray(v, dtype=np.float32)
    log_alpha = np.asarray(log_alpha, dtype=np.float32)
    beta = sigmoid(beta)
    batch, time, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    state = (
        np.zeros((batch, heads, value_dim, key_dim), dtype=np.float32)
        if initial_state is None
        else np.asarray(initial_state, dtype=np.float32).copy()
    )
    output = np.empty((batch, time, heads, value_dim), dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(key_dim))
    for token in range(time):
        state *= np.exp(log_alpha[:, token])[:, :, None, :]
        prediction = np.einsum("bhvk,bhk->bhv", state, k[:, token], optimize=True)
        error = v[:, token] - prediction
        state += (
            beta[:, token, :, None, None]
            * error[:, :, :, None]
            * k[:, token, :, None, :]
        )
        output[:, token] = np.einsum(
            "bhvk,bhk->bhv", state, q[:, token] * scale, optimize=True
        )
    return output, state


@dataclass(frozen=True)
class RouterResult:
    ids: np.ndarray
    weights: np.ndarray
    raw_scores: np.ndarray
    selection_scores: np.ndarray


def route(
    hidden: np.ndarray,
    weight: np.ndarray,
    correction_bias: np.ndarray,
    top_k: int,
    scaling_factor: float = 1.0,
) -> RouterResult:
    hidden = np.asarray(hidden, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    correction_bias = np.asarray(correction_bias, dtype=np.float32)
    raw = sigmoid(hidden @ weight.T)
    selection = raw + correction_bias
    ids = np.argsort(-selection, axis=-1, kind="stable")[..., :top_k]
    chosen = np.take_along_axis(raw, ids, axis=-1)
    weights = chosen / (chosen.sum(axis=-1, keepdims=True) + np.float32(1e-20))
    return RouterResult(
        ids=ids.astype(np.int64),
        weights=weights * np.float32(scaling_factor),
        raw_scores=raw,
        selection_scores=selection,
    )


_FP4 = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def unpack_e2m1(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    low = packed & np.uint8(0x0F)
    high = (packed >> np.uint8(4)) & np.uint8(0x0F)
    nibble = np.stack((low, high), axis=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )
    sign = np.where(nibble & np.uint8(0x08), -1.0, 1.0)
    return (_FP4[(nibble & np.uint8(0x07)).astype(np.int64)] * sign).astype(np.float32)


def decode_e8m0(scale: np.ndarray) -> np.ndarray:
    return np.exp2(np.asarray(scale, dtype=np.int32) - 127).astype(np.float32)


def expand_block32_scale(scale: np.ndarray) -> np.ndarray:
    """Expand one E8M0 scale per 32 logical E2M1 values."""
    return np.repeat(decode_e8m0(scale), 32, axis=-1)


def dequantize_mxfp4(
    packed: np.ndarray, scale: np.ndarray, group_size: int = 32
) -> np.ndarray:
    """Slow explicit dequantization of a logical [out, in] MXFP4 matrix."""
    unpacked = unpack_e2m1(packed)
    if group_size != 32:
        raise ValueError("Kimi K3 MXFP4 group size must be 32")
    expanded = expand_block32_scale(scale)
    if unpacked.shape != expanded.shape:
        raise ValueError(
            f"packed values and block scales disagree: {unpacked.shape} != {expanded.shape}"
        )
    return unpacked * expanded


def mxfp4_linear(
    value: np.ndarray, packed: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Correctness-first y = x @ dequantize(weight).T oracle."""
    weight = dequantize_mxfp4(packed, scale)
    return np.asarray(value, dtype=np.float32) @ weight.T
