"""GDN core attention — the FASTEST tpu-inference path, in one file, for ZML.

This is `tpu-inference`'s `RAGGED_GATED_DELTA_RULE_IMPL = chunked_kernel_pd`
(the fastest preset combo), bundled end-to-end. It contains exactly TWO Mosaic
kernels — the slower paths (P0 ref scan, P1 chunked-JAX, P3 token-by-token
recurrent kernel) are deliberately NOT here:

      hidden -> in_proj (caller)            <- NOT here; the caller's projection
      mixed_qkv, b, a
        |
        v   ragged_conv1d  (pure XLA; causal depthwise conv1d over ragged seqs;
        |                    maintains conv_state)
        v   --- the gated-delta-rule core: pick by `is_decode_only` ---
        |     decode-only batch  ->  fused_decoding_gdn  ("P4", Mosaic — the fast
        |                             decode kernel; `fused_gdn` is the wrapper)
        |     mixed / prefill     ->  recurrent_scan      ("P5", Mosaic v2: chunked
        |                             WY/UT prefill + recurrent decode in ONE
        |                             pallas_call — schedule table, sublane-aligned
        |                             chunking, token-by-token transition blocks,
        |                             fused SiLU+gate. Handles a mixed batch alone.)
        v
      o (-> RMSNormGated(o, z) -> out_proj, in the caller)

It is a *verbatim* concatenation of these tpu-inference modules (banners mark
the boundaries), with only these adaptations:
  * the Apache license headers collapsed into the one above this docstring;
  * `import` lines consolidated at the top of this file;
  * `from tpu_inference. ...` imports removed (those modules are sections below);
  * `get_default_block_sizes` -> `_get_default_block_sizes_decode` (the source
    module had a function of that name; same body, just renamed for clarity —
    `fused_gdn_recurrent_kernel.py` had a same-named one but that module is not
    included here);
  * `compute_schedule_table_v2.compute_schedule_table_v2(...)` ->
    `compute_schedule_table_v2(...)` (the module became a function here);
  * `_dispatch_with_distribution` ADAPTED: the original chains `fused_gdn` →
    decode kernel (P4) → recurrent kernel (P3); on the `chunked_kernel_pd` path
    the P3 leg is always a 0-grid no-op (decode-only batches), so it's elided
    and `fused_recurrent_gdn` / `calculate_chunk_indices` / the metadata helpers
    are NOT included. (See the function's docstring; the verbatim 2-leg version
    is in GDN-Google/tpu_inference/kernels/gdn/fused_gdn_kernel_wrapper.py.)
  * a small `_gdn_delta_rule_chunked_kernel_pd(...)` reimplementing the one
    `chunked_kernel_pd` branch of `ragged_gated_delta_rule_wrapper`;
  * `gdn_core_attention(...)` reimplementing `gdn_attention.run_jax_gdn_attention_local`
    minus the `jax.shard_map` (the harness runs one abstract TPU core);
  * a `build_args(cfg)` + `main()` harness adapter at the bottom (the
    abstract-TPU-mesh + CPU-lowering-rule pattern, same as the other
    `kernels/mosaic_tpu/py/*.py` files).

Otherwise nothing is trimmed — every comment, TODO, defensive `jnp.where(isinf...)`,
clamp value, eps, etc. is kept exactly as upstream.

WHAT TO PORT, IN WHAT ORDER (see GDN-Google/IMPLEMENTATIONS.md §4 for the
roadmap, GDN-Google/PORTING_PLAYBOOK.md for the general rules):
  1. `ragged_conv1d` + `_*_mixed_prefill` / `_*_decode_only` — pure XLA, no
     Mosaic; you can keep this in StableHLO. (It runs *before* the delta rule.)
  2. `fused_decoding_gdn` ("P4") + `validate_gdn_inputs` — the decode Mosaic
     kernel: `pltpu.emit_pipeline` over `bt`-token windows + a manual,
     double-buffered `pltpu.make_async_copy` gather/scatter of the recurrent
     state (indexed by `state_indices`) + `input_output_aliases` + fused
     L2-norm + fused gate transform. Smaller / simpler of the two Mosaic
     kernels — port this first. THIS replaces your current `gdn_decode.zig`.
  3. `recurrent_scan` ("P5") + `compute_schedule_table_v2` + `invert_triangular_matrix`
     + `inner_kernel` + `fused_kernel` + `create_block_specs` + `get_qkv_index_map_v2`
     — the fast chunked prefill (and mixed-batch) Mosaic kernel. The big one:
     budget real time for the schedule table, the `sublanesize`-aligned chunking,
     and `process_transition_prefill` (token-by-token at sequence boundaries that
     share a sublane). THIS replaces your current `gdn_prefill_varlen.zig`.
  4. `fused_gdn` / `_dispatch_with_distribution` / `ragged_gated_delta_rule`
     (the adapter) / `_gdn_delta_rule_chunked_kernel_pd` / `gdn_core_attention`
     — pure glue (no Mosaic); reimplement in whatever ties your DSL kernels
     together.
  (If you ever want the `recurrent_kernel_pd` fallback — slower, O(T) prefill,
   but simpler kernels — `fused_recurrent_gdn` ("P3") + `calculate_chunk_indices`
   are in GDN-Google/tpu_inference/kernels/gdn/fused_gdn_recurrent_kernel.py.)

Test oracle (the slow obvious `jax.lax.scan` to diff every Mosaic kernel
against): `GDN-Google/tpu_inference/layers/common/ragged_gated_delta_rule_ref.py`
(`ragged_gated_delta_rule`). It is NOT bundled here (it's not on the fast path)
but it's the thing you validate against — and remember the playbook's #1:
`precision=jax.lax.Precision.HIGHEST` + `preferred_element_type=jnp.float32` on
every matmul (these kernels already do; carry it over).

Source: vllm-project/tpu-inference @ main (extracted 2026-05). Modules:
  tpu_inference/layers/common/ragged_conv1d_jax.py
  tpu_inference/kernels/gdn/fused_gdn_kernel_common.py
  tpu_inference/kernels/gdn/fused_gdn_decode_kernel.py             (P4)
  tpu_inference/kernels/gdn/fused_gdn_kernel_wrapper.py            (_dispatch_with_distribution ADAPTED)
  tpu_inference/kernels/gdn/compute_schedule_v2.py
  tpu_inference/kernels/gdn/recurrent_scan_v2.py                   (P5)
  tpu_inference/layers/common/ragged_gated_delta_rule_wrapper.py   (the one combo, reimplemented)
  tpu_inference/layers/common/gdn_attention.py                     (the local glue, reimplemented)
  (NOT included: fused_gdn_recurrent_kernel.py / triangle_solver.py /
   ragged_gated_delta_rule_{ref,chunked}.py — the slower paths.)

Pair file in zig (to be written by you): kernels/mosaic_tpu/zig/gdn_fast.zig.
"""
from __future__ import annotations

import dataclasses
import enum
import functools
from functools import partial
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# =============================================================================
# tpu_inference/layers/common/ragged_conv1d_jax.py
#   The causal depthwise conv1d that runs BEFORE the delta rule (pure XLA).
# =============================================================================

"""Ragged convolution Jax implementation.

This file provides highly optimized 1D convolutions over ragged sequences on
TPUs in JAX.

Key design ideas:
1. Convolution via standard operators: Instead of mapping large index loops,
   it applies standard `lax.conv_general_dilated` on flat tokens to perform a
   depthwise 1-D convolution.
2. Ragged boundary fixup: To prevent cross-sequence pollution at padded
   boundaries in consecutive batch groups, we compute slice segments right where
   sequences transition and resolve border overlaps accurately.
3. State update: We update the convolutional state by selecting either the
   previous state or the current input based on whether the current position is
   within the state's range or not.
"""



def _fix_query_start_loc(query_start_loc, num_valid_seqs):
    """Fixes query_start_loc to be non-decreasing for invalid sequences."""
    last_valid_loc = query_start_loc[num_valid_seqs]
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    return jnp.where(valid_loc_mask, query_start_loc, last_valid_loc)


def _get_boundary_indices(starts, lengths, kernel_size, num_valid_seqs):
    """Computes indices for boundary fixup."""
    valid_mask = jnp.arange(starts.shape[0]) < num_valid_seqs
    starts = jnp.where(valid_mask, starts, 1)[:, None]
    lengths = lengths[:, None]
    k_range = jnp.arange(kernel_size - 1)[None, :]
    gather_indices = starts + jnp.minimum(k_range, lengths - 1)
    scatter_indices = jnp.where(
        (k_range < lengths) & valid_mask[:, None],
        starts + k_range,
        -1,
    )
    return gather_indices, scatter_indices


def _get_state_update_indices(query_start_loc, kernel_size, num_tokens):
    """Computes indices for updating the convolutional state."""
    lengths = query_start_loc[1:] - query_start_loc[:-1]

    k_range = jnp.arange(kernel_size - 1)

    safe_idx_x = (query_start_loc[1:, None] -
                  jnp.arange(kernel_size - 1, 0, -1)[None, :])
    safe_idx_x = jnp.clip(safe_idx_x, 0, num_tokens - 1)

    is_from_old_state = k_range[None, :] < (kernel_size - 1 - lengths)[:, None]

    idx_g = k_range[None, :] + lengths[:, None]
    idx_g = jnp.clip(idx_g, 0, kernel_size - 2)

    return safe_idx_x, is_from_old_state, idx_g


def _depthwise_conv1d_loop_and_bias(x, conv_weight, conv_bias):
    """Depthwise 1D convolution using loops over kernel size.

  Note that an alternative is to use `lax.conv_general_dilated`. However we use
  loops to enable fusing bias addition.
  """
    num_tokens = x.shape[0]
    kernel_size = conv_weight.shape[-1]
    out = None

    # Pad x on the left with kernel_size - 1 zeros
    padded_x = jnp.pad(x, ((kernel_size - 1, 0), (0, 0)))

    # Accumulate over kernel size
    for k in range(kernel_size):
        # Accumulation needs to be done in float32 to avoid accuracy loss
        x_slice = padded_x[k:k + num_tokens, :].astype(jnp.float32)
        weight_slice = conv_weight[:, 0, k].astype(jnp.float32)
        if out is None:
            if conv_bias is None:
                out = x_slice * weight_slice
            else:
                out = x_slice * weight_slice + conv_bias[jnp.newaxis, :]
        else:
            out += x_slice * weight_slice

    assert out is not None
    return out.astype(x.dtype)


def ragged_conv1d_mixed_prefill(
    x,
    conv_state,
    conv_weight,
    conv_bias,
    query_start_loc,
    state_indices,
    distribution,
    has_initial_state,
    *,
    kernel_size,
):
    """Applies 1D convolution, optimized for prefill."""
    num_tokens = x.shape[0]
    max_blocks = state_indices.shape[0]
    num_valid_seqs = distribution[2]

    # 1. Compute Convolution
    out = _depthwise_conv1d_loop_and_bias(x, conv_weight, conv_bias)

    # 2. Fixup Boundary Tokens
    query_start_loc = _fix_query_start_loc(query_start_loc, num_valid_seqs)
    starts = query_start_loc[:-1]
    lengths = query_start_loc[1:] - query_start_loc[:-1]
    gather_indices, scatter_indices = _get_boundary_indices(
        starts, lengths, kernel_size, num_valid_seqs)
    x_first = x[gather_indices]  # (max_blocks, K-1, dim)

    # Concatenate state and x_first along the spatial dimension
    gathered_state = conv_state[state_indices]  # (max_blocks, K-1, dim)

    # Mask the gathered conv state to zero for sequences without initial
    # state, so brand-new prefills see zeros instead of whatever a reused
    # slot still held.
    gathered_state = jnp.where(
        has_initial_state[:, None, None],
        gathered_state,
        jnp.zeros_like(gathered_state),
    )

    combined_tokens = jnp.concatenate([gathered_state, x_first],
                                      axis=1)  # (max_blocks, 2K - 2, dim)

    # Depthwise Convolution for Fixup
    b_out = lax.conv_general_dilated(
        combined_tokens,
        conv_weight,
        window_strides=(1, ),
        padding="VALID",
        dimension_numbers=("NWC", "OIW", "NWC"),
        feature_group_count=x.shape[-1],
        precision=lax.Precision.HIGHEST,
    ).reshape(-1, x.shape[-1])
    if conv_bias is not None:
        b_out += conv_bias[jnp.newaxis, :]

    # Scatter the updates. Note that scatter indices may contain -1, which will be
    # ignored by the scatter operation with mode="drop".
    out = out.at[scatter_indices.flatten()].set(b_out.astype(out.dtype),
                                                mode="drop",
                                                wrap_negative_indices=False)
    # Mask invalid tokens to 0
    total_valid_tokens = query_start_loc[num_valid_seqs]
    valid_token_mask = jnp.arange(num_tokens) < total_valid_tokens
    out = jnp.where(valid_token_mask[:, jnp.newaxis], out, 0.0)
    # 3. Update State
    true_valid_seq_mask = jnp.arange(max_blocks) < num_valid_seqs
    safe_idx_x, is_from_old_state, idx_g = _get_state_update_indices(
        query_start_loc, kernel_size, num_tokens)

    x_tokens = x[safe_idx_x]
    r_grid = jnp.arange(max_blocks)[:, None]
    state_tokens = gathered_state[r_grid, idx_g]

    new_state_extracted = jnp.where(is_from_old_state[..., None], state_tokens,
                                    x_tokens)

    updated_conv_state = conv_state.at[state_indices].set(
        jnp.where(
            true_valid_seq_mask[:, None, None],
            new_state_extracted,
            conv_state[state_indices],
        ))

    return out.astype(x.dtype), updated_conv_state


def ragged_conv1d_decode_only(
    x,
    conv_state,
    conv_weight,
    conv_bias,
    query_start_loc,
    state_indices,
    distribution,
    has_initial_state,
    *,
    kernel_size,
):
    """Apply conv1d for decode-only case (All valid reqs have seq_len=1)."""
    num_tokens = x.shape[0]

    token_idx = jnp.arange(num_tokens)
    req_state_indices = state_indices[token_idx]
    gathered_state = conv_state[
        req_state_indices]  # (num_tokens, kernel_size - 1, dim)

    # Concat old state and new token to form (num_tokens, kernel_size, dim)
    lhs = jnp.concatenate([gathered_state, x[:, jnp.newaxis, :]], axis=1)

    out = jnp.einsum(
        "nkd,dk->nd",
        lhs,
        conv_weight[:, 0, :],
        precision=lax.Precision.HIGHEST,
    )

    if conv_bias is not None:
        out = out + conv_bias

    num_valid_seqs = distribution[2]

    # Drop oldest state and append new state
    new_state_extracted = jnp.concatenate(
        [gathered_state[:, 1:, :], x[:, jnp.newaxis, :]], axis=1)

    token_idx = jnp.arange(num_tokens)
    valid_mask = token_idx < num_valid_seqs
    states_to_set = jnp.where(
        valid_mask[:, jnp.newaxis, jnp.newaxis],
        new_state_extracted,
        gathered_state,
    )

    updated_conv_state = conv_state.at[req_state_indices].set(states_to_set)

    out = jnp.where(valid_mask[:, jnp.newaxis], out, 0.0)

    return out.astype(x.dtype), updated_conv_state


# Donate conv_state to avoid "copy" op by XLA
@jax.jit(donate_argnames=("conv_state", ), static_argnames=("kernel_size", ))
@jax.named_scope("ragged_conv1d_jax")
def ragged_conv1d(
    x: jax.Array,
    conv_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    has_initial_state: jax.Array,
    *,
    kernel_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Applies 1D convolution over ragged sequences and updates state.

    Args:
      x: Input tensor of shape `(num_tokens, dim)`.
      conv_state: Combined convolutional state of shape `(max_blocks, kernel_size
        - 1, dim)`.
      conv_weight: Convolutional weight of shape `(dim, 1, kernel_size)`.
      conv_bias: Optional convolutional bias of shape `(dim,)`.
      query_start_loc: Tensor of shape `(num_seqs + 1,)` containing the start
        indices of each sequence, with the last element being the total number of
        valid tokens.
      state_indices: Tensor of shape `(max_blocks,)` mapping request index to
        state index.
      distribution: Distribution tensor containing number of valid sequences at
        index 2.
      has_initial_state: Boolean tensor of shape `(max_reqs,)`. ``True`` when the
        request has prior conv state to use (chunked-prefill continuation or
        prefix-cache hit). ``False`` for brand-new prefills, in which case the
        gathered conv state is treated as zeros — matching GPU's
        ``causal_conv1d_fn(has_initial_state=...)`` semantics. Without this
        masking the conv would consume whatever a reused mamba slot still held
        from a prior request, silently corrupting the first ``kernel_size - 1``
        outputs of every new request.
      kernel_size: The size of the convolution kernel.

    Returns:
      A tuple containing:
      - output: The output tensor of shape `(num_tokens, dim)`.
      - updated_conv_state: The updated convolutional state of shape `(max_blocks,
        kernel_size - 1, dim)`.
    """
    is_decode_only = distribution[0] == distribution[2]

    def decode_only_branch(_):
        return ragged_conv1d_decode_only(
            x,
            conv_state,
            conv_weight,
            conv_bias,
            query_start_loc,
            state_indices,
            distribution,
            has_initial_state,
            kernel_size=kernel_size,
        )

    def mixed_prefill_branch(_):
        return ragged_conv1d_mixed_prefill(
            x,
            conv_state,
            conv_weight,
            conv_bias,
            query_start_loc,
            state_indices,
            distribution,
            has_initial_state,
            kernel_size=kernel_size,
        )

    return jax.lax.cond(is_decode_only,
                        decode_only_branch,
                        mixed_prefill_branch,
                        operand=None)

# =============================================================================
# tpu_inference/kernels/gdn/fused_gdn_kernel_common.py
#   Shared shape / dtype / TPU-alignment validation for the fused kernels.
# =============================================================================

"""Shared input validation for fused GDN kernels."""




def validate_gdn_inputs(
    q,
    k,
    v,
    g,
    initial_state,
    state_indices,
    *,
    b=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
):
    """Validate shapes, dtypes, and TPU alignment for fused GDN kernels.

    Args:
        q: ``[T, H_qk, K]``.
        k: ``[T, H_qk, K]``.
        v: ``[T, H_v, V]``.
        g: ``[T, H_v, K]`` float32.
        initial_state: ``[num_states, H_v, K, V]`` float32.
        state_indices: ``[max_num_req]`` int32.
        b: ``[T, H_v, num_lanes]`` or ``None``.
        use_gate_in_kernel: Whether gate transformation is applied inside kernel.
        A_log: ``[H_v, num_lanes]`` float32 or ``None``.
        dt_bias: ``[H_v, num_lanes]`` float32 or ``None``.

    Returns:
        ``(T, H_qk, H_v, K, V, dtype, num_states, num_lanes, packing)``.
    """
    T, H_qk, K = q.shape
    H_v = v.shape[1]
    V = v.shape[2]
    dtype = q.dtype
    num_states = initial_state.shape[0]
    num_lanes = pltpu.get_tpu_info().num_lanes
    packing = 32 // dtypes.itemsize_bits(dtype)

    # Shape checks
    if k.shape != (T, H_qk, K):
        raise ValueError(f"k shape {k.shape} != q shape {q.shape}")
    if H_v % H_qk != 0:
        raise ValueError(f"H_v={H_v} must be a multiple of H_qk={H_qk}")
    if v.shape != (T, H_v, V):
        raise ValueError(f"v shape {v.shape} must be [{T}, {H_v}, {V}]")
    if g.shape != (T, H_v, K):
        raise ValueError(f"g shape {g.shape} must be [{T}, {H_v}, {K}]")
    if initial_state.shape[1:] != (H_v, K, V):
        raise ValueError(
            f"initial_state trailing dims {initial_state.shape[1:]} "
            f"must be ({H_v}, {K}, {V})")
    if b is not None and (b.ndim != 3 or b.shape[0] != T or b.shape[1] != H_v):
        raise ValueError(f"b shape {b.shape} must be [{T}, {H_v}, ...]")

    # TPU alignment
    if K % num_lanes != 0 or V % num_lanes != 0:
        raise ValueError(f"K={K}, V={V} must be multiples of {num_lanes}")
    if H_qk % packing != 0:
        raise ValueError(
            f"H_qk={H_qk} must be a multiple of packing={packing}")
    if H_v % packing != 0:
        raise ValueError(f"H_v={H_v} must be a multiple of packing={packing}")

    # Dtype checks
    if k.dtype != dtype or v.dtype != dtype:
        raise ValueError(f"q/k/v must share the same dtype, got q={dtype}, "
                         f"k={k.dtype}, v={v.dtype}")
    if g.dtype != jnp.float32:
        raise ValueError(f"g must be float32, got {g.dtype}")
    if initial_state.dtype not in (jnp.float32, jnp.bfloat16, jnp.float16):
        raise ValueError(
            f"initial_state must be float32, bfloat16, or float16, "
            f"got {initial_state.dtype}")
    if state_indices.dtype != jnp.int32:
        raise ValueError(
            f"state_indices must be int32, got {state_indices.dtype}")

    # Gate-in-kernel checks
    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("A_log is required when use_gate_in_kernel=True")
        if dt_bias is not None and (dt_bias.ndim != 2
                                    or dt_bias.shape[0] != H_v):
            raise ValueError(
                f"dt_bias shape {dt_bias.shape} must be [{H_v}, ...]")
        if dt_bias is not None and dt_bias.dtype != jnp.float32:
            raise ValueError(f"dt_bias must be float32, got {dt_bias.dtype}")

    return T, H_qk, H_v, K, V, dtype, num_states, num_lanes, packing

# =============================================================================
# tpu_inference/kernels/gdn/fused_gdn_decode_kernel.py        ("P4" — fast decode)
#   Mosaic: bt-token-windowed emit_pipeline, per-token recurrent update, manual
#   double-buffered gather/scatter DMA of the recurrent state, fuses L2norm+gate.
#   NOTE: this module's get_default_block_sizes -> _get_default_block_sizes_decode.
# =============================================================================

"""Fused recurrent GDN decoding kernel for TPU.

Processes ``bt`` decode tokens per pipeline step using ``emit_pipeline``
for q/k/v/g/b tiling, with bulk manual DMA for state load/store via
``state_indices``.
"""






def _get_default_block_sizes_decode(
    H_qk: int,
    H_v: int,
    K: int,
    V: int,
    dtype,
    use_gate_in_kernel: bool,
    has_dt_bias: bool,
    vmem_bytes_limit: int,
    state_dtype=jnp.float32,
) -> int:
    """Choose bt to maximize VMEM utilization within vmem_bytes_limit.

    Accounts for state scratch ``(bt, H_v, K, V)`` of ``state_dtype``, optional
    a_log / dt_bias, and bt-proportional tiles that ``emit_pipeline``
    double-buffers (q, k, v, g, b, o).
    """
    ibits = dtypes.itemsize_bits(dtype)
    sbits = dtypes.itemsize_bits(state_dtype)

    # Fixed (not bt-dependent), in bits
    num_lanes = pltpu.get_tpu_info().num_lanes
    fixed_bits = 0
    if use_gate_in_kernel:
        fixed_bits += 2 * H_v * num_lanes * 32  # a_log: (H_v, num_lanes) f32
    if has_dt_bias:
        fixed_bits += 2 * H_v * num_lanes * 32  # dt_bias: (H_v, num_lanes) f32

    # bt-proportional (in bits):
    #   state scratch: (2*bt, H_v, K, V) state_dtype (double buffer)
    #   pipeline tiles (×2 for emit_pipeline double buffering):
    #     q(bt,H_qk,K) + k(bt,H_qk,K)           -> 2·H_qk·K·ibits
    #     g(bt,H_v,K) float32                     -> H_v·K·32
    #     v(bt,H_v,V) + o(bt,H_v,V)              -> 2·H_v·V·ibits
    #     b(bt,H_v,num_lanes)                     -> H_v·num_lanes·ibits
    per_bt_bits = 2 * H_v * K * V * sbits + 2 * (
        2 * H_qk * K * ibits + H_v * K * 32 + 2 * H_v * V * ibits +
        H_v * num_lanes * ibits)

    bt = max(1, (vmem_bytes_limit * 8 - fixed_bits) // per_bt_bits)
    # Round down to nearest power of 2
    return 1 << (bt.bit_length() - 1)


# ── Outer kernel ──────────────────────────────────────────────────────


def _decode_kernel_main(
    q_hbm,  # [T, H_qk, K]
    k_hbm,  # [T, H_qk, K]
    v_hbm,  # [T, H_v, V]
    g_hbm,  # [T, H_v, K] float32
    b_hbm,  # [T, H_v, num_lanes]
    state_indices_ref,  # [max_num_req] int32 (SMEM)
    a_log_hbm,  # [H_v, num_lanes] or None
    dt_bias_hbm,  # [H_v, num_lanes] or None
    distribution_ref,  # [2] int32 (SMEM)
    _state_init_ref,  # [num_states, H_v, K, V] aliased to state_hbm
    o_hbm,  # [T, H_v, V]
    state_hbm,  # [num_states, H_v, K, V]
    h_bufs,  # [2, bt, H_v, K, V] VMEM scratch
    h_load_sems,
    h_store_sems,
    *,
    H_qk: int,
    H_v: int,
    K: int,
    V: int,
    scale: float,
    use_qk_l2norm: bool,
    use_gate_in_kernel: bool,
    lower_bound: float | None,
    bt: int,
):
    decode_end = distribution_ref[0]
    nb_t = (decode_end + bt - 1) // bt
    repeat_factor = H_v // H_qk

    bounded_bt = pl.BoundedSlice(bt)

    def token_map(i):
        t_start = i * bt
        t_size = jnp.minimum(bt, decode_end - t_start)
        return (pl.ds(t_start, t_size), 0, 0)

    qk_spec = pl.BlockSpec((bounded_bt, H_qk, K), token_map)
    g_spec = pl.BlockSpec((bounded_bt, H_v, K), token_map)
    v_spec = pl.BlockSpec((bounded_bt, H_v, V), token_map)
    if b_hbm is not None:
        b_last = b_hbm.shape[2]
        b_spec = pl.BlockSpec((bounded_bt, H_v, b_last), token_map)
    else:
        b_spec = None

    if use_gate_in_kernel and a_log_hbm is not None:
        a_log_spec = pl.BlockSpec((H_v, a_log_hbm.shape[1]), lambda _: (0, 0))
    else:
        a_log_spec = None
    dt_bias_spec = (pl.BlockSpec((H_v, dt_bias_hbm.shape[1]), lambda _:
                                 (0, 0)) if dt_bias_hbm is not None else None)

    # ── Prologue: start loading first bt-block's states ──
    for i_t in range(bt):

        @pl.when(i_t < decode_end)
        def _first_load():
            si = state_indices_ref[i_t]
            pltpu.make_async_copy(
                state_hbm.at[pl.ds(si, 1), :, :, :],
                h_bufs.at[0, pl.ds(i_t, 1), :, :, :],
                h_load_sems.at[0],
            ).start()

    # ── Inner kernel (runs per bt-block) ──
    def _inner_kernel(
        q_ref,  # [<=bt, H_qk, K]
        k_ref,  # [<=bt, H_qk, K]
        v_ref,  # [<=bt, H_v, V]
        g_ref,  # [<=bt, H_v, K]
        b_ref,  # [<=bt, H_v, num_lanes]
        a_log_ref,  # [H_v, num_lanes] or None
        dt_bias_ref,  # [H_v, num_lanes] or None
        o_ref,  # [<=bt, H_v, V]
        h_bufs_s,
        state_indices_s,  # [max_num_req] int32 (SMEM)
        h_load_sems_s,
        h_store_sems_s,
    ):
        block_id = pl.program_id(0)
        t_start = block_id * bt
        block_len = jnp.minimum(bt, decode_end - t_start)
        buf_idx = block_id % 2
        next_buf_idx = (block_id + 1) % 2

        if use_gate_in_kernel:
            a_val = jnp.exp(a_log_ref[:, 0].astype(jnp.float32))
            if dt_bias_ref is not None:
                dt_bias_tile = dt_bias_ref[...].astype(
                    jnp.float32)  # [H_v, num_lanes]
                if K > dt_bias_tile.shape[-1]:
                    dt_bias_val = jnp.concatenate(
                        [dt_bias_tile] * (K // dt_bias_tile.shape[-1]),
                        axis=-1)
                else:
                    dt_bias_val = dt_bias_tile

        # ── Step 1: Prefetch next bt-block's states ──
        next_t_start = t_start + bt
        next_block_len = jnp.maximum(
            jnp.minimum(bt, decode_end - next_t_start), 0)
        for i_t in range(bt):

            @pl.when(i_t < next_block_len)
            def _prefetch():
                next_si = state_indices_s[next_t_start + i_t]
                pltpu.make_async_copy(
                    state_hbm.at[pl.ds(next_si, 1), :, :, :],
                    h_bufs_s.at[next_buf_idx,
                                pl.ds(i_t, 1), :, :, :],
                    h_load_sems_s.at[next_buf_idx],
                ).start()

        # ── Step 2: Wait for current bt-block's state loads ──
        pltpu.make_async_copy(
            state_hbm.at[pl.ds(0, block_len), :, :, :],
            h_bufs_s.at[buf_idx, pl.ds(0, block_len), :, :, :],
            h_load_sems_s.at[buf_idx],
        ).wait()

        # ── Step 3: Compute ──
        for i_t in range(bt):

            @pl.when(i_t < block_len)
            def _process_token():
                h0 = h_bufs_s[buf_idx, i_t].astype(jnp.float32)
                q_t = q_ref[i_t].astype(jnp.float32)
                k_t = k_ref[i_t].astype(jnp.float32)
                v_t = v_ref[i_t].astype(jnp.float32)
                g_t = g_ref[i_t].astype(jnp.float32)
                if b_ref is not None:
                    b_tile = b_ref[i_t].astype(jnp.float32)  # [H_v, num_lanes]
                    if V > b_tile.shape[-1]:
                        beta_t = jax.nn.sigmoid(
                            jnp.concatenate([b_tile] * (V // b_tile.shape[-1]),
                                            axis=-1))  # [H_v, V]
                    else:
                        beta_t = jax.nn.sigmoid(
                            b_tile)  # [H_v, num_lanes] (== [H_v, V])

                if use_qk_l2norm:
                    q_t = q_t / jnp.sqrt(
                        jnp.sum(q_t * q_t, axis=-1, keepdims=True) + 1e-6)
                    k_t = k_t / jnp.sqrt(
                        jnp.sum(k_t * k_t, axis=-1, keepdims=True) + 1e-6)
                q_t = q_t * scale

                # GQA: repeat q/k from H_qk to H_v heads
                if repeat_factor > 1:
                    q_t = jnp.repeat(q_t, repeat_factor, axis=0)
                    k_t = jnp.repeat(k_t, repeat_factor, axis=0)

                if use_gate_in_kernel:
                    g_val = g_t
                    if dt_bias_ref is not None:
                        g_val = g_val + dt_bias_val
                    if lower_bound is not None:
                        gk = lower_bound / (1.0 +
                                            jnp.exp(-(a_val[:, None] * g_val)))
                    else:
                        gk = -a_val[:, None] * jax.nn.softplus(g_val)
                else:
                    gk = g_t

                h_pre = h0 * jnp.exp(gk[:, :, None])
                kh = jax.lax.dot_general(
                    k_t.reshape(H_v, 1, K),
                    h_pre,
                    (((2, ), (1, )), ((0, ), (0, ))),
                    preferred_element_type=jnp.float32,
                ).reshape(H_v, V)
                v_diff = v_t - kh
                b_v = beta_t * v_diff if b_ref is not None else v_diff

                # Algebraic identity to skip the post-rank-1-update matmul:
                # o = q @ (h_pre + outer(k, b_v))
                #   = q @ h_pre + (q . k) * b_v
                # (q . k)[h] is a per-head scalar, so the second term is a
                # cheap HV*V scaled-add instead of a full HV*K*V matmul.
                # This lets MXU(o) and VPU(rank-1 update) run in parallel.
                o_step1 = jax.lax.dot_general(
                    q_t.reshape(H_v, 1, K),
                    h_pre,
                    (((2, ), (1, )), ((0, ), (0, ))),
                    preferred_element_type=jnp.float32,
                ).reshape(H_v, V)
                qk_dot = jnp.sum(q_t * k_t, axis=-1, keepdims=True)
                o_t = o_step1 + qk_dot * b_v
                h_new = h_pre + k_t[:, :, None] * b_v[:, None, :]

                o_ref[i_t] = o_t.astype(o_ref.dtype)
                h_bufs_s[buf_idx, i_t] = h_new.astype(h_bufs_s.dtype)

        # ── Step 4: Wait for stores from 2 blocks ago (same buffer set) ──
        prev_t_start = jnp.maximum((block_id - 2) * bt, 0)
        prev_block_len = jnp.where(
            block_id >= 2,
            jnp.minimum(bt, decode_end - prev_t_start),
            0,
        )

        @pl.when(prev_block_len > 0)
        def _wait_prev_store():
            pltpu.make_async_copy(
                h_bufs_s.at[buf_idx,
                            pl.ds(0, prev_block_len), :, :, :],
                state_hbm.at[pl.ds(0, prev_block_len), :, :, :],
                h_store_sems_s.at[buf_idx],
            ).wait()

        # ── Step 5: Start storing current bt-block's states ──
        for i_t in range(bt):

            @pl.when(i_t < block_len)
            def _start_store():
                si = state_indices_s[t_start + i_t]
                pltpu.make_async_copy(
                    h_bufs_s.at[buf_idx, pl.ds(i_t, 1), :, :, :],
                    state_hbm.at[pl.ds(si, 1), :, :, :],
                    h_store_sems_s.at[buf_idx],
                ).start()

    pltpu.emit_pipeline(
        _inner_kernel,
        grid=(nb_t, ),
        in_specs=[
            qk_spec, qk_spec, v_spec, g_spec, b_spec, a_log_spec, dt_bias_spec
        ],
        out_specs=v_spec,
    )(
        q_hbm,
        k_hbm,
        v_hbm,
        g_hbm,
        b_hbm,
        a_log_hbm,
        dt_bias_hbm,
        o_hbm,
        scratches=[h_bufs, state_indices_ref, h_load_sems, h_store_sems],
    )

    # ── Epilogue: drain outstanding stores ──
    last_buf_idx = (nb_t - 1) % 2
    other_buf_idx = nb_t % 2
    last_block_len = jnp.minimum(bt, decode_end - (nb_t - 1) * bt)
    pltpu.make_async_copy(
        h_bufs.at[last_buf_idx,
                  pl.ds(0, last_block_len), :, :, :],
        state_hbm.at[pl.ds(0, last_block_len), :, :, :],
        h_store_sems.at[last_buf_idx],
    ).wait()

    other_block_len = jnp.where(
        nb_t >= 2,
        jnp.minimum(bt, decode_end - (nb_t - 2) * bt),
        0,
    )

    @pl.when(other_block_len > 0)
    def _drain_other():
        pltpu.make_async_copy(
            h_bufs.at[other_buf_idx,
                      pl.ds(0, other_block_len), :, :, :],
            state_hbm.at[pl.ds(0, other_block_len), :, :, :],
            h_store_sems.at[other_buf_idx],
        ).wait()


# ── Public API ───────────────────────────────────────────────────────


@functools.partial(
    jax.jit,
    static_argnames=[
        "scale",
        "use_qk_l2norm_in_kernel",
        "use_gate_in_kernel",
        "lower_bound",
    ],
)
def fused_decoding_gdn(
    q: jax.Array,  # [T, H_qk, K]
    k: jax.Array,  # [T, H_qk, K]
    v: jax.Array,  # [T, H_v, V]
    g: jax.Array,  # [T, H_v, K] float32
    initial_state: jax.Array,  # [num_states, H_v, K, V] float32
    state_indices: jax.Array,  # [max_num_req] int32
    distribution: jax.Array,  # [2] int32
    b: jax.Array | None,  # [T, H_v, num_lanes] or None
    *,
    scale: float,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: jax.Array | None = None,  # [H_v, num_lanes] float32 or None
    dt_bias: jax.Array | None = None,  # [H_v, num_lanes] float32 or None
    lower_bound: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    r"""Fused recurrent GDN single-step decode.

    Args:
        q: Queries ``[T, H_qk, K]``.
        k: Keys ``[T, H_qk, K]``.
        v: Values ``[T, H_v, V]``.
        g: Per-key gating ``[T, H_v, K]``, float32.
        initial_state: State cache ``[num_states, H_v, K, V]`` float32.
        state_indices: ``i32[max_num_req]`` — indices into the state cache.
        distribution: ``i32[2]`` — ``(decode_end, total)``.
        b: Raw betas ``[T, H_v, num_lanes]`` (sigmoid applied inside kernel).
        scale: Scale factor.
        use_qk_l2norm_in_kernel: L2-normalize q, k inside the kernel.
        use_gate_in_kernel: Apply gate transformation inside kernel.
        A_log: Per-head log gate ``[H_v, num_lanes]`` float32.
        dt_bias: Per-head bias ``[H_v, num_lanes]`` float32.
        lower_bound: If set, use sigmoid gate instead of softplus.

    Returns:
        ``(o, updated_state)`` — *o* is ``[T, H_v, V]``,
        *updated_state* is ``[num_states, H_v, K, V]``.
    """
    T, H_qk, H_v, K, V, dtype, num_states, num_lanes, _ = validate_gdn_inputs(
        q,
        k,
        v,
        g,
        initial_state,
        state_indices,
        b=b,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
    )

    vmem_bytes_limit = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)
    bt = _get_default_block_sizes_decode(
        H_qk,
        H_v,
        K,
        V,
        dtype,
        use_gate_in_kernel,
        dt_bias is not None,
        vmem_bytes_limit,
        state_dtype=initial_state.dtype,
    )

    any_spec = pl.BlockSpec(memory_space=pl.ANY)
    smem_spec = pl.BlockSpec(memory_space=pltpu.SMEM)

    decode_end = distribution[0]
    grid_dim = jnp.where(decode_end > 0, 1, 0)

    n_b = b is not None
    n_gate = (A_log is not None) + (dt_bias is not None)

    scope_name = f"decoding_gdn-bt_{bt}"

    o, state = pl.pallas_call(
        functools.partial(
            _decode_kernel_main,
            H_qk=H_qk,
            H_v=H_v,
            K=K,
            V=V,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            lower_bound=lower_bound,
            bt=bt,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                *([any_spec] * 4),  # q, k, v, g
                any_spec if b is not None else None,  # b
                smem_spec,  # state_indices
                any_spec if A_log is not None else None,
                any_spec if dt_bias is not None else None,
                smem_spec,  # distribution
                any_spec,  # state_init
            ],
            out_specs=[any_spec, any_spec],
            grid=(grid_dim, ),
            scratch_shapes=[
                # h_bufs match HBM dtype so the DMAs don't need conversion.
                # The per-token compute path upcasts to fp32 on each load
                # (see h0 = h_bufs_s[..., i_t].astype(fp32) above), so on-chip
                # math is fp32 regardless of HBM storage dtype.
                pltpu.VMEM((2, bt, H_v, K, V),
                           initial_state.dtype),  # h_bufs (double buffer)
                pltpu.SemaphoreType.DMA((2, )),  # h_load_sems
                pltpu.SemaphoreType.DMA((2, )),  # h_store_sems
            ],
        ),
        input_output_aliases={
            2: 0,
            6 + n_b + n_gate: 1
        },
        out_shape=[
            jax.ShapeDtypeStruct((T, H_v, V), dtype),
            jax.ShapeDtypeStruct((num_states, H_v, K, V), initial_state.dtype),
        ],
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes,
        ),
        name=scope_name,
    )(
        q,
        k,
        v,
        g,
        b,
        state_indices,
        A_log,
        dt_bias,
        distribution,
        initial_state,
    )

    return o, state

# =============================================================================
# tpu_inference/kernels/gdn/fused_gdn_kernel_wrapper.py
#   fused_gdn = the fused-kernel public API; + the ragged_gated_delta_rule adapter
#   that matches the _ref/_chunked signature. (`_dispatch_with_distribution` is
#   ADAPTED here — the original also chains the P3 recurrent kernel; on the
#   chunked_kernel_pd path that leg is always a 0-grid no-op, so it's elided.
#   See the function's docstring and GDN-Google/.../fused_gdn_kernel_wrapper.py.)
# =============================================================================

"""Fused GDN kernel wrapper — dispatch and public API."""






def _dispatch_with_distribution(
    q,
    k,
    v,
    cu_seqlens,
    g,
    initial_state,
    state_indices,
    b,
    has_initial_state,
    *,
    scale,
    use_qk_l2norm,
    use_gate_in_kernel,
    A_log,
    dt_bias,
    lower_bound,
    distribution,
):
    """Run the decode (P4) kernel and return its result.

    >>> ADAPTED for `gdn_fast.py` (the fastest `chunked_kernel_pd` path). <<<
    The original `_dispatch_with_distribution` (verbatim in
    GDN-Google/tpu_inference/kernels/gdn/fused_gdn_kernel_wrapper.py) runs the
    decode kernel (P4) and *then chains* `fused_recurrent_gdn` (P3) for the
    prefill tokens, both updating the state cache in-place. Here, `fused_gdn` is
    only ever called from the `decode_only_branch` of
    `_gdn_delta_rule_chunked_kernel_pd` (the `chunked_kernel_pd` combo uses
    `recurrent_scan` v2 for the mixed/prefill branch). On a decode-only batch
    `distribution = (decode_end, prefill_end, mixed_end)` has `decode_end ==
    mixed_end`, the adapter passes `(decode_end, decode_end)` to `fused_gdn`, so
    the P3 leg would run with `n_seqs = distribution[1] - distribution[0] = 0`
    — a 0-grid no-op whose aliased output `o_r` equals `o_d`. So we drop the P3
    leg entirely (the slow O(T) recurrent kernel is not on the fast path; if you
    ever want the `recurrent_kernel_pd` fallback, port P3 from GDN-Google/).

    `has_initial_state` is now unused on the decode path: decode tokens always
    have a valid prior state (a request must finish prefill before it can
    decode), so masking is unnecessary. (Kept in the signature so callers — the
    `ragged_gated_delta_rule` adapter — don't need changing.)
    """
    del cu_seqlens, has_initial_state  # only used by the elided P3 leg
    # ── Decode kernel → updates state in-place ──
    # NOTE: pass `initial_state` through as-is. The kernels upcast to fp32
    # on VMEM load for compute precision; HBM storage stays at the array's
    # dtype. An `astype(jnp.float32)` here would materialize an fp32 copy
    # of a bf16 state and undo the storage win before the kernel runs.
    o_d, state_1 = fused_decoding_gdn(
        q,
        k,
        v,
        g.astype(jnp.float32),
        initial_state,
        state_indices,
        distribution,
        b,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )
    return o_d, state_1


# ── Public API ──


@functools.partial(
    jax.jit,
    static_argnames=[
        "scale",
        "use_qk_l2norm_in_kernel",
        "use_gate_in_kernel",
        "lower_bound",
    ],
    donate_argnames=["v", "initial_state"],
)
def fused_gdn(
    q: jax.Array,  # [T, H_qk, K]
    k: jax.Array,  # [T, H_qk, K]
    v: jax.Array,  # [T, H_v, V]
    cu_seqlens: jax.Array,  # [max_num_req+1] int32
    g: jax.Array,  # [T, H_v, K] or [T, H_v]
    initial_state: jax.Array,  # [num_states, H_v, K, V]
    state_indices: jax.Array,  # [max_num_req] int32
    distribution: jax.Array,  # [2] int32
    b: jax.Array | None = None,  # [T, H_v] or None
    has_initial_state: jax.Array | None = None,  # [max_num_req] bool
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    A_log: jax.Array | None = None,  # [H_v] float32 or None
    dt_bias: jax.Array | None = None,  # [H_v] float32 or None
    lower_bound: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    r"""Fused recurrent GDN forward pass.

    Supports GQA: ``H_v`` (value heads from ``v``) can be a multiple of
    ``H_qk`` (query/key heads from ``q``/``k``).  The kernel repeats
    q/k internally.

    Args:
        q: Queries ``[T, H_qk, K]``.
        k: Keys ``[T, H_qk, K]``.
        v: Values ``[T, H_v, V]``.
        cu_seqlens: Cumulative sequence lengths ``[max_num_req+1]``.
        g: Gating ``[T, H_v, K]`` or ``[T, H_v]`` (broadcast to K).
        initial_state: State cache ``[num_states, H_v, K, V]``.
        state_indices: ``i32[max_num_req]`` — indices into the state cache.
        distribution: ``i32[2]`` — ``(decode_end, total)``.
        b: Raw betas ``[T, H_v]`` (sigmoid applied inside kernel).
            ``None`` means beta=1 (no beta gating).
        has_initial_state: Boolean tensor of shape ``[max_num_req]``.
            ``True`` when the request's recurrent slot already holds a
            valid prior state (chunked-prefill continuation, prefix-cache
            hit, or running decode). ``False`` for brand-new prefills,
            whose slot is zeroed inside the recurrent kernel before the
            update so stale data from a previous tenant doesn't leak.
            ``None`` (default) is treated as all-True, preserving the
            pre-fix behaviour for callers that don't manage slot reuse.
        scale: Scale factor.  Default ``K ** -0.5``.
        use_qk_l2norm_in_kernel: L2-normalize q, k inside the kernel.
        use_gate_in_kernel: Apply gate transformation inside kernel.
        A_log: Per-head log gate ``[H_v]`` float32.
        dt_bias: Per-head bias ``[H_v]`` float32. Optional.
            Broadcast to ``[H_v, num_lanes]`` internally.
        lower_bound: If set, use sigmoid gate instead of softplus.

    Returns:
        ``(o, updated_state)`` — *o* is ``[T, H_v, V]``,
        *updated_state* is ``[num_states, H_v, K, V]`` with final states
        written back at the corresponding ``state_indices`` positions.
    """
    T, H_qk, K = q.shape
    H_v = v.shape[1]

    # Broadcast g from [T, H_v] to [T, H_v, K] if needed.
    if g.shape == (T, H_v):
        g = jnp.broadcast_to(g[..., None], (T, H_v, K))
    elif g.shape != (T, H_v, K):
        raise ValueError(
            f"g shape {g.shape} must be [{T}, {H_v}, {K}] or [{T}, {H_v}]")

    # Validate pre-broadcast inputs.
    if b is not None and b.shape != (T, H_v):
        raise ValueError(f"b shape {b.shape} must be [{T}, {H_v}]")
    if A_log is not None and A_log.shape != (H_v, ):
        raise ValueError(f"A_log shape {A_log.shape} must be [{H_v}]")
    if dt_bias is not None and dt_bias.shape != (H_v, ):
        raise ValueError(f"dt_bias shape {dt_bias.shape} must be [{H_v}]")

    cu_seqlens = cu_seqlens.astype(jnp.int32)
    state_indices = state_indices.astype(jnp.int32)

    if scale is None:
        scale = K**-0.5
    num_lanes = pltpu.get_tpu_info().num_lanes
    if b is not None:
        b = jnp.broadcast_to(b[:, :, None],
                             (T, H_v, num_lanes))  # [T, H_v, num_lanes]
    if dt_bias is not None:
        dt_bias = jnp.broadcast_to(dt_bias[:, None], (H_v, num_lanes)).astype(
            jnp.float32)  # [H_v, num_lanes]
    distribution = distribution.astype(jnp.int32)

    if A_log is not None:
        A_log = jnp.broadcast_to(A_log[:, None], (H_v, num_lanes)).astype(
            jnp.float32)  # [H_v, num_lanes]

    # Public contract is Boolean (matching the chunked / ref impls);
    # cast to int32 here for SMEM compatibility — the recurrent kernel
    # checks `has_init == 0` to decide whether to zero h0. Default to
    # all-True (no masking), matching the pre-fix behaviour.
    max_num_req = state_indices.shape[0]
    if has_initial_state is None:
        has_initial_state = jnp.ones((max_num_req, ), dtype=jnp.int32)
    else:
        has_initial_state = has_initial_state.astype(jnp.int32)

    o, state = _dispatch_with_distribution(
        q,
        k,
        v,
        cu_seqlens,
        g,
        initial_state,
        state_indices,
        b,
        has_initial_state,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        distribution=distribution,
    )

    return o, state


def ragged_gated_delta_rule(
    mixed_qkv,
    b,
    a,
    recurrent_state,
    A_log,
    dt_bias,
    query_start_loc,
    state_indices,
    distribution,
    has_initial_state=None,
    *,
    n_kq,
    n_v,
    d_k,
    d_v,
):
    """Adapter matching the ragged_gated_delta_rule_{ref,chunked} interface.

    Internally reshapes inputs and delegates to :func:`fused_gdn`.

    Args:
        mixed_qkv: ``(num_tokens, 2*n_kq*d_k + n_v*d_v)`` post-conv/silu.
        b: ``(num_tokens, n_v)`` — raw beta (sigmoid applied in kernel).
        a: ``(num_tokens, n_v)`` — raw alpha (gate transform in kernel).
        recurrent_state: ``(num_states, n_v, d_k, d_v)``.
        A_log: ``(n_v,)`` float32.
        dt_bias: ``(n_v,)`` float32.
        query_start_loc: ``(num_seqs+1,)`` int32.
        state_indices: ``(num_seqs,)`` int32.
        distribution: ``(3,)`` int32 — ``(decode_end, prefill_end, mixed_end)``.
        has_initial_state: Optional Boolean tensor of shape
            ``(max_reqs,)``. ``True`` when the request's slot already
            holds a valid prior recurrent state; ``False`` for brand-new
            prefills (the recurrent kernel zeros h0 for those slots so
            stale data from a reused mamba slot doesn't leak). ``None``
            (default) is treated as all-True, preserving the
            pre-PR-#2408 behaviour. Pass it when you want the same
            stale-slot guard the chunked and ref impls already enforce.
        n_kq: Number of key/query heads.
        n_v: Number of value heads.
        d_k: Key dimension.
        d_v: Value dimension.

    Returns:
        ``(updated_recurrent_state, output)`` where
        *updated_recurrent_state* is ``(num_states, n_v, d_k, d_v)`` and
        *output* is ``(num_tokens, n_v*d_v)``.
    """
    num_tokens = mixed_qkv.shape[0]
    key_dim = n_kq * d_k

    q = mixed_qkv[..., :key_dim].reshape(num_tokens, n_kq, d_k)
    k = mixed_qkv[..., key_dim:key_dim * 2].reshape(num_tokens, n_kq, d_k)
    v = mixed_qkv[..., key_dim * 2:].reshape(num_tokens, n_v, d_v)

    g = a

    # (decode_end, prefill_end, mixed_end) → (decode_end, total)
    fused_distribution = jnp.stack([distribution[0], distribution[2]])

    output, new_recurrent_state = fused_gdn(
        q,
        k,
        v,
        cu_seqlens=query_start_loc,
        g=g,
        initial_state=recurrent_state,
        state_indices=state_indices,
        distribution=fused_distribution,
        b=b,
        has_initial_state=has_initial_state,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        A_log=A_log,
        dt_bias=dt_bias,
    )

    output = output.reshape(num_tokens, n_v * d_v)
    return new_recurrent_state, output

# =============================================================================
# tpu_inference/kernels/gdn/compute_schedule_v2.py
#   Builds the per-grid-step schedule table that drives recurrent_scan (P5).
#   (pure JAX; runs once per forward.)
# =============================================================================




def compute_schedule_table_v2(
    query_start_loc: jax.Array,
    decode_tokens: int | jax.Array,
    num_valid_seqs: int | jax.Array,
    max_tokens: int,
    chunk_size: int,
    BT: int | None = None,
    alignment: int = 8,
) -> tuple[jax.Array, jax.Array]:
    """Compute number of iterations in grid and work each iteration will do

  At high level
    - each iteration of grid is either prefill and or decode
    - grid moves in size of bt decode tokens (sequence) backwards starting from
    boundary
    - and prefill moves in chunk sized tokens forward from boundary to end
  Input characteristics
    - each sequence start and end may not be sublane aligned,
    boundary between decode and prefill maybe in shared sublane
    - sequence may not divide chunk size

  hardware req
    - offset for each block has to be sublane aligned

  So for this we have transition blocks at boundaries between prefill sequences,
  including first one with decode, token by token math is done here instead of
  chunk wise

  TODO: optimize table ,
    remove metadata which can be derived from other metadata or loop indices,
    like
        block offset can be derived from block idx and sequence start,
        block count can be derived from block idx and sequence start/end.
        also some metadata is only used for prefill or decode and can be stored
        in separate tables or encoded in same table with fewer bits.
        dtype of some metadata can be reduced to save space, for example
        block_is_first and block_is_last can be stored in 2 bits together,
        Sublane token by token metadata can be optimized by only storing
        boundaries
  """
    if BT is None:
        BT = chunk_size

    num_decode_batches = (decode_tokens + BT - 1) // BT
    num_seqs = query_start_loc.shape[0] - 1

    max_blocks = (max_tokens + chunk_size - 1) // chunk_size
    safe_max_blocks = int(max_blocks + num_seqs * 2)

    # =========================================================================
    # 1. Get each prefill sequence's effective start for chunkwise math
    # =========================================================================
    r_idx = jnp.arange(num_seqs)
    is_last_seq = r_idx == num_seqs - 1
    seq_start = query_start_loc[:-1]
    seq_end = query_start_loc[1:]
    num_tokens = query_start_loc[num_valid_seqs]

    # create vector of sequence ends
    prev_seq_end = jnp.pad(seq_end[:-1], (1, 0), constant_values=0)
    effective_start = jnp.where(
        prev_seq_end % alignment != 0,
        (prev_seq_end // alignment) * alignment + alignment,
        prev_seq_end,
    )

    # if seq_len < sublane size
    is_decode_boundary = prev_seq_end == decode_tokens
    is_swallowed = (effective_start >= seq_end) & (~is_decode_boundary)

    # compute the effective end of the rounded up to nearest sublane
    next_aligned_start = (seq_end // alignment) * alignment
    needs_transition = ((seq_end % alignment != 0) & (~is_last_seq) &
                        (~is_swallowed))

    is_decode_boundary = prev_seq_end == decode_tokens

    needs_start_transition = ((prev_seq_end % alignment != 0) & (~is_swallowed)
                              & is_decode_boundary)

    effective_end = jnp.where(needs_transition, next_aligned_start, seq_end)
    effective_end = jnp.maximum(effective_start, effective_end)

    # Block counts per sequence
    num_regular_blocks = (effective_end - effective_start + chunk_size -
                          1) // chunk_size
    total_blocks_per_seq = (num_regular_blocks +
                            needs_transition.astype(jnp.int32) +
                            needs_start_transition.astype(jnp.int32))
    total_blocks_per_seq = jnp.where(is_swallowed, 0, total_blocks_per_seq)

    # Calculate the last perfectly aligned decode boundary
    is_pure_decode = seq_end <= decode_tokens
    total_blocks_per_seq = jnp.where(is_pure_decode, 0, total_blocks_per_seq)

    # Starting block index for each sequence
    base_idx = jnp.cumsum(total_blocks_per_seq) - total_blocks_per_seq
    total_prefill_blocks = jnp.sum(total_blocks_per_seq)

    # =========================================================================
    # 2. shows up as gathers
    # create block table
    # =========================================================================
    b_idx = jnp.arange(safe_max_blocks)
    prefill_valid_mask = b_idx < total_prefill_blocks

    # map grid index to sequence/request,
    # key for previous metadata arrays constructed to gather by sequence
    r_for_block = jnp.sum(b_idx[:, None] >= base_idx[None, :], axis=-1) - 1
    r_for_block = jnp.minimum(jnp.maximum(r_for_block, 0), num_seqs - 1)

    # index of block within blocks for a sequence
    local_b = b_idx - base_idx[r_for_block]

    start_trans_offset = (seq_start[r_for_block] // alignment) * alignment

    is_start_trans = needs_start_transition[r_for_block] & (local_b == 0)

    # Adjust local_b for regular blocks if there was a start transition
    adj_local_b = jnp.where(needs_start_transition[r_for_block], local_b - 1,
                            local_b)

    is_end_trans = needs_transition[r_for_block] & (
        adj_local_b == num_regular_blocks[r_for_block])

    reg_offset = effective_start[r_for_block] + adj_local_b * chunk_size
    reg_count = jnp.minimum(chunk_size,
                            effective_end[r_for_block] - reg_offset)
    #   reg_is_last = reg_offset + reg_count >= seq_end[r_for_block]
    #   reg_is_first = reg_offset == seq_start[r_for_block]

    trans_offset = next_aligned_start[r_for_block]

    # Apply predication
    block_offset = jnp.where(
        is_start_trans,
        start_trans_offset,
        jnp.where(is_end_trans, trans_offset, reg_offset),
    )

    block_count = jnp.where(
        is_start_trans,
        effective_start[r_for_block] - seq_start[r_for_block],
        jnp.where(is_end_trans, alignment, reg_count),
    )

    is_trans_block = is_start_trans | is_end_trans

    # =========================================================================
    # 3. Metadata for shared sublane tiles
    # =========================================================================
    last_valid_loc = query_start_loc[num_valid_seqs]
    valid_loc_mask = jnp.arange(query_start_loc.shape[0]) <= num_valid_seqs
    fixed_query_start_loc = jnp.where(valid_loc_mask, query_start_loc,
                                      last_valid_loc)
    glob_idxs = block_offset[:, None] + jnp.arange(alignment)[None, :]

    # [safe_max_blocks, sublane size, num_seqs]
    valid_mask = glob_idxs < num_tokens
    t_reqs = (
        jnp.sum(glob_idxs[:, :, None] >= fixed_query_start_loc[None, None, :],
                axis=-1) - 1)
    # there could be padding in query_start_loc
    last_valid_seq = jnp.max(
        jnp.where(total_blocks_per_seq > 0, jnp.arange(num_seqs), -1))
    t_reqs = jnp.where(valid_mask, t_reqs, last_valid_seq)
    t_reqs = jnp.minimum(jnp.maximum(t_reqs, 0), num_seqs - 1)

    is_first_tok = (glob_idxs == query_start_loc[t_reqs]).astype(jnp.int32)
    is_last_tok = (glob_idxs == query_start_loc[t_reqs + 1] - 1).astype(
        jnp.int32)

    # =========================================================================
    # 4. Decode blocks metadata
    # =========================================================================
    decode_valid_mask = b_idx < num_decode_batches
    decode_batch_idx = jnp.where(decode_valid_mask,
                                 (num_decode_batches - 1) - b_idx, 0)
    decode_offsets = decode_batch_idx * BT
    decode_req_ids = decode_batch_idx * BT
    decode_counts = jnp.where(decode_valid_mask,
                              jnp.minimum(BT, decode_tokens - decode_offsets),
                              0)

    # Mask out invalid prefill
    prefill_valid_ints = prefill_valid_mask.astype(jnp.int32)
    block_offset = jnp.where(prefill_valid_mask, block_offset, 0)
    r_for_block = jnp.where(prefill_valid_mask, r_for_block, 0)
    block_count = jnp.where(prefill_valid_mask, block_count, 0)
    block_is_first = block_offset <= seq_start[r_for_block]
    block_is_last = (block_offset + block_count) >= seq_end[r_for_block]
    block_is_first = jnp.where(prefill_valid_mask, block_is_first, False)
    block_is_last = jnp.where(prefill_valid_mask, block_is_last, False)
    is_trans_block = jnp.where(prefill_valid_mask, is_trans_block, False)
    t_reqs = jnp.where(prefill_valid_mask[:, None], t_reqs, 0)
    is_first_tok = jnp.where(prefill_valid_mask[:, None], is_first_tok, 0)
    is_last_tok = jnp.where(prefill_valid_mask[:, None], is_last_tok, 0)

    # =========================================================================
    # 5. Merge all
    # =========================================================================
    # Columns mapping:
    # 0: prefill_valid_ints - 1 if this grid block has valid prefill work,
    # .                  0 otherwise
    # 1: block_offset - start index of prefill start in tile, usually 0
    #                     but in shared sublane case its not
    # 2: r_for_block - request ID (sequence index) this prefill block belongs to
    # 3: block_count - number of valid tokens in this prefill block
    # 4: decode_valid_mask - 1 if this step has valid decode work, 0 otherwise
    # 5: decode_offsets - start index for the decode batch
    # 6: decode_req_ids - starting request ID in decode batch
    # 7: decode_counts - number of valid decode requests in this batch
    # 8: block_is_last - 1 if this is the last block for the request, 0 otherwise
    # 9: block_is_first - 1 if first block for request, 0 otherwise
    # 10: is_trans_block - 1 if this is a transition block, 0 otherwise
    cols = [
        prefill_valid_ints,  # 0
        block_offset,  # 1
        r_for_block,  # 2
        block_count,  # 3
        decode_valid_mask.astype(jnp.int32),  # 4
        decode_offsets,  # 5
        decode_req_ids,  # 6
        decode_counts,  # 7
        block_is_last.astype(jnp.int32),  # 8
        block_is_first.astype(jnp.int32),  # 9
        is_trans_block.astype(jnp.int32),  # 10
    ]

    # 11 to 11 + alignment - 1: Request ID for each token in the sublane tile
    for i in range(alignment):
        cols.append(t_reqs[:, i])  # e.g., 11-18 if alignment=8
    # 11 + alignment to 11 + 2*alignment - 1: 1 if token is first in request
    for i in range(alignment):
        cols.append(is_first_tok[:, i])  # e.g., 19-26
    # 11 + 2*alignment to 11 + 3*alignment - 1: 1 if token is last in request
    for i in range(alignment):
        cols.append(is_last_tok[:, i])  # e.g., 27-34

    final_table = jnp.stack(cols, axis=1)
    total_blocks = jnp.maximum(total_prefill_blocks, num_decode_batches)

    return final_table, total_blocks

# =============================================================================
# tpu_inference/kernels/gdn/recurrent_scan_v2.py              ("P5" — fast prefill)
#   Mosaic: chunked WY/UT prefill + recurrent decode in ONE pallas_call (an inner
#   emit_pipeline over the schedule table); fused SiLU + gate; sublane-aligned
#   chunking; token-by-token transition blocks at sequence boundaries; in-kernel
#   gather DMA of state; inlines its own triangular inverse (invert_triangular_matrix).
#   NOTE: compute_schedule_table_v2.compute_schedule_table_v2(...) -> compute_schedule_table_v2(...).
# =============================================================================






def invert_triangular_matrix(A, block_size=16):
    """Inverts a unit lower triangular matrix A block-wise.

  Args:
    A: Unit lower triangular matrix of shape (B, N, N).
    block_size: Size of the blocks for Gaussian elimination.

  Returns:
    Inverse of A, of shape (B, N, N).
  """
    B, N, _ = A.shape
    num_blocks = N // block_size

    def local_forward_sub(A_mat, b_mat):
        x_list = []
        for i in range(block_size):
            b_i = b_mat[:, i, :]
            if i == 0:
                x_i = b_i
            else:
                stacked_x = jnp.stack(x_list, axis=1)
                all_prev_A = A_mat[:, i, :i]
                prev_sum = jnp.sum(all_prev_A[..., None] * stacked_x, axis=1)
                x_i = b_i - prev_sum
            x_list.append(x_i)
        return jnp.stack(x_list, axis=1)

    x_blocks = []
    for i in range(num_blocks):
        start, end = i * block_size, (i + 1) * block_size
        e_block = jnp.eye(N, dtype=A.dtype)[start:end, :]
        e_block = jnp.broadcast_to(e_block, (B, block_size, N))

        if i == 0:
            target_b = e_block
        else:
            interaction_A = A[:, start:end, :start]
            solved_x = jnp.concatenate(x_blocks, axis=1)
            prev_sum = jnp.matmul(interaction_A,
                                  solved_x,
                                  precision=jax.lax.Precision.HIGHEST)
            target_b = e_block - prev_sum

        local_A = A[:, start:end, start:end]
        x_block = local_forward_sub(local_A, target_b)
        x_blocks.append(x_block)

    return jnp.concatenate(x_blocks, axis=1)


def inner_kernel(
    # VMEM: (C, D) where D = 2*n_kq*d_k + n_v*d_v. QKV tokens for Prefill chunk
    prefill_qkv_ref,
    # VMEM: (C, D) where D = 2*n_kq*d_k + n_v*d_v. QKV tokens for Decode batch
    decode_qkv_ref,
    # VMEM: (C, 128). Raw a values for Prefill chunk
    prefill_a_raw_ref,
    # VMEM: (BT, 128). Raw a values for Decode batch
    decode_a_raw_ref,
    # VMEM: (C, 128). Raw b values for Prefill chunk
    prefill_b_raw_ref,
    # VMEM: (BT, 128). Raw b values for Decode batch
    decode_b_raw_ref,
    # VMEM: (n_v,). A_log for gate computation
    a_log_ref,
    # VMEM: (n_v,). dt_bias for gate computation
    dt_bias_ref,
    # VMEM: (C, n_v * d_v). Scanned outputs for prefill
    prefill_output_ref,
    # VMEM: (BT, n_v * d_v). Scanned outputs for decode
    decode_output_ref,
    # SMEM: (max_blocks, 8). Schedule table
    schedule_table,
    # SMEM: (max_reqs,). State indices
    state_indices,
    # SMEM: (max_reqs,). Whether each request has prior recurrent state
    has_initial_state,
    *,
    # HBM: (B, n_v, d_k, d_v). All recurrent states
    recurrent_state_in,
    recurrent_state_out,
    # Chunk size for prefill
    C: int,
    # Batch size for decode
    BT: int,
    #  Number of key/query heads
    n_kq: int,
    #  Number of value heads
    n_v: int,
    #  Key dimension
    d_k: int,
    #  Value dimension
    d_v: int,
    use_qk_norm_in_gdn: bool,
    sublanesize: int,
    # VMEM scratchpad: (2, n_v, d_k, d_v). To carry state across chunks
    # (double buffered)
    prefill_scratch,
    # VMEM scratchpad: (1, 2, n_v, d_k, d_v). TODO: double
    # buffer or x buffer to to loop over BT in decode without overwriting state and using async copy for state load/store)
    decode_state_scratch,
    # VMEM scratchpad: (1, n_v, d_k, d_v). dtype = recurrent_state dtype
    # TODO: if output dtype of state is always f32 then this can be removed.
    state_commit_scratch,
    # VMEM scratchpad: (BT, n_v * d_v). To hold decode outputs before DMA
    decode_output_scratch,
    # Array of C semaphores for decode state loads
    decode_read_semaphores,
    # 1 semaphore for decode state stores
    decode_write_semaphore,
    # 1 semaphore for prefill DMA (stores only)
    prefill_semaphore,
    # Number of decode tokens (requests) in the batch
    decode_tokens,
):
    """Inner kernel for recurrent scan processing both prefill and decode.

  This function is called for each step in the schedule table and dispatches
  work to either `process_decode` or
  `process_regular_prefill`/`process_transition_prefill`.
  """
    step = pl.program_id(0)

    # READ table

    prefill_valid = schedule_table[step, 0][...]
    prefill_req_id = schedule_table[step, 2][...]

    decode_valid = schedule_table[step, 4][...]
    decode_offset = schedule_table[step, 5][...]
    decode_req_id = schedule_table[step, 6][...]
    decode_count = schedule_table[step, 7][...]

    prefill_offset = schedule_table[step, 1][...]
    is_transition = schedule_table[step, 10][...]

    is_last_chunk = schedule_table[step, 8][...]
    is_first_chunk = schedule_table[step, 9][...]

    def l2_normalize(x, eps=1e-6):
        norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)
        return x / norm

    # 2. Decode Branch
    # check current iteration had decode work
    @pl.when(decode_valid > 0)
    def decode_wrapper():

        def get_target_idx(b):
            safe_req_id = jnp.minimum(decode_req_id + b,
                                      state_indices.shape[0] - 1)
            return state_indices[safe_req_id][...]

        def process_decode(b, _):
            # token by token check if decode token or not
            is_valid = b < decode_count

            @pl.when(is_valid)
            def do_work():
                target_idx = get_target_idx(b)

                # Load state TODO: make async
                copy_op = pltpu.make_async_copy(
                    src_ref=recurrent_state_in.at[pl.ds(target_idx, 1)],
                    dst_ref=state_commit_scratch,
                    sem=decode_read_semaphores.at[0],
                )
                copy_op.start()
                copy_op.wait()
                decode_state_scratch[pl.ds(
                    0, 1)] = state_commit_scratch[...].astype(jnp.float32)

                key_dim = n_kq * d_k
                b_aligned = (b // sublanesize) * sublanesize
                # Workaround: Upcast to fp32 to avoid NaNs
                qkv_block_data = decode_qkv_ref[
                    pl.ds(b_aligned, sublanesize), :].astype(jnp.float32)
                mask = (jnp.arange(sublanesize) == (b % sublanesize)).astype(
                    qkv_block_data.dtype)[:, None]
                qkv_row = jnp.sum(qkv_block_data * mask, axis=0, keepdims=True)
                # Fused SiLU
                qkv_row = jax.nn.silu(qkv_row)
                q = qkv_row[:, :key_dim].reshape(n_kq, d_k)
                k = qkv_row[:, key_dim:2 * key_dim].reshape(n_kq, d_k)
                v = qkv_row[:, 2 * key_dim:].reshape(n_v, d_v)

                if use_qk_norm_in_gdn:
                    q = l2_normalize(q)
                    k = l2_normalize(k)

                # Head repetition
                repeat_factor = n_v // n_kq
                if repeat_factor > 1:
                    q = jnp.repeat(q, repeat_factor, axis=0)
                    k = jnp.repeat(k, repeat_factor, axis=0)

                scale = d_k**-0.5
                q = q * scale

                b_aligned = (b // sublanesize) * sublanesize

                g_block_new = decode_a_raw_ref[
                    pl.ds(b_aligned, sublanesize), :]
                beta_block_new = decode_b_raw_ref[
                    pl.ds(b_aligned, sublanesize), :]

                mask_new = (jnp.arange(sublanesize) == (
                    b % sublanesize)).astype(g_block_new.dtype)[:, None]

                curr_g_slice_new = jnp.sum(g_block_new * mask_new,
                                           axis=0,
                                           keepdims=True)
                curr_beta_slice_new = jnp.sum(beta_block_new * mask_new,
                                              axis=0,
                                              keepdims=True)

                a_raw_new = curr_g_slice_new[:, :n_v].reshape(n_v).astype(
                    jnp.float32)
                b_raw_new = (curr_beta_slice_new[:, :n_v].reshape(n_v).astype(
                    jnp.float32))

                # Compute gate
                curr_beta = jax.nn.sigmoid(b_raw_new)
                curr_g = -jnp.exp(a_log_ref[...].astype(
                    jnp.float32)) * jax.nn.softplus(
                        a_raw_new + dt_bias_ref[...].astype(jnp.float32))
                curr_g = jnp.maximum(curr_g, -100.0)
                decay = jnp.exp(curr_g)

                current_state = decode_state_scratch[0]

                # TODO: compare MXU vs VPU, MXU doesn't support FP32, VPU does
                # (n_v, d_k, 1) * (n_v, 1, d_v) -> (n_v, d_k, d_v)
                out_list = []
                new_state_list = []
                for h in range(n_v):
                    q_h = q[h:h + 1, :]  # (1, d_k)
                    k_h = k[h:h + 1, :]  # (1, d_k)
                    v_h = v[h:h + 1, :]  # (1, d_v)

                    state_h = current_state[h]  # (d_k, d_v)

                    k_state_h = pl.dot(
                        k_h, state_h,
                        precision=jax.lax.Precision.HIGHEST)  # (1, d_v)

                    # v_diff_h = v_h - decay[h].astype(jnp.float32) * k_state_h
                    decay_k_state = jnp.where(
                        jnp.isinf(k_state_h),
                        0.0,
                        decay[h].astype(jnp.float32) * k_state_h,
                    )
                    v_diff_h = v_h - decay_k_state
                    v_new_h = curr_beta[h].astype(jnp.float32) * v_diff_h

                    q_state_h = pl.dot(
                        q_h, state_h,
                        precision=jax.lax.Precision.HIGHEST)  # (1, d_v)

                    q_k_h = jnp.sum(q_h * k_h, axis=-1,
                                    keepdims=True)  # (1, 1)

                    # Defensive code to handle NaNs and infs in state,
                    # Saw similar issue while trying newton schulz
                    # which can happen due to large decay or long sequences.
                    # TODO: analyze perf impact and risk of removing this.
                    decay_q_state = jnp.where(jnp.isinf(q_state_h), 0.0,
                                              decay[h] * q_state_h)
                    out_h = decay_q_state + q_k_h * v_new_h
                    out_list.append(out_h)

                    k_v_new_h = pl.dot(k_h,
                                       v_new_h,
                                       trans_a=True,
                                       precision=jax.lax.Precision.HIGHEST
                                       )  # (d_k, 1) @ (1, d_v) -> (d_k, d_v)
                    # Defensive code to handle NaNs and infs in state,
                    # which can happen due to large decay or long sequences.
                    # In such cases, we reset the state contribution to zero and rely solely on the new value
                    # TODO: analyze perf impact and risk of removing this.
                    decay_state = jnp.where(jnp.isinf(state_h), 0.0,
                                            state_h * decay[h])
                    new_state_h = decay_state + k_v_new_h
                    new_state_list.append(new_state_h)

                out = jnp.concatenate(out_list, axis=0)  # (n_v, d_v)
                new_state = jnp.stack(new_state_list,
                                      axis=0)  # (n_v, d_k, d_v)

                # TODO: remove VPU path if MXU is certified path
                # decay_exp = decay[..., None]  # (n_v, 1)

                # k_state = jnp.sum(k[..., None] * current_state, axis=1)  # (n_v, d_v)
                # v_diff = v - decay_exp * k_state
                # v_new = curr_beta[..., None] * v_diff  # (n_v, d_v)

                # q_state = jnp.sum(q[..., None] * current_state, axis=1)  # (n_v, d_v)
                # q_k = jnp.sum(q * k, axis=-1, keepdims=True)  # (n_v, 1)

                # out = decay_exp * q_state + q_k * v_new  # (n_v, d_v)
                # k_v_new = k[..., None] * v_new[:, None, :]
                # new_state = current_state * decay_exp[..., None] + k_v_new

                decode_state_scratch[pl.ds(
                    0, 1)] = new_state[None, ...].astype(current_state.dtype)

                # Accumulate output in scratchpad
                current_output = decode_output_scratch[...]
                mask = (jnp.arange(BT) == b).astype(current_output.dtype)[:,
                                                                          None]
                new_output = jnp.where(
                    mask,
                    out.reshape(1, n_v * d_v),
                    current_output,
                )
                decode_output_scratch[...] = new_output.astype(
                    current_output.dtype)

                # Store state (Synchronous)
                state_commit_scratch[0] = decode_state_scratch[0].astype(
                    state_commit_scratch.dtype)
                copy_op = pltpu.make_async_copy(
                    src_ref=state_commit_scratch,
                    dst_ref=recurrent_state_out.at[pl.ds(target_idx, 1)],
                    sem=decode_write_semaphore.at[0],
                )
                copy_op.start()
                copy_op.wait()

                return None

            return None

        # loop over bt, could be for loop, BT is static anyway, unroll
        jax.lax.fori_loop(0, BT, process_decode, None)

        # Mask and write accumulated outputs to HBM
        mask = (jnp.arange(BT)
                < decode_count).astype(decode_output_scratch.dtype)[:, None]
        decode_output_scratch_masked = decode_output_scratch[...] * mask
        decode_output_ref[...] = decode_output_scratch_masked

        return None

    # Prefill Branch
    # Process prefill if there is valid prefill work in this step
    @pl.when(prefill_valid > 0)
    def process_prefill():
        # TODO: eliminate k.transpose in matmuls by directly slicing in the right shape above

        # not used meaningfully, because dma is sync.
        # intention is to index into scratch for storing state and not overwrite each other
        prefill_slot = prefill_req_id % 2

        def process_regular_prefill():
            # 1. Initialize state if first chunk of the request in this step
            @pl.when(is_first_chunk > 0)
            def init_state():
                has_init = has_initial_state[prefill_req_id][...]

                def load_from_hbm():
                    state_idx = state_indices[prefill_req_id][...]
                    copy_op = pltpu.make_async_copy(
                        src_ref=recurrent_state_in.at[pl.ds(state_idx, 1)],
                        dst_ref=state_commit_scratch,
                        sem=prefill_semaphore.at[prefill_slot],
                    )
                    copy_op.start()
                    copy_op.wait()
                    prefill_scratch[prefill_slot] = state_commit_scratch[
                        0].astype(prefill_scratch.dtype)

                def zero_state():
                    prefill_scratch[prefill_slot] = jnp.zeros(
                        (n_v, d_k, d_v), dtype=prefill_scratch.dtype)

                jax.lax.cond(has_init > 0, load_from_hbm, zero_state)
                return None

            ### Preparataion for chunk wise math,
            ### this kernel design could be optimized lot by not doing this every chunk
            # 1. Extract Q, K, V, g, beta for the chunk
            key_dim = n_kq * d_k

            # Workaround: Upcast to fp32 to avoid NaNs in long sequences
            qkv_chunk = prefill_qkv_ref[...].astype(jnp.float32)  # (C, d)
            # Fused SiLU
            qkv_chunk = jax.nn.silu(qkv_chunk)
            q = qkv_chunk[:, :key_dim]
            k = qkv_chunk[:, key_dim:2 * key_dim]
            v = qkv_chunk[:, 2 * key_dim:]

            # Load a, b
            a_raw_chunk = prefill_a_raw_ref[...]  # (C, 128)
            b_raw_chunk = prefill_b_raw_ref[...]  # (C, 128)

            # Slice and transpose to match expected shape (n_v, C),
            # TODO: this transpose can be eliminated
            a_raw_processed = a_raw_chunk[:, :n_v].T
            b_raw_processed = b_raw_chunk[:, :n_v].T

            # Compute gates in VMEM in full fp32, not sure if needed.
            beta = jax.nn.sigmoid(b_raw_processed)
            g = -jnp.exp(a_log_ref[...][:, None].astype(
                jnp.float32)) * jax.nn.softplus(a_raw_processed + dt_bias_ref[
                    ...][:, None].astype(jnp.float32))
            # Workaround: Clamp g to avoid underflow to negative inf
            # g is always negative, from above line
            # for long prefill sequence this negative value will get more negative
            # pow(e,-100) is close to 0.
            g = jnp.maximum(g, -100.0)
            prefill_count = schedule_table[step, 3][...]
            mask_float = (jnp.arange(C) < prefill_count).astype(q.dtype)
            q = jnp.where(mask_float[:, None] > 0, q, 0.0)
            k = jnp.where(mask_float[:, None] > 0, k, 0.0)
            g = jnp.where(mask_float[None, :] > 0, g, 0.0)
            v = jnp.where(mask_float[:, None] > 0, v, 0.0)
            beta = jnp.where(mask_float[None, :] > 0, beta, 0.0)

            q = q.reshape(C, n_kq, d_k)
            k = k.reshape(C, n_kq, d_k)
            v = v.reshape(C, n_v, d_v)

            if use_qk_norm_in_gdn:
                q = l2_normalize(q)
                k = l2_normalize(k)

            repeat_factor = n_v // n_kq
            if repeat_factor > 1:
                q = jnp.repeat(q, repeat_factor, axis=1)
                k = jnp.repeat(k, repeat_factor, axis=1)

            # TODO: eliminate these transposes by directly slicing in the right
            # shape above,
            q = q.transpose(1, 0, 2)
            k = k.transpose(1, 0, 2)
            v = v.transpose(1, 0, 2)

            scale = d_k**-0.5
            q = q * scale

            g_cumsum_list = []
            current_sum = jnp.zeros((n_v, ), dtype=jnp.float32)
            # cumsum not implemented in pallas
            for i in range(C):
                current_sum = current_sum + g[:, i].astype(jnp.float32)
                g_cumsum_list.append(current_sum)
            g_cumsum = jnp.stack(g_cumsum_list, axis=-1)
            k_beta = k * beta[..., None]

            S = jnp.matmul(
                k_beta.astype(jnp.float32),
                k.transpose(0, 2, 1).astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )

            g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
            i = jnp.arange(C)[:, None]
            j = jnp.arange(C)[None, :]
            mask_float = (i > j).astype(jnp.float32)

            # Defensive code to handle large positive g_diff which can cause
            # overflow in exp,
            # TODO: analyze if this is a common case and if we can remove this or do
            # by other means (like clipping g values before cumsum or using a
            # different data type for g/g_cumsum)
            g_diff_safe = jnp.minimum(g_diff, 0.0)
            S = jnp.where(mask_float[None, :, :] > 0, S * jnp.exp(g_diff_safe),
                          0.0)

            S_q = jnp.matmul(
                q.astype(jnp.float32),
                k.transpose(0, 2, 1).astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )
            mask_float_q = (i >= j).astype(jnp.float32)
            g_diff_Sq = g_diff_safe * mask_float_q[None, ...] + (
                1.0 - mask_float_q[None, ...]) * (-1e30)
            S_q = S_q * jnp.exp(g_diff_Sq)
            S_q = S_q * mask_float_q[None, ...]

            I_plus_S = jnp.eye(C, dtype=jnp.float32)[None, ...] + S
            # TODO: call the function in kernels file
            A_inv = invert_triangular_matrix(I_plus_S, block_size=16)

            # UW
            v_beta = v * beta[..., None]
            u = jnp.matmul(A_inv,
                           v_beta.astype(jnp.float32),
                           precision=jax.lax.Precision.HIGHEST)

            k_beta_g = k_beta * jnp.exp(g_cumsum)[..., None]
            w = jnp.matmul(
                A_inv,
                k_beta_g.astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )

            q_g = q * jnp.exp(g_cumsum)[..., None]
            current_state = prefill_scratch[prefill_slot]
            attn_inter = jnp.matmul(
                q_g.astype(jnp.float32),
                current_state.astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )
            v_prime = jnp.matmul(
                w,
                current_state.astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )
            v_new = u - v_prime
            term2 = jnp.matmul(S_q, v_new, precision=jax.lax.Precision.HIGHEST)
            o_c = attn_inter + term2

            g_i_last_exp = jnp.exp(g_cumsum[..., -1, None, None])
            g_diff_exp_state = jnp.exp(g_cumsum[..., -1, None] -
                                       g_cumsum)[..., None]
            k_i_g_diff = k * g_diff_exp_state

            update_term = jnp.matmul(
                k_i_g_diff.transpose(0, 2, 1).astype(jnp.float32),
                v_new,
                precision=jax.lax.Precision.HIGHEST,
            )
            h_new = current_state * g_i_last_exp + update_term

            prefill_scratch[prefill_slot] = h_new.astype(prefill_scratch.dtype)

            # Store state only if it's the last chunk of the request
            @pl.when(is_last_chunk > 0)
            def store_state():
                # TODO: if dtype of state in HBM is always f32,
                # then we can eliminate this copy and directly write from scratch to HBM
                state_commit_scratch[0] = prefill_scratch[prefill_slot].astype(
                    state_commit_scratch.dtype)
                state_idx = state_indices[prefill_req_id][...]
                copy_op = pltpu.make_async_copy(
                    src_ref=state_commit_scratch,
                    dst_ref=recurrent_state_out.at[pl.ds(state_idx, 1)],
                    sem=prefill_semaphore.at[prefill_slot],
                )
                copy_op.start()
                copy_op.wait()
                return None

            # TODO: eliminate this transpose and reshape by directly writing in the right shape above
            o_c_tr = o_c.transpose(1, 0, 2)
            o_c_flat = o_c_tr.reshape(C, n_v * d_v)

            prefill_count = schedule_table[step, 3][...]
            mask_float = (jnp.arange(C) < prefill_count).astype(o_c_flat.dtype)
            o_c_flat_masked = o_c_flat * mask_float[:, None]
            prefill_output_ref[...] = o_c_flat_masked.astype(
                prefill_output_ref.dtype)
            return None

        def process_transition_prefill():
            # this is processing prefill sequences in a sublane that has multiple sequences
            C_trans = sublanesize
            key_dim = n_kq * d_k

            # Workaround: Upcast to fp32 to avoid NaNs
            qkv_chunk = prefill_qkv_ref[:C_trans, :].astype(jnp.float32)
            # Fused SiLU TODO: maybe 'SiLU' needs to be parametrized,
            qkv_chunk = jax.nn.silu(qkv_chunk)
            q = qkv_chunk[:, :key_dim]
            k = qkv_chunk[:, key_dim:2 * key_dim]
            v = qkv_chunk[:, 2 * key_dim:]

            # Load untransposed a and b
            a_raw_chunk = prefill_a_raw_ref[...]  # (C, 128)
            b_raw_chunk = prefill_b_raw_ref[...]  # (C, 128)

            # Slice and transpose to match expected shape (n_v, C_trans)
            a_raw_processed = a_raw_chunk[:C_trans, :n_v].T
            b_raw_processed = b_raw_chunk[:C_trans, :n_v].T

            # NOTE: b is upcasted to f32 in ref before sigmoid, beta is bf16
            beta_chunk = jax.nn.sigmoid(b_raw_processed)
            # NOTE: a is upcasted to f32 before add to dt_bias
            g_chunk = -jnp.exp(a_log_ref[...][:, None].astype(
                jnp.float32)) * jax.nn.softplus(a_raw_processed + dt_bias_ref[
                    ...][:, None].astype(jnp.float32))
            g_chunk = jnp.maximum(g_chunk, -100.0)
            q = q.reshape(C_trans, n_kq, d_k)
            k = k.reshape(C_trans, n_kq, d_k)
            v = v.reshape(C_trans, n_v, d_v)

            if use_qk_norm_in_gdn:
                q = l2_normalize(q)
                k = l2_normalize(k)

            repeat_factor = n_v // n_kq
            if repeat_factor > 1:
                q = jnp.repeat(q, repeat_factor, axis=1)
                k = jnp.repeat(k, repeat_factor, axis=1)

            # TODO: eliminate these transposes by directly slicing in the right shape above,
            q = q.transpose(1, 0, 2)
            k = k.transpose(1, 0, 2)
            v = v.transpose(1, 0, 2)

            scale = d_k**-0.5
            q = q * scale

            # state indice for req
            first_req_id = schedule_table[step, 11][...]
            first_is_first = schedule_table[step, 11 + C_trans][...]
            first_slot = first_req_id % 2
            first_has_init = has_initial_state[first_req_id][...]

            @pl.when((first_is_first > 0) & (first_has_init > 0))
            def load_first_state():
                state_idx = state_indices[first_req_id][...]
                copy_op = pltpu.make_async_copy(
                    src_ref=recurrent_state_in.at[pl.ds(state_idx, 1)],
                    dst_ref=state_commit_scratch,
                    sem=prefill_semaphore.at[first_slot],
                )
                copy_op.start()
                copy_op.wait()
                prefill_scratch[first_slot] = state_commit_scratch[0].astype(
                    prefill_scratch.dtype)

            h = prefill_scratch[first_slot]
            h = jnp.where((first_is_first > 0) & (first_has_init == 0),
                          jnp.zeros_like(h), h)

            current_r = first_req_id
            sequence_valid = True

            # loop over token by token
            for i in range(sublanesize):
                # read transition token metadata
                t_req = schedule_table[step, 11 + i][...]
                # get sequence index for token i in sublane
                t_is_first = schedule_table[step, 11 + C_trans + i][...]
                t_is_last = schedule_table[step, 11 + 2 * C_trans + i][...]

                is_new_seq = t_req != current_r
                sequence_valid = jnp.where(is_new_seq, True, sequence_valid)

                # Ignore tokens that belong to decode requests,
                # (assumes decode tokens are at packed at head)
                is_decode_token = t_req < decode_tokens
                sequence_valid = jnp.where(is_decode_token, False,
                                           sequence_valid)

                c_slot = current_r % 2

                h0 = prefill_scratch[0]
                h1 = prefill_scratch[1]
                prefill_scratch[0] = jnp.where(c_slot == 0, h, h0)
                prefill_scratch[1] = jnp.where(c_slot == 1, h, h1)

                # prefill_scratch in f32, state_commit might be in bf16
                state_commit_scratch[0] = prefill_scratch[c_slot].astype(
                    state_commit_scratch.dtype)

                def do_write():
                    # TODO: Make async
                    state_idx = state_indices[current_r][...]
                    copy_op = pltpu.make_async_copy(
                        src_ref=state_commit_scratch,
                        dst_ref=recurrent_state_out.at[pl.ds(state_idx, 1)],
                        sem=prefill_semaphore.at[c_slot],
                    )
                    copy_op.start()
                    copy_op.wait()
                    return None

                is_current_r_prefill = current_r >= decode_tokens
                should_write = is_current_r_prefill & is_new_seq
                jax.lax.cond(should_write, do_write, lambda: None)

                t_slot = t_req % 2
                t_has_init = has_initial_state[t_req][...]

                def load_t_state():
                    state_idx = state_indices[t_req][...]
                    copy_op = pltpu.make_async_copy(
                        src_ref=recurrent_state_in.at[pl.ds(state_idx, 1)],
                        dst_ref=state_commit_scratch,
                        sem=prefill_semaphore.at[t_slot],
                    )
                    copy_op.start()
                    copy_op.wait()
                    prefill_scratch[t_slot] = state_commit_scratch[0].astype(
                        prefill_scratch.dtype)

                should_load_t = (t_is_first > 0) & (t_has_init > 0)
                jax.lax.cond(should_load_t, load_t_state, lambda: None)

                h0_new = prefill_scratch[0]
                h1_new = prefill_scratch[1]
                new_h = jnp.where(t_slot == 0, h0_new, h1_new)

                new_h = jnp.where((t_is_first > 0) & (t_has_init == 0),
                                  jnp.zeros_like(new_h), new_h)
                h = new_h

                current_r = t_req

                k_i = k[:, i, :]
                v_i = v[:, i, :]
                g_i = g_chunk[:, i]
                beta_i = beta_chunk[:, i]
                q_i = q[:, i, :]

                decay = jnp.exp(g_i)[..., None]

                k_state = jnp.sum(k_i[..., None] * h, axis=1)
                v_diff = v_i - decay * k_state
                v_new = beta_i[:, None] * v_diff

                q_state = jnp.sum(q_i[..., None] * h, axis=1)
                q_k = jnp.sum(q_i * k_i, axis=-1, keepdims=True)

                out_i = decay * q_state + q_k * v_new

                k_v_new = k_i[..., None] * v_new[:, None, :]
                h_new = h * decay[..., None] + k_v_new

                h = jnp.where(sequence_valid, h_new, h)

                # Mask output BEFORE invalidating the sequence for the next token
                out_i = jnp.where(sequence_valid, out_i, 0.0)

                sequence_valid = jnp.where(t_is_last > 0, False,
                                           sequence_valid)

                prefill_output_ref[i, :] = out_i.reshape(n_v * d_v).astype(
                    prefill_output_ref.dtype)

            final_slot = current_r % 2
            prefill_scratch[final_slot] = h
            state_commit_scratch[0] = h.astype(state_commit_scratch.dtype)

            is_current_r_prefill = current_r >= decode_tokens

            # Store state if the current request is a prefill
            @pl.when(is_current_r_prefill)
            def do_final_write():
                # TODO: make async
                state_idx = state_indices[current_r][...]
                copy_op = pltpu.make_async_copy(
                    src_ref=state_commit_scratch,
                    dst_ref=recurrent_state_out.at[pl.ds(state_idx, 1)],
                    sem=prefill_semaphore.at[final_slot],
                )
                copy_op.start()
                copy_op.wait()
                return None

            return None

        is_transition = schedule_table[step, 10][...]

        def process_prefill_dispatch():
            return jax.lax.cond(
                is_transition > 0,
                lambda _: process_transition_prefill(),
                lambda _: process_regular_prefill(),
                operand=None,
            )

        process_prefill_dispatch()
        return None

    # For transition block at boundary of decode and prefill we will have overlap
    # decode block BT contains prefill tokens
    # sublane size transition prefill block contains some decode tokens in the sublane
    # so we need to stitch the outputs so they don't overwrite each other in global index
    # we exchange decode and prefill outputs so
    # prefill output ref has decode token outputs at decode token indexes in its out ref
    # decode output ref has prefill token outputs have prefill token indexes in its out ref
    def do_stitch():
        local_start = prefill_offset - decode_offset
        local_split = decode_tokens - prefill_offset

        # Need to hint compiler, or it complains in DMA added by emit pipeline
        safe_local_start = pl.multiple_of(local_start, sublanesize)

        decode_overlap = decode_output_ref[
            pl.ds(safe_local_start, sublanesize), :]
        prefill_arr = prefill_output_ref[pl.ds(0, sublanesize), :]

        # 3. Build sublane size mask
        iota = jax.lax.broadcasted_iota(jnp.int32, (sublanesize, ), 0)
        is_decode_mask = (iota < local_split).astype(jnp.int32)[:, None]

        # 4. Merge the tensors
        merged_overlap = jnp.where(is_decode_mask, decode_overlap, prefill_arr)

        decode_output_ref[
            pl.ds(safe_local_start, sublanesize), :] = merged_overlap
        prefill_output_ref[pl.ds(0, sublanesize), :] = merged_overlap

        return None

    is_first_block = pl.program_id(0) == 0
    needs_stitching = (is_transition > 0) & is_first_block & (decode_valid > 0)
    jax.lax.cond(needs_stitching, do_stitch, lambda: None)


def get_qkv_index_map_v2(
    step,
    schedule_table,
    valid_col,
    offset_col,
    count_col,
    alignment=16,
    block_size=64,
    sink_offset=0,
):
    valid = schedule_table[step, valid_col][...]
    offset = schedule_table[step, offset_col][...]
    offset = pl.multiple_of(offset, alignment)

    safe_offset = jnp.where(valid > 0, offset, sink_offset)
    safe_offset = pl.multiple_of(safe_offset, alignment)

    return (pl.ds(safe_offset, block_size), 0)


def create_block_specs(
    schedule_table,
    chunk_size,
    BT,
    d,
    n_v,
    d_v,
    alignment=16,
    sink_offset=0,
):
    """Creates block specs for recurrent scan kernel."""

    prefill_qkv_index_map = functools.partial(
        get_qkv_index_map_v2,
        schedule_table=schedule_table,
        valid_col=0,
        offset_col=1,
        count_col=3,
        alignment=alignment,
        block_size=chunk_size,
        sink_offset=sink_offset,
    )

    decode_qkv_index_map = functools.partial(
        get_qkv_index_map_v2,
        schedule_table=schedule_table,
        valid_col=4,
        offset_col=5,
        count_col=7,
        alignment=alignment,
        block_size=BT,
        sink_offset=sink_offset,
    )

    prefill_qkv_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(chunk_size), d),
        index_map=prefill_qkv_index_map,
    )
    decode_qkv_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(BT), d),
        index_map=decode_qkv_index_map,
    )

    prefill_output_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(chunk_size), n_v * d_v),
        index_map=prefill_qkv_index_map,
    )
    decode_output_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(BT), n_v * d_v),
        index_map=decode_qkv_index_map,
    )

    a_log_spec = pl.BlockSpec(block_shape=(n_v, ), index_map=lambda _: (0, ))
    dt_bias_spec = pl.BlockSpec(block_shape=(n_v, ), index_map=lambda _: (0, ))
    prefill_a_raw_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(chunk_size), 128),
        index_map=prefill_qkv_index_map,
    )
    decode_a_raw_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(BT), 128),
        index_map=decode_qkv_index_map,
    )
    prefill_b_raw_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(chunk_size), 128),
        index_map=prefill_qkv_index_map,
    )
    decode_b_raw_spec = pl.BlockSpec(
        block_shape=(pl.BoundedSlice(BT), 128),
        index_map=decode_qkv_index_map,
    )

    return [
        prefill_qkv_spec,
        decode_qkv_spec,
        prefill_a_raw_spec,
        decode_a_raw_spec,
        prefill_b_raw_spec,
        decode_b_raw_spec,
        a_log_spec,
        dt_bias_spec,
    ], [prefill_output_spec, decode_output_spec]


def fused_kernel(
    mixed_qkv_ref,
    aliased_recurrent_state_ref,
    state_indices_ref,
    has_initial_state_ref,
    a_raw_ref,
    b_raw_ref,
    a_log_ref,
    dt_bias_ref,
    schedule_table_ref,
    decode_tokens_ref,
    total_blocks_ref,
    recurrent_state_ref,
    output_ref,
    *,
    C: int,
    BT: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    use_qk_norm_in_gdn: bool,
    sublanesize: int,
):
    """Fused kernel for recurrent scan."""
    decode_tokens = decode_tokens_ref[0]
    total_blocks = total_blocks_ref[0]

    d = mixed_qkv_ref.shape[-1]
    pad_size = max(C, BT)
    sink_offset = mixed_qkv_ref.shape[0] - pad_size

    in_specs, out_specs = create_block_specs(
        schedule_table_ref,
        C,
        BT,
        d,
        n_v,
        d_v,
        alignment=sublanesize,
        sink_offset=sink_offset,
    )

    def _run_with_scratch(
        scratch_ref,
        decode_state_scratch_ref,
        state_commit_scratch_ref,
        decode_output_scratch_ref,
        decode_read_sems,
        decode_write_sem,
        prefill_sem,
    ):

        pipeline_func = pltpu.emit_pipeline(
            body=functools.partial(
                inner_kernel,
                C=C,
                BT=BT,
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
                use_qk_norm_in_gdn=use_qk_norm_in_gdn,
                sublanesize=sublanesize,
                prefill_scratch=scratch_ref,
                decode_state_scratch=decode_state_scratch_ref,
                decode_output_scratch=decode_output_scratch_ref,
                state_commit_scratch=state_commit_scratch_ref,
                decode_read_semaphores=decode_read_sems,
                decode_write_semaphore=decode_write_sem,
                prefill_semaphore=prefill_sem,
                decode_tokens=decode_tokens,
                recurrent_state_in=aliased_recurrent_state_ref,
                recurrent_state_out=recurrent_state_ref,
            ),
            grid=(total_blocks, ),
            in_specs=in_specs,
            out_specs=out_specs,
        )

        pipeline_func(
            mixed_qkv_ref,
            mixed_qkv_ref,
            a_raw_ref,
            a_raw_ref,
            b_raw_ref,
            b_raw_ref,
            a_log_ref,
            dt_bias_ref,
            output_ref,
            output_ref,
            scratches=[
                schedule_table_ref, state_indices_ref, has_initial_state_ref
            ],
        )

    pl.run_scoped(
        # TODO: Move this to outer pallas call and get rid of run_scoped
        _run_with_scratch,
        pltpu.VMEM((2, n_v, d_k, d_v),
                   jnp.float32),  # prefill_scratch (double buffered)
        pltpu.VMEM((1, n_v, d_k, d_v), jnp.float32),  # decode_state_scratch
        pltpu.VMEM((1, n_v, d_k, d_v),
                   recurrent_state_ref.dtype),  # state_commit_scratch
        pltpu.VMEM((BT, n_v * d_v),
                   mixed_qkv_ref.dtype),  # decode_output_scratch
        pltpu.SemaphoreType.DMA((1, )),  # decode_read_semaphores
        pltpu.SemaphoreType.DMA((1, )),  # decode_write_semaphore
        pltpu.SemaphoreType.DMA((2, )),  # prefill_semaphore
    )


@functools.partial(
    jax.jit,
    static_argnames=[
        "n_kq",
        "n_v",
        "d_k",
        "d_v",
        "chunk_size",
        "BT",
        "use_qk_norm_in_gdn",
    ],
)
def recurrent_scan(
    mixed_qkv: jax.Array,
    b: jax.Array,
    a: jax.Array,
    recurrent_state: jax.Array,
    A_log: jax.Array,
    dt_bias: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    chunk_size: int = 128,
    BT: int = 128,
    use_qk_norm_in_gdn: bool = True,
    has_initial_state: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Fused recurrent scan kernel for GDN on TPU v7.

  Args:
    mixed_qkv: jax.Array of shape [num_tokens, 2 * n_kq * d_k + n_v * d_v].
      Packed Query, Key, and Value tokens.
    b: jax.Array of shape [num_tokens, n_v]. Input for beta gate.
    a: jax.Array of shape [num_tokens, n_v]. Input for g gate.
    recurrent_state: jax.Array of shape [max_reqs, n_v, d_k, d_v]. Current
      recurrent states.
    A_log: jax.Array of shape [n_v]. Log of parameter A.
    dt_bias: jax.Array of shape [n_v]. Bias for dt.
    query_start_loc: jax.Array of shape [num_requests + 1]. Start indices of
      each request in mixed_qkv.
    state_indices: jax.Array of shape [num_requests] or larger. Mapping from
      request ID to state index.
    distribution: jax.Array of shape [2]. Contains [decode_tokens,
      total_tokens].
    n_kq: Number of query/key heads.
    n_v: Number of value heads.
    d_k: Dimension of query/key features.
    d_v: Dimension of value features.
    chunk_size: Block size for processing (default 128).
    BT: Block size for decode requests (default 128).
    use_qk_norm_in_gdn: Whether to use QK normalization.

  Returns:
    A tuple containing:
      - Updated recurrent state of shape [max_reqs, n_v, d_k, d_v].
      - The mixed_qkv array of shape [num_tokens, 2 * n_kq * d_k + n_v * d_v].
  """
    if has_initial_state is None:
        has_initial_state = jnp.zeros(state_indices.shape[0], dtype=jnp.int32)
    else:
        has_initial_state = has_initial_state.astype(jnp.int32)

    num_tokens = mixed_qkv.shape[0]
    tpu_info = pltpu.get_tpu_info()
    sublanesize = 4 // mixed_qkv.itemsize * tpu_info.num_sublanes

    # Pad token dimension so invalid pipeline steps DMA into a safe sink area.
    # Sink offset must be aligned to sublanesize for Mosaic tile compatibility.
    block_size = max(chunk_size, BT)
    sink_offset = ((num_tokens + sublanesize - 1) // sublanesize) * sublanesize
    pad_rows = sink_offset + block_size - num_tokens
    mixed_qkv = jnp.pad(mixed_qkv, ((0, pad_rows), (0, 0)))

    # Pad raw a and b to (num_tokens + pad_rows, 128) for sublanes
    a_padded = jnp.pad(a, ((0, pad_rows), (0, 128 - n_v)))
    b_padded = jnp.pad(b, ((0, pad_rows), (0, 128 - n_v)))

    # decode_tokens: scalar, number of decode tokens.
    # Assuming length 1 per decode request, this is also the number of decode
    # requests.
    decode_tokens = distribution[0]
    schedule_table, total_blocks = (
        compute_schedule_table_v2(
            query_start_loc,
            decode_tokens,
            distribution[2],
            num_tokens,
            chunk_size,
            BT,
            alignment=sublanesize,
        ))

    # sublane,128
    decode_tokens_arr = jnp.expand_dims(decode_tokens, 0)
    total_blocks_arr = jnp.expand_dims(total_blocks, 0)

    grid_spec = pl.GridSpec(
        grid=(1, ),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.SMEM),
            pl.BlockSpec(block_shape=(1, ), index_map=lambda _: (0, )),
            pl.BlockSpec(block_shape=(1, ), index_map=lambda _: (0, )),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ],
    )

    updated_recurrent_state, output_padded = pl.pallas_call(
        functools.partial(
            fused_kernel,
            C=chunk_size,
            BT=BT,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            use_qk_norm_in_gdn=use_qk_norm_in_gdn,
            sublanesize=sublanesize,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(recurrent_state.shape, recurrent_state.dtype),
            jax.ShapeDtypeStruct((sink_offset + block_size, n_v * d_v),
                                 mixed_qkv.dtype),
        ),
        grid_spec=grid_spec,
        input_output_aliases={1: 0},
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
    )(
        mixed_qkv,
        recurrent_state,
        state_indices,
        has_initial_state,
        a_padded,
        b_padded,
        A_log,
        dt_bias,
        schedule_table,
        decode_tokens_arr,
        total_blocks_arr,
    )
    return updated_recurrent_state, output_padded[:num_tokens]

# =============================================================================
# The `chunked_kernel_pd` dispatch — reimplemented from the one matching branch
# of `tpu_inference/layers/common/ragged_gated_delta_rule_wrapper.py`
# (`RaggedGatedDeltaRuleConfig(prefill_impl='recurrent_scan_v2', decode_impl='fused')`,
#  i.e. `RaggedGatedDeltaRuleImpl.CHUNKED_KERNEL_PD`).
#
# Verbatim logic from `ragged_gated_delta_rule_wrapper.ragged_gated_delta_rule_wrapper`
# with the impl-selector enum / config dataclass / other-impl branches removed —
# `use_qk_norm_in_gdn` is hard-wired True and `chunk_size`/`BT` to `chunk_size`,
# matching what `gdn_attention.py` passes for this combo.
# =============================================================================


@functools.partial(
    jax.jit,
    donate_argnames=('recurrent_state', ),
    static_argnames=('n_kq', 'n_v', 'd_k', 'd_v', 'chunk_size'),
)
@jax.named_scope('gdn_delta_rule_chunked_kernel_pd')
def _gdn_delta_rule_chunked_kernel_pd(
    mixed_qkv: jnp.ndarray,
    b: jnp.ndarray,
    a: jnp.ndarray,
    recurrent_state: jnp.ndarray,
    A_log: jnp.ndarray,
    dt_bias: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    state_indices: jnp.ndarray,
    distribution: jnp.ndarray,
    has_initial_state: jnp.ndarray,
    *,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    chunk_size: int = 64,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (updated_recurrent_state, output) — state first, matching the
    `_ref` / `_chunked` / `recurrent_scan` / `fused`-adapter convention.

    `distribution` is the 3-tuple `(decode_end, prefill_end, mixed_end)`;
    decode-only iff `distribution[0] == distribution[2]`. `has_initial_state`
    is `bool[max_reqs]` — True iff the request's recurrent slot already holds a
    valid prior state (continuation / prefix-cache hit / running decode); False
    for brand-new prefills on a freshly-allocated (possibly reused) slot, whose
    contents the kernels then zero before the update.
    """
    is_decode_only = distribution[0] == distribution[2]

    def decode_only_branch(_):
        # decode_impl == 'fused'
        qkv_in = jax.nn.silu(mixed_qkv)
        new_state, output = ragged_gated_delta_rule(  # the fused-kernel adapter, defined above
            mixed_qkv=qkv_in,
            b=b,
            a=a,
            recurrent_state=recurrent_state,
            A_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            distribution=distribution,
            has_initial_state=has_initial_state,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
        return new_state, output

    def mixed_prefill_branch(_):
        # prefill_impl == 'recurrent_scan_v2'  (the v2 kernel fuses SiLU, so raw mixed_qkv)
        return recurrent_scan(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            recurrent_state=recurrent_state,
            A_log=A_log,
            dt_bias=dt_bias,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            distribution=distribution,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            chunk_size=chunk_size,
            BT=chunk_size,
            use_qk_norm_in_gdn=True,
            has_initial_state=has_initial_state,
        )

    return jax.lax.cond(is_decode_only,
                        decode_only_branch,
                        mixed_prefill_branch,
                        operand=None)


# =============================================================================
# The end-to-end GDN "core attention" — reimplemented from
# `tpu_inference/layers/common/gdn_attention.py::run_jax_gdn_attention_local`,
# minus the `jax.shard_map` (the harness lowers one abstract TPU core, and the
# DSL conversion will handle sharding separately if needed).
#
# `gdn_attention_op.py` (the torch<->JAX bridge) wraps THIS: it reorders
# mixed_qkv / conv_weight from [QQ|KK|VV] to interleaved [QK|QK|...|VV] for TP,
# slices conv_state to [:, :kernel_size-1, :], pulls (conv_state, recurrent_state)
# from the kv-cache, calls run_jax_gdn_attention (the shard_map version of this),
# writes (new_conv_state, new_recurrent_state) back, and copies output ->
# core_attn_out. Then the layer does RMSNormGated(core_attn_out, z) -> out_proj.
# =============================================================================


def gdn_core_attention(
    mixed_qkv: jnp.ndarray,        # [num_tokens, dim], dim = 2*n_kq*d_k + n_v*d_v
    b: jnp.ndarray,                # [num_tokens, n_v]   raw beta (sigmoid applied in the kernel)
    a: jnp.ndarray,                # [num_tokens, n_v]   raw alpha (gate transform applied in the kernel)
    conv_state: jnp.ndarray,       # [num_blocks, kernel_size-1, dim], num_blocks >= max_reqs+1, slot 0 = null
    recurrent_state: jnp.ndarray,  # [num_blocks, n_v, d_k, d_v]
    conv_weight: jnp.ndarray,      # [dim, 1, kernel_size]
    conv_bias: Optional[jnp.ndarray],  # [dim] or None
    A_log: jnp.ndarray,            # [n_v]  float32  (per-head log gate)
    dt_bias: jnp.ndarray,          # [n_v]  float32  (per-head bias)
    query_start_loc: jnp.ndarray,  # [max_reqs+1] int32  (prefix sums of per-request token counts; padded tail = last value)
    state_indices: jnp.ndarray,    # [max_reqs]   int32  (persistent-batch slot -> physical slot in the state pool)
    distribution: jnp.ndarray,     # [3]          int32  (decode_end, prefill_end, mixed_end)
    seq_lens: jnp.ndarray,         # [max_reqs]   int32  (total seq length per request = computed + scheduled)
    *,
    n_kq: int,                     # number of key/query heads
    n_v: int,                      # number of value heads  (multiple of n_kq -> GQA)
    d_k: int,                      # key/query head dim   (multiple of num_lanes = 128)
    d_v: int,                      # value head dim       (multiple of 128)
    kernel_size: int,              # causal conv width    (e.g. 4)
    chunk_size: int = 64,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """conv1d -> gated-delta-rule (chunked_kernel_pd). Returns
    ((new_conv_state, new_recurrent_state), output) with
    output: [num_tokens, n_v*d_v].

    `has_initial_state[i] = (seq_lens[i] - query_lens[i]) > 0` — i.e. the
    request has prior computed tokens (continuation / prefix-cache hit / decode);
    False for brand-new prefills. Threaded into both the conv1d and the delta
    rule so a freshly-allocated (reused) slot can't leak its previous tenant's
    state; mirrors GPU's `initial_state[~has_initial_state, ...] = 0`.
    """
    max_reqs = seq_lens.shape[0]
    query_lens = query_start_loc[1:max_reqs + 1] - query_start_loc[:max_reqs]
    has_initial_state = (seq_lens - query_lens) > 0

    # 1. Causal depthwise conv1d over the ragged token stream (pure XLA),
    #    also updating the conv_state cache in-place (donated).
    out_mixed_qkv, new_conv_state = ragged_conv1d(
        mixed_qkv,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
    )

    # 2. Gated delta rule — the fastest combo: v2 Mosaic kernel for the
    #    mixed/prefill branch, fused-decode Mosaic kernel for the decode-only
    #    branch (which `jax.lax.cond` selects at runtime; both branches are
    #    traced, so the lowered module contains every pallas_call).
    new_recurrent_state, output = _gdn_delta_rule_chunked_kernel_pd(
        mixed_qkv=out_mixed_qkv,
        b=b,
        a=a,
        recurrent_state=recurrent_state,
        A_log=A_log,
        dt_bias=dt_bias,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        distribution=distribution,
        has_initial_state=has_initial_state,
        n_kq=n_kq,
        n_v=n_v,
        d_k=d_k,
        d_v=d_v,
        chunk_size=chunk_size,
    )

    return (new_conv_state, new_recurrent_state), output


# =============================================================================
# dsl-harness adapter — `build_args(cfg)` + standalone `main()`.
# Mirrors kernels/mosaic_tpu/py/{ragged_paged,gdn_decode,gdn_prefill_varlen}.py:
# the harness runner imports this module, reads `build_args(cfg) -> (args,
# static_kwargs)`, then `jax.jit(gdn_core_attention, backend="cpu",
# static_argnames=...).lower(*args, **static_kwargs)` under an abstract TPU mesh
# (with the pallas_call TPU lowering rule registered on the cpu backend), and
# captures the printed Mosaic module(s). `gdn_core_attention` traces BOTH cond
# branches, so the lowering emits every pallas_call (P4, P3 + its metadata
# kernel, P5) — the harness's `_MODULE_RE.findall(...)[-1]` captures the last
# one; to inspect a specific kernel in isolation, call it directly (see the
# `--only` flag in main()).
# =============================================================================


_DTYPE_MAP = {
    "i8": jnp.int8, "i16": jnp.int16, "i32": jnp.int32,
    "f16": jnp.float16, "bf16": jnp.bfloat16, "f32": jnp.float32,
}


def _dtype(name: Any, default: jnp.dtype) -> jnp.dtype:
    return _DTYPE_MAP.get(name, default) if isinstance(name, str) else default


def _gdn_shapes(cfg: Dict[str, Any]):
    """Build the (ShapeDtypeStruct args, static_kwargs) for gdn_core_attention.

    cfg keys (all optional, with Qwen3.5-ish defaults):
      dtype / state_dtype / conv_state_dtype : "bf16" | "f16" | "f32"
      num_k_heads (n_kq) / num_v_heads (n_v) / head_k_dim (d_k) / head_v_dim (d_v)
      kernel_size : causal conv width (default 4)
      chunk_size  : v2 chunk / decode-batch size (default 64)
      decode_lens : list[int] — one per decode request (each must be 1)
      prefill_lens: list[int] — one per prefill request (>= 1)
      context_lens: list[int] — prior computed tokens per request, len == num_seqs;
                    must be > 0 for decode requests; 0 for fresh prefills.
                    (default: [128]*num_decodes + [0]*num_prefills)
      num_blocks  : recurrent/conv state pool size (default num_seqs + 1)
      with_conv_bias : bool (default True)
    """
    dtype = _dtype(cfg.get("dtype"), jnp.bfloat16)
    state_dtype = _dtype(cfg.get("state_dtype"), jnp.bfloat16)
    conv_state_dtype = _dtype(cfg.get("conv_state_dtype"), dtype)
    n_kq = int(cfg.get("num_k_heads", 16))
    n_v = int(cfg.get("num_v_heads", 32))
    d_k = int(cfg.get("head_k_dim", 128))
    d_v = int(cfg.get("head_v_dim", 128))
    kernel_size = int(cfg.get("kernel_size", 4))
    chunk_size = int(cfg.get("chunk_size", 64))
    if n_v % n_kq != 0:
        raise ValueError(f"n_v={n_v} must be a multiple of n_kq={n_kq}")
    dim = 2 * n_kq * d_k + n_v * d_v

    decode_lens = [int(x) for x in cfg.get("decode_lens", [1, 1, 1, 1])]
    prefill_lens = [int(x) for x in cfg.get("prefill_lens", [256, 128])]
    if any(x != 1 for x in decode_lens):
        raise ValueError(f"decode_lens entries must all be 1, got {decode_lens}")
    seqlens = decode_lens + prefill_lens          # scheduled token counts this step
    num_seqs = len(seqlens)
    decode_end = len(decode_lens)
    mixed_end = prefill_end = num_seqs            # no padding-within-valid in this harness config
    distribution_val = (decode_end, prefill_end, mixed_end)

    context_lens = cfg.get("context_lens")
    if context_lens is None:
        context_lens = [128] * decode_end + [0] * len(prefill_lens)
    context_lens = [int(x) for x in context_lens]
    if len(context_lens) != num_seqs:
        raise ValueError(f"context_lens must have length {num_seqs}, got {len(context_lens)}")
    if any(c <= 0 for c in context_lens[:decode_end]):
        raise ValueError("decode requests must have context_lens > 0")
    seq_lens_val = [seqlens[i] + context_lens[i] for i in range(num_seqs)]

    num_tokens = sum(seqlens)
    num_blocks = int(cfg.get("num_blocks", num_seqs + 1))
    if num_blocks < num_seqs + 1:
        raise ValueError(f"num_blocks={num_blocks} must be >= num_seqs+1={num_seqs+1}")
    with_conv_bias = bool(cfg.get("with_conv_bias", True))

    mixed_qkv = jax.ShapeDtypeStruct((num_tokens, dim), dtype)
    b = jax.ShapeDtypeStruct((num_tokens, n_v), jnp.float32)
    a = jax.ShapeDtypeStruct((num_tokens, n_v), jnp.float32)
    conv_state = jax.ShapeDtypeStruct((num_blocks, kernel_size - 1, dim), conv_state_dtype)
    recurrent_state = jax.ShapeDtypeStruct((num_blocks, n_v, d_k, d_v), state_dtype)
    conv_weight = jax.ShapeDtypeStruct((dim, 1, kernel_size), dtype)
    conv_bias = jax.ShapeDtypeStruct((dim,), dtype) if with_conv_bias else None
    A_log = jax.ShapeDtypeStruct((n_v,), jnp.float32)
    dt_bias = jax.ShapeDtypeStruct((n_v,), jnp.float32)
    query_start_loc = jax.ShapeDtypeStruct((num_seqs + 1,), jnp.int32)
    state_indices = jax.ShapeDtypeStruct((num_seqs,), jnp.int32)
    distribution = jax.ShapeDtypeStruct((3,), jnp.int32)
    seq_lens = jax.ShapeDtypeStruct((num_seqs,), jnp.int32)

    args = [mixed_qkv, b, a, conv_state, recurrent_state, conv_weight, conv_bias,
            A_log, dt_bias, query_start_loc, state_indices, distribution, seq_lens]
    static_kwargs = {
        "n_kq": n_kq, "n_v": n_v, "d_k": d_k, "d_v": d_v,
        "kernel_size": kernel_size, "chunk_size": chunk_size,
    }
    # Echo the derived ints so the caller can build the matching runtime values.
    static_kwargs["_meta"] = {
        "num_tokens": num_tokens, "dim": dim, "num_seqs": num_seqs,
        "num_blocks": num_blocks, "distribution": distribution_val,
        "seq_lens": tuple(seq_lens_val), "seqlens": tuple(seqlens),
    }
    return args, static_kwargs


__all__ = ["gdn_core_attention", "build_args"]


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    """Harness entry: returns (args, static_kwargs) for `gdn_core_attention`.

    `_meta` is a debug echo (not a kernel arg); strip it for the actual jit call.
    """
    args, static_kwargs = _gdn_shapes(cfg)
    static_kwargs = {k: v for k, v in static_kwargs.items() if k != "_meta"}
    return args, static_kwargs


# ---------------------------------------------------------------------------
# Standalone CLI: `python gdn_fast.py [--only {full,decode,prefill}] [--print-mlir]`
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    import contextlib
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--state-dtype", default="bf16")
    p.add_argument("--num-k-heads", type=int, default=16)
    p.add_argument("--num-v-heads", type=int, default=32)
    p.add_argument("--head-k-dim", type=int, default=128)
    p.add_argument("--head-v-dim", type=int, default=128)
    p.add_argument("--kernel-size", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--decode-lens", default="1,1,1,1",
                   help="comma list, each must be 1 (one decode token per request)")
    p.add_argument("--prefill-lens", default="256,128", help="comma list")
    p.add_argument("--context-lens", default=None,
                   help="comma list, one per request; >0 for decodes, 0 for fresh prefills")
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--no-conv-bias", action="store_true")
    p.add_argument("--only", choices=["full", "decode", "prefill"],
                   default="full",
                   help="full = gdn_core_attention (conv1d + cond[P4 | P5]); "
                        "decode = fused_decoding_gdn (P4) in isolation; "
                        "prefill = recurrent_scan (P5) in isolation")
    p.add_argument("--print-mlir", action="store_true")
    args = p.parse_args()

    def _split_ints(s):
        return [int(x) for x in s.split(",") if x.strip() != ""]

    cfg: Dict[str, Any] = {
        "dtype": args.dtype, "state_dtype": args.state_dtype,
        "num_k_heads": args.num_k_heads, "num_v_heads": args.num_v_heads,
        "head_k_dim": args.head_k_dim, "head_v_dim": args.head_v_dim,
        "kernel_size": args.kernel_size, "chunk_size": args.chunk_size,
        "decode_lens": _split_ints(args.decode_lens),
        "prefill_lens": _split_ints(args.prefill_lens),
        "with_conv_bias": not args.no_conv_bias,
    }
    if args.context_lens is not None:
        cfg["context_lens"] = _split_ints(args.context_lens)
    if args.num_blocks is not None:
        cfg["num_blocks"] = args.num_blocks

    pos_args, static_full = _gdn_shapes(cfg)
    meta = static_full.pop("_meta")
    n_kq, n_v, d_k, d_v = static_full["n_kq"], static_full["n_v"], static_full["d_k"], static_full["d_v"]
    chunk_size = static_full["chunk_size"]
    num_tokens, num_blocks, num_seqs = meta["num_tokens"], meta["num_blocks"], meta["num_seqs"]
    decode_end = meta["distribution"][0]
    dtype = _dtype(cfg["dtype"], jnp.bfloat16)
    state_dtype = _dtype(cfg.get("state_dtype"), jnp.bfloat16)
    key_dim = n_kq * d_k
    # The harness lowers under an abstract "TPU v5 lite" (128 lanes). The kernels
    # themselves read `pltpu.get_tpu_info().num_lanes` at lowering time (correct
    # for whatever target); we only need this constant to build the dummy
    # ShapeDtypeStructs for the `--only decode` isolation path (where
    # `b`/`A_log`/`dt_bias` are already broadcast to the lane width by the caller).
    _NUM_LANES = 128
    nl = _NUM_LANES
    _A_LOG_BC = jnp.broadcast_to(jnp.zeros((n_v,), jnp.float32)[:, None], (n_v, nl))
    _DT_BIAS_BC = jnp.broadcast_to(jnp.zeros((n_v,), jnp.float32)[:, None], (n_v, nl))

    # --- abstract-TPU-mesh + CPU pallas_call lowering shim (same as the other files) ---
    from jax._src.interpreters import mlir
    from jax._src.pallas import pallas_call

    def _cpu_lowering_tpu_wrapper(ctx, *in_nodes, **params):
        params.pop("backend", None)
        params.pop("which_linear", None)
        if "out_shapes" in params:
            params["out_avals"] = params.pop("out_shapes")
        from jax._src.pallas.mosaic.pallas_call_registration import (
            pallas_call_tpu_lowering_rule,
        )
        return pallas_call_tpu_lowering_rule(ctx, *in_nodes, **params)

    mlir.register_lowering(pallas_call.pallas_call_p,
                           _cpu_lowering_tpu_wrapper, platform="cpu")

    @contextlib.contextmanager
    def _tpu_abstract_mesh_context():
        mesh = jax.sharding.AbstractMesh(
            axis_sizes=(1,), axis_names=("tpu_core",),
            abstract_device=jax.sharding.AbstractDevice(
                device_kind="TPU v5 lite", num_cores=1,
            ),
        )
        with jax.sharding.use_abstract_mesh(mesh):
            yield

    _scale = float(d_k) ** -0.5
    if args.only == "full":
        fn = gdn_core_attention
        call_args, call_static = pos_args, static_full
        static_names = list(static_full.keys())
    elif args.only == "decode":
        # P4 (`fused_decoding_gdn`) in isolation. The caller (`fused_gdn`) has
        # already broadcast `b` -> [T, H_v, num_lanes] and A_log/dt_bias ->
        # [H_v, num_lanes]; wrap so we can `.lower()` it like the others.
        # q/k: [T,H_qk,K]; v: [T,H_v,V]; g: [T,H_v,K] f32; state: [num_blocks,H_v,K,V];
        # state_indices: [max_reqs] i32; distribution: [2] i32 (decode_end, total).
        def _decode_entry(q, k, v, g, initial_state, state_indices, distribution, b):
            return fused_decoding_gdn(
                q, k, v, g, initial_state, state_indices, distribution, b,
                scale=_scale, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                A_log=_A_LOG_BC, dt_bias=_DT_BIAS_BC, lower_bound=None)
        fn = _decode_entry
        call_args = [
            jax.ShapeDtypeStruct((num_tokens, n_kq, d_k), dtype),          # q
            jax.ShapeDtypeStruct((num_tokens, n_kq, d_k), dtype),          # k
            jax.ShapeDtypeStruct((num_tokens, n_v, d_v), dtype),           # v
            jax.ShapeDtypeStruct((num_tokens, n_v, d_k), jnp.float32),     # g
            jax.ShapeDtypeStruct((num_blocks, n_v, d_k, d_v), state_dtype),# initial_state
            jax.ShapeDtypeStruct((num_seqs,), jnp.int32),                  # state_indices
            jax.ShapeDtypeStruct((2,), jnp.int32),                         # distribution (decode_end, total)
            jax.ShapeDtypeStruct((num_tokens, n_v, nl), dtype),            # b (already lane-broadcast)
        ]
        call_static = {}
        static_names = []
    elif args.only == "prefill":
        # P5 (`recurrent_scan`) in isolation. mixed_qkv: [T, 2*n_kq*d_k + n_v*d_v];
        # b,a: [T,n_v]; recurrent_state: [num_blocks,n_v,d_k,d_v]; A_log/dt_bias: [n_v];
        # query_start_loc: [num_seqs+1] i32; state_indices: [num_seqs] i32; distribution: [3] i32.
        fn = recurrent_scan
        dim = 2 * key_dim + n_v * d_v
        call_args = [
            jax.ShapeDtypeStruct((num_tokens, dim), dtype),                # mixed_qkv
            jax.ShapeDtypeStruct((num_tokens, n_v), jnp.float32),          # b
            jax.ShapeDtypeStruct((num_tokens, n_v), jnp.float32),          # a
            jax.ShapeDtypeStruct((num_blocks, n_v, d_k, d_v), state_dtype),# recurrent_state
            jax.ShapeDtypeStruct((n_v,), jnp.float32),                     # A_log
            jax.ShapeDtypeStruct((n_v,), jnp.float32),                     # dt_bias
            jax.ShapeDtypeStruct((num_seqs + 1,), jnp.int32),              # query_start_loc
            jax.ShapeDtypeStruct((num_seqs,), jnp.int32),                  # state_indices
            jax.ShapeDtypeStruct((3,), jnp.int32),                         # distribution
        ]
        call_static = {"n_kq": n_kq, "n_v": n_v, "d_k": d_k, "d_v": d_v,
                       "chunk_size": chunk_size, "BT": chunk_size, "use_qk_norm_in_gdn": True}
        static_names = list(call_static.keys())
    else:  # unreachable — `choices` restricts `--only` to {full, decode, prefill}
        raise ValueError(f"unknown --only: {args.only!r}")

    with _tpu_abstract_mesh_context():
        lowered = jax.jit(fn, backend="cpu", static_argnames=tuple(static_names)).lower(
            *call_args, **call_static)

    if args.print_mlir:
        print(lowered.compiler_ir())
    else:
        print(f"lowered ok: only={args.only}  "
              f"n_kq={n_kq} n_v={n_v} d_k={d_k} d_v={d_v} "
              f"num_tokens={num_tokens} num_seqs={num_seqs} num_blocks={num_blocks} "
              f"decode_end={decode_end} chunk_size={chunk_size}  (meta={meta})")


if __name__ == "__main__":
    main()
