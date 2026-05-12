"""GDN fused decode Mosaic kernel ("P4") — harness reference for the Zig DSL conversion.

Self-contained extract of `fused_decoding_gdn` (the fast single-token recurrent
GDN decode kernel) + its `validate_gdn_inputs` helper, verbatim from
tpu-inference (see GDN-Google/tpu_inference/kernels/gdn/fused_gdn_decode_kernel.py
and fused_gdn_kernel_common.py), with only these adaptations:
  * the two source modules' Apache headers collapsed into this docstring;
  * imports consolidated at the top;
  * `from tpu_inference. ...` import removed (validate_gdn_inputs is a section below);
  * `get_default_block_sizes` -> `_get_default_block_sizes_decode` (the source name);
  * `debug=True` added to the `pl.pallas_call` so the harness runner captures the
    post-lowering Mosaic module from stdout (`tools/dsl-harness/mosaic_tpu/runner.py`);
  * a thin `gdn_decode_fwd(...)` wrapper (makes A_log/dt_bias positional, hardwires
    `use_qk_l2norm_in_kernel=True`, `use_gate_in_kernel=True`, `scale=d_k**-0.5`) +
    `build_args(cfg)` + `main()` harness adapter at the bottom.

Pair file (the Zig DSL conversion, to be matched IR-wise): zig/gdn_decode_fast.zig.
The full bundled reference (conv1d + this + P5 + glue) is py/gdn_fast.py.
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
        debug=True,  # harness: print the post-lowering Mosaic module to stdout
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
# dsl-harness adapter — thin entry wrapper + build_args + main.
# =============================================================================


def gdn_decode_fwd(
    q: jax.Array,                 # [T, H_qk, K]
    k: jax.Array,                 # [T, H_qk, K]
    v: jax.Array,                 # [T, H_v, V]
    g: jax.Array,                 # [T, H_v, K]  float32  (raw alpha; gate transform applied in-kernel)
    initial_state: jax.Array,     # [num_states, H_v, K, V]
    state_indices: jax.Array,     # [max_num_req]  int32
    distribution: jax.Array,      # [2]  int32  (decode_end, total)
    b: jax.Array,                 # [T, H_v, num_lanes]  (raw beta; sigmoid in-kernel)
    A_log: jax.Array,             # [H_v, num_lanes]  float32
    dt_bias: jax.Array,           # [H_v, num_lanes]  float32
    *,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    lower_bound: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Calls `fused_decoding_gdn` with feature flags routed from the harness cfg
    (scale = d_k ** -0.5 always; flags default to the Qwen3.5 GDN settings).
    Returns (o [T, H_v, V], updated_state [num_states, H_v, K, V])."""
    scale = float(q.shape[-1]) ** -0.5
    return fused_decoding_gdn(
        q, k, v, g, initial_state, state_indices, distribution, b,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )


_DTYPE_MAP = {
    "i8": jnp.int8, "i16": jnp.int16, "i32": jnp.int32,
    "f16": jnp.float16, "bf16": jnp.bfloat16, "f32": jnp.float32,
}


def _dtype(name: Any, default) -> Any:
    return _DTYPE_MAP.get(name, default) if isinstance(name, str) else default


def _decode_shapes(cfg: Dict[str, Any]):
    """(ShapeDtypeStruct args, static_kwargs={}) for gdn_decode_fwd, + a `_meta` echo.

    cfg keys (all optional):
      dtype / state_dtype : "bf16" | "f16" | "f32"  (q/k/v/b dtype; state dtype)
      num_k_heads / num_v_heads / head_k_dim / head_v_dim
      num_decode_tokens : how many decode tokens (= decode requests) this step
      num_states : recurrent-state pool size (default num_decode_tokens + 1)
    Notes: bt (the emit_pipeline window) is auto-picked by
    `_get_default_block_sizes_decode` from the (abstract-mesh) VMEM size and the
    state-scratch footprint — larger num_v_heads / head dims => smaller bt. The
    func name in the lowered IR is `decoding_gdn-bt_<bt>`.
    """
    dtype = _dtype(cfg.get("dtype"), jnp.bfloat16)
    state_dtype = _dtype(cfg.get("state_dtype"), jnp.bfloat16)
    n_kq = int(cfg.get("num_k_heads", 16))
    n_v = int(cfg.get("num_v_heads", 32))
    d_k = int(cfg.get("head_k_dim", 128))
    d_v = int(cfg.get("head_v_dim", 128))
    if n_v % n_kq != 0:
        raise ValueError(f"n_v={n_v} must be a multiple of n_kq={n_kq}")
    num_tokens = int(cfg.get("num_decode_tokens", 4))
    num_states = int(cfg.get("num_states", num_tokens + 1))
    nl = 128  # num_lanes; the kernel reads pltpu.get_tpu_info().num_lanes at lowering time (128 everywhere)

    # Feature flags — routed to static_kwargs of `gdn_decode_fwd`. The Zig Cfg
    # mirrors the same defaults so a vanilla cfg matches Qwen3.5 GDN.
    use_qk_l2norm = bool(cfg.get("use_qk_l2norm", True))
    use_gate_in_kernel = bool(cfg.get("use_gate_in_kernel", True))
    has_dt_bias = bool(cfg.get("has_dt_bias", True))
    has_b = bool(cfg.get("has_b", True))
    lower_bound = cfg.get("lower_bound")
    lower_bound = float(lower_bound) if lower_bound is not None else None

    # Optional operands — `None` makes Pallas drop the operand from the lowered
    # func.func signature entirely. `A_log` needs `use_gate_in_kernel`; `dt_bias`
    # additionally needs `has_dt_bias`; `b` is independently gated by `has_b`.
    q = jax.ShapeDtypeStruct((num_tokens, n_kq, d_k), dtype)
    k = jax.ShapeDtypeStruct((num_tokens, n_kq, d_k), dtype)
    v = jax.ShapeDtypeStruct((num_tokens, n_v, d_v), dtype)
    g = jax.ShapeDtypeStruct((num_tokens, n_v, d_k), jnp.float32)
    initial_state = jax.ShapeDtypeStruct((num_states, n_v, d_k, d_v), state_dtype)
    state_indices = jax.ShapeDtypeStruct((num_tokens,), jnp.int32)
    distribution = jax.ShapeDtypeStruct((2,), jnp.int32)         # (decode_end, total)
    b = jax.ShapeDtypeStruct((num_tokens, n_v, nl), dtype) if has_b else None
    A_log = jax.ShapeDtypeStruct((n_v, nl), jnp.float32) if use_gate_in_kernel else None
    dt_bias = jax.ShapeDtypeStruct((n_v, nl), jnp.float32) if (use_gate_in_kernel and has_dt_bias) else None

    args = [q, k, v, g, initial_state, state_indices, distribution, b, A_log, dt_bias]
    static_kwargs: Dict[str, Any] = {
        "use_qk_l2norm_in_kernel": use_qk_l2norm,
        "use_gate_in_kernel": use_gate_in_kernel,
        "lower_bound": lower_bound,
    }
    static_kwargs["_meta"] = {
        "num_tokens": num_tokens, "num_states": num_states,
        "n_kq": n_kq, "n_v": n_v, "d_k": d_k, "d_v": d_v, "num_lanes": nl,
    }
    return args, static_kwargs


__all__ = ["gdn_decode_fwd", "fused_decoding_gdn", "build_args"]


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    """Harness entry: (args, static_kwargs) for `gdn_decode_fwd`. `_meta` stripped."""
    args, static_kwargs = _decode_shapes(cfg)
    static_kwargs = {kk: vv for kk, vv in static_kwargs.items() if kk != "_meta"}
    return args, static_kwargs


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
    p.add_argument("--num-decode-tokens", type=int, default=4)
    p.add_argument("--num-states", type=int, default=None)
    p.add_argument("--print-mlir", action="store_true",
                   help="print compiler_ir() (StableHLO). The Mosaic module is printed by debug=True regardless.")
    args = p.parse_args()

    cfg: Dict[str, Any] = {
        "dtype": args.dtype, "state_dtype": args.state_dtype,
        "num_k_heads": args.num_k_heads, "num_v_heads": args.num_v_heads,
        "head_k_dim": args.head_k_dim, "head_v_dim": args.head_v_dim,
        "num_decode_tokens": args.num_decode_tokens,
    }
    if args.num_states is not None:
        cfg["num_states"] = args.num_states
    pos_args, static_full = _decode_shapes(cfg)
    meta = static_full.pop("_meta")

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

    with _tpu_abstract_mesh_context():
        lowered = jax.jit(gdn_decode_fwd, backend="cpu").lower(*pos_args)

    if args.print_mlir:
        print(lowered.compiler_ir())
    else:
        print(f"lowered ok: {meta}")


if __name__ == "__main__":
    main()
