"""Faithful Pallas TPU port of `examples/triton_emitter/kernels_zig/gdn.zig`.

Same constexpr configuration as the Triton port:

  USE_G=True, USE_GK=False, USE_GV=False
  USE_QK_L2NORM_IN_KERNEL=True
  IS_BETA_HEADWISE=True
  USE_INITIAL_STATE=True
  STORE_FINAL_STATE=True
  USE_EXP2=False, TRANSPOSE_STATE=False, IS_VARLEN=True

Layout matches the Triton kernel's varlen contract:

  q, k         : [1, T_total, H,  K]   bf16     # token-major flat tensors
  v, o         : [1, T_total, HV, V]   bf16
  g            : [1, T_total, HV]      f32
  beta         : [1, T_total, HV]      bf16
  h0, ht       : [N, HV, K, V]         f32      # per-sequence recurrent state
  cu_seqlens   : [N+1]                 i32      # bos/eos for each sequence

Grid is `(V // BV, N * HV)`; per-program work is one value-block × one
value-head × one sequence, exactly as in `gdn.zig`. With `debug=True`,
`pl.pallas_call` prints the Mosaic IR module during lowering.

Run:
    uv run --with "jax" --with "jaxlib" python gdn.py
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


# Demo shapes — pinned to small values for fast lowering.
NUM_SEQUENCES = 2                              # N
SEQ_LEN = 8                                    # per-sequence length
T_TOTAL = NUM_SEQUENCES * SEQ_LEN
NUM_QK_HEADS = 2                               # H
NUM_V_HEADS = 4                                # HV
HV_PER_H = NUM_V_HEADS // NUM_QK_HEADS         # HV // H
KEY_DIM = 128                                  # K
VALUE_DIM = 128                                # V
BK = KEY_DIM                                   # full-K tile
BV = VALUE_DIM                                 # full-V tile
SCALE = 1.0 / math.sqrt(KEY_DIM)


def kernel(q_ref, k_ref, v_ref, g_ref, beta_ref, h0_ref, cu_seqlens_ref,
           o_ref, ht_ref):
    # i_v, i_nh = program_id(0), program_id(1)
    i_v = pl.program_id(0)
    i_nh = pl.program_id(1)
    # i_n = i_nh // HV; i_hv = i_nh % HV; i_h = i_hv // (HV // H)
    # Use truncating div/rem (not Python floor-div) to mirror Triton's div.S/rem.S
    # and avoid the sign-checking floor-divide expansion.
    i_n = lax.div(i_nh, jnp.int32(NUM_V_HEADS))
    i_hv = lax.rem(i_nh, jnp.int32(NUM_V_HEADS))
    i_h = lax.div(i_hv, jnp.int32(HV_PER_H))

    # bos, eos = cu_seqlens[i_n], cu_seqlens[i_n + 1]; T = eos - bos
    bos = cu_seqlens_ref[i_n]
    eos = cu_seqlens_ref[i_n + 1]
    T = eos - bos

    # o_k = arange(0, BK);  o_v = i_v * BV + arange(0, BV)
    o_k = jnp.arange(BK, dtype=jnp.int32)
    o_v = i_v * BV + jnp.arange(BV, dtype=jnp.int32)
    mask_k = o_k < KEY_DIM
    mask_v = o_v < VALUE_DIM
    mask_h = mask_k[:, None] & mask_v[None, :]

    # USE_INITIAL_STATE: b_h = masked load of h0[i_n, i_hv, o_k, o_v] → f32
    h0_tile = h0_ref[i_n, i_hv, :, pl.ds(i_v * BV, BV)].astype(jnp.float32)
    b_h_init = jnp.where(mask_h, h0_tile, 0.0)

    def body(_, b_h):
        # The Triton kernel carries (p_q, p_k, p_v, p_g, p_beta, p_o, b_h)
        # and bumps each pointer per iter. Here the loop index `_t` reaches
        # the same locations via structured indexing; b_h is the only carried
        # value (the recurrent state).
        # Pallas does not expose `_t` directly to fori_loop bodies as a
        # closure variable, so reconstruct the absolute token index `bos+t`.
        # `_` is the integer iteration index from fori_loop.
        t = _

        # b_q = load(q[bos+t, i_h, :], mask=mask_k).to(f32)
        b_q = jnp.where(
            mask_k,
            q_ref[0, bos + t, i_h, :].astype(jnp.float32),
            jnp.float32(0.0),
        )
        # b_k = load(k[bos+t, i_h, :], mask=mask_k).to(f32)
        b_k = jnp.where(
            mask_k,
            k_ref[0, bos + t, i_h, :].astype(jnp.float32),
            jnp.float32(0.0),
        )
        # b_v = load(v[bos+t, i_hv, o_v], mask=mask_v).to(f32)
        b_v = jnp.where(
            mask_v,
            v_ref[0, bos + t, i_hv, pl.ds(i_v * BV, BV)].astype(jnp.float32),
            jnp.float32(0.0),
        )

        # USE_QK_L2NORM_IN_KERNEL
        b_q = b_q / jnp.sqrt(jnp.sum(b_q * b_q) + 1e-6)
        b_k = b_k / jnp.sqrt(jnp.sum(b_k * b_k) + 1e-6)
        # b_q *= scale
        b_q = b_q * SCALE

        # IS_BETA_HEADWISE
        b_beta = beta_ref[0, bos + t, i_hv].astype(jnp.float32)
        # USE_G  (USE_EXP2=False ⇒ exp(g))
        b_g = g_ref[0, bos + t, i_hv].astype(jnp.float32)
        b_h = b_h * jnp.exp(b_g)

        # TRANSPOSE_STATE=False
        # b_v = b_beta * (b_v - sum(b_h * b_k[:, None], 0))
        # b_h += b_k[:, None] * b_v
        # b_o = sum(b_h * b_q[:, None], 0)
        b_v = b_beta * (b_v - jnp.sum(b_h * b_k[:, None], axis=0))
        b_h = b_h + b_k[:, None] * b_v
        b_o = jnp.sum(b_h * b_q[:, None], axis=0)

        # store(o[bos+t, i_hv, o_v], b_o, mask=mask_v)
        o_ref[0, bos + t, i_hv, pl.ds(i_v * BV, BV)] = b_o.astype(o_ref.dtype)
        return b_h

    b_h_final = lax.fori_loop(0, T, body, b_h_init)

    # STORE_FINAL_STATE: store(ht[i_n, i_hv, o_k, o_v], b_h, mask=mask_h)
    ht_ref[i_n, i_hv, :, pl.ds(i_v * BV, BV)] = b_h_final.astype(ht_ref.dtype)


def main() -> None:
    q = jnp.zeros((1, T_TOTAL, NUM_QK_HEADS, KEY_DIM), dtype=jnp.bfloat16)
    k = jnp.zeros((1, T_TOTAL, NUM_QK_HEADS, KEY_DIM), dtype=jnp.bfloat16)
    v = jnp.zeros((1, T_TOTAL, NUM_V_HEADS, VALUE_DIM), dtype=jnp.bfloat16)
    g = jnp.zeros((1, T_TOTAL, NUM_V_HEADS), dtype=jnp.float32)
    beta = jnp.zeros((1, T_TOTAL, NUM_V_HEADS), dtype=jnp.bfloat16)
    h0 = jnp.zeros((NUM_SEQUENCES, NUM_V_HEADS, KEY_DIM, VALUE_DIM),
                   dtype=jnp.float32)
    cu_seqlens = jnp.arange(NUM_SEQUENCES + 1, dtype=jnp.int32) * SEQ_LEN

    vmem = pl.BlockSpec(memory_space=pltpu.MemorySpace.VMEM)
    smem = pl.BlockSpec(memory_space=pltpu.MemorySpace.SMEM)

    launch = pl.pallas_call(
        kernel,
        grid=(VALUE_DIM // BV, NUM_SEQUENCES * NUM_V_HEADS),
        in_specs=[vmem, vmem, vmem, vmem, vmem, vmem, smem],
        out_specs=(vmem, vmem),
        out_shape=(
            jax.ShapeDtypeStruct(v.shape, v.dtype),
            jax.ShapeDtypeStruct(h0.shape, h0.dtype),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
        ),
        name="fused_recurrent_gated_delta_rule_fwd_kernel_ptr",
        debug=True,
    )

    # `lowering_platforms=("tpu",)` forces TPU lowering on a non-TPU host.
    jax.jit(launch).trace(q, k, v, g, beta, h0, cu_seqlens).lower(
        lowering_platforms=("tpu",)
    )


if __name__ == "__main__":
    main()
