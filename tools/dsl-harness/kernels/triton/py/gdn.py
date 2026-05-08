# Vendored from Flash-Linear-Attention @ 9c3188511cfc63171dc59d5e3d4395976ab4be5d
# with a `build_args(cfg)` adapter for the harness.

import math
from typing import Any, Dict, Tuple

import triton
import triton.language as tl

from triton_helpers import FakeTensor


@triton.jit
def dummy_kernel(x_ptr, y_ptr):
    x = tl.load(x_ptr)
    tl.store(y_ptr, x)


@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))


@triton.jit
def exp2(x):
    pass


@triton.jit
def fused_recurrent_gated_delta_rule_fwd_kernel_ptr(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    h0_ptr,
    cu_seqlens_ptr,
    o_ptr,
    ht_ptr,
    gk: tl.constexpr,
    gv: tl.constexpr,
    scale: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    fused_recurrent_gated_delta_rule_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        gk,
        gv,
        beta_ptr,
        o_ptr,
        h0_ptr,
        ht_ptr,
        cu_seqlens_ptr,
        scale,
        T,
        H,
        HV,
        K,
        V,
        BK,
        BV,
        USE_G,
        USE_GK,
        USE_GV,
        USE_QK_L2NORM_IN_KERNEL,
        IS_BETA_HEADWISE,
        USE_INITIAL_STATE,
        STORE_FINAL_STATE,
        USE_EXP2,
        TRANSPOSE_STATE,
        IS_VARLEN,
    )


@triton.jit
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    gk,
    gv,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    TRANSPOSE_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if USE_G:
        p_g = g + bos * HV + i_hv
    if USE_GK:
        p_gk = gk + (bos * HV + i_hv) * K + o_k
    if USE_GV:
        p_gv = gv + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * HV + i_hv
    else:
        p_beta = beta + (bos * HV + i_hv) * V + o_v

    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    if TRANSPOSE_STATE:
        mask_h = mask_v[:, None] & mask_k[None, :]
    else:
        mask_h = mask_k[:, None] & mask_v[None, :]

    if TRANSPOSE_STATE:
        b_h = tl.zeros([BV, BK], dtype=tl.float32)
    else:
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if TRANSPOSE_STATE:
            p_h0 = h0 + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)

        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            if USE_EXP2:
                b_h *= exp2(b_g)
            else:
                b_h *= exp(b_g)

        if USE_GK:
            b_gk = tl.load(p_gk).to(tl.float32)
            if USE_EXP2:
                if TRANSPOSE_STATE:
                    b_h *= exp2(b_gk[None, :])
                else:
                    b_h *= exp2(b_gk[:, None])
            else:
                if TRANSPOSE_STATE:
                    b_h *= exp(b_gk[None, :])
                else:
                    b_h *= exp(b_gk[:, None])

        if USE_GV:
            b_gv = tl.load(p_gv).to(tl.float32)
            if USE_EXP2:
                if TRANSPOSE_STATE:
                    b_h *= exp2(b_gv[:, None])
                else:
                    b_h *= exp2(b_gv[None, :])
            else:
                if TRANSPOSE_STATE:
                    b_h *= exp(b_gv[:, None])
                else:
                    b_h *= exp(b_gv[None, :])

        if TRANSPOSE_STATE:
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[None, :], 1))
            b_h += b_v[:, None] * b_k[None, :]
            b_o = tl.sum(b_h * b_q[None, :], 1)
        else:
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
            b_h += b_k[:, None] * b_v
            b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        if USE_G:
            p_g += HV
        if USE_GK:
            p_gk += HV * K
        if USE_GV:
            p_gv += HV * V
        p_beta += HV * (1 if IS_BETA_HEADWISE else V)
        p_o += HV * V

    if STORE_FINAL_STATE:
        if TRANSPOSE_STATE:
            p_ht = ht + i_nh * K * V + o_v[:, None] * K + o_k[None, :]
        else:
            p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    # Defaults match the qwen3_5 constexpr config (USE_G=True, IS_VARLEN=True, …).
    num_tokens = int(cfg.get("T", 64))
    num_qk_heads = int(cfg.get("H", 4))
    num_v_heads = int(cfg.get("HV", 16))
    key_dim = int(cfg.get("K", 32))
    value_dim = int(cfg.get("V", 64))
    num_sequences = 2  # NS+1 below assumes 2 sequences.
    bk = int(cfg.get("BK", 32))
    bv = int(cfg.get("BV", 8))
    scale = float(cfg.get("scale", 1.0 / math.sqrt(key_dim)))

    use_g = bool(cfg.get("USE_G", True))
    use_gk = bool(cfg.get("USE_GK", False))
    use_gv = bool(cfg.get("USE_GV", False))
    use_qk_l2norm_in_kernel = bool(cfg.get("USE_QK_L2NORM_IN_KERNEL", True))
    is_beta_headwise = bool(cfg.get("IS_BETA_HEADWISE", True))
    use_initial_state = bool(cfg.get("USE_INITIAL_STATE", True))
    store_final_state = bool(cfg.get("STORE_FINAL_STATE", True))
    use_exp2 = bool(cfg.get("USE_EXP2", False))
    transpose_state = bool(cfg.get("TRANSPOSE_STATE", False))
    is_varlen = bool(cfg.get("IS_VARLEN", True))

    qk_shape = (1, num_tokens, num_qk_heads, key_dim)
    v_shape = (1, num_tokens, num_v_heads, value_dim)
    gbeta_shape = (1, num_tokens, num_v_heads)
    state_shape = (num_sequences, num_v_heads, key_dim, value_dim)
    cu_seqlens_shape = (num_sequences + 1,)

    args = [
        FakeTensor("bf16", qk_shape),       # q_ptr
        FakeTensor("bf16", qk_shape),       # k_ptr
        FakeTensor("bf16", v_shape),        # v_ptr
        FakeTensor("fp32", gbeta_shape),    # g_ptr
        FakeTensor("bf16", gbeta_shape),    # beta_ptr
        FakeTensor("fp32", state_shape),    # h0_ptr
        FakeTensor("i32", cu_seqlens_shape), # cu_seqlens_ptr
        FakeTensor("bf16", v_shape),        # o_ptr
        FakeTensor("fp32", state_shape),    # ht_ptr
        None,                                 # gk (constexpr)
        None,                                 # gv (constexpr)
        scale,                                # scale (constexpr)
        num_tokens,                           # T
        num_qk_heads,                         # H
        num_v_heads,                          # HV
        key_dim,                              # K
        value_dim,                            # V
        bk,                                   # BK
        bv,                                   # BV
        use_g,
        use_gk,
        use_gv,
        use_qk_l2norm_in_kernel,
        is_beta_headwise,
        use_initial_state,
        store_final_state,
        use_exp2,
        transpose_state,
        is_varlen,
    ]
    return args, {}
