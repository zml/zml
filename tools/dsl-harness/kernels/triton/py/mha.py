# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Any, Dict, Tuple

import triton
import triton.language as tl

from triton_helpers import FakeTensor


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    return pid


@triton.jit
def _compute_fp8_scaling_factors(x, fp8_max: tl.constexpr):
    # compute fp8 scaling and descaling factor for a block
    x_amax = tl.max(tl.abs(x))  # NOTE: abs deals with negative values
    x_amax = tl.where(x_amax <= 1e-9, 1e-9, x_amax)
    scale_x = fp8_max / x_amax
    descale_x = x_amax / fp8_max
    return scale_x, descale_x

@triton.jit
def _cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def _load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & (
            offset_second[None, :] < boundary_second
        )
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor


@triton.jit
def _compute_alibi_block(
    alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n, transpose=False
):
    # when seqlen_k and seqlen_q are different we want the diagonal to stick to the bottom right of the attention matrix
    # for casual mask we want something like this where (1 is kept and 0 is masked)
    # seqlen_q = 2 and seqlen_k = 5
    #   1 1 1 1 0
    #   1 1 1 1 1
    # seqlen_q = 5 and seqlen_k = 2
    #        0 0
    #        0 0
    #        0 0
    #        1 0
    #        1 1
    # for alibi the diagonal is 0 indicating no penalty for attending to that spot and increasing penalty for attending further from the diagonal
    # e.g. alibi_slope = 1, seqlen_q = 2, seqlen_k = 5, offs_m = [0, 1, 2, 3], offs_n = [0, 1, 2, 3, 4], transpose = False
    # 1. offs_m[:,None] = [[0],
    #                       [1],
    # 2. offs_m[:,None] + seqlen_k = [[5],
    #                                  [6],
    # 3. offs_m[:,None] + seqlen_k - seqlen_q = [[3],
    #                                             [4],
    # 4. offs_m[:,None] + seqlen_k - seqlen_q - offs_n[None,:] = [[3], - [[0, 1, 2, 3, 4]] =  [[ 3, 2, 1, 0,-1],
    #                                                            [4],                           [ 4, 3, 2, 1, 0]]
    # 5. -1 * alibi_slope * tl.abs(relative_pos_block) = [[ -3, -2, -1, 0,-1],
    #                                                     [ -4, -3, -2, -1, 0]],
    relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    q_pe,
    k_ptrs,
    k_pe_ptrs,
    v_ptrs,
    stride_kn,
    stride_vk,
    stride_sn,
    start_m,
    seqlen_k,
    seqlen_q,
    dropout_p,
    sd_mask_ptrs,
    dropout_mask_ptrs,
    philox_seed,
    philox_ptrs,
    block_min,
    block_max,
    offs_n_causal,
    masked_blocks,
    n_extra_tokens,
    alibi_slope,
    descale_q,
    descale_k,
    descale_v,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    PRELOAD_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_POW2: tl.constexpr,
    BLOCK_DMODEL_PE: tl.constexpr,  # it's zero or a power of 2
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_SCORES: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    ENABLE_PIPELINING: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    HAS_PE: tl.constexpr = BLOCK_DMODEL_PE > 0

    # loop over k, v, and update accumulator

    num_stages: tl.constexpr = (
        None if ENABLE_PIPELINING else 1
    )  # Set num_stages==1 if we want to disable pipelining
    for start_n in tl.range(block_min, block_max, BLOCK_N, num_stages=num_stages):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL_POW2)
        k = _load_fn(k_ptrs, k_offs_k, k_offs_n, BLOCK_DMODEL, seqlen_k)
        if HAS_PE:
            k_pe = _load_fn(
                k_pe_ptrs,
                None,
                k_offs_n,
                (BLOCK_DMODEL + BLOCK_DMODEL_PE),
                seqlen_k,
            )
        if PRELOAD_V:
            v = _load_fn(v_ptrs, k_offs_n, k_offs_k, seqlen_k, BLOCK_DMODEL)

        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        mask = tl.full([BLOCK_M, BLOCK_N], True, dtype=tl.int1)
        if MASK_STEPS:
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
            # last step might get wasted but that is okay. check if this masking works For
            # that case.

            # remove the old if condition
            # if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
            # Though this will unconditionally compute mask_partial at runtime,
            # the causal for loop does not have the if-else block any more, which
            # helps instruction scheduling and register pressure.
            bound_cond = (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0)
            size_n = start_n + OFFS_N[None, :]
            mask_partial = size_n < seqlen_k
            mask = tl.where(bound_cond, mask_partial, mask)

        # compute masks
        q_mask = OFFS_M[:, None] < seqlen_q
        k_mask = (start_n + tl.arange(0, BLOCK_N))[None, :] < seqlen_k
        p_mask = q_mask & k_mask
        qk_scale = SM_SCALE * RCP_LN2
        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if HAS_PE:
            qk += tl.dot(q_pe, k_pe)
        qk += tl.dot(q, k)
        if IS_FP8:
            qk = qk * (qk_scale * descale_q * descale_k)
        else:
            qk = qk * qk_scale
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            mask = mask & causal_mask

        if SLIDING_WINDOW > 0:
            k_pos = start_n + tl.arange(0, BLOCK_N)
            q_adj = OFFS_M + seqlen_k - seqlen_q
            window_mask = k_pos[None, :] >= (q_adj[:, None] - SLIDING_WINDOW)
            mask = mask & window_mask

        qk = tl.where(mask, qk, float("-inf"))

        if alibi_slope is not None:
            # Compute the global position of each token within the sequence
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = _compute_alibi_block(
                alibi_slope, seqlen_q, seqlen_k, global_m_positions, global_n_positions
            )
            qk += alibi_block * RCP_LN2
        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        # Compute scaled QK and softmax probabilities
        p = tl.math.exp2(qk - m_ij[:, None])

        if SLIDING_WINDOW > 0:
            # When all qk in a row are -inf (fully out-of-window block) and m_i was -inf,
            # exp2(-inf - (-inf)) = NaN. Sanitize by zeroing masked elements.
            p = tl.where(mask, p, 0.0)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            rng_output = tl.rand(
                philox_seed, philox_ptrs
            )  # TODO: use tl.randint for better performance
            dropout_mask = rng_output > dropout_p
            tl.store(dropout_mask_ptrs, dropout_mask, mask=p_mask)

            # return scores with negative values for dropped vals
            sd_mask = tl.where(dropout_mask, p, -p)
            tl.store(sd_mask_ptrs, sd_mask, mask=p_mask)

            # apply dropout mask in place
            p = tl.where(dropout_mask, p, 0.0)
        elif RETURN_SCORES:
            # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
            tl.store(sd_mask_ptrs, p, mask=p_mask)

        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        alpha = tl.math.exp2(m_i - m_ij)
        if SLIDING_WINDOW > 0:
            # When m_i == m_ij == -inf, exp2(-inf - (-inf)) = NaN. alpha should be 1.0
            # (no rescaling needed since max didn't change).
            alpha = tl.where(m_i == m_ij, 1.0, alpha)
        acc = acc * alpha[:, None]
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        if not PRELOAD_V:
            v = _load_fn(v_ptrs, k_offs_n, k_offs_k, seqlen_k, BLOCK_DMODEL)
        if IS_FP8:
            scale_p, descale_p = _compute_fp8_scaling_factors(p, FP8_MAX)
            acc += (
                tl.dot((p * scale_p).to(v.type.element_ty), v) * descale_p * descale_v
            )
        else:
            acc += tl.dot(p.to(v.type.element_ty), v)

        k_ptrs += BLOCK_N * stride_kn
        if HAS_PE:
            k_pe_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if RETURN_SCORES:
            sd_mask_ptrs += BLOCK_N * stride_sn

        if ENABLE_DROPOUT:
            dropout_mask_ptrs += BLOCK_N * stride_sn
            philox_ptrs += BLOCK_N * stride_sn

    return acc, l_i, m_i


@triton.jit
def _attn_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    descale_q_ptr,
    descale_k_ptr,
    descale_v_ptr,
    out_ptr,
    alibi_slopes_ptr,
    s_dmask_ptr,
    dropout_mask_ptr,
    softmax_lse_ptr,
    sink_ptr,
    stride_qz_in,
    stride_qh_in,
    stride_qm_in,
    stride_qk_in,
    stride_kz_in,
    stride_kh_in,
    stride_kn_in,
    stride_kk_in,
    stride_vz_in,
    stride_vh_in,
    stride_vn_in,
    stride_vk_in,
    stride_descale_q_z_in,
    stride_descale_k_z_in,
    stride_descale_v_z_in,
    stride_oz_in,
    stride_oh_in,
    stride_om_in,
    stride_on_in,
    stride_alibi_z_in,
    stride_alibi_h_in,
    stride_sd_z_in,
    stride_sd_h_in,
    stride_sd_m_in,
    stride_sd_n_in,
    stride_lse_z_in,
    stride_lse_h_in,
    stride_lse_m_in,
    sm_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    dropout_p,
    philox_seed,
    philox_offset_base_in,
    SEQLEN_Q,
    SEQLEN_K,
    IS_CAUSAL: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    PRELOAD_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_POW2: tl.constexpr,
    BLOCK_DMODEL_PE: tl.constexpr,  # it's zero or a power of 2
    RETURN_SCORES: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
    VARLEN: tl.constexpr,
    BATCH,
    NUM_XCD: tl.constexpr,
    USE_INT64_STRIDES: tl.constexpr,
    ENABLE_SINK: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
):
    NUM_BLOCKS = (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M
    # calculate offsets
    wid = tl.program_id(
        0
    )  # workgroup id ranging: 0,1,2,...., (BATCH * NUM_Q_HEADS * NUM_BLOCKS - 1)
    # num blocks along seqlen

    off_q_head = wid % NUM_Q_HEADS
    off_q_head = remap_xcd(off_q_head, NUM_Q_HEADS, NUM_XCD)
    start_m = (wid // NUM_Q_HEADS) % NUM_BLOCKS
    off_z = (wid // (NUM_BLOCKS * NUM_Q_HEADS)) % BATCH

    # offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_POW2)
    HAS_PE: tl.constexpr = BLOCK_DMODEL_PE > 0
    if HAS_PE:
        offs_pe = BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL_PE)

    # NOTE:
    # Workaround for int64 strides, In the absence of strides being int64, parts of the offset
    # computation is done in 32 bit and overflows resulting in segfaults
    # If input strides are defined as int64, it disables vectorized loads which drops perf
    # If we define new strides as stride_x = stride_x_in.to(tl.int64), that does not work
    # because strides are tl.constexpr and cannot be upcasted
    # If we define new strides as stride_x: tl.int64 = stride_x_in, segfault remains
    # The permanent solution is to enable upcasting of tl.constexpr
    # In the meantime, the following workaround provides correctness and does not drop perf
    if USE_INT64_STRIDES:
        stride_qz = tl.cast(stride_qz_in, tl.int64)
        stride_qh = tl.cast(stride_qh_in, tl.int64)
        stride_qm = tl.cast(stride_qm_in, tl.int64)
        stride_qk = tl.cast(stride_qk_in, tl.int64)
        stride_kz = tl.cast(stride_kz_in, tl.int64)
        stride_kh = tl.cast(stride_kh_in, tl.int64)
        stride_kn = tl.cast(stride_kn_in, tl.int64)
        stride_kk = tl.cast(stride_kk_in, tl.int64)
        stride_vz = tl.cast(stride_vz_in, tl.int64)
        stride_vh = tl.cast(stride_vh_in, tl.int64)
        stride_vn = tl.cast(stride_vn_in, tl.int64)
        stride_vk = tl.cast(stride_vk_in, tl.int64)
        if IS_FP8:
            stride_descale_q_z = tl.cast(stride_descale_q_z_in, tl.int64)
            stride_descale_k_z = tl.cast(stride_descale_k_z_in, tl.int64)
            stride_descale_v_z = tl.cast(stride_descale_v_z_in, tl.int64)
        stride_oz = tl.cast(stride_oz_in, tl.int64)
        stride_oh = tl.cast(stride_oh_in, tl.int64)
        stride_om = tl.cast(stride_om_in, tl.int64)
        stride_on = tl.cast(stride_on_in, tl.int64)
        stride_alibi_z = tl.cast(stride_alibi_z_in, tl.int64)
        stride_alibi_h = tl.cast(stride_alibi_h_in, tl.int64)

        # NOTE: philox offset is need in dropout pointer calculations
        philox_offset_base = tl.cast(philox_offset_base_in, tl.int64)
        stride_sd_z = tl.cast(stride_sd_z_in, tl.int64)
        stride_sd_h = tl.cast(stride_sd_h_in, tl.int64)
        stride_sd_m = tl.cast(stride_sd_m_in, tl.int64)
        stride_sd_n = tl.cast(stride_sd_n_in, tl.int64)
        stride_lse_z = tl.cast(stride_lse_z_in, tl.int64)
        stride_lse_h = tl.cast(stride_lse_h_in, tl.int64)
        stride_lse_m = tl.cast(stride_lse_m_in, tl.int64)
    else:
        stride_qz = stride_qz_in
        stride_qm = stride_qm_in
        stride_qk = stride_qk_in
        stride_qh = stride_qh_in
        stride_kz = stride_kz_in
        stride_kh = stride_kh_in
        stride_kn = stride_kn_in
        stride_kk = stride_kk_in
        stride_vz = stride_vz_in
        stride_vh = stride_vh_in
        stride_vn = stride_vn_in
        stride_vk = stride_vk_in
        stride_descale_q_z = stride_descale_q_z_in
        stride_descale_k_z = stride_descale_k_z_in
        stride_descale_v_z = stride_descale_v_z_in
        stride_oz = stride_oz_in
        stride_oh = stride_oh_in
        stride_om = stride_om_in
        stride_on = stride_on_in
        stride_alibi_z = stride_alibi_z_in
        stride_alibi_h = stride_alibi_h_in
        philox_offset_base = philox_offset_base_in
        stride_sd_z = stride_sd_z_in
        stride_sd_h = stride_sd_h_in
        stride_sd_m = stride_sd_m_in
        stride_sd_n = stride_sd_n_in
        stride_lse_z = stride_lse_z_in
        stride_lse_h = stride_lse_h_in
        stride_lse_m = stride_lse_m_in

    tl.assume(stride_qz_in >= 0)
    tl.assume(stride_qh_in >= 0)
    tl.assume(stride_qm_in >= 0)
    tl.assume(stride_qk_in >= 0)
    tl.assume(stride_kz_in >= 0)
    tl.assume(stride_kh_in >= 0)
    tl.assume(stride_kn_in >= 0)
    tl.assume(stride_kk_in >= 0)
    tl.assume(stride_vz_in >= 0)
    tl.assume(stride_vh_in >= 0)
    tl.assume(stride_vn_in >= 0)
    tl.assume(stride_vk_in >= 0)
    if IS_FP8:
        tl.assume(stride_descale_q_z_in >= 0)
        tl.assume(stride_descale_k_z_in >= 0)
        tl.assume(stride_descale_v_z_in >= 0)
        tl.assume(stride_oz_in >= 0)
        tl.assume(stride_oh_in >= 0)
        tl.assume(stride_om_in >= 0)
        tl.assume(stride_on_in >= 0)
        tl.assume(stride_alibi_z_in >= 0)
        tl.assume(stride_alibi_h_in >= 0)
    # NOTE: philox offset is need in dropout pointer calculations
    tl.assume(philox_offset_base_in >= 0)
    tl.assume(stride_sd_z_in >= 0)
    tl.assume(stride_sd_h_in >= 0)
    tl.assume(stride_sd_m_in >= 0)
    tl.assume(stride_sd_n_in >= 0)
    tl.assume(stride_lse_z_in >= 0)
    tl.assume(stride_lse_h_in >= 0)
    tl.assume(stride_lse_m_in >= 0)

    if VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)

        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        # We have a one-size-fits-all grid in id(0). Some seqlens might be too
        # small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = SEQLEN_Q
        seqlen_k = SEQLEN_K

    n_blocks = _cdiv_fn(seqlen_k, BLOCK_N)

    # Now we compute whether we need to exit early due to causal masking.
    # This is because for seqlen_q > seqlen_k, M rows of the attn scores
    # are completely masked, resulting in 0s written to the output, and
    # inf written to LSE. We don't need to do any GEMMs in this case.
    # This block of code determines what N is, and if this WG is operating
    # on those M rows.
    if IS_CAUSAL:
        # If seqlen_q == seqlen_k, the attn scores are a square matrix.
        # If seqlen_q != seqlen_k, attn scores are rectangular which means
        # the causal mask boundary is bottom right aligned, and ends at either
        # the top edge (seqlen_q < seqlen_k) or left edge.

        # This captures the decrease in n_blocks if we have a rectangular attn matrix
        n_blocks_seqlen = _cdiv_fn(
            (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N
        )

        # This is what adjusts the block_max for the current WG, only
        # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
        n_blocks = min(n_blocks, n_blocks_seqlen)

        # If we have no blocks after adjusting for seqlen deltas, this WG is part of
        # the blocks that are all 0. We exit early.
        if n_blocks <= 0:
            offs_out = (
                off_z * stride_oz
                + off_q_head * stride_oh
                + cu_seqlens_q_start * stride_om
                + offs_m[:, None] * stride_om
                + offs_d[None, :] * stride_on
            )
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=out_ptr.type.element_ty)
            out_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_DMODEL)
            tl.store(out_ptr + offs_out, acc, mask=out_mask)

            if softmax_lse_ptr is not None:
                offs_lse = (
                    off_z * stride_lse_z
                    + off_q_head * stride_lse_h
                    + cu_seqlens_q_start * stride_lse_m
                    + offs_m * stride_lse_m
                )
                lse_mask = offs_m < SEQLEN_Q
                lse = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)
                tl.store(softmax_lse_ptr + offs_lse, lse, mask=lse_mask)
                # TODO: Should dropout and return encoded softmax be handled here too?

            return

    grp_sz: tl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
    if grp_sz != 1:  # Grouped Query Attention
        off_k_head = off_q_head // grp_sz
    else:
        off_k_head = off_q_head

    # q,k,v offsets
    q_offs = (
        off_z * stride_qz
        + off_q_head * stride_qh
        + cu_seqlens_q_start * stride_qm
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk
    )
    q_ptrs = q_ptr + q_offs
    if HAS_PE:
        q_pe_offs = (
            off_z * stride_qz
            + off_q_head * stride_qh
            + cu_seqlens_q_start * stride_qm
            + offs_m[:, None] * stride_qm
            + offs_pe[None, :] * stride_qk
        )
        q_pe_ptrs = q_ptr + q_pe_offs
    else:
        q_pe_ptrs = None

    k_offs = (
        off_z * stride_kz
        + off_k_head * stride_kh
        + cu_seqlens_k_start * stride_kn
        + offs_d[:, None] * stride_kk
        + offs_n[None, :] * stride_kn
    )
    k_ptrs = k_ptr + k_offs
    if HAS_PE:
        k_pe_offs = (
            off_z * stride_kz
            + off_k_head * stride_kh
            + cu_seqlens_k_start * stride_kn
            + offs_pe[:, None] * stride_kk
            + offs_n[None, :] * stride_kn
        )
        k_pe_ptrs = k_ptr + k_pe_offs
    else:
        k_pe_ptrs = None

    v_offs = (
        off_z * stride_vz
        + off_k_head * stride_vh
        + cu_seqlens_k_start * stride_vn
        + offs_n[:, None] * stride_vn
        + offs_d[None, :] * stride_vk
    )
    v_ptrs = v_ptr + v_offs

    # alibi slopes
    if alibi_slopes_ptr is not None:
        alibi_offs = off_z * stride_alibi_z + off_q_head * stride_alibi_h
        alibi_slope = tl.load(alibi_slopes_ptr + alibi_offs)
    else:
        alibi_slope = None

    # s_dmask (return_scores)
    if s_dmask_ptr is not None:
        s_dmask_offs = (
            off_z * stride_sd_z
            + off_q_head * stride_sd_h
            + offs_m[:, None] * stride_sd_m
            + offs_n[None, :] * stride_sd_n
        )
        s_dmask_ptrs = s_dmask_ptr + s_dmask_offs
    else:
        s_dmask_ptrs = None

    # dropout
    if dropout_mask_ptr is not None:
        dropout_mask_offs = (
            off_z * stride_sd_z
            + off_q_head * stride_sd_h
            + offs_m[:, None] * stride_sd_m
            + offs_n[None, :] * stride_sd_n
        )
        dropout_mask_ptrs = dropout_mask_ptr + dropout_mask_offs
        philox_ptrs = (
            philox_offset_base
            + off_z * stride_sd_z
            + off_q_head * stride_sd_h
            + offs_m[:, None] * stride_sd_m
            + offs_n[None, :] * stride_sd_n
        )
    else:
        dropout_mask_ptrs = None
        philox_ptrs = None

    if ENABLE_SINK:
        RCP_LN2: tl.constexpr = 1.4426950408889634
        m_i_value = tl.load(sink_ptr + off_q_head).to(tl.float32) * RCP_LN2
    else:
        m_i_value = float("-inf")

    m_i = tl.full([BLOCK_M], m_i_value, dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=tl.float32)
    if BLOCK_DMODEL == BLOCK_DMODEL_POW2:
        q_mask = offs_m[:, None] < seqlen_q
    else:
        q_mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_DMODEL)

    if BLOCK_M >= NUM_Q_HEADS:
        q_cache_mod: tl.constexpr = ".cg"
    else:
        q_cache_mod: tl.constexpr = ""

    if HAS_PE:
        q_pe = tl.load(q_pe_ptrs, mask=q_mask, other=0.0, cache_modifier=q_cache_mod)
    else:
        q_pe = None
    q = tl.load(q_ptrs, mask=q_mask, other=0.0, cache_modifier=q_cache_mod)
    if IS_FP8:
        descale_q = tl.load(descale_q_ptr + off_z * stride_descale_q_z + off_q_head)
        descale_k = tl.load(descale_k_ptr + off_z * stride_descale_k_z + off_k_head)
        descale_v = tl.load(descale_v_ptr + off_z * stride_descale_v_z + off_k_head)
    else:
        descale_q, descale_k, descale_v = 1.0, 1.0, 1.0

    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N

    # if CAUSAL, then determine masked_blocks and full blocks
    # Here we compute how many full and masked blocks we have.
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    skipped_blocks = 0
    if SLIDING_WINDOW > 0:
        # Skip K blocks that are fully left of the earliest key position
        # reachable by this Q block. The first retained block can still be
        # partially outside the window, so we keep the per-element mask below.
        window_start_n = start_m * BLOCK_M + seqlen_k - seqlen_q - SLIDING_WINDOW
        skipped_blocks = tl.maximum(window_start_n, 0) // BLOCK_N
        skipped_blocks = tl.minimum(skipped_blocks, n_blocks)
    if IS_CAUSAL:
        # There are always at least BLOCK_M // BLOCK_N masked blocks.
        # Additionally there might be one more due to dissimilar seqlens.
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        # Padding on Q does not need to be masked in the FA loop.
        masked_blocks = padded_block_k
    # if IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
    # In this case we might exceed n_blocks so pick the min.
    visible_blocks = n_blocks - skipped_blocks
    masked_blocks = min(masked_blocks, visible_blocks)
    n_full_blocks = visible_blocks - masked_blocks
    block_min = skipped_blocks * BLOCK_N
    block_max = n_blocks * BLOCK_N
    if skipped_blocks > 0:
        k_ptrs += skipped_blocks * BLOCK_N * stride_kn
        if HAS_PE:
            k_pe_ptrs += skipped_blocks * BLOCK_N * stride_kn
        v_ptrs += skipped_blocks * BLOCK_N * stride_vn
        if RETURN_SCORES:
            s_dmask_ptrs += skipped_blocks * BLOCK_N * stride_sd_n
        if ENABLE_DROPOUT:
            dropout_mask_ptrs += skipped_blocks * BLOCK_N * stride_sd_n
            philox_ptrs += skipped_blocks * BLOCK_N * stride_sd_n
    # Compute for full blocks. Here we set causal to false regardless of its actual
    # value because there is no masking. Similarly we do not need padding.
    if n_full_blocks > 0:
        block_max = block_min + n_full_blocks * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_pe,
            k_ptrs,
            k_pe_ptrs,
            v_ptrs,
            stride_kn,
            stride_vn,
            stride_sd_n,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            s_dmask_ptrs,
            dropout_mask_ptrs,
            philox_seed,
            philox_ptrs,
            block_min,
            block_max,
            0,
            0,
            0,
            alibi_slope,
            descale_q,
            descale_k,
            descale_v,
            offs_m,
            offs_n,
            PRELOAD_V,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE,
            sm_scale,
            False,
            MASK_STEPS=False,
            ENABLE_DROPOUT=ENABLE_DROPOUT,
            RETURN_SCORES=RETURN_SCORES,
            PADDED_HEAD=BLOCK_DMODEL != BLOCK_DMODEL_POW2,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            ENABLE_PIPELINING=True,
            SLIDING_WINDOW=SLIDING_WINDOW,
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    # Remaining blocks, if any, are full / not masked.
    if masked_blocks > 0:
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seqlen_q - seqlen_k)
        else:
            offs_n_causal = 0
        k_ptrs += n_full_blocks * BLOCK_N * stride_kn
        if HAS_PE:
            k_pe_ptrs += n_full_blocks * BLOCK_N * stride_kn
        v_ptrs += n_full_blocks * BLOCK_N * stride_vn
        if RETURN_SCORES:
            s_dmask_ptrs += n_full_blocks * BLOCK_N * stride_sd_n
        if ENABLE_DROPOUT:
            dropout_mask_ptrs += n_full_blocks * BLOCK_N * stride_sd_n
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_pe,
            k_ptrs,
            k_pe_ptrs,
            v_ptrs,
            stride_kn,
            stride_vn,
            stride_sd_n,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            s_dmask_ptrs,
            dropout_mask_ptrs,
            philox_seed,
            philox_ptrs,
            block_min,
            block_max,
            offs_n_causal,
            masked_blocks,
            n_extra_tokens,
            alibi_slope,
            descale_q,
            descale_k,
            descale_v,
            offs_m,
            offs_n,
            PRELOAD_V,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE,
            sm_scale,
            IS_CAUSAL,
            MASK_STEPS=True,
            ENABLE_DROPOUT=ENABLE_DROPOUT,
            RETURN_SCORES=RETURN_SCORES,
            PADDED_HEAD=BLOCK_DMODEL != BLOCK_DMODEL_POW2,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            ENABLE_PIPELINING=False,
            SLIDING_WINDOW=SLIDING_WINDOW,
        )
    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    if ENABLE_DROPOUT:
        dropout_scale = 1 / (1 - dropout_p)
        acc = acc * dropout_scale
    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full(
                (BLOCK_DMODEL_POW2,), causal_start_idx, dtype=tl.int32
            )
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))

    # write back LSE(Log Sum Exponents), the log of the normalization constant
    overflow_size = end_m_idx - seqlen_q
    if softmax_lse_ptr is not None:
        LN2: tl.constexpr = 0.6931471824645996
        # compute log-sum-exp in base 2 units
        softmax_lse = m_i + tl.math.log2(l_i)
        # convert back to natural units
        softmax_lse *= LN2

        if IS_CAUSAL:
            # zero out nans caused by -infs when doing causal
            lse_causal_mask = (start_m_idx + tl.arange(0, BLOCK_M)) < causal_start_idx
            softmax_lse = tl.where(lse_causal_mask, 0.0, softmax_lse)

        # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
        # This is only true for the last M block. For others, overflow_size will be -ve
        offs_lse = (
            off_z * stride_lse_z
            + off_q_head * stride_lse_h
            + cu_seqlens_q_start * stride_lse_m
            + offs_m * stride_lse_m
        )
        if overflow_size > 0:
            boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
            lse_mask = tl.arange(0, BLOCK_M) < boundary
            tl.store(
                softmax_lse_ptr + offs_lse, softmax_lse, mask=lse_mask
            )  # the log of the normalization constant
        else:
            tl.store(
                softmax_lse_ptr + offs_lse, softmax_lse
            )  # the log of the normalization constant

    # write back O
    offs_out = (
        off_z * stride_oz
        + off_q_head * stride_oh
        + cu_seqlens_q_start * stride_om
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_on
    )
    out_mask = tl.full([BLOCK_M, 1], 1, dtype=tl.int1)
    if overflow_size > 0:
        out_mask = out_mask & (offs_m[:, None] < seqlen_q)
    if BLOCK_DMODEL != BLOCK_DMODEL_POW2:
        out_mask = out_mask & (offs_d[None, :] < BLOCK_DMODEL)
    op = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_out, op, mask=out_mask)


def build_args(cfg: Dict[str, Any]) -> Tuple[list, Dict[str, Any]]:
    """Harness adapter: returns (positional_args, kwargs) for warmup."""
    BATCH = 1
    SEQLEN_Q = 512
    SEQLEN_K = 512
    NUM_Q_HEADS = int(cfg.get("NUM_Q_HEADS", 32))
    NUM_K_HEADS = int(cfg.get("NUM_K_HEADS", 8))
    HEAD_DIM = int(cfg.get("BLOCK_DMODEL", 128))
    BLOCK_M = int(cfg.get("BLOCK_M", 128))
    BLOCK_N = int(cfg.get("BLOCK_N", 64))
    BLOCK_DMODEL_POW2 = int(cfg.get("BLOCK_DMODEL_POW2", HEAD_DIM))

    # Contiguous layout: [batch, heads, seq, dim]
    q = FakeTensor("bf16", (BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_DIM))
    k = FakeTensor("bf16", (BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_DIM))
    v = FakeTensor("bf16", (BATCH, NUM_K_HEADS, SEQLEN_K, HEAD_DIM))
    out = FakeTensor("bf16", (BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_DIM))
    descale_q = FakeTensor("fp32", (BATCH,))
    descale_k = FakeTensor("fp32", (BATCH,))
    descale_v = FakeTensor("fp32", (BATCH,))
    alibi_slopes = FakeTensor("fp32", (BATCH, NUM_Q_HEADS))
    s_dmask = FakeTensor("fp32", (BATCH, NUM_Q_HEADS, SEQLEN_Q, SEQLEN_K))
    dropout_mask = FakeTensor("fp32", (BATCH, NUM_Q_HEADS, SEQLEN_Q, SEQLEN_K))
    softmax_lse = FakeTensor("fp32", (BATCH, NUM_Q_HEADS, SEQLEN_Q))
    sink = FakeTensor("fp32", (BATCH, NUM_Q_HEADS, SEQLEN_Q, HEAD_DIM))
    cu_seqlens_q = FakeTensor("i32", (2,))
    cu_seqlens_k = FakeTensor("i32", (2,))

    args = [
        q,                          # q_ptr
        k,                          # k_ptr
        v,                          # v_ptr
        descale_q,                  # descale_q_ptr
        descale_k,                  # descale_k_ptr
        descale_v,                  # descale_v_ptr
        out,                        # out_ptr
        alibi_slopes,               # alibi_slopes_ptr
        s_dmask,                    # s_dmask_ptr
        dropout_mask,               # dropout_mask_ptr
        softmax_lse,                # softmax_lse_ptr
        sink,                       # sink_ptr
        # strides (i32 scalars)
        q.stride(0),               # stride_qz_in
        q.stride(1),               # stride_qh_in
        q.stride(2),               # stride_qm_in
        q.stride(3),               # stride_qk_in
        k.stride(0),               # stride_kz_in
        k.stride(1),               # stride_kh_in
        k.stride(2),               # stride_kn_in
        k.stride(3),               # stride_kk_in
        v.stride(0),               # stride_vz_in
        v.stride(1),               # stride_vh_in
        v.stride(2),               # stride_vn_in
        v.stride(3),               # stride_vk_in
        descale_q.stride(0),       # stride_descale_q_z_in
        descale_k.stride(0),       # stride_descale_k_z_in
        descale_v.stride(0),       # stride_descale_v_z_in
        out.stride(0),             # stride_oz_in
        out.stride(1),             # stride_oh_in
        out.stride(2),             # stride_om_in
        out.stride(3),             # stride_on_in
        alibi_slopes.stride(0),    # stride_alibi_z_in
        alibi_slopes.stride(1),    # stride_alibi_h_in
        s_dmask.stride(0),         # stride_sd_z_in
        s_dmask.stride(1),         # stride_sd_h_in
        s_dmask.stride(2),         # stride_sd_m_in
        s_dmask.stride(3),         # stride_sd_n_in
        softmax_lse.stride(0),     # stride_lse_z_in
        softmax_lse.stride(1),     # stride_lse_h_in
        softmax_lse.stride(2),     # stride_lse_m_in
        # other scalars
        1.0 / (HEAD_DIM ** 0.5),   # sm_scale
        cu_seqlens_q,              # cu_seqlens_q
        cu_seqlens_k,              # cu_seqlens_k
        0.0,                       # dropout_p
        0,                         # philox_seed
        0,                         # philox_offset_base_in
        SEQLEN_Q,                  # SEQLEN_Q
        SEQLEN_K,                  # SEQLEN_K
    ]
    kwargs: Dict[str, Any] = {
        "IS_CAUSAL": bool(cfg.get("IS_CAUSAL", True)),
        "NUM_Q_HEADS": NUM_Q_HEADS,
        "NUM_K_HEADS": NUM_K_HEADS,
        "PRELOAD_V": bool(cfg.get("PRELOAD_V", True)),
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_DMODEL": HEAD_DIM,
        "BLOCK_DMODEL_POW2": BLOCK_DMODEL_POW2,
        "BLOCK_DMODEL_PE": int(cfg.get("BLOCK_DMODEL_PE", 0)),
        "RETURN_SCORES": bool(cfg.get("RETURN_SCORES", False)),
        "ENABLE_DROPOUT": bool(cfg.get("ENABLE_DROPOUT", False)),
        "IS_FP8": bool(cfg.get("IS_FP8", False)),
        "FP8_MAX": float(cfg.get("FP8_MAX", 0.0)),
        "VARLEN": bool(cfg.get("VARLEN", False)),
        "BATCH": BATCH,
        "NUM_XCD": int(cfg.get("NUM_XCD", 8)),
        "USE_INT64_STRIDES": bool(cfg.get("USE_INT64_STRIDES", True)),
        "ENABLE_SINK": bool(cfg.get("ENABLE_SINK", False)),
        "SLIDING_WINDOW": int(cfg.get("SLIDING_WINDOW", 0)),
        "num_warps": 4,
        "num_stages": 1,
    }
    return args, kwargs

