// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Zig port of `tools/dsl-harness/kernels/triton/py/mha.py:_attn_fwd`.
//
// Pinned config:
//   IS_CAUSAL=false, VARLEN=false, IS_FP8=false, ENABLE_DROPOUT=false,
//   ENABLE_SINK=false, SLIDING_WINDOW=0, HAS_PE=false (BLOCK_DMODEL_PE=0),
//   USE_INT64_STRIDES=false, RETURN_SCORES=false, PRELOAD_V=true, NUM_XCD=8.

const std = @import("std");

const mlir = @import("mlir");
const dialects = @import("mlir/dialects");
const ttir = dialects.ttir;

const zml = @import("../../zml.zig");
const tri = zml.kernel.triton;
const Builder = tri.Builder;
const Value = tri.Value;
const DType = tri.DType;

/// `log_2(e)` — pre-multiply with this to turn `exp(x)` into `exp2(x * RCP_LN2)`.
const RCP_LN2: f32 = 1.4426950408889634;

// ============================================================================
// Helper: remap_xcd
// ============================================================================

/// Direct port of Python `remap_xcd(pid, GRID_MN, NUM_XCDS=8)`.
fn remapXcd(b: *Builder, pid: Value, GRID_MN: Value, NUM_XCDS: Value) Value {
    // pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    const pids_per_xcd = GRID_MN.add(NUM_XCDS).sub(1).div(NUM_XCDS);
    // tall_xcds = GRID_MN % NUM_XCDS
    const tall_xcds_raw = GRID_MN.rem(NUM_XCDS);
    // tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    const tall_xcds = b.where(tall_xcds_raw.eq(0), NUM_XCDS, tall_xcds_raw);
    // xcd = pid % NUM_XCDS
    const xcd = pid.rem(NUM_XCDS);
    // local_pid = pid // NUM_XCDS
    const local_pid = pid.div(NUM_XCDS);

    // if xcd < tall_xcds:
    //     pid = xcd * pids_per_xcd + local_pid
    // else:
    //     pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
    var ie = b.openIfElse(xcd.lt(tall_xcds), .{ b.scalarTy(.i32) });
    {
        // then: tall xcd
        const new_pid = xcd.mul(pids_per_xcd).add(local_pid);
        ie.yieldThen(.{new_pid});
    }
    {
        // else: short xcd
        const new_pid = tall_xcds.mul(pids_per_xcd)
            .add(xcd.sub(tall_xcds).mul(pids_per_xcd.sub(1)))
            .add(local_pid);
        ie.yieldElse(.{new_pid});
    }
    return ie.results[0];
}

// ============================================================================
// Helper: _load_fn
// ============================================================================

/// Direct port of Python `_load_fn`. For the pinned config, the call sites
/// always pass concrete mask arguments (offset_first, offset_second, or both
/// may be null at the call site). Since Python uses `is not None` checks at
/// comptime, we mirror with a comptime-dispatched overload.
fn loadFnBothMask(
    b: *Builder,
    ptrs: Value,
    offset_first: Value,
    offset_second: Value,
    boundary_first: anytype,
    boundary_second: anytype,
) Value {
    // mask = (offset_first[:, None] < boundary_first) & (offset_second[None, :] < boundary_second)
    const mask = offset_first.expandDims(1).lt(boundary_first)
        .bitAnd(offset_second.expandDims(0).lt(boundary_second));
    return b.loadOpts(ptrs, .{ .mask = mask, .other = b.zeros(ptrs.shape().constSlice(), .f32) });
}

fn loadFnFirstMaskOnly(
    b: *Builder,
    ptrs: Value,
    offset_first: Value,
    boundary_first: anytype,
) Value {
    // mask = offset_first[:, None] < boundary_first
    const mask = offset_first.expandDims(1).lt(boundary_first);
    return b.loadOpts(ptrs, .{ .mask = mask, .other = b.zeros(ptrs.shape().constSlice(), .f32) });
}

fn loadFnSecondMaskOnly(
    b: *Builder,
    ptrs: Value,
    offset_second: Value,
    boundary_second: anytype,
) Value {
    // mask = offset_second[None, :] < boundary_second
    const mask = offset_second.expandDims(0).lt(boundary_second);
    return b.loadOpts(ptrs, .{ .mask = mask, .other = b.zeros(ptrs.shape().constSlice(), .f32) });
}

fn loadFnNoMask(
    b: *Builder,
    ptrs: Value,
) Value {
    return b.load(ptrs);
}

// ============================================================================
// Helper: _attn_fwd_inner
// ============================================================================

/// Direct port of Python `_attn_fwd_inner` for the pinned config:
///   IS_CAUSAL=false, ENABLE_DROPOUT=false, RETURN_SCORES=false,
///   IS_FP8=false, PRELOAD_V=true, SLIDING_WINDOW=0, HAS_PE=false.
fn attnFwdInner(
    b: *Builder,
    acc_in: Value,
    l_i_in: Value,
    m_i_in: Value,
    q: Value,
    k_ptrs_in: Value,
    v_ptrs_in: Value,
    stride_kn: Value,
    stride_vk: Value,
    seqlen_k: Value,
    block_min: Value,
    block_max: Value,
    n_extra_tokens: Value,
    BLOCK_M: i32,
    BLOCK_N: i32,
    BLOCK_DMODEL: i32,
    BLOCK_DMODEL_POW2: i32,
    SM_SCALE: f32,
    DTYPE: DType,
    MASK_STEPS: bool,
    PADDED_HEAD: bool,
) struct { Value, Value, Value } {
    const RCP_LN2_: f32 = 1.4426950408889634;
    const qk_scale: f32 = SM_SCALE * RCP_LN2_;
    const BM: i64 = BLOCK_M;
    const BN: i64 = BLOCK_N;

    // loop over k, v, and update accumulator
    // Pinned: IS_CAUSAL=false, ENABLE_DROPOUT=false, RETURN_SCORES=false,
    // IS_FP8=false, PRELOAD_V=true, SLIDING_WINDOW=0

    // openFor(block_min, block_max, BLOCK_N, .{acc, l_i, m_i, k_ptrs, v_ptrs})
    var loop = b.openFor(block_min, block_max, @as(i32, @intCast(BLOCK_N)), .{ acc_in, l_i_in, m_i_in, k_ptrs_in, v_ptrs_in });
    {
        const start_n = loop.iv;
        const acc = loop.carried[0];
        const l_i = loop.carried[1];
        const m_i = loop.carried[2];
        const k_ptrs = loop.carried[3];
        const v_ptrs = loop.carried[4];

        // Load k
        const k = if (MASK_STEPS and PADDED_HEAD) blk: {
            const k_offs_n = start_n.add(b.arange(0, BLOCK_N, .i32));
            const k_offs_k = b.arange(0, BLOCK_DMODEL_POW2, .i32);
            break :blk loadFnBothMask(b, k_ptrs, k_offs_k, k_offs_n, @as(i32, @intCast(BLOCK_DMODEL)), seqlen_k);
        } else if (MASK_STEPS) blk: {
            const k_offs_n = start_n.add(b.arange(0, BLOCK_N, .i32));
            break :blk loadFnSecondMaskOnly(b, k_ptrs, k_offs_n, seqlen_k);
        } else if (PADDED_HEAD) blk: {
            const k_offs_k = b.arange(0, BLOCK_DMODEL_POW2, .i32);
            break :blk loadFnFirstMaskOnly(b, k_ptrs, k_offs_k, @as(i32, @intCast(BLOCK_DMODEL)));
        } else loadFnNoMask(b, k_ptrs);

        // PRELOAD_V=true: load v alongside k
        const v = if (MASK_STEPS and PADDED_HEAD) blk: {
            const k_offs_n = start_n.add(b.arange(0, BLOCK_N, .i32));
            const k_offs_k = b.arange(0, BLOCK_DMODEL_POW2, .i32);
            break :blk loadFnBothMask(b, v_ptrs, k_offs_n, k_offs_k, seqlen_k, @as(i32, @intCast(BLOCK_DMODEL)));
        } else if (MASK_STEPS) blk: {
            const k_offs_n = start_n.add(b.arange(0, BLOCK_N, .i32));
            break :blk loadFnFirstMaskOnly(b, v_ptrs, k_offs_n, seqlen_k);
        } else if (PADDED_HEAD) blk: {
            const k_offs_k = b.arange(0, BLOCK_DMODEL_POW2, .i32);
            break :blk loadFnSecondMaskOnly(b, v_ptrs, k_offs_k, @as(i32, @intCast(BLOCK_DMODEL)));
        } else loadFnNoMask(b, v_ptrs);

        // mask = tl.full([BLOCK_M, BLOCK_N], True, dtype=tl.int1)
        var mask = b.full(&.{ BM, BN }, 1, .i1);

        if (MASK_STEPS) {
            // bound_cond = (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0)
            const bound_cond = start_n.add(@as(i32, @intCast(BLOCK_N))).eq(block_max)
                .bitAnd(n_extra_tokens.ne(0));
            // size_n = start_n + OFFS_N[None, :]
            const size_n = start_n.add(b.arange(0, BLOCK_N, .i32).expandDims(0));
            // mask_partial = size_n < seqlen_k
            const mask_partial = size_n.lt(seqlen_k);
            // mask = tl.where(bound_cond, mask_partial, mask)
            mask = b.where(bound_cond, mask_partial, mask);
        }

        // -- compute qk ----
        // qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        var qk = b.zeros(&.{ BM, BN }, .f32);
        // qk += tl.dot(q, k)
        qk = b.dot(q, k, qk);
        // IS_FP8=false: qk = qk * qk_scale
        qk = qk.mul(qk_scale);

        // IS_CAUSAL=false: no causal mask
        // SLIDING_WINDOW=0: no window mask

        // qk = tl.where(mask, qk, float("-inf"))
        qk = b.where(mask, qk, b.full(&.{ BM, BN }, -std.math.inf(f32), .f32));

        // alibi_slope is None for our pinned config (alibi_slopes_ptr is not used in the outer)

        // get max scores so far
        // m_ij = tl.maximum(m_i, tl.max(qk, 1))
        const m_ij = m_i.maximum(b.maxOpts(qk, .{ .axis = 1 }));

        // Compute scaled QK and softmax probabilities
        // p = tl.math.exp2(qk - m_ij[:, None])
        const p = b.exp2(qk.sub(m_ij.expandDims(1)));

        // ENABLE_DROPOUT=false: no dropout

        // l_ij = tl.sum(p, 1)
        const l_ij = b.sumOpts(p, .{ .axis = 1 });

        // -- update output accumulator --
        // alpha = tl.math.exp2(m_i - m_ij)
        const alpha = b.exp2(m_i.sub(m_ij));
        // acc = acc * alpha[:, None]
        const new_acc_scaled = acc.mul(alpha.expandDims(1));
        // l_i = l_i * alpha + l_ij
        const new_l_i = l_i.mul(alpha).add(l_ij);
        // m_i = m_ij
        const new_m_i = m_ij;
        // IS_FP8=false: acc += tl.dot(p.to(v.type.element_ty), v)
        const new_acc = b.dot(p.to(DTYPE), v, new_acc_scaled);

        // k_ptrs += BLOCK_N * stride_kn
        const new_k_ptrs = k_ptrs.addPtr(b.liftAs(@as(i32, @intCast(BLOCK_N)), .i32).mul(stride_kn));
        // v_ptrs += BLOCK_N * stride_vk
        const new_v_ptrs = v_ptrs.addPtr(b.liftAs(@as(i32, @intCast(BLOCK_N)), .i32).mul(stride_vk));

        loop.yield(.{ new_acc, new_l_i, new_m_i, new_k_ptrs, new_v_ptrs });
    }

    return .{ loop.results[0], loop.results[1], loop.results[2] };
}

// ============================================================================
// Main kernel: _attn_fwd
// ============================================================================

pub const MhaFwd = struct {
    pub const Config = struct {
        /// Element dtype for Q/K/V/O pointers.
        dtype: DType = .bf16,
        /// Number of query heads.
        NUM_Q_HEADS: i32 = 32,
        /// Number of key/value heads (for GQA).
        NUM_K_HEADS: i32 = 8,
        /// Block size along the query sequence dimension.
        BLOCK_M: i32 = 128,
        /// Block size along the key sequence dimension.
        BLOCK_N: i32 = 64,
        /// Actual head dimension.
        BLOCK_DMODEL: i32 = 128,
        /// Padded head dimension (next power-of-2).
        BLOCK_DMODEL_POW2: i32 = 128,
        /// Sequence length along Q.
        SEQLEN_Q: i32 = 1024,
        /// Sequence length along K.
        SEQLEN_K: i32 = 1024,
        /// Softmax scale factor (typically 1/sqrt(d)).
        sm_scale: f32 = 0.08838834764831843, // 1/sqrt(128)
        /// Batch size.
        BATCH: i32 = 1,
        /// Number of XCDs on the GPU (MI300X=8).
        NUM_XCD: i32 = 8,
        /// Whether to preload V alongside K.
        PRELOAD_V: bool = true,
    };

    pub const Kernel = tri.Kernel(Config, .{
        .name = "_attn_fwd",
        .inputs = &.{
            "q_ptr",
            "k_ptr",
            "v_ptr",
            "descale_q_ptr",
            "descale_k_ptr",
            "descale_v_ptr",
            "alibi_slopes_ptr",
            "s_dmask_ptr",
            "dropout_mask_ptr",
            "softmax_lse_ptr",
            "sink_ptr",
            // strides — q
            "stride_qz_in",
            "stride_qh_in",
            "stride_qm_in",
            "stride_qk_in",
            // strides — k
            "stride_kz_in",
            "stride_kh_in",
            "stride_kn_in",
            "stride_kk_in",
            // strides — v
            "stride_vz_in",
            "stride_vh_in",
            "stride_vn_in",
            "stride_vk_in",
            // strides — descale (unused but in signature)
            "stride_descale_q_z_in",
            "stride_descale_k_z_in",
            "stride_descale_v_z_in",
            // strides — output
            "stride_oz_in",
            "stride_oh_in",
            "stride_om_in",
            "stride_on_in",
            // strides — alibi
            "stride_alibi_z_in",
            "stride_alibi_h_in",
            // strides — sd_mask / dropout
            "stride_sd_z_in",
            "stride_sd_h_in",
            "stride_sd_m_in",
            "stride_sd_n_in",
            // strides — lse
            "stride_lse_z_in",
            "stride_lse_h_in",
            "stride_lse_m_in",
            // runtime scalars
            "sm_scale",
            "cu_seqlens_q",
            "cu_seqlens_k",
            "dropout_p",
            "philox_seed",
            "philox_offset_base_in",
        },
        .outputs = &.{"out"},
        .run = run,
    });

    fn run(b: *Builder, cfg: Config) tri.FinishError!void {
        // Pinned config: IS_CAUSAL=false, VARLEN=false, IS_FP8=false,
        // ENABLE_DROPOUT=false, ENABLE_SINK=false, SLIDING_WINDOW=0,
        // HAS_PE=false, USE_INT64_STRIDES=false, RETURN_SCORES=false,
        // PRELOAD_V=true.

        const BLOCK_M: i32 = cfg.BLOCK_M;
        const BLOCK_N: i32 = cfg.BLOCK_N;
        const BLOCK_DMODEL: i32 = cfg.BLOCK_DMODEL;
        const BLOCK_DMODEL_POW2: i32 = cfg.BLOCK_DMODEL_POW2;
        const NUM_Q_HEADS: i32 = cfg.NUM_Q_HEADS;
        const NUM_K_HEADS: i32 = cfg.NUM_K_HEADS;
        const BATCH: i32 = cfg.BATCH;
        const SEQLEN_Q: i32 = cfg.SEQLEN_Q;
        const SEQLEN_K: i32 = cfg.SEQLEN_K;
        const NUM_XCD: i32 = cfg.NUM_XCD;
        const PADDED_HEAD = BLOCK_DMODEL != BLOCK_DMODEL_POW2;

        const NUM_BLOCKS: i32 = @divTrunc(SEQLEN_Q + BLOCK_M - 1, BLOCK_M);

        const a = try b.declareArgs(.{
            .q_ptr = .{ .ptr = cfg.dtype },
            .k_ptr = .{ .ptr = cfg.dtype },
            .v_ptr = .{ .ptr = cfg.dtype },
            .descale_q_ptr = .{ .ptr = .f32 },
            .descale_k_ptr = .{ .ptr = .f32 },
            .descale_v_ptr = .{ .ptr = .f32 },
            .alibi_slopes_ptr = .{ .ptr = .f32 },
            .s_dmask_ptr = .{ .ptr = .f32 },
            .dropout_mask_ptr = .{ .ptr = .f32 },
            .softmax_lse_ptr = .{ .ptr = .f32 },
            .sink_ptr = .{ .ptr = .f32 },
            // strides — q
            .stride_qz_in = .{ .scalar = .i32 },
            .stride_qh_in = .{ .scalar = .i32 },
            .stride_qm_in = .{ .scalar = .i32 },
            .stride_qk_in = .{ .scalar = .i32 },
            // strides — k
            .stride_kz_in = .{ .scalar = .i32 },
            .stride_kh_in = .{ .scalar = .i32 },
            .stride_kn_in = .{ .scalar = .i32 },
            .stride_kk_in = .{ .scalar = .i32 },
            // strides — v
            .stride_vz_in = .{ .scalar = .i32 },
            .stride_vh_in = .{ .scalar = .i32 },
            .stride_vn_in = .{ .scalar = .i32 },
            .stride_vk_in = .{ .scalar = .i32 },
            // strides — descale
            .stride_descale_q_z_in = .{ .scalar = .i32 },
            .stride_descale_k_z_in = .{ .scalar = .i32 },
            .stride_descale_v_z_in = .{ .scalar = .i32 },
            // strides — output
            .stride_oz_in = .{ .scalar = .i32 },
            .stride_oh_in = .{ .scalar = .i32 },
            .stride_om_in = .{ .scalar = .i32 },
            .stride_on_in = .{ .scalar = .i32 },
            // strides — alibi
            .stride_alibi_z_in = .{ .scalar = .i32 },
            .stride_alibi_h_in = .{ .scalar = .i32 },
            // strides — sd_mask / dropout
            .stride_sd_z_in = .{ .scalar = .i32 },
            .stride_sd_h_in = .{ .scalar = .i32 },
            .stride_sd_m_in = .{ .scalar = .i32 },
            .stride_sd_n_in = .{ .scalar = .i32 },
            // strides — lse
            .stride_lse_z_in = .{ .scalar = .i32 },
            .stride_lse_h_in = .{ .scalar = .i32 },
            .stride_lse_m_in = .{ .scalar = .i32 },
            // runtime scalars
            .sm_scale = .{ .scalar = .f32 },
            .cu_seqlens_q = .{ .ptr = .i32 },
            .cu_seqlens_k = .{ .ptr = .i32 },
            .dropout_p = .{ .scalar = .f32 },
            .philox_seed = .{ .scalar = .i64 },
            .philox_offset_base_in = .{ .scalar = .i32 },
            // output
            .out_ptr = .{ .ptr = cfg.dtype },
        });

        // USE_INT64_STRIDES=false: strides used as-is (i32)
        const stride_qz = a.stride_qz_in;
        const stride_qh = a.stride_qh_in;
        const stride_qm = a.stride_qm_in;
        const stride_qk = a.stride_qk_in;
        const stride_kz = a.stride_kz_in;
        const stride_kh = a.stride_kh_in;
        const stride_kn = a.stride_kn_in;
        const stride_kk = a.stride_kk_in;
        const stride_vz = a.stride_vz_in;
        const stride_vh = a.stride_vh_in;
        const stride_vn = a.stride_vn_in;
        const stride_vk = a.stride_vk_in;
        const stride_oz = a.stride_oz_in;
        const stride_oh = a.stride_oh_in;
        const stride_om = a.stride_om_in;
        const stride_on = a.stride_on_in;

        // tl.assume on all stride inputs
        b.assume(a.stride_qz_in.ge(0));
        b.assume(a.stride_qh_in.ge(0));
        b.assume(a.stride_qm_in.ge(0));
        b.assume(a.stride_qk_in.ge(0));
        b.assume(a.stride_kz_in.ge(0));
        b.assume(a.stride_kh_in.ge(0));
        b.assume(a.stride_kn_in.ge(0));
        b.assume(a.stride_kk_in.ge(0));
        b.assume(a.stride_vz_in.ge(0));
        b.assume(a.stride_vh_in.ge(0));
        b.assume(a.stride_vn_in.ge(0));
        b.assume(a.stride_vk_in.ge(0));
        b.assume(a.stride_oz_in.ge(0));
        b.assume(a.stride_oh_in.ge(0));
        b.assume(a.stride_om_in.ge(0));
        b.assume(a.stride_on_in.ge(0));
        b.assume(a.stride_alibi_z_in.ge(0));
        b.assume(a.stride_alibi_h_in.ge(0));
        b.assume(a.philox_offset_base_in.ge(0));
        b.assume(a.stride_sd_z_in.ge(0));
        b.assume(a.stride_sd_h_in.ge(0));
        b.assume(a.stride_sd_m_in.ge(0));
        b.assume(a.stride_sd_n_in.ge(0));
        b.assume(a.stride_lse_z_in.ge(0));
        b.assume(a.stride_lse_h_in.ge(0));
        b.assume(a.stride_lse_m_in.ge(0));

        // calculate offsets
        // wid = tl.program_id(0)
        const wid = b.programId(.x);

        // off_q_head = wid % NUM_Q_HEADS
        var off_q_head = wid.rem(NUM_Q_HEADS);
        // off_q_head = remap_xcd(off_q_head, NUM_Q_HEADS, NUM_XCD)
        off_q_head = remapXcd(b, off_q_head, b.liftAs(NUM_Q_HEADS, .i32), b.liftAs(NUM_XCD, .i32));
        // start_m = (wid // NUM_Q_HEADS) % NUM_BLOCKS
        const start_m = wid.div(NUM_Q_HEADS).rem(NUM_BLOCKS);
        // off_z = (wid // (NUM_BLOCKS * NUM_Q_HEADS)) % BATCH
        const off_z = wid.div(NUM_BLOCKS * NUM_Q_HEADS).rem(BATCH);

        // offsets
        // offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        const offs_m = start_m.mul(BLOCK_M).add(b.arange(0, BLOCK_M, .i32));
        // offs_n = tl.arange(0, BLOCK_N)
        const offs_n = b.arange(0, BLOCK_N, .i32);
        // offs_d = tl.arange(0, BLOCK_DMODEL_POW2)
        const offs_d = b.arange(0, BLOCK_DMODEL_POW2, .i32);

        // HAS_PE=false: skip pe offsets

        // VARLEN=false:
        const cu_seqlens_q_start = b.liftAs(0, .i32);
        const cu_seqlens_k_start = b.liftAs(0, .i32);
        const seqlen_q = b.liftAs(SEQLEN_Q, .i32);
        const seqlen_k = b.liftAs(SEQLEN_K, .i32);

        // n_blocks = _cdiv_fn(seqlen_k, BLOCK_N)
        var n_blocks = seqlen_k.add(BLOCK_N - 1).div(BLOCK_N);

        // IS_CAUSAL=false: no causal early exit, n_blocks unchanged

        // grp_sz: tl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
        const grp_sz: i32 = @divTrunc(NUM_Q_HEADS, NUM_K_HEADS);
        // if grp_sz != 1: off_k_head = off_q_head // grp_sz else: off_k_head = off_q_head
        const off_k_head = if (grp_sz != 1) off_q_head.div(grp_sz) else off_q_head;

        // q,k,v offsets
        // q_offs = off_z * stride_qz + off_q_head * stride_qh + cu_seqlens_q_start * stride_qm
        //        + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        const q_offs = off_z.mul(stride_qz)
            .add(off_q_head.mul(stride_qh))
            .add(cu_seqlens_q_start.mul(stride_qm))
            .add(offs_m.expandDims(1).mul(stride_qm))
            .add(offs_d.expandDims(0).mul(stride_qk));
        // q_ptrs = q_ptr + q_offs
        const q_ptrs = a.q_ptr.addPtr(q_offs);

        // HAS_PE=false: skip q_pe_ptrs

        // k_offs = off_z * stride_kz + off_k_head * stride_kh + cu_seqlens_k_start * stride_kn
        //        + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
        const k_offs = off_z.mul(stride_kz)
            .add(off_k_head.mul(stride_kh))
            .add(cu_seqlens_k_start.mul(stride_kn))
            .add(offs_d.expandDims(1).mul(stride_kk))
            .add(offs_n.expandDims(0).mul(stride_kn));
        // k_ptrs = k_ptr + k_offs
        const k_ptrs = a.k_ptr.addPtr(k_offs);

        // v_offs = off_z * stride_vz + off_k_head * stride_vh + cu_seqlens_k_start * stride_vn
        //        + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        const v_offs = off_z.mul(stride_vz)
            .add(off_k_head.mul(stride_vh))
            .add(cu_seqlens_k_start.mul(stride_vn))
            .add(offs_n.expandDims(1).mul(stride_vn))
            .add(offs_d.expandDims(0).mul(stride_vk));
        // v_ptrs = v_ptr + v_offs
        const v_ptrs = a.v_ptr.addPtr(v_offs);

        // alibi slopes — alibi_slopes_ptr is not None but alibi_slope is not used by _attn_fwd_inner
        // when IS_CAUSAL=false and no alibi block is computed. For the pinned config, alibi_slope = None
        // (alibi_slopes_ptr is in the signature but we do not use it).

        // ENABLE_SINK=false
        // m_i_value = float("-inf")
        // m_i = tl.full([BLOCK_M], m_i_value, dtype=tl.float32)
        const m_i = b.full(&.{BLOCK_M}, -std.math.inf(f32), .f32);
        // l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        const l_i = b.full(&.{BLOCK_M}, 1.0, .f32);
        // acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=tl.float32)
        const acc = b.zeros(&.{ BLOCK_M, BLOCK_DMODEL_POW2 }, .f32);

        // q_mask
        const q_mask = if (PADDED_HEAD)
            // (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < BLOCK_DMODEL)
            offs_m.expandDims(1).lt(seqlen_q).bitAnd(offs_d.expandDims(0).lt(@as(i32, BLOCK_DMODEL)))
        else
            // offs_m[:, None] < seqlen_q
            offs_m.expandDims(1).lt(seqlen_q);

        // q_cache_mod
        // if BLOCK_M >= NUM_Q_HEADS: q_cache_mod = ".cg" else: q_cache_mod = ""
        const q_cache_mod: ttir.CacheModifier = if (BLOCK_M >= NUM_Q_HEADS) .cg else .none;

        // HAS_PE=false: skip q_pe
        // q = tl.load(q_ptrs, mask=q_mask, other=0.0, cache_modifier=q_cache_mod)
        const q = b.loadOpts(q_ptrs, .{
            .mask = q_mask,
            .other = b.zeros(&.{ BLOCK_M, BLOCK_DMODEL_POW2 }, cfg.dtype),
            .cache_modifier = q_cache_mod,
        });

        // IS_FP8=false:
        // descale_q, descale_k, descale_v = 1.0, 1.0, 1.0
        // (not used in non-fp8 path)

        // n_extra_tokens = 0
        // if seqlen_k < BLOCK_N: n_extra_tokens = BLOCK_N - seqlen_k
        // elif seqlen_k % BLOCK_N: n_extra_tokens = seqlen_k % BLOCK_N
        var n_extra_tokens: Value = b.liftAs(0, .i32);
        if (SEQLEN_K < BLOCK_N) {
            n_extra_tokens = b.liftAs(BLOCK_N - SEQLEN_K, .i32);
        } else if (@rem(SEQLEN_K, BLOCK_N) != 0) {
            n_extra_tokens = b.liftAs(@rem(SEQLEN_K, BLOCK_N), .i32);
        }

        // padded_block_k = n_extra_tokens != 0
        const padded_block_k: bool = if (SEQLEN_K < BLOCK_N) true else (@rem(SEQLEN_K, BLOCK_N) != 0);

        // IS_CAUSAL=false:
        //   masked_blocks = padded_block_k  (as int: 1 if true, 0 if false)
        const masked_blocks_val: i32 = if (padded_block_k) 1 else 0;

        // SLIDING_WINDOW=0: skipped_blocks=0, visible_blocks=n_blocks
        const block_min = b.liftAs(0, .i32);
        var block_max = n_blocks.mul(BLOCK_N);

        // Compute for full blocks (n_full_blocks > 0)
        // For IS_CAUSAL=false, if padded_block_k is false, all blocks are full.
        // If padded_block_k is true, the last block is masked.
        if (masked_blocks_val == 0) {
            // All blocks are full, no masked pass needed
            const result = attnFwdInner(
                b, acc, l_i, m_i, q,
                k_ptrs, v_ptrs,
                stride_kn, stride_vn,
                seqlen_k,
                block_min, block_max,
                n_extra_tokens,
                BLOCK_M, BLOCK_N, BLOCK_DMODEL, BLOCK_DMODEL_POW2,
                cfg.sm_scale,
                cfg.dtype,
                false, // MASK_STEPS
                PADDED_HEAD,
            );
            const final_acc = result[0];
            const final_l_i = result[1];

            // epilogue
            // l_recip = 1 / l_i[:, None]
            const l_recip = b.full(&.{ BLOCK_M, 1 }, 1.0, .f32).div(final_l_i.expandDims(1));
            // acc = acc * l_recip
            const acc_out = final_acc.mul(l_recip);

            // write back O
            const offs_out = off_z.mul(stride_oz)
                .add(off_q_head.mul(stride_oh))
                .add(cu_seqlens_q_start.mul(stride_om))
                .add(offs_m.expandDims(1).mul(stride_om))
                .add(offs_d.expandDims(0).mul(stride_on));

            // out_mask
            var out_mask = b.full(&.{ BLOCK_M, 1 }, 1, .i1);
            // Always mask by seqlen_q (overflow rows beyond seqlen_q)
            out_mask = out_mask.bitAnd(offs_m.expandDims(1).lt(seqlen_q));
            if (PADDED_HEAD) {
                out_mask = out_mask.bitAnd(offs_d.expandDims(0).lt(@as(i32, BLOCK_DMODEL)));
            }

            // op = acc.to(out_ptr.dtype.element_ty)
            const op = acc_out.to(cfg.dtype);
            b.storeOpts(a.out_ptr.addPtr(offs_out), op, .{ .mask = out_mask });
        } else {
            // There are full blocks followed by one masked block
            const full_block_max = n_blocks.sub(masked_blocks_val).mul(BLOCK_N);

            // Full blocks pass
            const result_full = attnFwdInner(
                b, acc, l_i, m_i, q,
                k_ptrs, v_ptrs,
                stride_kn, stride_vn,
                seqlen_k,
                block_min, full_block_max,
                n_extra_tokens,
                BLOCK_M, BLOCK_N, BLOCK_DMODEL, BLOCK_DMODEL_POW2,
                cfg.sm_scale,
                cfg.dtype,
                false, // MASK_STEPS
                PADDED_HEAD,
            );

            // Update pointers for masked pass
            // k_ptrs += n_full_blocks * BLOCK_N * stride_kn
            const n_full_blocks_rt = n_blocks.sub(masked_blocks_val);
            const k_ptrs_masked = k_ptrs.addPtr(n_full_blocks_rt.mul(BLOCK_N).mul(stride_kn));
            // v_ptrs += n_full_blocks * BLOCK_N * stride_vn
            const v_ptrs_masked = v_ptrs.addPtr(n_full_blocks_rt.mul(BLOCK_N).mul(stride_vn));

            // Masked blocks pass
            block_max = n_blocks.mul(BLOCK_N);
            const result_masked = attnFwdInner(
                b, result_full[0], result_full[1], result_full[2], q,
                k_ptrs_masked, v_ptrs_masked,
                stride_kn, stride_vn,
                seqlen_k,
                full_block_max, block_max,
                n_extra_tokens,
                BLOCK_M, BLOCK_N, BLOCK_DMODEL, BLOCK_DMODEL_POW2,
                cfg.sm_scale,
                cfg.dtype,
                true, // MASK_STEPS
                PADDED_HEAD,
            );

            const final_acc = result_masked[0];
            const final_l_i = result_masked[1];

            // epilogue
            // l_recip = 1 / l_i[:, None]
            const l_recip = b.full(&.{ BLOCK_M, 1 }, 1.0, .f32).div(final_l_i.expandDims(1));
            // acc = acc * l_recip
            const acc_out = final_acc.mul(l_recip);

            // write back O
            const offs_out = off_z.mul(stride_oz)
                .add(off_q_head.mul(stride_oh))
                .add(cu_seqlens_q_start.mul(stride_om))
                .add(offs_m.expandDims(1).mul(stride_om))
                .add(offs_d.expandDims(0).mul(stride_on));

            var out_mask = b.full(&.{ BLOCK_M, 1 }, 1, .i1);
            out_mask = out_mask.bitAnd(offs_m.expandDims(1).lt(seqlen_q));
            if (PADDED_HEAD) {
                out_mask = out_mask.bitAnd(offs_d.expandDims(0).lt(@as(i32, BLOCK_DMODEL)));
            }

            const op = acc_out.to(cfg.dtype);
            b.storeOpts(a.out_ptr.addPtr(offs_out), op, .{ .mask = out_mask });
        }
    }
};
