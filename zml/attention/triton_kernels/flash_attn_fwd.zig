//! Zig port of AMD's `flash_attn:_attn_fwd` (composable kernel for
//! multi-head attention). This is the forward-pass kernel from
//! ROCm/flash-attention's Triton implementation.
//!
//! Currently pinned to:
//!   IS_CAUSAL=true, PRELOAD_V=true, ENABLE_DROPOUT=false,
//!   RETURN_SCORES=false, IS_FP8=false, VARLEN=false,
//!   USE_INT64_STRIDES=true, ENABLE_SINK=false, SLIDING_WINDOW=0,
//!   BLOCK_DMODEL_PE=0 (no PE).
//!   PADDED_HEAD=false (BLOCK_DMODEL == BLOCK_DMODEL_POW2).

const std = @import("std");

const zml = @import("../../zml.zig");
const tri = zml.kernel.triton;

pub const FlashAttnFwd = struct {
    pub const Cfg = struct {
        // Pointer dtypes
        q_dtype: tri.DType = .bf16,
        k_dtype: tri.DType = .bf16,
        v_dtype: tri.DType = .bf16,
        o_dtype: tri.DType = .bf16,
        // tl.constexpr fields
        IS_CAUSAL: bool = true,
        NUM_Q_HEADS: i32 = 32,
        NUM_K_HEADS: i32 = 8,
        PRELOAD_V: bool = true,
        BLOCK_M: i32 = 128,
        BLOCK_N: i32 = 64,
        BLOCK_DMODEL: i32 = 128,
        BLOCK_DMODEL_POW2: i32 = 128,
        BLOCK_DMODEL_PE: i32 = 0,
        RETURN_SCORES: bool = false,
        ENABLE_DROPOUT: bool = false,
        IS_FP8: bool = false,
        FP8_MAX: f32 = 0.0,
        VARLEN: bool = false,
        NUM_XCD: i32 = 8,
        USE_INT64_STRIDES: bool = true,
        ENABLE_SINK: bool = false,
        SLIDING_WINDOW: i32 = 0,
    };

    pub const Kernel = tri.Kernel(Cfg, .{
        .name = "_attn_fwd",
        .inputs = &.{
            "q_ptr",                 "k_ptr",                 "v_ptr",
            "descale_q_ptr",         "descale_k_ptr",         "descale_v_ptr",
            "alibi_slopes_ptr",      "s_dmask_ptr",           "dropout_mask_ptr",
            "softmax_lse_ptr",       "sink_ptr",              "stride_qz_in",
            "stride_qh_in",          "stride_qm_in",          "stride_qk_in",
            "stride_kz_in",          "stride_kh_in",          "stride_kn_in",
            "stride_kk_in",          "stride_vz_in",          "stride_vh_in",
            "stride_vn_in",          "stride_vk_in",          "stride_descale_q_z_in",
            "stride_descale_k_z_in", "stride_descale_v_z_in", "stride_oz_in",
            "stride_oh_in",          "stride_om_in",          "stride_on_in",
            "stride_alibi_z_in",     "stride_alibi_h_in",     "stride_sd_z_in",
            "stride_sd_h_in",        "stride_sd_m_in",        "stride_sd_n_in",
            "stride_lse_z_in",       "stride_lse_h_in",       "stride_lse_m_in",
            "sm_scale",              "cu_seqlens_q",          "cu_seqlens_k",
            "dropout_p",             "philox_seed",           "philox_offset_base_in",
            "SEQLEN_Q",              "SEQLEN_K",              "BATCH",
        },
        .outputs = &.{"out"},
        .run = run,
    });

    // -----------------------------------------------------------------------
    // Helper: remap_xcd (Python `remap_xcd`)
    // -----------------------------------------------------------------------
    fn remapXcd(b: *tri.Builder, pid: tri.Value, GRID_MN: i32, NUM_XCDS: i32) tri.Value {
        const grid_mn = b.liftAs(@as(i32, GRID_MN), .i32);
        const num_xcds = b.liftAs(@as(i32, NUM_XCDS), .i32);
        const pids_per_xcd = grid_mn.add(NUM_XCDS - 1).div(NUM_XCDS);
        const tall_xcds_raw = grid_mn.rem(num_xcds);
        const tall_xcds = b.select(tall_xcds_raw.eq(0), num_xcds, tall_xcds_raw);
        const xcd = pid.rem(num_xcds);
        const local_pid = pid.div(num_xcds);
        const cond = xcd.lt(tall_xcds);
        const pid_tall = xcd.mul(pids_per_xcd).add(local_pid);
        const pid_short = tall_xcds.mul(pids_per_xcd).add(xcd.sub(tall_xcds).mul(pids_per_xcd.sub(1))).add(local_pid);
        return b.select(cond, pid_tall, pid_short);
    }

    // -----------------------------------------------------------------------
    // Helper: _cdiv_fn (Python `_cdiv_fn`)
    // -----------------------------------------------------------------------
    fn cdivFn(x: tri.Value, y: tri.Value) tri.Value {
        return x.add(y).sub(1).div(y);
    }

    // -----------------------------------------------------------------------
    // Main kernel body
    // -----------------------------------------------------------------------
    fn run(b: *tri.Builder, cfg: Cfg) tri.FinishError!void {
        std.debug.assert(cfg.IS_CAUSAL);
        std.debug.assert(!cfg.IS_FP8);
        std.debug.assert(!cfg.VARLEN);
        std.debug.assert(!cfg.ENABLE_DROPOUT);
        std.debug.assert(!cfg.RETURN_SCORES);
        std.debug.assert(cfg.BLOCK_DMODEL_PE == 0);
        std.debug.assert(!cfg.ENABLE_SINK);
        std.debug.assert(cfg.SLIDING_WINDOW == 0);
        std.debug.assert(cfg.USE_INT64_STRIDES);
        std.debug.assert(cfg.PRELOAD_V);
        std.debug.assert(cfg.BLOCK_DMODEL == cfg.BLOCK_DMODEL_POW2);

        const BLOCK_M = cfg.BLOCK_M;
        const BLOCK_N = cfg.BLOCK_N;
        const BLOCK_DMODEL_POW2 = cfg.BLOCK_DMODEL_POW2;
        const NUM_Q_HEADS = cfg.NUM_Q_HEADS;
        const NUM_K_HEADS = cfg.NUM_K_HEADS;
        const NUM_XCD = cfg.NUM_XCD;

        const BLOCK_M_i64: i64 = @intCast(BLOCK_M);
        const BLOCK_N_i64: i64 = @intCast(BLOCK_N);
        _ = BLOCK_N_i64;
        const BLOCK_DMODEL_POW2_i64: i64 = @intCast(BLOCK_DMODEL_POW2);

        const a = try b.declareArgs(.{
            .q_ptr = .{ .ptr = cfg.q_dtype },
            .k_ptr = .{ .ptr = cfg.k_dtype },
            .v_ptr = .{ .ptr = cfg.v_dtype },
            .descale_q_ptr = .{ .ptr = .f32 },
            .descale_k_ptr = .{ .ptr = .f32 },
            .descale_v_ptr = .{ .ptr = .f32 },
            .alibi_slopes_ptr = .{ .ptr = .f32 },
            .s_dmask_ptr = .{ .ptr = .f32 },
            .dropout_mask_ptr = .{ .ptr = .f32 },
            .softmax_lse_ptr = .{ .ptr = .f32 },
            .sink_ptr = .{ .ptr = .f32 },
            .stride_qz_in = .{ .scalar = .i32 },
            .stride_qh_in = .{ .scalar = .i32 },
            .stride_qm_in = .{ .scalar = .i32 },
            .stride_qk_in = .{ .scalar = .i32 },
            .stride_kz_in = .{ .scalar = .i32 },
            .stride_kh_in = .{ .scalar = .i32 },
            .stride_kn_in = .{ .scalar = .i32 },
            .stride_kk_in = .{ .scalar = .i32 },
            .stride_vz_in = .{ .scalar = .i32 },
            .stride_vh_in = .{ .scalar = .i32 },
            .stride_vn_in = .{ .scalar = .i32 },
            .stride_vk_in = .{ .scalar = .i32 },
            .stride_descale_q_z_in = .{ .scalar = .i32 },
            .stride_descale_k_z_in = .{ .scalar = .i32 },
            .stride_descale_v_z_in = .{ .scalar = .i32 },
            .stride_oz_in = .{ .scalar = .i32 },
            .stride_oh_in = .{ .scalar = .i32 },
            .stride_om_in = .{ .scalar = .i32 },
            .stride_on_in = .{ .scalar = .i32 },
            .stride_alibi_z_in = .{ .scalar = .i32 },
            .stride_alibi_h_in = .{ .scalar = .i32 },
            .stride_sd_z_in = .{ .scalar = .i32 },
            .stride_sd_h_in = .{ .scalar = .i32 },
            .stride_sd_m_in = .{ .scalar = .i32 },
            .stride_sd_n_in = .{ .scalar = .i32 },
            .stride_lse_z_in = .{ .scalar = .i32 },
            .stride_lse_h_in = .{ .scalar = .i32 },
            .stride_lse_m_in = .{ .scalar = .i32 },
            .sm_scale = .{ .scalar = .f32 },
            .cu_seqlens_q = .{ .ptr = .i32 },
            .cu_seqlens_k = .{ .ptr = .i32 },
            .dropout_p = .{ .scalar = .f32 },
            .philox_seed = .{ .scalar = .i64 },
            .philox_offset_base_in = .{ .scalar = .i32 },
            .SEQLEN_Q = .{ .scalar = .i32 },
            .SEQLEN_K = .{ .scalar = .i32 },
            .BATCH = .{ .scalar = .i32 },
            .out_ptr = .{ .ptr = cfg.o_dtype },
        });

        const SEQLEN_Q = a.SEQLEN_Q;
        const SEQLEN_K = a.SEQLEN_K;

        const wid = b.programId(.x);

        const off_q_head_raw = wid.rem(NUM_Q_HEADS);
        const off_q_head = remapXcd(b, off_q_head_raw, NUM_Q_HEADS, NUM_XCD);

        const NUM_BLOCKS = SEQLEN_Q.add(BLOCK_M - 1).div(BLOCK_M);
        const start_m = wid.div(NUM_Q_HEADS).rem(NUM_BLOCKS);

        const off_z = wid.div(NUM_BLOCKS.mul(NUM_Q_HEADS)).rem(a.BATCH);

        const offs_m = start_m.mul(BLOCK_M).add(b.arange(0, BLOCK_M, .i32));
        const offs_n = b.arange(0, BLOCK_N, .i32);
        const offs_d = b.arange(0, BLOCK_DMODEL_POW2, .i32);

        const stride_qz = a.stride_qz_in.to(.i64);
        const stride_qh = a.stride_qh_in.to(.i64);
        const stride_qm = a.stride_qm_in.to(.i64);
        const stride_qk = a.stride_qk_in.to(.i64);
        const stride_kz = a.stride_kz_in.to(.i64);
        const stride_kh = a.stride_kh_in.to(.i64);
        const stride_kn = a.stride_kn_in.to(.i64);
        const stride_kk = a.stride_kk_in.to(.i64);
        const stride_vz = a.stride_vz_in.to(.i64);
        const stride_vh = a.stride_vh_in.to(.i64);
        const stride_vn = a.stride_vn_in.to(.i64);
        const stride_vk = a.stride_vk_in.to(.i64);
        const stride_oz = a.stride_oz_in.to(.i64);
        const stride_oh = a.stride_oh_in.to(.i64);
        const stride_om = a.stride_om_in.to(.i64);
        const stride_on = a.stride_on_in.to(.i64);
        const philox_offset_base = a.philox_offset_base_in.to(.i64);
        _ = philox_offset_base;
        const stride_sd_z = a.stride_sd_z_in.to(.i64);
        _ = stride_sd_z;
        const stride_sd_h = a.stride_sd_h_in.to(.i64);
        _ = stride_sd_h;
        const stride_sd_m = a.stride_sd_m_in.to(.i64);
        _ = stride_sd_m;
        const stride_sd_n = a.stride_sd_n_in.to(.i64);
        _ = stride_sd_n;
        const stride_lse_z = a.stride_lse_z_in.to(.i64);
        const stride_lse_h = a.stride_lse_h_in.to(.i64);
        const stride_lse_m = a.stride_lse_m_in.to(.i64);

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
        b.assume(a.philox_offset_base_in.ge(0));
        b.assume(a.stride_sd_z_in.ge(0));
        b.assume(a.stride_sd_h_in.ge(0));
        b.assume(a.stride_sd_m_in.ge(0));
        b.assume(a.stride_sd_n_in.ge(0));
        b.assume(a.stride_lse_z_in.ge(0));
        b.assume(a.stride_lse_h_in.ge(0));
        b.assume(a.stride_lse_m_in.ge(0));

        // VARLEN=false
        const cu_seqlens_q_start = b.liftAs(@as(i32, 0), .i32);
        const cu_seqlens_k_start = b.liftAs(@as(i32, 0), .i32);
        const seqlen_q = SEQLEN_Q;
        const seqlen_k = SEQLEN_K;

        const BLOCK_N_val = b.liftAs(@as(i32, BLOCK_N), .i32);
        var n_blocks = cdivFn(seqlen_k, BLOCK_N_val);

        // IS_CAUSAL
        const n_blocks_seqlen = cdivFn(
            start_m.add(1).mul(BLOCK_M).add(seqlen_k).sub(seqlen_q),
            BLOCK_N_val,
        );
        n_blocks = n_blocks.minimum(n_blocks_seqlen);

        // Early exit if no blocks
        const no_blocks = n_blocks.le(0);
        {
            var scope = b.openReturnIf(no_blocks);
            {
                const offs_out = off_z.to(.i64).mul(stride_oz)
                    .add(off_q_head.to(.i64).mul(stride_oh))
                    .add(cu_seqlens_q_start.to(.i64).mul(stride_om))
                    .add(b.expandDims(offs_m, 1).to(.i64).mul(stride_om))
                    .add(b.expandDims(offs_d, 0).to(.i64).mul(stride_on));
                const acc_zero = b.zeros(&.{ BLOCK_M_i64, BLOCK_DMODEL_POW2_i64 }, cfg.o_dtype);
                const out_mask = b.expandDims(offs_m.lt(seqlen_q), 1).bitAnd(b.expandDims(offs_d.lt(BLOCK_DMODEL_POW2), 0));
                b.storeOpts(a.out_ptr.addPtr(offs_out), acc_zero, .{ .mask = out_mask });

                const offs_lse = off_z.to(.i64).mul(stride_lse_z)
                    .add(off_q_head.to(.i64).mul(stride_lse_h))
                    .add(cu_seqlens_q_start.to(.i64).mul(stride_lse_m))
                    .add(offs_m.to(.i64).mul(stride_lse_m));
                const lse_mask = offs_m.lt(SEQLEN_Q);
                const lse_zero = b.full(&.{BLOCK_M_i64}, 0.0, .f32);
                b.storeOpts(a.softmax_lse_ptr.addPtr(offs_lse), lse_zero, .{ .mask = lse_mask });

                scope.yieldReturn(.{});
            }
        }

        const grp_sz: i32 = @divTrunc(NUM_Q_HEADS, NUM_K_HEADS);
        const off_k_head = if (grp_sz != 1)
            off_q_head.div(grp_sz)
        else
            off_q_head;

        const q_offs = off_z.to(.i64).mul(stride_qz)
            .add(off_q_head.to(.i64).mul(stride_qh))
            .add(cu_seqlens_q_start.to(.i64).mul(stride_qm))
            .add(b.expandDims(offs_m, 1).to(.i64).mul(stride_qm))
            .add(b.expandDims(offs_d, 0).to(.i64).mul(stride_qk));
        const q_ptrs = a.q_ptr.addPtr(q_offs);

        const k_offs = off_z.to(.i64).mul(stride_kz)
            .add(off_k_head.to(.i64).mul(stride_kh))
            .add(cu_seqlens_k_start.to(.i64).mul(stride_kn))
            .add(b.expandDims(offs_d, 1).to(.i64).mul(stride_kk))
            .add(b.expandDims(offs_n, 0).to(.i64).mul(stride_kn));
        var k_ptrs = a.k_ptr.addPtr(k_offs);

        const v_offs = off_z.to(.i64).mul(stride_vz)
            .add(off_k_head.to(.i64).mul(stride_vh))
            .add(cu_seqlens_k_start.to(.i64).mul(stride_vn))
            .add(b.expandDims(offs_n, 1).to(.i64).mul(stride_vn))
            .add(b.expandDims(offs_d, 0).to(.i64).mul(stride_vk));
        var v_ptrs = a.v_ptr.addPtr(v_offs);

        var m_i = b.full(&.{BLOCK_M_i64}, -std.math.inf(f32), .f32);
        var l_i = b.full(&.{BLOCK_M_i64}, 1.0, .f32);
        var acc = b.zeros(&.{ BLOCK_M_i64, BLOCK_DMODEL_POW2_i64 }, .f32);

        const q_mask = b.expandDims(offs_m.lt(seqlen_q), 1);

        const q = b.loadOpts(q_ptrs, .{
            .mask = q_mask,
            .other = b.zeros(&.{ BLOCK_M_i64, BLOCK_DMODEL_POW2_i64 }, cfg.q_dtype),
        });

        const sk_lt_bn = seqlen_k.lt(BLOCK_N);
        const n_extra_if_lt = BLOCK_N_val.sub(seqlen_k);
        const sk_mod_bn = seqlen_k.rem(BLOCK_N_val);
        const n_extra_else = b.select(sk_mod_bn.ne(0), sk_mod_bn, b.liftAs(@as(i32, 0), .i32));
        const n_extra_tokens = b.select(sk_lt_bn, n_extra_if_lt, n_extra_else);

        const padded_block_k = n_extra_tokens.ne(0);
        const BLOCK_M_val = b.liftAs(@as(i32, BLOCK_M), .i32);
        const is_modulo_mn_not = padded_block_k.bitOr(seqlen_q.rem(BLOCK_M_val).ne(0));
        const base_masked: i32 = @divTrunc(BLOCK_M, BLOCK_N);
        const masked_blocks_raw = b.select(is_modulo_mn_not, b.liftAs(@as(i32, base_masked + 1), .i32), b.liftAs(@as(i32, base_masked), .i32));

        const masked_blocks = masked_blocks_raw.minimum(n_blocks);
        const n_full_blocks = n_blocks.sub(masked_blocks);
        var block_min = b.liftAs(@as(i32, 0), .i32);
        var block_max = n_blocks.mul(BLOCK_N_val);

        // Full blocks
        {
            const has_full = n_full_blocks.gt(0);
            var i_scope = b.openIfElse(has_full, .{
                acc.type_(),
                l_i.type_(),
                m_i.type_(),
                k_ptrs.type_(),
                v_ptrs.type_(),
                block_min.type_(),
                block_max.type_(),
            });
            {
                const full_block_max = block_min.add(n_full_blocks.mul(BLOCK_N_val));
                const full_result = attnFwdInner(
                    b,
                    cfg,
                    acc,
                    l_i,
                    m_i,
                    q,
                    k_ptrs,
                    v_ptrs,
                    stride_kn,
                    stride_vn,
                    start_m,
                    seqlen_k,
                    seqlen_q,
                    a.sm_scale,
                    block_min,
                    full_block_max,
                    b.liftAs(@as(i32, 0), .i32),
                    b.liftAs(@as(i32, 0), .i32),
                    b.liftAs(@as(i32, 0), .i32),
                    offs_m,
                    offs_n,
                    false,
                    false,
                );
                const new_block_min = full_block_max;
                const new_block_max = n_blocks.mul(BLOCK_N_val);
                const new_k_ptrs = k_ptrs.addPtr(n_full_blocks.to(.i64).mul(@as(i64, BLOCK_N)).mul(stride_kn));
                const new_v_ptrs = v_ptrs.addPtr(n_full_blocks.to(.i64).mul(@as(i64, BLOCK_N)).mul(stride_vn));
                i_scope.yieldThen(.{ full_result.acc, full_result.l_i, full_result.m_i, new_k_ptrs, new_v_ptrs, new_block_min, new_block_max });
            }
            {
                i_scope.yieldElse(.{ acc, l_i, m_i, k_ptrs, v_ptrs, block_min, block_max });
            }
            acc = i_scope.results[0];
            l_i = i_scope.results[1];
            m_i = i_scope.results[2];
            k_ptrs = i_scope.results[3];
            v_ptrs = i_scope.results[4];
            block_min = i_scope.results[5];
            block_max = i_scope.results[6];
        }

        // Masked blocks
        {
            const has_masked = masked_blocks.gt(0);
            var i_scope = b.openIfElse(has_masked, .{ acc.type_(), l_i.type_(), m_i.type_() });
            {
                const offs_n_causal = offs_n.add(seqlen_q.sub(seqlen_k));
                const masked_result = attnFwdInner(
                    b,
                    cfg,
                    acc,
                    l_i,
                    m_i,
                    q,
                    k_ptrs,
                    v_ptrs,
                    stride_kn,
                    stride_vn,
                    start_m,
                    seqlen_k,
                    seqlen_q,
                    a.sm_scale,
                    block_min,
                    block_max,
                    offs_n_causal,
                    masked_blocks,
                    n_extra_tokens,
                    offs_m,
                    offs_n,
                    true,
                    true,
                );
                i_scope.yieldThen(.{ masked_result.acc, masked_result.l_i, masked_result.m_i });
            }
            {
                i_scope.yieldElse(.{ acc, l_i, m_i });
            }
            acc = i_scope.results[0];
            l_i = i_scope.results[1];
            m_i = i_scope.results[2];
        }

        // Epilogue
        const l_recip = b.full(&.{ BLOCK_M_i64, 1 }, 1.0, .f32).div(b.expandDims(l_i, 1));
        acc = acc.mul(l_recip);

        const end_m_idx = start_m.add(1).mul(BLOCK_M);
        const start_m_idx = start_m.mul(BLOCK_M);
        const causal_start_idx = seqlen_q.sub(seqlen_k);

        {
            const need_fixup = causal_start_idx.gt(start_m_idx).bitAnd(causal_start_idx.lt(end_m_idx));
            var fixup = b.openIfElse(need_fixup, .{acc.type_()});
            {
                const out_mask_boundary = b.full(&.{BLOCK_DMODEL_POW2_i64}, causal_start_idx, .i32);
                const mask_m_offsets = start_m_idx.add(b.arange(0, BLOCK_M, .i32));
                const out_ptrs_mask = b.expandDims(mask_m_offsets, 1).ge(b.expandDims(out_mask_boundary, 0));
                const z = b.zeros(&.{ BLOCK_M_i64, BLOCK_DMODEL_POW2_i64 }, .f32);
                fixup.yieldThen(.{b.where(out_ptrs_mask, acc, z)});
            }
            {
                fixup.yieldElse(.{acc});
            }
            acc = fixup.results[0];
        }

        // Write back LSE
        const overflow_size = end_m_idx.sub(seqlen_q);
        {
            const LN2: f32 = 0.6931471824645996;
            var softmax_lse = m_i.add(b.log2(l_i));
            softmax_lse = softmax_lse.mul(LN2);

            const lse_causal_mask = start_m_idx.add(b.arange(0, BLOCK_M, .i32)).lt(causal_start_idx);
            softmax_lse = b.where(lse_causal_mask, b.full(&.{BLOCK_M_i64}, 0.0, .f32), softmax_lse);

            const offs_lse = off_z.to(.i64).mul(stride_lse_z)
                .add(off_q_head.to(.i64).mul(stride_lse_h))
                .add(cu_seqlens_q_start.to(.i64).mul(stride_lse_m))
                .add(offs_m.to(.i64).mul(stride_lse_m));

            const overflow_pos = overflow_size.gt(0);
            var lse_scope = b.openIfElse(overflow_pos, .{});
            {
                const boundary = b.full(&.{BLOCK_M_i64}, BLOCK_M, .i32).sub(overflow_size);
                const lse_mask = b.arange(0, BLOCK_M, .i32).lt(boundary);
                b.storeOpts(a.softmax_lse_ptr.addPtr(offs_lse), softmax_lse, .{ .mask = lse_mask });
                lse_scope.yieldThen(.{});
            }
            {
                b.store(a.softmax_lse_ptr.addPtr(offs_lse), softmax_lse);
                lse_scope.yieldElse(.{});
            }
        }

        // Write back O
        const offs_out = off_z.to(.i64).mul(stride_oz)
            .add(off_q_head.to(.i64).mul(stride_oh))
            .add(cu_seqlens_q_start.to(.i64).mul(stride_om))
            .add(b.expandDims(offs_m, 1).to(.i64).mul(stride_om))
            .add(b.expandDims(offs_d, 0).to(.i64).mul(stride_on));

        var out_mask = b.full(&.{ BLOCK_M_i64, 1 }, 1, .i1);
        const overflow_pos = overflow_size.gt(0);
        out_mask = b.where(overflow_pos, out_mask.bitAnd(b.expandDims(offs_m.lt(seqlen_q), 1)), out_mask);
        const op = acc.to(cfg.o_dtype);
        b.storeOpts(a.out_ptr.addPtr(offs_out), op, .{ .mask = out_mask });
    }

    // -----------------------------------------------------------------------
    // _attn_fwd_inner
    // -----------------------------------------------------------------------
    const InnerResult = struct {
        acc: tri.Value,
        l_i: tri.Value,
        m_i: tri.Value,
    };

    fn attnFwdInner(
        b: *tri.Builder,
        cfg: Cfg,
        acc_init: tri.Value,
        l_i_init: tri.Value,
        m_i_init: tri.Value,
        q: tri.Value,
        k_ptrs_init: tri.Value,
        v_ptrs_init: tri.Value,
        stride_kn: tri.Value,
        stride_vk: tri.Value,
        start_m: tri.Value,
        seqlen_k: tri.Value,
        seqlen_q: tri.Value,
        sm_scale: tri.Value,
        block_min: tri.Value,
        block_max: tri.Value,
        offs_n_causal: tri.Value,
        masked_blocks_arg: tri.Value,
        n_extra_tokens: tri.Value,
        OFFS_M: tri.Value,
        OFFS_N: tri.Value,
        comptime IS_CAUSAL: bool,
        comptime MASK_STEPS: bool,
    ) InnerResult {
        _ = masked_blocks_arg;
        _ = start_m;
        _ = seqlen_q;

        const BLOCK_M = cfg.BLOCK_M;
        const BLOCK_N = cfg.BLOCK_N;
        const BLOCK_DMODEL_POW2 = cfg.BLOCK_DMODEL_POW2;

        const BLOCK_M_i64: i64 = @intCast(BLOCK_M);
        const BLOCK_N_i64: i64 = @intCast(BLOCK_N);
        const BLOCK_DMODEL_POW2_i64: i64 = @intCast(BLOCK_DMODEL_POW2);

        const RCP_LN2: f32 = 1.4426950408889634;

        var loop = b.openFor(block_min, block_max, @as(i32, BLOCK_N), .{
            acc_init, l_i_init, m_i_init, k_ptrs_init, v_ptrs_init,
        });
        {
            var acc_loop = loop.carried[0];
            var l_i_loop = loop.carried[1];
            var m_i_loop = loop.carried[2];
            const k_ptrs_loop = loop.carried[3];
            const v_ptrs_loop = loop.carried[4];
            const start_n = loop.iv;

            const k = if (MASK_STEPS) blk: {
                const k_offs_n = start_n.add(b.arange(0, BLOCK_N, .i32));
                const mask = b.expandDims(k_offs_n.lt(seqlen_k), 0);
                break :blk b.loadOpts(k_ptrs_loop, .{
                    .mask = mask,
                    .other = b.zeros(&.{ BLOCK_DMODEL_POW2_i64, BLOCK_N_i64 }, cfg.k_dtype),
                });
            } else b.load(k_ptrs_loop);

            const v = if (MASK_STEPS) blk: {
                const k_offs_n = start_n.add(b.arange(0, BLOCK_N, .i32));
                const mask = b.expandDims(k_offs_n.lt(seqlen_k), 1);
                break :blk b.loadOpts(v_ptrs_loop, .{
                    .mask = mask,
                    .other = b.zeros(&.{ BLOCK_N_i64, BLOCK_DMODEL_POW2_i64 }, cfg.v_dtype),
                });
            } else b.load(v_ptrs_loop);

            var mask = b.full(&.{ BLOCK_M_i64, BLOCK_N_i64 }, 1, .i1);

            if (MASK_STEPS) {
                const bound_cond = start_n.add(BLOCK_N).eq(block_max).bitAnd(n_extra_tokens.ne(0));
                const size_n = start_n.add(b.expandDims(OFFS_N, 0));
                const mask_partial = size_n.lt(seqlen_k);
                mask = b.where(bound_cond, mask_partial, mask);
            }

            var qk = b.zeros(&.{ BLOCK_M_i64, BLOCK_N_i64 }, .f32);
            qk = b.dot(q, k, qk);
            qk = qk.mul(sm_scale.mul(RCP_LN2));

            if (IS_CAUSAL) {
                const causal_boundary = start_n.add(offs_n_causal);
                const causal_mask = b.expandDims(OFFS_M, 1).ge(b.expandDims(causal_boundary, 0));
                mask = mask.bitAnd(causal_mask);
            }

            qk = b.where(mask, qk, b.full(&.{ BLOCK_M_i64, BLOCK_N_i64 }, -std.math.inf(f32), .f32));

            const m_ij = m_i_loop.maximum(b.maxOpts(qk, .{ .axis = 1 }));

            const p = b.exp2(qk.sub(b.expandDims(m_ij, 1)));

            const l_ij = b.sumOpts(p, .{ .axis = 1 });

            const alpha = b.exp2(m_i_loop.sub(m_ij));
            acc_loop = acc_loop.mul(b.expandDims(alpha, 1));
            l_i_loop = l_i_loop.mul(alpha).add(l_ij);
            const m_i_new = m_ij;

            acc_loop = b.dot(p.to(cfg.v_dtype), v, acc_loop);

            const new_k_ptrs = k_ptrs_loop.addPtr(b.liftAs(@as(i64, BLOCK_N), .i64).mul(stride_kn));
            const new_v_ptrs = v_ptrs_loop.addPtr(b.liftAs(@as(i64, BLOCK_N), .i64).mul(stride_vk));

            loop.yield(.{ acc_loop, l_i_loop, m_i_new, new_k_ptrs, new_v_ptrs });
        }

        return .{
            .acc = loop.results[0],
            .l_i = loop.results[1],
            .m_i = loop.results[2],
        };
    }
};
