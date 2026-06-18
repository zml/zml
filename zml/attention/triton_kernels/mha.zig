const std = @import("std");

const dialects = @import("mlir/dialects");
const ttir = dialects.ttir;

const zml = @import("../../zml.zig");
const tri = zml.kernel.triton;
const Builder = tri.Builder;
const Value = tri.Value;
const DType = tri.DType;

const RCP_LN2: f32 = 1.4426950408889634;
const LN2: f32 = 0.6931471824645996;

pub const MhaFwd = struct {
    pub const Config = struct {
        q_dtype: DType = .bf16,
        kv_dtype: DType = .bf16,
        out_dtype: DType = .bf16,

        SEQLEN_Q: i64 = 128,
        SEQLEN_K: i64 = 128,
        IS_CAUSAL: bool = true,
        NUM_Q_HEADS: i64 = 8,
        NUM_K_HEADS: i64 = 8,
        PRELOAD_V: bool = false,
        BLOCK_M: i64 = 64,
        BLOCK_N: i64 = 64,
        BLOCK_DMODEL: i64 = 64,
        BLOCK_DMODEL_POW2: i64 = 64,
        BLOCK_DMODEL_PE: i64 = 0,
        IS_FP8: bool = false,
        FP8_MAX: f32 = 448.0,
        VARLEN: bool = false,
        BATCH: i64 = 1,
        NUM_XCD: i64 = 8,
        USE_INT64_STRIDES: bool = true,
        ENABLE_SINK: bool = false,
        SLIDING_WINDOW: i64 = 0,
        HEAD_STRIDE_ALIGNED_8: bool = true,
    };

    pub const Kernel = tri.Kernel(Config, .{
        .name = "mha",
        // The reference Python signature has output pointers interleaved with
        // inputs. Keep this kernel emit-only at the declarative layer; the
        // harness wires XLA manually with aliased output operands.
        .inputs = &.{
            "q_ptr",
            "k_ptr",
            "v_ptr",
            "descale_q_ptr",
            "descale_k_ptr",
            "descale_v_ptr",
            "out_ptr",
            "alibi_slopes_ptr",
            "softmax_lse_ptr",
            "sink_ptr",
            "stride_qz_in_ptr",
            "stride_qh_in_ptr",
            "stride_qm_in_ptr",
            "stride_qk_in_ptr",
            "stride_kz_in_ptr",
            "stride_kh_in_ptr",
            "stride_kn_in_ptr",
            "stride_kk_in_ptr",
            "stride_vz_in_ptr",
            "stride_vh_in_ptr",
            "stride_vn_in_ptr",
            "stride_vk_in_ptr",
            "stride_descale_q_z_in_ptr",
            "stride_descale_k_z_in_ptr",
            "stride_descale_v_z_in_ptr",
            "stride_oz_in_ptr",
            "stride_oh_in_ptr",
            "stride_om_in_ptr",
            "stride_on_in_ptr",
            "stride_alibi_z_in_ptr",
            "stride_alibi_h_in_ptr",
            "stride_lse_z_in_ptr",
            "stride_lse_h_in_ptr",
            "stride_lse_m_in_ptr",
            "sm_scale_ptr",
            "cu_seqlens_q",
            "cu_seqlens_k",
        },
        .outputs = &.{},
        .run = run,
    });

    fn run(b: *Builder, cfg: Config) tri.FinishError!void {
        std.debug.assert(!cfg.PRELOAD_V);
        std.debug.assert(cfg.BLOCK_DMODEL_PE == 0);
        std.debug.assert(cfg.USE_INT64_STRIDES);

        const a = try b.declareArgs(.{
            .q_ptr = .{ .ptr = cfg.q_dtype },
            .k_ptr = .{ .ptr = cfg.kv_dtype },
            .v_ptr = .{ .ptr = cfg.kv_dtype },
            .descale_q_ptr = .{ .ptr = .f32 },
            .descale_k_ptr = .{ .ptr = .f32 },
            .descale_v_ptr = .{ .ptr = .f32 },
            .out_ptr = .{ .ptr = cfg.out_dtype },
            .alibi_slopes_ptr = .{ .ptr = .f32 },
            .softmax_lse_ptr = .{ .ptr = .f32 },
            .sink_ptr = .{ .ptr = .f32 },
            .stride_qz_in_ptr = .{ .ptr = .i64 },
            .stride_qh_in_ptr = .{ .ptr = .i64 },
            .stride_qm_in_ptr = .{ .ptr = .i64 },
            .stride_qk_in_ptr = .{ .ptr = .i64 },
            .stride_kz_in_ptr = .{ .ptr = .i64 },
            .stride_kh_in_ptr = .{ .ptr = .i64 },
            .stride_kn_in_ptr = .{ .ptr = .i64 },
            .stride_kk_in_ptr = .{ .ptr = .i64 },
            .stride_vz_in_ptr = .{ .ptr = .i64 },
            .stride_vh_in_ptr = .{ .ptr = .i64 },
            .stride_vn_in_ptr = .{ .ptr = .i64 },
            .stride_vk_in_ptr = .{ .ptr = .i64 },
            .stride_descale_q_z_in_ptr = .{ .ptr = .i64 },
            .stride_descale_k_z_in_ptr = .{ .ptr = .i64 },
            .stride_descale_v_z_in_ptr = .{ .ptr = .i64 },
            .stride_oz_in_ptr = .{ .ptr = .i64 },
            .stride_oh_in_ptr = .{ .ptr = .i64 },
            .stride_om_in_ptr = .{ .ptr = .i64 },
            .stride_on_in_ptr = .{ .ptr = .i64 },
            .stride_alibi_z_in_ptr = .{ .ptr = .i64 },
            .stride_alibi_h_in_ptr = .{ .ptr = .i64 },
            .stride_lse_z_in_ptr = .{ .ptr = .i64 },
            .stride_lse_h_in_ptr = .{ .ptr = .i64 },
            .stride_lse_m_in_ptr = .{ .ptr = .i64 },
            .sm_scale_ptr = .{ .ptr = .f32 },
            .cu_seqlens_q = .{ .ptr = .i32 },
            .cu_seqlens_k = .{ .ptr = .i32 },
        });

        const stride_qz_in = b.load(a.stride_qz_in_ptr);
        const stride_qh_in = b.load(a.stride_qh_in_ptr);
        const stride_qm_in = b.load(a.stride_qm_in_ptr);
        const stride_qk_in = b.load(a.stride_qk_in_ptr);
        const stride_kz_in = b.load(a.stride_kz_in_ptr);
        const stride_kh_in = b.load(a.stride_kh_in_ptr);
        const stride_kn_in = b.load(a.stride_kn_in_ptr);
        const stride_kk_in = b.load(a.stride_kk_in_ptr);
        const stride_vz_in = b.load(a.stride_vz_in_ptr);
        const stride_vh_in = b.load(a.stride_vh_in_ptr);
        const stride_vn_in = b.load(a.stride_vn_in_ptr);
        const stride_vk_in = b.load(a.stride_vk_in_ptr);
        const stride_descale_q_z_in = b.load(a.stride_descale_q_z_in_ptr);
        const stride_descale_k_z_in = b.load(a.stride_descale_k_z_in_ptr);
        const stride_descale_v_z_in = b.load(a.stride_descale_v_z_in_ptr);
        const stride_oz_in = b.load(a.stride_oz_in_ptr);
        const stride_oh_in = b.load(a.stride_oh_in_ptr);
        const stride_om_in = b.load(a.stride_om_in_ptr);
        const stride_on_in = b.load(a.stride_on_in_ptr);
        const stride_alibi_z_in = b.load(a.stride_alibi_z_in_ptr);
        const stride_alibi_h_in = b.load(a.stride_alibi_h_in_ptr);
        const stride_lse_z_in = b.load(a.stride_lse_z_in_ptr);
        const stride_lse_h_in = b.load(a.stride_lse_h_in_ptr);
        const stride_lse_m_in = b.load(a.stride_lse_m_in_ptr);
        const sm_scale = b.load(a.sm_scale_ptr);

        const stride_qz = stride_qz_in.to(.i64);
        const stride_qh = stride_qh_in.to(.i64);
        const stride_qm = stride_qm_in.to(.i64);
        const stride_qk = stride_qk_in.to(.i64);
        const stride_kz = stride_kz_in.to(.i64);
        const stride_kh = stride_kh_in.to(.i64);
        const stride_kn = stride_kn_in.to(.i64);
        const stride_kk = stride_kk_in.to(.i64);
        const stride_vz = stride_vz_in.to(.i64);
        const stride_vh = stride_vh_in.to(.i64);
        const stride_vn = stride_vn_in.to(.i64);
        const stride_vk = stride_vk_in.to(.i64);
        const stride_descale_q_z = if (cfg.IS_FP8) stride_descale_q_z_in.to(.i64) else stride_descale_q_z_in;
        const stride_descale_k_z = if (cfg.IS_FP8) stride_descale_k_z_in.to(.i64) else stride_descale_k_z_in;
        const stride_descale_v_z = if (cfg.IS_FP8) stride_descale_v_z_in.to(.i64) else stride_descale_v_z_in;
        const stride_oz = stride_oz_in.to(.i64);
        const stride_oh = stride_oh_in.to(.i64);
        const stride_om = stride_om_in.to(.i64);
        const stride_on = stride_on_in.to(.i64);
        const stride_alibi_z = stride_alibi_z_in.to(.i64);
        const stride_alibi_h = stride_alibi_h_in.to(.i64);
        const stride_lse_z = stride_lse_z_in.to(.i64);
        const stride_lse_h = stride_lse_h_in.to(.i64);
        const stride_lse_m = stride_lse_m_in.to(.i64);

        mhaFwd(
            b,
            a.q_ptr,
            a.k_ptr,
            a.v_ptr,
            a.descale_q_ptr,
            a.descale_k_ptr,
            a.descale_v_ptr,
            a.out_ptr,
            a.alibi_slopes_ptr,
            a.softmax_lse_ptr,
            a.sink_ptr,
            stride_qz,
            stride_qh,
            stride_qm,
            stride_qk,
            stride_kz,
            stride_kh,
            stride_kn,
            stride_kk,
            stride_vz,
            stride_vh,
            stride_vn,
            stride_vk,
            stride_descale_q_z,
            stride_descale_k_z,
            stride_descale_v_z,
            stride_oz,
            stride_oh,
            stride_om,
            stride_on,
            stride_alibi_z,
            stride_alibi_h,
            stride_lse_z,
            stride_lse_h,
            stride_lse_m,
            sm_scale,
            a.cu_seqlens_q,
            a.cu_seqlens_k,
            cfg,
        );
    }
};

fn ceilDivComptime(x: i64, y: i64) i64 {
    return @divFloor(x + y - 1, y);
}

fn remapXcd(b: *Builder, pid: Value, grid_mn: i64, num_xcds: i64) Value {
    const pids_per_xcd: i32 = @intCast(ceilDivComptime(grid_mn, num_xcds));
    var tall_xcds: i32 = @intCast(@mod(grid_mn, num_xcds));
    if (tall_xcds == 0) tall_xcds = @intCast(num_xcds);

    const num_xcds_i32: i32 = @intCast(num_xcds);
    const xcd = pid.rem(num_xcds_i32);
    const local_pid = pid.div(num_xcds_i32);

    var i = b.openIfElse(xcd.lt(tall_xcds), .{b.scalarTy(.i32)});
    {
        i.yieldThen(.{xcd.mul(pids_per_xcd).add(local_pid)});
    }
    {
        i.yieldElse(.{
            b.liftAs(tall_xcds, .i32).mul(pids_per_xcd)
                .add(xcd.sub(tall_xcds).mul(pids_per_xcd - 1))
                .add(local_pid),
        });
    }

    return i.results[0];
}

fn loadK(
    b: *Builder,
    ptrs: Value,
    start_n: Value,
    offs_n: Value,
    seqlen_k: anytype,
    cfg: MhaFwd.Config,
    comptime mask_steps: bool,
) Value {
    if (!mask_steps) return b.load(ptrs);

    const k_offs_n = start_n.add(offs_n);
    const mask = k_offs_n.expandDims(0).lt(seqlen_k);
    return b.loadOpts(ptrs, .{
        .mask = mask,
        .other = b.zeros(&.{ cfg.BLOCK_DMODEL_POW2, cfg.BLOCK_N }, cfg.kv_dtype),
    });
}

fn loadV(
    b: *Builder,
    ptrs: Value,
    start_n: Value,
    offs_n: Value,
    seqlen_k: anytype,
    cfg: MhaFwd.Config,
    comptime mask_steps: bool,
) Value {
    if (!mask_steps) return b.load(ptrs);

    const k_offs_n = start_n.add(offs_n);
    const mask = k_offs_n.expandDims(1).lt(seqlen_k);
    return b.loadOpts(ptrs, .{
        .mask = mask,
        .other = b.zeros(&.{ cfg.BLOCK_N, cfg.BLOCK_DMODEL_POW2 }, cfg.kv_dtype),
    });
}

fn computeAlibiBlock(
    b: *Builder,
    alibi_slope: Value,
    seqlen_q: anytype,
    seqlen_k: anytype,
    offs_m: Value,
    offs_n: Value,
) Value {
    const relative_pos_block = offs_m.expandDims(1)
        .add(seqlen_k)
        .sub(seqlen_q)
        .sub(offs_n.expandDims(0));
    return alibi_slope.mul(-1.0).mul(b.abs(relative_pos_block).to(.f32));
}

fn computeNExtraTokens(b: *Builder, seqlen_k: Value, block_n: i32) Value {
    var too_short = b.openIfElse(seqlen_k.lt(block_n), .{b.scalarTy(.i32)});
    {
        too_short.yieldThen(.{b.liftAs(block_n, .i32).sub(seqlen_k)});
    }
    {
        const rem = seqlen_k.rem(block_n);
        var has_rem = b.openIfElse(rem.ne(0), .{b.scalarTy(.i32)});
        {
            has_rem.yieldThen(.{rem});
        }
        {
            has_rem.yieldElse(.{b.liftAs(0, .i32)});
        }
        too_short.yieldElse(.{has_rem.results[0]});
    }
    return too_short.results[0];
}

fn addVarlenScalarOffset(base: Value, start: Value, stride: Value, cfg: MhaFwd.Config) Value {
    return if (cfg.VARLEN) base.add(start.to(.i64).mul(stride)) else base;
}

fn attnFwdInner(
    b: *Builder,
    acc_init: Value,
    l_i_init: Value,
    m_i_init: Value,
    q: Value,
    k_ptrs_init: Value,
    v_ptrs_init: Value,
    stride_kn: Value,
    stride_vk: Value,
    start_m: Value,
    seqlen_k: anytype,
    seqlen_q: anytype,
    block_min: Value,
    block_max: Value,
    offs_n_causal: Value,
    n_extra_tokens: anytype,
    alibi_slope: Value,
    descale_q: Value,
    descale_k: Value,
    descale_v: Value,
    sm_scale: Value,
    offs_m: Value,
    offs_n: Value,
    cfg: MhaFwd.Config,
    comptime is_causal: bool,
    comptime mask_steps: bool,
) [3]Value {
    const BLOCK_N_I32: i32 = @intCast(cfg.BLOCK_N);

    var loop = b.openFor(block_min, block_max, BLOCK_N_I32, .{
        acc_init,
        l_i_init,
        m_i_init,
        k_ptrs_init,
        v_ptrs_init,
    });
    if (mask_steps) loop.opts.num_stages = 1;
    {
        const start_n = loop.iv;
        var acc = loop.carried[0];
        var l_i = loop.carried[1];
        var m_i = loop.carried[2];
        const k_ptrs = loop.carried[3];
        const v_ptrs = loop.carried[4];

        const k = loadK(b, k_ptrs, start_n, offs_n, seqlen_k, cfg, mask_steps);

        var mask = if (mask_steps)
            b.full(&.{ 1, cfg.BLOCK_N }, 1, .i1)
        else
            b.full(&.{ cfg.BLOCK_M, cfg.BLOCK_N }, 1, .i1);
        if (mask_steps and (cfg.VARLEN or @mod(cfg.SEQLEN_K, cfg.BLOCK_N) != 0 or cfg.SEQLEN_K < cfg.BLOCK_N)) {
            var bound_cond = start_n.add(BLOCK_N_I32).eq(block_max);
            if (cfg.VARLEN) {
                bound_cond = bound_cond.bitAnd(n_extra_tokens.ne(0));
            }
            const size_n = start_n.splatTo(&.{ 1, cfg.BLOCK_N }).add(offs_n.expandDims(0));
            const mask_partial = size_n.lt(seqlen_k);
            mask = b.where(bound_cond, mask_partial, mask);
        }

        const qk_scale = sm_scale.mul(RCP_LN2);
        var qk = b.dot(q, k, b.zeros(&.{ cfg.BLOCK_M, cfg.BLOCK_N }, .f32));
        qk = if (cfg.IS_FP8)
            qk.mul(qk_scale.mul(descale_q).mul(descale_k))
        else
            qk.mul(qk_scale);

        if (is_causal) {
            const causal_boundary = start_n.add(offs_n_causal);
            const causal_mask = offs_m.expandDims(1).ge(causal_boundary.expandDims(0));
            mask = mask.bitAnd(causal_mask);
        }
        if (cfg.SLIDING_WINDOW > 0) {
            const k_pos = start_n.add(b.arange(0, cfg.BLOCK_N, .i32));
            const q_adj = offs_m.add(seqlen_k).sub(seqlen_q);
            const window_mask = k_pos.expandDims(0).ge(q_adj.expandDims(1).sub(@as(i32, @intCast(cfg.SLIDING_WINDOW))));
            mask = mask.bitAnd(window_mask);
        }

        qk = b.where(mask, qk, b.full(&.{ cfg.BLOCK_M, cfg.BLOCK_N }, -std.math.inf(f32), .f32));

        const global_m_positions = start_m.mul(@as(i32, @intCast(cfg.BLOCK_M))).add(b.arange(0, cfg.BLOCK_M, .i32));
        const global_n_positions = start_n.add(b.arange(0, cfg.BLOCK_N, .i32));
        const alibi_block = computeAlibiBlock(b, alibi_slope, seqlen_q, seqlen_k, global_m_positions, global_n_positions);
        qk = qk.add(alibi_block.mul(RCP_LN2));

        const m_ij = m_i.maximum(b.maxOpts(qk, .{ .axis = 1 }));
        var p = b.exp2(qk.sub(m_ij.expandDims(1)));
        if (cfg.SLIDING_WINDOW > 0) {
            p = b.where(mask, p, b.zeros(&.{ cfg.BLOCK_M, cfg.BLOCK_N }, .f32));
        }
        const l_ij = b.sumOpts(p, .{ .axis = 1 });

        var alpha = b.exp2(m_i.sub(m_ij));
        if (cfg.SLIDING_WINDOW > 0) {
            alpha = b.where(m_i.eq(m_ij), b.full(&.{cfg.BLOCK_M}, 1.0, .f32), alpha);
        }
        acc = acc.mul(alpha.expandDims(1));
        l_i = l_i.mul(alpha).add(l_ij);
        m_i = m_ij;

        const v = loadV(b, v_ptrs, start_n, offs_n, seqlen_k, cfg, mask_steps);
        if (cfg.IS_FP8) {
            var p_amax = b.max(p.abs());
            p_amax = b.where(p_amax.le(1e-9), b.liftAs(1e-9, .f32), p_amax);
            const scale_p = b.liftAs(cfg.FP8_MAX, .f32).div(p_amax);
            const descale_p = p_amax.div(cfg.FP8_MAX);
            const scaled = b.dot(
                p.mul(scale_p).to(cfg.kv_dtype),
                v,
                b.zeros(&.{ cfg.BLOCK_M, cfg.BLOCK_DMODEL_POW2 }, .f32),
            );
            acc = acc.add(scaled.mul(descale_p).mul(descale_v));
        } else {
            acc = b.dot(p.to(cfg.kv_dtype), v, acc);
        }

        const new_k_ptrs = k_ptrs.addPtr(stride_kn.mul(BLOCK_N_I32));
        const new_v_ptrs = v_ptrs.addPtr(stride_vk.mul(BLOCK_N_I32));

        loop.yield(.{ acc, l_i, m_i, new_k_ptrs, new_v_ptrs });
    }

    return .{ loop.results[0], loop.results[1], loop.results[2] };
}

fn mhaFwd(
    b: *Builder,
    q_ptr: Value,
    k_ptr: Value,
    v_ptr: Value,
    descale_q_ptr: Value,
    descale_k_ptr: Value,
    descale_v_ptr: Value,
    out_ptr: Value,
    alibi_slopes_ptr: Value,
    softmax_lse_ptr: Value,
    sink_ptr: Value,
    stride_qz: Value,
    stride_qh: Value,
    stride_qm: Value,
    stride_qk: Value,
    stride_kz: Value,
    stride_kh: Value,
    stride_kn: Value,
    stride_kk: Value,
    stride_vz: Value,
    stride_vh: Value,
    stride_vn: Value,
    stride_vk: Value,
    stride_descale_q_z: Value,
    stride_descale_k_z: Value,
    stride_descale_v_z: Value,
    stride_oz: Value,
    stride_oh: Value,
    stride_om: Value,
    stride_on: Value,
    stride_alibi_z: Value,
    stride_alibi_h: Value,
    stride_lse_z: Value,
    stride_lse_h: Value,
    stride_lse_m: Value,
    sm_scale: Value,
    cu_seqlens_q: Value,
    cu_seqlens_k: Value,
    cfg: MhaFwd.Config,
) void {
    const NUM_BLOCKS: i64 = ceilDivComptime(cfg.SEQLEN_Q, cfg.BLOCK_M);
    const BLOCK_M_I32: i32 = @intCast(cfg.BLOCK_M);
    const BLOCK_N_I32: i32 = @intCast(cfg.BLOCK_N);
    const BLOCK_DMODEL_I32: i32 = @intCast(cfg.BLOCK_DMODEL);
    const SEQLEN_Q_I32: i32 = @intCast(cfg.SEQLEN_Q);
    const SEQLEN_K_I32: i32 = @intCast(cfg.SEQLEN_K);
    const NUM_Q_HEADS_I32: i32 = @intCast(cfg.NUM_Q_HEADS);
    const NUM_BLOCKS_I32: i32 = @intCast(NUM_BLOCKS);

    const wid = b.programId(.x);

    var off_q_head = wid.rem(NUM_Q_HEADS_I32);
    off_q_head = remapXcd(b, off_q_head, cfg.NUM_Q_HEADS, cfg.NUM_XCD);
    const start_m = wid.div(NUM_Q_HEADS_I32).rem(NUM_BLOCKS_I32);
    const off_z = wid.div(NUM_BLOCKS_I32 * NUM_Q_HEADS_I32).rem(@as(i32, @intCast(cfg.BATCH)));

    const offs_m = start_m.mul(BLOCK_M_I32).add(b.arange(0, cfg.BLOCK_M, .i32));
    const offs_n = b.arange(0, cfg.BLOCK_N, .i32);
    const offs_d = b.arange(0, cfg.BLOCK_DMODEL_POW2, .i32);

    b.assume(stride_qz.ge(0));
    b.assume(stride_qh.ge(0));
    b.assume(stride_qm.ge(0));
    b.assume(stride_qk.ge(0));
    b.assume(stride_kz.ge(0));
    b.assume(stride_kh.ge(0));
    b.assume(stride_kn.ge(0));
    b.assume(stride_kk.ge(0));
    b.assume(stride_vz.ge(0));
    b.assume(stride_vh.ge(0));
    b.assume(stride_vn.ge(0));
    b.assume(stride_vk.ge(0));
    if (cfg.IS_FP8) {
        b.assume(stride_descale_q_z.ge(0));
        b.assume(stride_descale_k_z.ge(0));
        b.assume(stride_descale_v_z.ge(0));
        b.assume(stride_oz.ge(0));
        b.assume(stride_oh.ge(0));
        b.assume(stride_om.ge(0));
        b.assume(stride_on.ge(0));
        b.assume(stride_alibi_z.ge(0));
        b.assume(stride_alibi_h.ge(0));
    }
    b.assume(stride_lse_z.ge(0));
    b.assume(stride_lse_h.ge(0));
    b.assume(stride_lse_m.ge(0));

    const cu_seqlens_q_start: Value = if (cfg.VARLEN)
        b.load(cu_seqlens_q.addPtr(off_z))
    else
        b.liftAs(0, .i32);
    const seqlen_q: Value = if (cfg.VARLEN) blk: {
        const cu_seqlens_q_end = b.load(cu_seqlens_q.addPtr(off_z.add(1)));
        break :blk cu_seqlens_q_end.sub(cu_seqlens_q_start);
    } else b.liftAs(SEQLEN_Q_I32, .i32);

    if (cfg.VARLEN) {
        var ret = b.openReturnIf(start_m.mul(BLOCK_M_I32).gt(seqlen_q));
        ret.inline_return = true;
        {
            ret.yieldReturn(.{});
        }
    }

    const cu_seqlens_k_start: Value = if (cfg.VARLEN)
        b.load(cu_seqlens_k.addPtr(off_z))
    else
        b.liftAs(0, .i32);
    const seqlen_k: Value = if (cfg.VARLEN) blk: {
        const cu_seqlens_k_end = b.load(cu_seqlens_k.addPtr(off_z.add(1)));
        break :blk cu_seqlens_k_end.sub(cu_seqlens_k_start);
    } else b.liftAs(SEQLEN_K_I32, .i32);

    var n_blocks = if (cfg.VARLEN)
        seqlen_k.cdiv(BLOCK_N_I32)
    else
        b.liftAs(ceilDivComptime(cfg.SEQLEN_K, cfg.BLOCK_N), .i32);

    if (cfg.IS_CAUSAL) {
        const n_blocks_seqlen = if (cfg.VARLEN)
            start_m.add(1).mul(BLOCK_M_I32).add(seqlen_k).sub(seqlen_q).cdiv(BLOCK_N_I32)
        else
            start_m.add(1).mul(BLOCK_M_I32).add(SEQLEN_K_I32).sub(SEQLEN_Q_I32).cdiv(BLOCK_N_I32);
        n_blocks = n_blocks.minimum(n_blocks_seqlen);

        var ret = b.openReturnIf(n_blocks.le(0));
        ret.inline_return = true;
        {
            var offs_out_base = off_z.mul(stride_oz)
                .add(off_q_head.mul(stride_oh));
            offs_out_base = addVarlenScalarOffset(offs_out_base, cu_seqlens_q_start, stride_om, cfg);
            const offs_out = offs_out_base
                .add(offs_m.expandDims(1).mul(stride_om))
                .add(offs_d.expandDims(0).mul(stride_on));
            const acc = b.zeros(&.{ cfg.BLOCK_M, cfg.BLOCK_DMODEL_POW2 }, cfg.out_dtype);
            const out_mask = offs_m.expandDims(1).lt(seqlen_q)
                .bitAnd(offs_d.expandDims(0).lt(BLOCK_DMODEL_I32));
            b.storeOpts(out_ptr.addPtr(offs_out), acc, .{ .mask = out_mask });

            var offs_lse_base = off_z.mul(stride_lse_z)
                .add(off_q_head.mul(stride_lse_h));
            offs_lse_base = addVarlenScalarOffset(offs_lse_base, cu_seqlens_q_start, stride_lse_m, cfg);
            const offs_lse = offs_lse_base.add(offs_m.mul(stride_lse_m));
            const lse_mask = offs_m.lt(SEQLEN_Q_I32);
            const lse = b.full(&.{cfg.BLOCK_M}, 0.0, .f32);
            b.storeOpts(softmax_lse_ptr.addPtr(offs_lse), lse, .{ .mask = lse_mask });
            ret.yieldReturn(.{});
        }
    }

    const grp_sz: i32 = @intCast(@divTrunc(cfg.NUM_Q_HEADS, cfg.NUM_K_HEADS));
    const off_k_head = if (grp_sz != 1) off_q_head.div(grp_sz) else off_q_head;

    const qh_off = off_q_head.mul(stride_qh);
    const kh_off = off_k_head.mul(stride_kh);
    const vh_off = off_k_head.mul(stride_vh);

    var q_base = off_z.mul(stride_qz).add(qh_off);
    q_base = addVarlenScalarOffset(q_base, cu_seqlens_q_start, stride_qm, cfg);
    const q_offs = q_base
        .add(offs_m.expandDims(1).mul(stride_qm))
        .add(offs_d.expandDims(0).mul(stride_qk));
    const q_ptrs = q_ptr.addPtr(q_offs);

    var k_base = off_z.mul(stride_kz).add(kh_off);
    k_base = addVarlenScalarOffset(k_base, cu_seqlens_k_start, stride_kn, cfg);
    const k_offs = k_base
        .add(offs_d.expandDims(1).mul(stride_kk))
        .add(offs_n.expandDims(0).mul(stride_kn));
    const k_ptrs = k_ptr.addPtr(k_offs);

    var v_base = off_z.mul(stride_vz).add(vh_off);
    v_base = addVarlenScalarOffset(v_base, cu_seqlens_k_start, stride_vn, cfg);
    const v_offs = v_base
        .add(offs_n.expandDims(1).mul(stride_vn))
        .add(offs_d.expandDims(0).mul(stride_vk));
    const v_ptrs = v_ptr.addPtr(v_offs);

    const alibi_offs = off_z.mul(stride_alibi_z).add(off_q_head.mul(stride_alibi_h));
    const alibi_slope = b.load(alibi_slopes_ptr.addPtr(alibi_offs));

    const m_i = if (cfg.ENABLE_SINK)
        b.load(sink_ptr.addPtr(off_q_head)).to(.f32).mul(RCP_LN2).splatTo(&.{cfg.BLOCK_M})
    else
        b.full(&.{cfg.BLOCK_M}, -std.math.inf(f32), .f32);
    const l_i = b.full(&.{cfg.BLOCK_M}, 1.0, .f32);
    const acc = b.zeros(&.{ cfg.BLOCK_M, cfg.BLOCK_DMODEL_POW2 }, .f32);

    const q_mask = if (cfg.BLOCK_DMODEL == cfg.BLOCK_DMODEL_POW2)
        offs_m.expandDims(1).lt(seqlen_q)
    else
        offs_m.expandDims(1).lt(seqlen_q).bitAnd(offs_d.expandDims(0).lt(BLOCK_DMODEL_I32));

    const q_cache_mod: ttir.CacheModifier = if (cfg.BLOCK_M >= cfg.NUM_Q_HEADS) .cg else .none;
    const q = b.loadOpts(q_ptrs, .{
        .mask = q_mask,
        .other = b.zeros(&.{ cfg.BLOCK_M, cfg.BLOCK_DMODEL_POW2 }, cfg.q_dtype),
        .cache_modifier = q_cache_mod,
    });
    const descale_q = if (cfg.IS_FP8)
        b.load(descale_q_ptr.addPtr(off_z.mul(stride_descale_q_z).add(off_q_head)))
    else
        sm_scale;
    const descale_k = if (cfg.IS_FP8)
        b.load(descale_k_ptr.addPtr(off_z.mul(stride_descale_k_z).add(off_k_head)))
    else
        sm_scale;
    const descale_v = if (cfg.IS_FP8)
        b.load(descale_v_ptr.addPtr(off_z.mul(stride_descale_v_z).add(off_k_head)))
    else
        sm_scale;

    const n_extra_tokens = if (cfg.VARLEN)
        computeNExtraTokens(b, seqlen_k, BLOCK_N_I32)
    else
        b.liftAs(
            if (cfg.SEQLEN_K < cfg.BLOCK_N)
                cfg.BLOCK_N - cfg.SEQLEN_K
            else if (@mod(cfg.SEQLEN_K, cfg.BLOCK_N) != 0)
                @mod(cfg.SEQLEN_K, cfg.BLOCK_N)
            else
                0,
            .i32,
        );

    var block_min = b.liftAs(0, .i32);
    var k_ptrs_loop = k_ptrs;
    var v_ptrs_loop = v_ptrs;
    const visible_blocks = if (cfg.SLIDING_WINDOW > 0) blk: {
        const window_start_n = start_m.mul(BLOCK_M_I32)
            .add(seqlen_k)
            .sub(seqlen_q)
            .sub(@as(i32, @intCast(cfg.SLIDING_WINDOW)));
        const skipped_blocks = window_start_n.maximum(0).div(BLOCK_N_I32).minimum(n_blocks);
        block_min = skipped_blocks.mul(BLOCK_N_I32);
        var skip_if = b.openIfElse(skipped_blocks.gt(0), .{ k_ptrs.type_(), v_ptrs.type_() });
        {
            const skipped = skipped_blocks.mul(BLOCK_N_I32);
            skip_if.yieldThen(.{
                k_ptrs.addPtr(skipped.to(.i64).mul(stride_kn)),
                v_ptrs.addPtr(skipped.to(.i64).mul(stride_vn)),
            });
        }
        {
            skip_if.yieldElse(.{ k_ptrs, v_ptrs });
        }
        k_ptrs_loop = skip_if.results[0];
        v_ptrs_loop = skip_if.results[1];
        break :blk n_blocks.sub(skipped_blocks);
    } else n_blocks;

    const masked_blocks_base: Value = if (cfg.VARLEN) blk: {
        if (cfg.IS_CAUSAL) {
            const is_modulo_mn = n_extra_tokens.eq(0).bitAnd(seqlen_q.rem(BLOCK_M_I32).eq(0));
            const not_is_modulo_mn = b.xori(is_modulo_mn, b.liftAs(1, .i1));
            break :blk b.liftAs(@divTrunc(cfg.BLOCK_M, cfg.BLOCK_N), .i32).add(not_is_modulo_mn.to(.i32));
        }
        break :blk n_extra_tokens.ne(0).to(.i32);
    } else blk: {
        const n_extra_tokens_static: i64 = if (cfg.SEQLEN_K < cfg.BLOCK_N)
            cfg.BLOCK_N - cfg.SEQLEN_K
        else if (@mod(cfg.SEQLEN_K, cfg.BLOCK_N) != 0)
            @mod(cfg.SEQLEN_K, cfg.BLOCK_N)
        else
            0;
        const padded_block_k = n_extra_tokens_static != 0;
        const is_modulo_mn = !padded_block_k and (@mod(cfg.SEQLEN_Q, cfg.BLOCK_M) == 0);
        break :blk if (cfg.IS_CAUSAL)
            b.liftAs(@divTrunc(cfg.BLOCK_M, cfg.BLOCK_N) + @intFromBool(!is_modulo_mn), .i32)
        else
            b.liftAs(@intFromBool(padded_block_k), .i32);
    };

    var masked_blocks: Value = masked_blocks_base;
    masked_blocks = masked_blocks.minimum(visible_blocks);
    const n_full_blocks = visible_blocks.sub(masked_blocks);

    const all_block_max = n_blocks.mul(BLOCK_N_I32);

    var full_if = b.openIfElse(n_full_blocks.gt(0), .{
        b.tensorTy(&.{cfg.BLOCK_M}, .f32),
        b.tensorTy(&.{cfg.BLOCK_M}, .f32),
        b.tensorTy(&.{ cfg.BLOCK_M, cfg.BLOCK_DMODEL_POW2 }, .f32),
        b.scalarTy(.i32),
        b.scalarTy(.i32),
    });
    {
        const full_block_max = block_min.add(n_full_blocks.mul(BLOCK_N_I32));
        const full = attnFwdInner(
            b,
            acc,
            l_i,
            m_i,
            q,
            k_ptrs_loop,
            v_ptrs_loop,
            stride_kn,
            stride_vn,
            start_m,
            seqlen_k,
            seqlen_q,
            block_min,
            full_block_max,
            b.liftAs(0, .i32),
            n_extra_tokens,
            alibi_slope,
            descale_q,
            descale_k,
            descale_v,
            sm_scale,
            offs_m,
            offs_n,
            cfg,
            false,
            false,
        );
        full_if.yieldThen(.{ full[2], full[1], full[0], full_block_max, all_block_max });
    }
    {
        full_if.yieldElse(.{ m_i, l_i, acc, block_min, all_block_max });
    }

    var masked_if = b.openIfElse(masked_blocks.gt(0), .{
        b.tensorTy(&.{cfg.BLOCK_M}, .f32),
        b.tensorTy(&.{cfg.BLOCK_M}, .f32),
        b.tensorTy(&.{ cfg.BLOCK_M, cfg.BLOCK_DMODEL_POW2 }, .f32),
    });
    {
        const offs_n_causal = if (cfg.IS_CAUSAL)
            offs_n.add(seqlen_q.sub(seqlen_k))
        else
            b.liftAs(0, .i32);
        const skipped = n_full_blocks.mul(BLOCK_N_I32);
        const k_ptrs_masked = if (cfg.SLIDING_WINDOW > 0)
            k_ptrs_loop.addPtr(skipped.to(.i64).mul(stride_kn))
        else
            k_ptr.addPtr(k_offs.add(skipped.to(.i64).mul(stride_kn)));
        const v_ptrs_masked = if (cfg.SLIDING_WINDOW > 0)
            v_ptrs_loop.addPtr(skipped.to(.i64).mul(stride_vn))
        else
            v_ptr.addPtr(v_offs.add(skipped.to(.i64).mul(stride_vn)));
        const masked = if (cfg.IS_CAUSAL)
            attnFwdInner(
                b,
                full_if.results[2],
                full_if.results[1],
                full_if.results[0],
                q,
                k_ptrs_masked,
                v_ptrs_masked,
                stride_kn,
                stride_vn,
                start_m,
                seqlen_k,
                seqlen_q,
                full_if.results[3],
                full_if.results[4],
                offs_n_causal,
                n_extra_tokens,
                alibi_slope,
                descale_q,
                descale_k,
                descale_v,
                sm_scale,
                offs_m,
                offs_n,
                cfg,
                true,
                true,
            )
        else
            attnFwdInner(
                b,
                full_if.results[2],
                full_if.results[1],
                full_if.results[0],
                q,
                k_ptrs_masked,
                v_ptrs_masked,
                stride_kn,
                stride_vn,
                start_m,
                seqlen_k,
                seqlen_q,
                full_if.results[3],
                full_if.results[4],
                b.liftAs(0, .i32),
                n_extra_tokens,
                alibi_slope,
                descale_q,
                descale_k,
                descale_v,
                sm_scale,
                offs_m,
                offs_n,
                cfg,
                false,
                true,
            );
        masked_if.yieldThen(.{ masked[2], masked[1], masked[0] });
    }
    {
        masked_if.yieldElse(.{ full_if.results[0], full_if.results[1], full_if.results[2] });
    }

    const m_final = masked_if.results[0];
    const l_final = masked_if.results[1];
    var acc_final = masked_if.results[2];

    const l_recip = b.full(&.{ cfg.BLOCK_M, 1 }, 1.0, .f32).div(l_final.expandDims(1));
    acc_final = acc_final.mul(l_recip);

    const end_m_idx = start_m.add(1).mul(BLOCK_M_I32);
    const start_m_idx = start_m.mul(BLOCK_M_I32);
    const causal_start_idx = if (cfg.VARLEN)
        seqlen_q.sub(seqlen_k)
    else
        b.liftAs(@as(i32, @intCast(cfg.SEQLEN_Q - cfg.SEQLEN_K)), .i32);
    if (cfg.IS_CAUSAL) {
        const cleanup_cond = causal_start_idx.gt(start_m_idx).bitAnd(causal_start_idx.lt(end_m_idx));
        var cleanup_if = b.openIfElse(cleanup_cond, .{
            b.tensorTy(&.{ cfg.BLOCK_M, cfg.BLOCK_DMODEL_POW2 }, .f32),
        });
        {
            const out_mask_boundary = b.splat(causal_start_idx, &.{cfg.BLOCK_DMODEL_POW2});
            const mask_m_offsets = start_m_idx.add(b.arange(0, cfg.BLOCK_M, .i32));
            const out_ptrs_mask = mask_m_offsets.expandDims(1).ge(out_mask_boundary.expandDims(0));
            cleanup_if.yieldThen(.{
                b.where(
                    out_ptrs_mask,
                    acc_final,
                    b.zeros(&.{ cfg.BLOCK_M, cfg.BLOCK_DMODEL_POW2 }, .f32),
                ),
            });
        }
        {
            cleanup_if.yieldElse(.{acc_final});
        }
        acc_final = cleanup_if.results[0];
    }

    const overflow_size = end_m_idx.sub(seqlen_q);

    var softmax_lse = m_final.add(b.log2(l_final)).mul(LN2);
    if (cfg.IS_CAUSAL) {
        const lse_causal_mask = start_m_idx.add(b.arange(0, cfg.BLOCK_M, .i32)).lt(causal_start_idx);
        softmax_lse = b.where(lse_causal_mask, b.zeros(&.{cfg.BLOCK_M}, .f32), softmax_lse);
    }

    var offs_lse_base = off_z.mul(stride_lse_z)
        .add(off_q_head.mul(stride_lse_h));
    offs_lse_base = addVarlenScalarOffset(offs_lse_base, cu_seqlens_q_start, stride_lse_m, cfg);
    const offs_lse = offs_lse_base.add(offs_m.mul(stride_lse_m));
    var lse_if = b.openIfElse(overflow_size.gt(0), .{});
    {
        const boundary = b.splat(BLOCK_M_I32, &.{cfg.BLOCK_M}).sub(overflow_size);
        const lse_mask = b.arange(0, cfg.BLOCK_M, .i32).lt(boundary);
        b.storeOpts(softmax_lse_ptr.addPtr(offs_lse), softmax_lse, .{
            .mask = lse_mask,
        });
        lse_if.yieldThen(.{});
    }
    {
        b.store(softmax_lse_ptr.addPtr(offs_lse), softmax_lse);
        lse_if.yieldElse(.{});
    }

    var offs_out_base = off_z.mul(stride_oz)
        .add(off_q_head.mul(stride_oh));
    offs_out_base = addVarlenScalarOffset(offs_out_base, cu_seqlens_q_start, stride_om, cfg);
    const offs_out = offs_out_base
        .add(offs_m.expandDims(1).mul(stride_om))
        .add(offs_d.expandDims(0).mul(stride_on));
    const out_mask_init = b.full(&.{ cfg.BLOCK_M, 1 }, 1, .i1);
    var out_mask_if = b.openIfElse(overflow_size.gt(0), .{
        b.tensorTy(&.{ cfg.BLOCK_M, 1 }, .i1),
    });
    {
        out_mask_if.yieldThen(.{out_mask_init.bitAnd(offs_m.expandDims(1).lt(seqlen_q))});
    }
    {
        out_mask_if.yieldElse(.{out_mask_init});
    }
    var out_mask = out_mask_if.results[0];
    if (cfg.BLOCK_DMODEL != cfg.BLOCK_DMODEL_POW2) {
        out_mask = out_mask.bitAnd(offs_d.expandDims(0).lt(BLOCK_DMODEL_I32));
    }
    const op = acc_final.to(cfg.out_dtype);
    b.storeOpts(out_ptr.addPtr(offs_out), op, .{ .mask = out_mask });

}
