//! Routed MoE W4A16 split-K GEMM kernel.

const std = @import("std");

const dialects = @import("mlir/dialects");

const zml = @import("../../zml.zig");
const tri = zml.kernel.triton;

pub const Cfg = struct {
    a_dtype: tri.DType = .bf16,
    w_dtype: tri.DType = .i8,
    scale_dtype: tri.DType = .i8,
    BLOCK_M: i32 = 16,
    BLOCK_N: i32 = 64,
    BLOCK_K: i32 = 64,
    SPLIT_K: i32 = 4,
};

pub const Kernel = tri.Kernel(Cfg, .{
    .name = "_w4a16_gemm_splitk_kernel_ptr",
    .inputs = &.{
        "A",           "B",          "Scales",
        "TileExperts", "TileStarts", "TileEnds",
        "N",           "K",          "stride_am",
        "stride_ak",   "stride_be",  "stride_bk",
        "stride_bn",   "stride_se",  "stride_sk",
        "stride_sn",   "stride_cm",  "stride_cn",
        "CInit",
    },
    .outputs = &.{"C"},
    .run = run,
});

fn run(b: *tri.Builder, cfg: Cfg) tri.FinishError!void {
    std.debug.assert(cfg.a_dtype == .bf16);
    std.debug.assert(@mod(cfg.BLOCK_K, 32) == 0);

    const a = try b.declareArgs(.{
        .A = .{ .ptr = cfg.a_dtype },
        .B = .{ .ptr = cfg.w_dtype },
        .Scales = .{ .ptr = cfg.scale_dtype },
        .TileExperts = .{ .ptr = .i32 },
        .TileStarts = .{ .ptr = .i64 },
        .TileEnds = .{ .ptr = .i64 },
        .N_ptr = .{ .ptr = .i64 },
        .K_ptr = .{ .ptr = .i64 },
        .stride_am_ptr = .{ .ptr = .i64 },
        .stride_ak_ptr = .{ .ptr = .i64 },
        .stride_be_ptr = .{ .ptr = .i64 },
        .stride_bk_ptr = .{ .ptr = .i64 },
        .stride_bn_ptr = .{ .ptr = .i64 },
        .stride_se_ptr = .{ .ptr = .i64 },
        .stride_sk_ptr = .{ .ptr = .i64 },
        .stride_sn_ptr = .{ .ptr = .i64 },
        .stride_cm_ptr = .{ .ptr = .i64 },
        .stride_cn_ptr = .{ .ptr = .i64 },
        .CInit = .{ .ptr = .f32 },
        .C = .{ .ptr = .f32 },
    });

    const N = b.load(a.N_ptr);
    const K = b.load(a.K_ptr);
    const stride_am = b.load(a.stride_am_ptr);
    const stride_ak = b.load(a.stride_ak_ptr);
    const stride_be = b.load(a.stride_be_ptr);
    const stride_bk = b.load(a.stride_bk_ptr);
    const stride_bn = b.load(a.stride_bn_ptr);
    const stride_se = b.load(a.stride_se_ptr);
    const stride_sk = b.load(a.stride_sk_ptr);
    const stride_sn = b.load(a.stride_sn_ptr);
    const stride_cm = b.load(a.stride_cm_ptr);
    const stride_cn = b.load(a.stride_cn_ptr);

    const BLOCK_M: i64 = cfg.BLOCK_M;
    const BLOCK_N: i64 = cfg.BLOCK_N;
    const BLOCK_K: i64 = cfg.BLOCK_K;
    const SPLIT_K: i64 = cfg.SPLIT_K;

    const pid_m = b.programId(.x);
    const pid_n = b.programId(.y);
    const pid_k = b.programId(.z);

    const expert = b.load(a.TileExperts.addPtr(pid_m));
    const m_start = b.load(a.TileStarts.addPtr(pid_m));
    const m_end = b.load(a.TileEnds.addPtr(pid_m));
    const offs_m = m_start.add(b.arange(0, cfg.BLOCK_M, .i32));
    const offs_n = pid_n.mul(cfg.BLOCK_N).add(b.arange(0, cfg.BLOCK_N, .i32));
    const offs_k = b.arange(0, cfg.BLOCK_K, .i32);
    const offs_k_packed = b.arange(0, @divExact(cfg.BLOCK_K, 2), .i32);
    const offs_k_scale = b.arange(0, @divExact(cfg.BLOCK_K, 32), .i32);

    const offs_m_col = b.expandDims(offs_m, 1).to(.i64);
    const a_ptrs_col = a.A.addPtr(offs_m_col.mul(stride_am));
    const pid_k_block_k = pid_k.mul(cfg.BLOCK_K);
    const a_k_offsets = pid_k_block_k.add(offs_k);
    const a_k_row = b.expandDims(a_k_offsets, 0).to(.i64);
    const a_k_term = a_k_row.mul(stride_ak);
    const a_ptrs_2d = b.broadcastTo(a_ptrs_col, &.{ BLOCK_M, BLOCK_K });
    const a_k_term_2d = b.broadcastTo(a_k_term, &.{ BLOCK_M, BLOCK_K });
    const a_ptrs_init = a_ptrs_2d.addPtr(a_k_term_2d);

    const pid_k_block_k_packed = pid_k.mul(@divExact(cfg.BLOCK_K, 2));
    const b_k_offsets = pid_k_block_k_packed.add(offs_k_packed);
    const b_k_col = b.expandDims(b_k_offsets, 1).to(.i64);
    const b_ptrs_col = a.B.addPtr(expert.mul(stride_be)).addPtr(b_k_col.mul(stride_bk));
    const offs_n_row = b.expandDims(offs_n, 0).to(.i64);
    const b_n_term = offs_n_row.mul(stride_bn);
    const b_ptrs_2d = b.broadcastTo(b_ptrs_col, &.{ @divExact(BLOCK_K, 2), BLOCK_N });
    const b_n_term_2d = b.broadcastTo(b_n_term, &.{ @divExact(BLOCK_K, 2), BLOCK_N });
    const b_ptrs_init = b_ptrs_2d.addPtr(b_n_term_2d);

    const offs_n_col = b.expandDims(offs_n, 1).to(.i64);
    const scale_ptrs_col = a.Scales.addPtr(expert.mul(stride_se)).addPtr(offs_n_col.mul(stride_sn));
    const pid_k_block_k_scale = pid_k.mul(@divExact(cfg.BLOCK_K, 32));
    const scale_k_offsets_init = pid_k_block_k_scale.add(offs_k_scale);
    const scale_k_row = b.expandDims(scale_k_offsets_init, 0).to(.i64);
    const scale_k_term = scale_k_row.mul(stride_sk);
    const scale_ptrs_2d = b.broadcastTo(scale_ptrs_col, &.{ BLOCK_N, @divExact(BLOCK_K, 32) });
    const scale_k_term_2d = b.broadcastTo(scale_k_term, &.{ BLOCK_N, @divExact(BLOCK_K, 32) });
    const scale_ptrs_init = scale_ptrs_2d.addPtr(scale_k_term_2d);

    const acc_init = b.zeros(&.{ BLOCK_M, BLOCK_N }, .f32);
    const loop_start = pid_k_block_k.to(.i64);
    const loop_step = SPLIT_K * BLOCK_K;

    var loop = b.openFor(loop_start, K, loop_step, .{
        a_ptrs_init, b_ptrs_init, scale_ptrs_init, acc_init,
    });
    {
        const k = loop.iv;
        const a_ptrs = loop.carried[0];
        const b_ptrs = loop.carried[1];
        const scale_ptrs = loop.carried[2];
        const acc = loop.carried[3];

        const k_offsets = k.add(offs_k);
        const packed_k_offsets = k.div(@as(i64, 2)).add(offs_k_packed);
        const scale_k_offsets = k.div(@as(i64, 32)).add(offs_k_scale);

        const a_m_mask = offs_m_col.lt(m_end);
        const a_k_mask = b.expandDims(k_offsets, 0).lt(K);
        const a_mask = a_m_mask.bitAnd(a_k_mask);
        const a_val = b.loadOpts(a_ptrs, .{
            .mask = a_mask,
            .other = b.zeros(&.{ BLOCK_M, BLOCK_K }, cfg.a_dtype),
        });

        const b_k_mask = b.expandDims(packed_k_offsets, 1).lt(K.div(@as(i64, 2)));
        const b_n_mask = offs_n_row.lt(N);
        const b_mask = b_k_mask.bitAnd(b_n_mask);
        const b_val = b.loadOpts(b_ptrs, .{
            .mask = b_mask,
            .other = b.zeros(&.{ @divExact(BLOCK_K, 2), BLOCK_N }, cfg.w_dtype),
        });

        const scale_n_mask = offs_n_col.lt(N);
        const scale_k_mask = b.expandDims(scale_k_offsets, 0).lt(K.div(@as(i64, 32)));
        const scale_mask = scale_n_mask.bitAnd(scale_k_mask);
        const scales = b.loadOpts(scale_ptrs, .{
            .mask = scale_mask,
            .other = b.full(&.{ BLOCK_N, @divExact(BLOCK_K, 32) }, 127, cfg.scale_dtype),
        });

        const new_acc = b.dotScaledOpts(
            a_val,
            b_val,
            acc,
            null,
            scales,
            .bf16,
            .e2m1,
            .{ .fast_math = true },
        );

        const new_a_ptrs = a_ptrs.addPtr(stride_ak.mul(loop_step));
        const new_b_ptrs = b_ptrs.addPtr(stride_bk.mul(SPLIT_K * @divExact(BLOCK_K, 2)));
        const new_scale_ptrs = scale_ptrs.addPtr(stride_sk.mul(SPLIT_K * @divExact(BLOCK_K, 32)));
        loop.yield(.{ new_a_ptrs, new_b_ptrs, new_scale_ptrs, new_acc });
    }
    const acc = loop.results[3];

    const c_ptrs_col = a.C.addPtr(offs_m_col.mul(stride_cm));
    const c_n_term = offs_n_row.mul(stride_cn);
    const c_ptrs_2d = b.broadcastTo(c_ptrs_col, &.{ BLOCK_M, BLOCK_N });
    const c_n_term_2d = b.broadcastTo(c_n_term, &.{ BLOCK_M, BLOCK_N });
    const c_ptrs = c_ptrs_2d.addPtr(c_n_term_2d);
    const c_mask = offs_m_col.lt(m_end).bitAnd(offs_n_row.lt(N));
    _ = b.atomicRmwOpts(dialects.ttir.RMWOp.fadd, c_ptrs, acc, .{ .mask = c_mask });
}

test "emit routed W4A16 split-K kernel" {
    const ttir = try Kernel.emit(std.testing.allocator, .{ .BLOCK_M = 4 });
    defer std.testing.allocator.free(ttir);
    try std.testing.expect(std.mem.indexOf(u8, ttir, "_w4a16_gemm_splitk_kernel_ptr") != null);
}
