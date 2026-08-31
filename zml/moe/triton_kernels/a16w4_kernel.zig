const std = @import("std");

const zml = @import("../../zml.zig");
const tri = zml.kernel.triton;

pub const Cfg = struct {
    a_dtype: tri.DType = .bf16,
    wp_dtype: tri.DType = .i8,
    ws_dtype: tri.DType = .i8,
    c_dtype: tri.DType = .bf16,

    BLOCK_M: i32 = 16,
    BLOCK_N: i32 = 64,
    BLOCK_K: i32 = 64,
    SPLIT_K: i32 = 1,
    GROUP_M: i32 = 1,
    num_warps: i32 = 4,
    num_stages: i32 = 2,
};

pub const Kernel = tri.Kernel(Cfg, .{
    .name = "_gemm_a16mxfp4",
    .inputs = &.{
        "a_ptr",
        "wp_ptr",
        "ws_ptr",
        "tile_expert_ptr",
        "tile_mstart_ptr",
        "tile_mend_ptr",
        "NUM_M_TILES_ptr",
        "N_ptr",
        "K_ptr",
        "stride_am_ptr",
        "stride_ak_ptr",
        "stride_we_ptr",
        "stride_wk_ptr",
        "stride_wn_ptr",
        "stride_se_ptr",
        "stride_sk_ptr",
        "stride_sn_ptr",
        "stride_cm_ptr",
        "stride_cn_ptr",
    },
    .outputs = &.{"c"},
    .run = run,
});

fn run(b: *tri.Builder, cfg: Cfg) tri.FinishError!void {
    std.debug.assert(@mod(cfg.BLOCK_K, 32) == 0);

    const a = try b.declareArgs(.{
        .a_ptr = .{ .ptr = cfg.a_dtype },
        .wp_ptr = .{ .ptr = cfg.wp_dtype },
        .ws_ptr = .{ .ptr = cfg.ws_dtype },
        .tile_expert_ptr = .{ .ptr = .i32 },
        .tile_mstart_ptr = .{ .ptr = .i64 },
        .tile_mend_ptr = .{ .ptr = .i64 },
        .NUM_M_TILES_ptr = .{ .ptr = .i64 },
        .N_ptr = .{ .ptr = .i64 },
        .K_ptr = .{ .ptr = .i64 },
        .stride_am_ptr = .{ .ptr = .i64 },
        .stride_ak_ptr = .{ .ptr = .i64 },
        .stride_we_ptr = .{ .ptr = .i64 },
        .stride_wk_ptr = .{ .ptr = .i64 },
        .stride_wn_ptr = .{ .ptr = .i64 },
        .stride_se_ptr = .{ .ptr = .i64 },
        .stride_sk_ptr = .{ .ptr = .i64 },
        .stride_sn_ptr = .{ .ptr = .i64 },
        .stride_cm_ptr = .{ .ptr = .i64 },
        .stride_cn_ptr = .{ .ptr = .i64 },
        .c_ptr = .{ .ptr = cfg.c_dtype },
    });

    const NUM_M_TILES = b.load(a.NUM_M_TILES_ptr);
    const N_val = b.load(a.N_ptr);
    const K_val = b.load(a.K_ptr);
    const stride_am = b.load(a.stride_am_ptr);
    const stride_ak = b.load(a.stride_ak_ptr);
    const stride_we = b.load(a.stride_we_ptr);
    const stride_wk = b.load(a.stride_wk_ptr);
    const stride_wn = b.load(a.stride_wn_ptr);
    const stride_se = b.load(a.stride_se_ptr);
    const stride_sk = b.load(a.stride_sk_ptr);
    const stride_sn = b.load(a.stride_sn_ptr);
    const stride_cm = b.load(a.stride_cm_ptr);
    const stride_cn = b.load(a.stride_cn_ptr);

    // L2-friendly swizzle: group GROUP_M consecutive m-tiles (usually the same
    // expert) and sweep n within them, so each weight column tile is reused.
    const pid = b.programId(.x);
    const pid_k = b.programId(.y);
    const num_pid_n = N_val.cdiv(cfg.BLOCK_N);
    const width = b.liftAs(cfg.GROUP_M, .i64).mul(num_pid_n);
    const pid_i64 = pid.to(.i64);
    const group_id = pid_i64.div(width);
    const group_size = NUM_M_TILES.sub(group_id.mul(cfg.GROUP_M)).minimum(cfg.GROUP_M);
    const pid_m = group_id.mul(cfg.GROUP_M).add(pid_i64.rem(group_size));
    const pid_n = pid_i64.rem(width).div(group_size);

    const expert = b.load(a.tile_expert_ptr.addPtr(pid_m));
    const m_start = b.load(a.tile_mstart_ptr.addPtr(pid_m));
    const m_end = b.load(a.tile_mend_ptr.addPtr(pid_m));

    const offs_m = m_start.add(b.arange(0, cfg.BLOCK_M, .i32));
    const offs_n = pid_n.mul(cfg.BLOCK_N).add(b.arange(0, cfg.BLOCK_N, .i32));
    const m_mask = offs_m.lt(m_end);
    const n_mask = offs_n.lt(N_val);

    const BLOCK_K_PACKED: i32 = @divExact(cfg.BLOCK_K, 2);
    const BLOCK_K_SCALE: i32 = @divExact(cfg.BLOCK_K, 32);
    const offs_kh = b.arange(0, BLOCK_K_PACKED, .i32); // packed K rows within a tile
    const offs_kg = b.arange(0, BLOCK_K_SCALE, .i32); // scale groups within a tile

    const wp_base = a.wp_ptr.addPtr(expert.mul(stride_we));
    const ws_base = a.ws_ptr.addPtr(expert.mul(stride_se));

    const acc = b.zeros(&.{ cfg.BLOCK_M, cfg.BLOCK_N }, .f32);
    const num_k_tiles = K_val.div(cfg.BLOCK_K);
    const pid_k_i64 = pid_k.to(.i64);
    var loop = b.openFor(pid_k_i64, num_k_tiles, cfg.SPLIT_K, .{acc});
    {
        const t = loop.iv;
        const acc_loop = loop.carried[0];

        const k0 = t.mul(cfg.BLOCK_K);
        const offs_ak = k0.add(b.arange(0, cfg.BLOCK_K, .i32));
        const a_mask = m_mask.expandDims(1);
        const a_rows = offs_m.expandDims(1);
        const a_row_offsets = a_rows.mul(stride_am);
        const a_ptrs_col = a.a_ptr.addPtr(a_row_offsets);
        const a_cols = offs_ak.expandDims(0);
        const a_col_offsets = a_cols.mul(stride_ak);
        const a_ptrs_base = b.broadcastTo(a_ptrs_col, &.{ cfg.BLOCK_M, cfg.BLOCK_K });
        const a_col_offsets_full = b.broadcastTo(a_col_offsets, &.{ cfg.BLOCK_M, cfg.BLOCK_K });
        const a_ptrs = a_ptrs_base.addPtr(a_col_offsets_full);
        const a_mask_full = b.broadcastTo(a_mask, &.{ cfg.BLOCK_M, cfg.BLOCK_K });
        const a_tile = b.loadOpts(a_ptrs, .{
            .mask = a_mask_full,
            .other = b.zeros(&.{ cfg.BLOCK_M, cfg.BLOCK_K }, cfg.a_dtype),
        }); // (BM, BK) bf16

        const pj = k0.div(2).add(offs_kh);
        const w_mask = n_mask.expandDims(0);
        const w_rows = pj.expandDims(1);
        const w_row_offsets = w_rows.mul(stride_wk);
        const w_ptrs_col = wp_base.addPtr(w_row_offsets);
        const w_cols = offs_n.expandDims(0);
        const w_col_offsets = w_cols.mul(stride_wn);
        const w_ptrs_base = b.broadcastTo(w_ptrs_col, &.{ BLOCK_K_PACKED, cfg.BLOCK_N });
        const w_col_offsets_full = b.broadcastTo(w_col_offsets, &.{ BLOCK_K_PACKED, cfg.BLOCK_N });
        const w_ptrs = w_ptrs_base.addPtr(w_col_offsets_full);
        const w_mask_full = b.broadcastTo(w_mask, &.{ BLOCK_K_PACKED, cfg.BLOCK_N });
        const w = b.loadOpts(w_ptrs, .{
            .mask = w_mask_full,
            .other = b.zeros(&.{ BLOCK_K_PACKED, cfg.BLOCK_N }, cfg.wp_dtype),
        }); // (BK//2, BN) uint8

        const kg = k0.div(32).add(offs_kg);
        // rhs_scale must be laid out [N, K//32] for dot_scaled
        const sc_mask = n_mask.expandDims(1);
        const sc_rows = offs_n.expandDims(1);
        const sc_row_offsets = sc_rows.mul(stride_sn);
        const sc_ptrs_col = ws_base.addPtr(sc_row_offsets);
        const sc_cols = kg.expandDims(0);
        const sc_col_offsets = sc_cols.mul(stride_sk);
        const sc_ptrs_base = b.broadcastTo(sc_ptrs_col, &.{ cfg.BLOCK_N, BLOCK_K_SCALE });
        const sc_col_offsets_full = b.broadcastTo(sc_col_offsets, &.{ cfg.BLOCK_N, BLOCK_K_SCALE });
        const sc_ptrs = sc_ptrs_base.addPtr(sc_col_offsets_full);
        const sc_mask_full = b.broadcastTo(sc_mask, &.{ cfg.BLOCK_N, BLOCK_K_SCALE });
        const sc = b.loadOpts(sc_ptrs, .{
            .mask = sc_mask_full,
            .other = b.full(&.{ cfg.BLOCK_N, BLOCK_K_SCALE }, 127, cfg.ws_dtype),
        }); // (BN, BK//32) e8m0
        const new_acc = b.dotScaledOpts(a_tile, w, acc_loop, null, sc, .bf16, .e2m1, .{ .fast_math = true });

        loop.yield(.{new_acc});
    }

    const c_rows = offs_m.expandDims(1);
    const c_row_offsets = c_rows.mul(stride_cm);
    const c_ptrs_col = a.c_ptr.addPtr(c_row_offsets);
    const c_cols = offs_n.expandDims(0);
    const c_col_offsets = c_cols.mul(stride_cn);
    const c_ptrs_base = b.broadcastTo(c_ptrs_col, &.{ cfg.BLOCK_M, cfg.BLOCK_N });
    const c_col_offsets_full = b.broadcastTo(c_col_offsets, &.{ cfg.BLOCK_M, cfg.BLOCK_N });
    const c_ptrs = c_ptrs_base.addPtr(c_col_offsets_full);
    const c_mask = m_mask.expandDims(1).bitAnd(n_mask.expandDims(0));
    b.storeOpts(c_ptrs, loop.results[0].to(.bf16), .{ .mask = c_mask });
}
