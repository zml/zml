//! Faithful Mosaic-TPU port of `gdn.py`'s Pallas kernel.
//!
//! Same constexpr config as the Triton port at
//! `examples/triton_emitter/kernels_zig/gdn.zig`:
//!
//!   USE_G=true, USE_GK=false, USE_GV=false
//!   USE_QK_L2NORM_IN_KERNEL=true
//!   IS_BETA_HEADWISE=true
//!   USE_INITIAL_STATE=true
//!   STORE_FINAL_STATE=true
//!   USE_EXP2=false, TRANSPOSE_STATE=false, IS_VARLEN=true
//!
//! Layout matches `gdn.py` (and the Pallas IR it lowers to) op-for-op:
//!
//!   q, k         : [1, T_total, H,  K]   bf16   # token-major flat tensors
//!   v, o         : [1, T_total, HV, V]   bf16
//!   g            : [1, T_total, HV]      f32
//!   beta         : [1, T_total, HV]      bf16
//!   h0, ht       : [N, HV, K, V]         f32    # per-sequence recurrent state
//!   cu_seqlens   : [N+1]                 i32    # SMEM, scalar-prefetch style
//!
//! Grid is `(V // BV, N * HV)`; one program per value-block × value-head ×
//! sequence. The recurrent `b_h: vec<BKxBVxf32>` is the only loop-carried
//! value across the per-token scan.
//!
//! Source ordering and constant hoisting mirror what
//! `jax/_src/pallas/mosaic/lowering.py` emits: per-use re-emission of
//! `i_v * BV`, `bos + t`, and `index_cast` chains (JAX does not CSE in
//! `pallas_call` lowering); a single `<128x1xi1>` shape_cast for `mask_k`
//! and a leading-dim `vector.broadcast` for `mask_v` (the canonical TPU
//! broadcast pair).
//!
//! Run:
//!     bazel run //examples/mosaic_gdn:gated_delta_net

const std = @import("std");

const mlir = @import("mlir");
const mtt = @import("kernels/mosaic_tpu/builder");

// Demo shapes — match `gdn.py` defaults.
const NUM_SEQUENCES: i64 = 2; // N
const SEQ_LEN: i64 = 8; // per-sequence length
const T_TOTAL: i64 = NUM_SEQUENCES * SEQ_LEN;
const NUM_QK_HEADS: i64 = 2; // H
const NUM_V_HEADS: i64 = 4; // HV
const HV_PER_H: i64 = @divTrunc(NUM_V_HEADS, NUM_QK_HEADS);
const KEY_DIM: i64 = 128; // K
const VALUE_DIM: i64 = 128; // V
const BK: i64 = KEY_DIM;
const BV: i64 = VALUE_DIM;
const SCALE: f32 = 0.0883883461; // 1/sqrt(128)

fn buildKernelIr(allocator: std.mem.Allocator, ctx: *mlir.Context) ![:0]const u8 {
    var spec = try mtt.Builder.buildOpts(allocator, ctx, "fused_recurrent_gated_delta_rule_fwd_kernel_ptr", .{
        // Grid program-ids: (i_v, i_nh) — i32 scalars.
        .i_v = .{ .scalar = .i32 },
        .i_nh = .{ .scalar = .i32 },

        // Inputs.
        .q = .{ .ref = .{ .shape = &.{ 1, T_TOTAL, NUM_QK_HEADS, KEY_DIM }, .dtype = .bf16 } },
        .k = .{ .ref = .{ .shape = &.{ 1, T_TOTAL, NUM_QK_HEADS, KEY_DIM }, .dtype = .bf16 } },
        .v = .{ .ref = .{ .shape = &.{ 1, T_TOTAL, NUM_V_HEADS, VALUE_DIM }, .dtype = .bf16 } },
        .g = .{ .ref = .{ .shape = &.{ 1, T_TOTAL, NUM_V_HEADS }, .dtype = .f32 } },
        .beta = .{ .ref = .{ .shape = &.{ 1, T_TOTAL, NUM_V_HEADS }, .dtype = .bf16 } },
        .h0 = .{ .ref = .{ .shape = &.{ NUM_SEQUENCES, NUM_V_HEADS, KEY_DIM, VALUE_DIM }, .dtype = .f32 } },
        // SMEM resident — Pallas's scalar-prefetch path. cu_seqlens reads
        // are scalar `memref.load`.
        .cu_seqlens = .{ .ref = .{ .shape = &.{NUM_SEQUENCES + 1}, .dtype = .i32, .memory_space = .smem } },

        // Outputs.
        .o = .{ .ref = .{ .shape = &.{ 1, T_TOTAL, NUM_V_HEADS, VALUE_DIM }, .dtype = .bf16 } },
        .ht = .{ .ref = .{ .shape = &.{ NUM_SEQUENCES, NUM_V_HEADS, KEY_DIM, VALUE_DIM }, .dtype = .f32 } },
    }, &.{}, .{
        .dimension_semantics = &.{ .arbitrary, .arbitrary },
        // Pallas's `pallas_call` adds these for grid metadata. The
        // grid is `(VALUE_DIM/BV, NUM_SEQUENCES*NUM_V_HEADS) = (1, 8)`.
        .iteration_bounds = &.{ VALUE_DIM / BV, NUM_SEQUENCES * NUM_V_HEADS },
        .pallas_window_params = true,
    });
    defer spec.deinit();

    const k = &spec.kernel;
    const a = spec.args;

    // -------------------------------------------------------------------
    // Hoisted constants — order mirrors what Pallas emits at the top of
    // the function body (see the `%cst` / `%c{N}_i32` block).
    // -------------------------------------------------------------------
    const cst_scale_v = k.splat(@as(f32, SCALE), &.{BK}, .f32); // dense<SCALE> : <BKxf32>
    const cst_zero_kv = k.zeros(&.{ BK, BV }, .f32); // dense<0.0> : <BKxBVxf32>
    const cst_dim_v = k.splat(@as(i32, @intCast(KEY_DIM)), &.{BK}, .i32); // dense<128> : <BKxi32>
    const cst_zero_v = k.zeros(&.{BV}, .f32); // dense<0.0> : <BVxf32>
    const cst_eps = k.lift(@as(f32, 1e-6));
    const cst_zero_1 = k.zeros(&.{1}, .f32); // dense<0.0> : <1xf32>  (L2-norm acc)
    const c0_i32 = k.lift(@as(i32, 0));
    const c0 = k.cIndex(0);
    const c128_i32 = k.lift(@as(i32, @intCast(BV)));
    const c1_i32 = k.lift(@as(i32, 1));
    const c2_i32 = k.lift(@as(i32, @intCast(HV_PER_H)));
    const c4_i32 = k.lift(@as(i32, @intCast(NUM_V_HEADS)));

    // ---- decode (i_n, i_hv, i_h) from i_nh -----------------------------
    const i_n = k.divsi(a.i_nh, c4_i32);
    const i_hv = k.remsi(a.i_nh, c4_i32);
    const i_h = k.divsi(i_hv, c2_i32);

    // ---- bos = cu_seqlens[i_n], eos = cu_seqlens[i_n+1], T = eos-bos ---
    const i_n_idx_for_bos = k.toIndex(i_n);
    const bos = k.scalarLoad(a.cu_seqlens, &.{i_n_idx_for_bos});
    const i_n_p1 = k.addi(i_n, c1_i32);
    const i_n_p1_idx = k.toIndex(i_n_p1);
    const eos = k.scalarLoad(a.cu_seqlens, &.{i_n_p1_idx});
    const T = k.subi(eos, bos);

    // ---- iota / masks (Pallas-style 1-D iota) --------------------------
    const o_k = k.arange(BK, .i32); // <BK x i32>
    const iv_off = k.muli(a.i_v, c128_i32); // i_v * BV
    const iota_for_v = k.arange(BV, .i32); // <BV x i32>  (separate iota emit)
    const iv_off_v = k.broadcastTo(iv_off, &.{BV});
    const o_v = k.addi(iv_off_v, iota_for_v);
    const mask_k = k.cmpi(.slt, o_k, cst_dim_v);
    const mask_v = k.cmpi(.slt, o_v, cst_dim_v);
    const mask_k_2d = k.shapeCast(mask_k, &.{ BK, 1 });
    const mask_k_bc = k.broadcastTo(mask_k_2d, &.{ BK, BV });
    const mask_v_bc = k.broadcastTo(mask_v, &.{ BK, BV }); // leading-dim broadcast
    const mask_h = k.andi(mask_k_bc, mask_v_bc);

    // ---- USE_INITIAL_STATE: b_h = where(mask_h, h0[...], 0) -----------
    // Pallas re-emits `i_v * BV` here even though it computed it above.
    const iv_off_for_h0 = k.muli(a.i_v, c128_i32);
    const i_n_idx_for_h0 = k.toIndex(i_n);
    const i_hv_idx_for_h0 = k.toIndex(i_hv);
    const iv_off_idx_for_h0 = k.toIndex(iv_off_for_h0);
    const h0_tile_4d = k.vectorLoadShape(
        a.h0,
        &.{ i_n_idx_for_h0, i_hv_idx_for_h0, c0, iv_off_idx_for_h0 },
        &.{ 1, 1, BK, BV },
    );
    const h0_tile = k.shapeCast(h0_tile_4d, &.{ BK, BV });
    const b_h_init = k.select(mask_h, h0_tile, cst_zero_kv);

    // ---- scf.for t in 0..T iter_args(b_h) ------------------------------
    // i32 loop counter to match Pallas's `scf.for ... : i32`.
    var loop = k.openFor(c0_i32, T, c1_i32, .{b_h_init});
    {
        const t = loop.iv;
        var b_h = loop.carried[0];

        // --- masked load b_q = q[0, bos+t, i_h, :] -------------------
        const tpb_q = k.addi(bos, t);
        const tpb_q_idx = k.toIndex(tpb_q);
        const i_h_idx_q = k.toIndex(i_h);
        const q_4d = k.vectorLoadShape(a.q, &.{ c0, tpb_q_idx, i_h_idx_q, c0 }, &.{ 1, 1, 1, BK });
        const q_1d_bf16 = k.shapeCast(q_4d, &.{BK});
        const q_1d_f32 = q_1d_bf16.to(.f32);
        const b_q_loaded = k.select(mask_k, q_1d_f32, cst_zero_v);

        // --- masked load b_k = k[0, bos+t, i_h, :] -------------------
        const tpb_k = k.addi(bos, t);
        const tpb_k_idx = k.toIndex(tpb_k);
        const i_h_idx_k = k.toIndex(i_h);
        const k_4d = k.vectorLoadShape(a.k, &.{ c0, tpb_k_idx, i_h_idx_k, c0 }, &.{ 1, 1, 1, BK });
        const k_1d_bf16 = k.shapeCast(k_4d, &.{BK});
        const k_1d_f32 = k_1d_bf16.to(.f32);
        const b_k_loaded = k.select(mask_k, k_1d_f32, cst_zero_v);

        // --- masked load b_v = v[0, bos+t, i_hv, o_v] ----------------
        const tpb_v = k.addi(bos, t);
        const iv_off_v_load = k.muli(a.i_v, c128_i32);
        const tpb_v_idx = k.toIndex(tpb_v);
        const i_hv_idx_v = k.toIndex(i_hv);
        const iv_off_idx_v = k.toIndex(iv_off_v_load);
        const v_4d = k.vectorLoadShape(a.v, &.{ c0, tpb_v_idx, i_hv_idx_v, iv_off_idx_v }, &.{ 1, 1, 1, BV });
        const v_1d_bf16 = k.shapeCast(v_4d, &.{BV});
        const v_1d_f32 = v_1d_bf16.to(.f32);
        const b_v = k.select(mask_v, v_1d_f32, cst_zero_v);

        // --- L2-norm b_q (Pallas idiom: shape_cast<1xN> + multi_reduction[1] + extract[0]) ---
        const bq_sq = k.mulf(b_q_loaded, b_q_loaded);
        const bq_norm_sum = k.reduceToScalar(.add, bq_sq, cst_zero_1);
        const bq_norm_eps = k.addf(bq_norm_sum, cst_eps);
        const bq_norm = k.sqrt(bq_norm_eps);
        const bq_norm_v = k.broadcastTo(bq_norm, &.{BK});
        const b_q_normalized = k.divf(b_q_loaded, bq_norm_v);

        // --- L2-norm b_k -------------------------------------------
        const bk_sq = k.mulf(b_k_loaded, b_k_loaded);
        const bk_norm_sum = k.reduceToScalar(.add, bk_sq, cst_zero_1);
        const bk_norm_eps = k.addf(bk_norm_sum, cst_eps);
        const bk_norm = k.sqrt(bk_norm_eps);
        const bk_norm_v = k.broadcastTo(bk_norm, &.{BK});
        const b_k_normalized = k.divf(b_k_loaded, bk_norm_v);

        // b_q *= scale  (vec * vec, no broadcast needed since cst_scale_v is <BKxf32>)
        const b_q = k.mulf(b_q_normalized, cst_scale_v);

        // --- beta load -> extract -> extf ---------------------------
        const tpb_beta = k.addi(bos, t);
        const tpb_beta_idx = k.toIndex(tpb_beta);
        const i_hv_idx_beta = k.toIndex(i_hv);
        const beta_3d = k.vectorLoadShape(a.beta, &.{ c0, tpb_beta_idx, i_hv_idx_beta }, &.{ 1, 1, 1 });
        const beta_scalar = k.vectorExtract(beta_3d, &.{ 0, 0, 0 });
        const b_beta = beta_scalar.to(.f32);

        // --- g load -> extract -> exp -------------------------------
        const tpb_g = k.addi(bos, t);
        const tpb_g_idx = k.toIndex(tpb_g);
        const i_hv_idx_g = k.toIndex(i_hv);
        const g_3d = k.vectorLoadShape(a.g, &.{ c0, tpb_g_idx, i_hv_idx_g }, &.{ 1, 1, 1 });
        const b_g = k.vectorExtract(g_3d, &.{ 0, 0, 0 });
        const exp_g = k.exp(b_g);

        // --- b_h *= exp(b_g) ----------------------------------------
        const exp_g_kv = k.broadcastTo(exp_g, &.{ BK, BV });
        const b_h_decayed = k.mulf(b_h, exp_g_kv);

        // --- predicted_v = sum(b_h * b_k[:, None], axis=0) -----------
        const bk_for_pred_2d = k.shapeCast(b_k_normalized, &.{ BK, 1 });
        const bk_for_pred_bc = k.broadcastTo(bk_for_pred_2d, &.{ BK, BV });
        const hk = k.mulf(b_h_decayed, bk_for_pred_bc);
        const predicted_v = k.multiReduction(.add, hk, cst_zero_v, &.{0});

        // --- b_v_new = b_beta * (b_v - predicted_v) ------------------
        // Pallas emits: subf, then broadcast(b_beta), then mulf.
        const v_diff = k.subf(b_v, predicted_v);
        const b_beta_v = k.broadcastTo(b_beta, &.{BV});
        const b_v_new = k.mulf(b_beta_v, v_diff);

        // --- b_h += b_k[:, None] * b_v_new[None, :] ------------------
        // Pallas re-emits the b_k shape_cast/broadcast pair here.
        const bk_for_outer_2d = k.shapeCast(b_k_normalized, &.{ BK, 1 });
        const bk_for_outer_bc = k.broadcastTo(bk_for_outer_2d, &.{ BK, BV });
        const bv_for_outer_bc = k.broadcastTo(b_v_new, &.{ BK, BV }); // leading-dim broadcast
        const k_outer_v = k.mulf(bk_for_outer_bc, bv_for_outer_bc);
        const b_h_updated = k.addf(b_h_decayed, k_outer_v);

        // --- b_o = sum(b_h * b_q[:, None], axis=0) -------------------
        const bq_for_o_2d = k.shapeCast(b_q, &.{ BK, 1 });
        const bq_for_o_bc = k.broadcastTo(bq_for_o_2d, &.{ BK, BV });
        const hq = k.mulf(b_h_updated, bq_for_o_bc);
        const b_o_f32 = k.multiReduction(.add, hq, cst_zero_v, &.{0});
        const b_o_bf16 = b_o_f32.to(.bf16);

        // --- store o[0, bos+t, i_hv, o_v] = b_o ----------------------
        const tpb_o = k.addi(bos, t);
        const iv_off_o = k.muli(a.i_v, c128_i32);
        const tpb_o_idx = k.toIndex(tpb_o);
        const i_hv_idx_o = k.toIndex(i_hv);
        const iv_off_idx_o = k.toIndex(iv_off_o);
        const b_o_4d = k.shapeCast(b_o_bf16, &.{ 1, 1, 1, BV });
        k.vectorStoreAt(a.o, b_o_4d, &.{ c0, tpb_o_idx, i_hv_idx_o, iv_off_idx_o });

        b_h = b_h_updated;
        loop.yield(.{b_h});
    }
    const final_b_h = loop.results[0];

    // ---- STORE_FINAL_STATE: ht[i_n, i_hv, :, o_v] = b_h ----------------
    const iv_off_ht = k.muli(a.i_v, c128_i32);
    const i_n_idx_ht = k.toIndex(i_n);
    const i_hv_idx_ht = k.toIndex(i_hv);
    const iv_off_idx_ht = k.toIndex(iv_off_ht);
    const final_4d = k.shapeCast(final_b_h, &.{ 1, 1, BK, BV });
    k.vectorStoreAt(a.ht, final_4d, &.{ i_n_idx_ht, i_hv_idx_ht, c0, iv_off_idx_ht });

    return k.finish(&.{});
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (.{ "func", "arith", "scf", "math", "memref", "vector", "tpu" }) |d| {
        mlir.DialectHandle.fromString(d).insertDialect(registry);
    }

    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    const ir = try buildKernelIr(allocator, ctx);
    defer allocator.free(ir);

    var stdout_buf: [64 * 1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);
    defer stdout.interface.flush() catch {};
    try stdout.interface.print("{s}\n", .{ir});
}
