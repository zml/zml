//! Gated DeltaNet — packed-varlen prefill kernel.
//!
//! Zig DSL transcription of `kernels/mosaic_tpu/py/gdn_prefill_varlen.py`.
//!
//! Grid: `(H_v, total_chunks)` with `(parallel, arbitrary)` semantics.
//! Per-program: process one chunk of size `BT` for one v-head.
//!
//! Three SMEM-prefetched scalar arrays drive boundary state handling:
//!   `seq_id_smem`    [total_chunks] i32  — which sequence each chunk belongs to
//!   `is_first_smem`  [total_chunks] i32  — 1 if the chunk is the first of its sequence
//!   `is_last_smem`   [total_chunks] i32  — 1 if the chunk is the last of its sequence
//!
//! State `S` lives in a VMEM scratch ref persistent across the "arbitrary"
//! axis iterations. At the first chunk of each sequence we initialize from
//! `s_in_ref[seq_id]`; at the last chunk we write to `s_out_ref[seq_id]`.

const std = @import("std");

const mlir = @import("mlir");
const zml = @import("../../zml.zig");
const mtt = @import("kernels/mosaic_tpu/builder");

// =============================================================================
// Config
// =============================================================================

/// Maximum number of sequences a sweep config can encode. Cfg.seqlens is a
/// fixed-size array; trailing entries beyond `num_seqs` are ignored.
pub const MAX_SEQS: usize = 8;

pub const Cfg = struct {
    num_k_heads: i64,
    num_v_heads: i64,
    head_k_dim: i64 = 128,
    head_v_dim: i64 = 128,
    chunk_size: i64 = 64,
    /// Number of valid entries in `seqlens`. Python's harness side reads
    /// `cfg["seqlens"][:cfg["num_seqs"]]` to slice off the trailing zeros.
    num_seqs: i64,
    seqlens: [MAX_SEQS]i64,
    dtype: mtt.DType = .bf16,
    state_dtype: mtt.DType = .f32,
    scale: f32,
    l2_eps: f32 = 1e-6,

    pub fn gvaFactor(self: Cfg) i64 {
        return @divExact(self.num_v_heads, self.num_k_heads);
    }
    pub fn totalChunks(self: Cfg) i64 {
        var total: i64 = 0;
        var i: usize = 0;
        const n: usize = @intCast(self.num_seqs);
        while (i < n) : (i += 1) {
            const t = self.seqlens[i];
            total += @divTrunc(t + self.chunk_size - 1, self.chunk_size);
        }
        return total;
    }
    pub fn paddedTokens(self: Cfg) i64 {
        return self.totalChunks() * self.chunk_size;
    }
};

// =============================================================================
// Triangular solve helper (unrolled at comptime — produces 2*ceil(log2(BT))-1 matmuls).
// =============================================================================

fn nilpotentDoublingRounds(BT: i64) usize {
    if (BT < 1) @panic("BT must be >= 1");
    if (BT == 1) return 0;
    // ceil(log2(BT)) for BT >= 2.
    var n: i64 = BT - 1;
    var r: usize = 0;
    while (n > 0) : (n >>= 1) r += 1;
    return r;
}

/// `(I + A)^-1` for strictly-lower-triangular A (shape <BT x BT> f32).
/// `dnums` is the standard 2D matmul dim_numbers attribute that Pallas attaches
/// to every `tpu.matmul`. We accept it from the caller so the matmul calls
/// here match Python's IR exactly.
fn invEyePlusA(
    b: *mtt.Builder,
    A: mtt.Value,
    BT: i64,
    dnums: *const mlir.Attribute,
) mtt.Value {
    const rounds = nilpotentDoublingRounds(BT);

    // A_bar = -A.
    const zero_bt = b.zeros(&.{ BT, BT }, .f32);
    const Abar = b.subf(zero_bt, A);

    // eye_BT — `jnp.eye(BT)` lowers via Pallas to two full-rank tpu.iotas
    // (NOT the single-iota+shape_cast pattern used for arange-based masks).
    const i_idx = b.iota(&.{ BT, BT }, .i32, &.{0});
    const j_idx = b.iota(&.{ BT, BT }, .i32, &.{1});
    const eye_bt = b.arithUitofp(b.cmpi(.eq, i_idx, j_idx), .f32);

    // T = eye + A_bar
    var T = b.addf(eye_bt, Abar);
    if (rounds <= 1) return T;

    // P = A_bar @ A_bar
    var P = b.matmulOpts(Abar, Abar, b.zeros(&.{ BT, BT }, .f32), .{ .dimension_numbers = dnums });

    // for stage in 1 .. rounds:
    //   T = T + T @ P
    //   if stage + 1 < rounds: P = P @ P
    var stage: usize = 1;
    while (stage < rounds) : (stage += 1) {
        const TP = b.matmulOpts(T, P, b.zeros(&.{ BT, BT }, .f32), .{ .dimension_numbers = dnums });
        T = b.addf(T, TP);
        if (stage + 1 < rounds) {
            P = b.matmulOpts(P, P, b.zeros(&.{ BT, BT }, .f32), .{ .dimension_numbers = dnums });
        }
    }
    return T;
}

// =============================================================================
// Kernel body
// =============================================================================

pub fn run(b: *mtt.Builder, cfg: Cfg) mtt.FinishError!void {
    const BT = cfg.chunk_size;
    const K = cfg.head_k_dim;
    const V = cfg.head_v_dim;
    const total_chunks = cfg.totalChunks();
    const gva = cfg.gvaFactor();

    const a = try b.declareArgsOpts(.{
        // Grid program-ids (h_v on axis 0, c on axis 1).
        .h_v = .{ .scalar = .i32 },
        .c = .{ .scalar = .i32 },

        // Scalar-prefetched i32 SMEM arrays — order must match scalar_prefetch=3.
        .seq_id = .{ .ref = .{
            .shape = &.{total_chunks},
            .dtype = .i32,
            .memory_space = .smem,
            .role = .scalar_prefetch,
        } },
        .is_first = .{ .ref = .{
            .shape = &.{total_chunks},
            .dtype = .i32,
            .memory_space = .smem,
            .role = .scalar_prefetch,
        } },
        .is_last = .{ .ref = .{
            .shape = &.{total_chunks},
            .dtype = .i32,
            .memory_space = .smem,
            .role = .scalar_prefetch,
        } },

        // Per-program inputs. q/k/v are passed pre-cast to fp32 by the Python
        // wrapper (`q_p.astype(jnp.float32)` before pl.pallas_call), so the
        // kernel sees them as f32 even though the layer-level dtype is bf16.
        // The output `o` stays at the caller's bf16 (cast back via truncf).
        .q = .{ .ref = .{
            .shape = &.{ 1, BT, K },
            .dtype = .f32,
            .window = mtt.ArgSpec.WindowSpec{ .transform_returns = &[_]mtt.ArgSpec.TransformReturn{
                .{ .program_id_div = mtt.ArgSpec.ProgramIdDiv{ .pid = 0, .divisor = gva } }, // h_v // gva_factor
                .{ .program_id = 1 },                                   // c
                .zero,
            } },
        } },
        .k = .{ .ref = .{
            .shape = &.{ 1, BT, K },
            .dtype = .f32,
            .window = mtt.ArgSpec.WindowSpec{ .transform_returns = &[_]mtt.ArgSpec.TransformReturn{
                .{ .program_id_div = mtt.ArgSpec.ProgramIdDiv{ .pid = 0, .divisor = gva } },
                .{ .program_id = 1 },
                .zero,
            } },
        } },
        .v = .{ .ref = .{
            .shape = &.{ 1, BT, V },
            .dtype = .f32,
            .window = mtt.ArgSpec.WindowSpec{ .transform_returns = &[_]mtt.ArgSpec.TransformReturn{
                .{ .program_id = 0 },
                .{ .program_id = 1 },
                .zero,
            } },
        } },
        .g_cum = .{ .ref = .{
            .shape = &.{ 1, BT, 1 },
            .dtype = .f32,
            .window = mtt.ArgSpec.WindowSpec{ .transform_returns = &[_]mtt.ArgSpec.TransformReturn{
                .{ .program_id = 0 },
                .{ .program_id = 1 },
                .zero,
            } },
        } },
        .beta = .{ .ref = .{
            .shape = &.{ 1, BT, 1 },
            .dtype = .f32,
            .window = mtt.ArgSpec.WindowSpec{ .transform_returns = &[_]mtt.ArgSpec.TransformReturn{
                .{ .program_id = 0 },
                .{ .program_id = 1 },
                .zero,
            } },
        } },
        // Per-seq state — index by seq_id_smem[c].
        .s_in = .{ .ref = .{
            .shape = &.{ 1, 1, K, V },
            .dtype = cfg.state_dtype,
            .window = mtt.ArgSpec.WindowSpec{ .transform_returns = &[_]mtt.ArgSpec.TransformReturn{
                .{ .scalar_load_at_pid = mtt.ArgSpec.ScalarLoadAtPid{ .prefetch_idx = 0, .at_pid = 1 } }, // seq_id_smem[c]
                .{ .program_id = 0 },                                            // h_v
                .zero,
                .zero,
            } },
        } },

        // Outputs.
        .o = .{ .ref = .{
            .shape = &.{ 1, BT, V },
            .dtype = cfg.dtype,
            .role = .output,
            .window = mtt.ArgSpec.WindowSpec{ .transform_returns = &[_]mtt.ArgSpec.TransformReturn{
                .{ .program_id = 0 },
                .{ .program_id = 1 },
                .zero,
            } },
        } },
        .s_out = .{ .ref = .{
            .shape = &.{ 1, 1, K, V },
            .dtype = cfg.state_dtype,
            .role = .output,
            .window = mtt.ArgSpec.WindowSpec{ .transform_returns = &[_]mtt.ArgSpec.TransformReturn{
                .{ .scalar_load_at_pid = mtt.ArgSpec.ScalarLoadAtPid{ .prefetch_idx = 0, .at_pid = 1 } },
                .{ .program_id = 0 },
                .zero,
                .zero,
            } },
        } },

        // VMEM scratch persistent across the chunk axis.
        .s_scratch = .{ .ref = .{
            .shape = &.{ K, V },
            .dtype = .f32,
            .role = .scratch,
        } },
    }, &.{}, .{
        .dimension_semantics = &.{ .parallel, .arbitrary },
        .iteration_bounds = &.{ cfg.num_v_heads, total_chunks },
        .scalar_prefetch = 3,
        .scratch_operands = 1,
        .pallas_window_params = true,
    });

    const k = b;

    // --- @pl.when(is_first_smem[c] == 1): scratch = s_in_ref[0, 0] -----------
    {
        const c_idx = k.toIndex(a.c);
        const is_first_at_c = k.scalarLoad(a.is_first, &.{c_idx});
        const cond = k.cmpi(.eq, is_first_at_c, k.lift(@as(i32, 1)));
        var i = k.openIf(cond);
        // s_in is rank-4 [1, 1, K, V]; squeeze to [K, V] via shape_cast then store.
        const s_in_loaded = k.refLoad(a.s_in).to(.f32);                     // <1x1xKxVxf32>
        const s_in_2d = k.shapeCast(s_in_loaded, &.{ K, V });
        k.refStore(a.s_scratch, s_in_2d);
        i.yieldThen(.{});
    }

    // --- Per-chunk math (mirrors Python's _varlen_fused_kernel body). --------
    // q/k/v are pre-cast to f32 outside the kernel; load and squeeze to <BTxD>.
    var q = k.shapeCast(k.refLoad(a.q), &.{ BT, K });
    var kk = k.shapeCast(k.refLoad(a.k), &.{ BT, K });

    // L2 norm q, k along K axis.
    const acc_zero_bt = k.zeros(&.{BT}, .f32);
    const eps_2d = k.full(&.{ BT, 1 }, cfg.l2_eps, .f32);

    const q_sq = k.mulf(q, q);
    const q_sum_1d = k.multiReduction(.add, q_sq, acc_zero_bt, &.{1});
    const q_sum_2d = k.shapeCast(q_sum_1d, &.{ BT, 1 });
    const q_inv_2d = k.rsqrt(k.addf(q_sum_2d, eps_2d));
    q = k.mulf(q, k.broadcastTo(q_inv_2d, &.{ BT, K }));

    const k_sq = k.mulf(kk, kk);
    const k_sum_1d = k.multiReduction(.add, k_sq, acc_zero_bt, &.{1});
    const k_sum_2d = k.shapeCast(k_sum_1d, &.{ BT, 1 });
    const k_inv_2d = k.rsqrt(k.addf(k_sum_2d, eps_2d));
    kk = k.mulf(kk, k.broadcastTo(k_inv_2d, &.{ BT, K }));

    // q *= scale.
    q = k.mulf(q, k.full(&.{ BT, K }, cfg.scale, .f32));

    // v already f32; squeeze to <BTxV>.
    const v = k.shapeCast(k.refLoad(a.v), &.{ BT, V });
    // g_cum, beta loaded as <1xBTx1xf32>; reuse them via direct shape_cast
    // (matches Pallas's emit pattern — squeezing to 1D would add a hop).
    const g_cum_3d = k.refLoad(a.g_cum);                                    // <1xBTx1xf32>
    const g_cum = k.shapeCast(g_cum_3d, &.{BT});                            // <BTxf32> (used post-eq)
    const beta_3d = k.refLoad(a.beta);
    const beta = k.shapeCast(beta_3d, &.{BT});

    // Load state from scratch.
    const S = k.refLoad(a.s_scratch);                                       // <KxVxf32>

    // decay_diff[i,j] = min(g_cum[i] - g_cum[j], 0)
    // Pallas reshapes g_cum (1x64x1) to 64x1 and 1x64 directly — re-using the
    // original 3D load — instead of going through the squeezed 1D form.
    const g_cum_col_2d = k.shapeCast(g_cum_3d, &.{ BT, 1 });                // <BTx1>
    const g_cum_row_2d = k.shapeCast(g_cum_3d, &.{ 1, BT });                // <1xBT>
    const g_cum_col = k.broadcastTo(g_cum_col_2d, &.{ BT, BT });
    const g_cum_row = k.broadcastTo(g_cum_row_2d, &.{ BT, BT });
    const diff = k.subf(g_cum_col, g_cum_row);
    const zero_bt2 = k.zeros(&.{ BT, BT }, .f32);
    const decay_diff = k.minimumf(diff, zero_bt2);
    const decay = k.exp(decay_diff);

    // strict_lower / diag_or_below masks (BT x BT).
    // Pallas emits one `tpu.iota dim=1` (vector<1xBT>), then shape_casts to
    // <BTx1>, then broadcasts both to <BTxBT>. Single iota for both axes.
    const j_iota_1d = k.iota(&.{ 1, BT }, .i32, &.{1});                     // <1xBT>
    const i_iota_2d = k.shapeCast(j_iota_1d, &.{ BT, 1 });                  // <BTx1>
    const i_iota = k.broadcastTo(i_iota_2d, &.{ BT, BT });
    const j_iota = k.broadcastTo(j_iota_1d, &.{ BT, BT });
    // mask → f32 via uitofp from i1 (Pallas pattern; cleaner than select-with-splats).
    const strict_lower = k.arithUitofp(k.cmpi(.sgt, i_iota, j_iota), .f32);
    const diag_or_below = k.arithUitofp(k.cmpi(.sge, i_iota, j_iota), .f32);

    // Standard 2D matmul dim_numbers — Pallas always attaches them.
    const dnums_2d = k.dotDimensionNumbers(
        &.{1}, &.{0}, &.{0}, &.{1}, &.{ 0, 0, 1, 1 }, &.{}, &.{},
    );

    // kkt = k @ k.T via tpu.transpose + standard matmul (Pallas pattern).
    const kk_T = k.transpose(kk, &.{ 1, 0 });
    const kkt = k.matmulOpts(kk, kk_T, k.zeros(&.{ BT, BT }, .f32), .{ .dimension_numbers = dnums_2d });
    const a_step1 = k.mulf(kkt, strict_lower);
    const a_step2 = k.mulf(a_step1, decay);
    const beta_col = k.broadcastTo(k.shapeCast(beta, &.{ BT, 1 }), &.{ BT, BT });
    const A_strict = k.mulf(a_step2, beta_col);
    const T_inv = invEyePlusA(k, A_strict, BT, dnums_2d);

    // decay_g = exp(g_cum)
    const decay_g = k.exp(g_cum);

    // beta_v = beta[:, None] * v  (BT x V)
    const beta_col_v = k.broadcastTo(k.shapeCast(beta, &.{ BT, 1 }), &.{ BT, V });
    const beta_v = k.mulf(beta_col_v, v);

    // beta_k_decay = (beta * decay_g)[:, None] * k  (BT x K)
    const beta_decay_g = k.mulf(beta, decay_g);
    const beta_decay_g_col = k.broadcastTo(k.shapeCast(beta_decay_g, &.{ BT, 1 }), &.{ BT, K });
    const beta_k_decay = k.mulf(beta_decay_g_col, kk);

    // u = T_inv @ beta_v ; w = T_inv @ beta_k_decay
    const u = k.matmulOpts(T_inv, beta_v, k.zeros(&.{ BT, V }, .f32), .{ .dimension_numbers = dnums_2d });
    const w = k.matmulOpts(T_inv, beta_k_decay, k.zeros(&.{ BT, K }, .f32), .{ .dimension_numbers = dnums_2d });

    // wS = w @ S  (BT x V) ; v_new = u - wS
    const wS = k.matmulOpts(w, S, k.zeros(&.{ BT, V }, .f32), .{ .dimension_numbers = dnums_2d });
    const v_new = k.subf(u, wS);

    // qkT = q @ k.T (reuses kk_T from the kkt matmul above — Pallas dedups it).
    const qkT = k.matmulOpts(q, kk_T, k.zeros(&.{ BT, BT }, .f32), .{ .dimension_numbers = dnums_2d });
    const attn = k.mulf(k.mulf(qkT, diag_or_below), decay);

    // o_intra = attn @ v_new
    const o_intra = k.matmulOpts(attn, v_new, k.zeros(&.{ BT, V }, .f32), .{ .dimension_numbers = dnums_2d });
    // o_inter = (q * decay_g[:, None]) @ S
    const decay_g_col = k.broadcastTo(k.shapeCast(decay_g, &.{ BT, 1 }), &.{ BT, K });
    const q_scaled = k.mulf(q, decay_g_col);
    const o_inter = k.matmulOpts(q_scaled, S, k.zeros(&.{ BT, V }, .f32), .{ .dimension_numbers = dnums_2d });
    const o = k.addf(o_intra, o_inter);

    // g_last = g_cum[BT-1] — extract last element via vector.extract.
    // For now: use a strided slice on the BT axis at index BT-1.
    // Pallas emits `vector.extract %g_cum[BT-1] : vector<BTxf32>` which is a scalar f32.
    const g_last = k.vectorExtract(g_cum, &.{BT - 1});
    // decay_to_end = exp(g_last - g_cum) ; k_decayed = k * decay_to_end[:, None]
    const g_last_bc = k.broadcastTo(g_last, &.{BT});
    const decay_to_end = k.exp(k.subf(g_last_bc, g_cum));
    const decay_to_end_col = k.broadcastTo(k.shapeCast(decay_to_end, &.{ BT, 1 }), &.{ BT, K });
    const k_decayed = k.mulf(kk, decay_to_end_col);

    // S_new = exp(g_last) * S + k_decayed.T @ v_new
    const exp_g_last = k.exp(g_last);
    const exp_g_last_kv = k.broadcastTo(exp_g_last, &.{ K, V });
    const term1 = k.mulf(exp_g_last_kv, S);
    const k_decayed_T = k.transpose(k_decayed, &.{ 1, 0 });                 // <K x BT>
    const term2 = k.matmulOpts(k_decayed_T, v_new, k.zeros(&.{ K, V }, .f32), .{ .dimension_numbers = dnums_2d });
    const S_new = k.addf(term1, term2);

    k.refStore(a.s_scratch, S_new);
    // o_ref[0] = o.astype(o_ref.dtype)
    const o_3d = k.shapeCast(o.to(cfg.dtype), &.{ 1, BT, V });
    k.refStore(a.o, o_3d);

    // --- @pl.when(is_last_smem[c] == 1): s_out_ref[0, 0] = scratch -----------
    {
        const c_idx = k.toIndex(a.c);
        const is_last_at_c = k.scalarLoad(a.is_last, &.{c_idx});
        const cond = k.cmpi(.eq, is_last_at_c, k.lift(@as(i32, 1)));
        var i = k.openIf(cond);
        const final_state = k.refLoad(a.s_scratch);
        const final_4d = k.shapeCast(final_state.to(cfg.state_dtype), &.{ 1, 1, K, V });
        k.refStore(a.s_out, final_4d);
        i.yieldThen(.{});
    }
}

// =============================================================================
// Kernel factory + sweeps
// =============================================================================

pub const Kernel = zml.kernel.mosaic_tpu.Kernel(Cfg, .{
    .name = "gdn_prefill_varlen",
    // Program-id args (h_v, c) are part of the func.func signature inside the
    // kernel module but are NOT operands of `tpu_custom_call` — Mosaic synthesizes
    // them from `iteration_bounds`. So we omit them here. (Pallas does the same.)
    .inputs = &.{ "seq_id", "is_first", "is_last", "q", "k", "v", "g_cum", "beta", "s_in" },
    .outputs = &.{ "o", "s_out" },
    .run = run,
});

