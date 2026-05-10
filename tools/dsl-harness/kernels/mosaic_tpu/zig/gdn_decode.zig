//! Gated DeltaNet — decode kernel (single-token recurrent step).
//!
//! Zig DSL transcription of `kernels/mosaic_tpu/py/gdn_decode.py::_decode_kernel`.
//! L2 norm of q,k is INSIDE the kernel (matches Triton's
//! `USE_QK_L2NORM_IN_KERNEL=true`). State stored fp32 by default; output bf16.
//!
//! Grid: `(N / M_pp,)` programs, each processing M_pp slots of (b, h_v).
//! With in-kernel GVA (when H_v > H_qk), q/k arrive with M_pp/gva_factor rows
//! and we replicate inside the kernel via `tpu.repeat`.

const std = @import("std");

const mlir = @import("mlir");
const zml = @import("zml");
const mtt = @import("kernels/mosaic_tpu/builder");
const harness = @import("harness");

// =============================================================================
// Config
// =============================================================================

/// Field names match the Python harness's `build_args(cfg)` keys —
/// `tools/dsl-harness/kernels/mosaic_tpu/py/gdn_decode.py`. The harness
/// JSON-stringifies the Zig Cfg and passes it to the Python runner as
/// the `cfg` dict.
pub const Cfg = struct {
    /// Batch size.
    batch: i64 = 1,
    /// Number of q/k heads (per batch).
    num_k_heads: i64,
    /// Number of v heads (per batch). `gva_factor = num_v_heads / num_k_heads`.
    num_v_heads: i64,
    /// Head-K dim (= NUM_LANES = 128 typically).
    head_k_dim: i64 = 128,
    /// Head-V dim.
    head_v_dim: i64 = 128,
    /// Slot count per Pallas program. (8, 128) tile rule:
    /// `M_pp / gva_factor >= 8`.
    M_pp: i64,
    /// q/k/v/out element dtype.
    dtype: mtt.DType = .bf16,
    /// State dtype. Default fp32 (matches Triton `h_dtype: .f32`).
    state_dtype: mtt.DType = .f32,
    /// `1 / sqrt(head_k_dim)`, applied post-L2-norm.
    scale: f32,
    /// L2 epsilon.
    l2_eps: f32 = 1e-6,
    /// `g` clip range, mirrors Python's `_DECODE_CLIP`.
    g_clip: f32 = 20.0,

    pub fn n(self: Cfg) i64 {
        return self.batch * self.num_v_heads;
    }
    pub fn nQk(self: Cfg) i64 {
        return self.batch * self.num_k_heads;
    }
    pub fn gvaFactor(self: Cfg) i64 {
        return @divExact(self.num_v_heads, self.num_k_heads);
    }
    pub fn mPpQk(self: Cfg) i64 {
        return @divExact(self.M_pp, self.gvaFactor());
    }
    pub fn nProgs(self: Cfg) i64 {
        return @divExact(self.n(), self.M_pp);
    }
};

// =============================================================================
// Kernel body
// =============================================================================

pub fn run(b: *mtt.Builder, cfg: Cfg) mtt.FinishError!void {
    const m_pp_qk = cfg.mPpQk();
    const gva_factor = cfg.gvaFactor();

    // Per the DSL convention (matches Pallas's `func.func` arg types), the
    // `.shape` on each `.ref` is the BLOCK shape, not the full HBM array shape.
    // The window's `transform_returns` selects which grid program-id each block
    // dim maps to. Since block_shape == ref.shape here, no explicit
    // `WindowSpec.block_shape` is needed.

    const a = try b.declareArgsOpts(.{
        // Grid program-id (block index along the N/M_pp axis).
        .pid = .{ .scalar = .i32 },

        // q/k blocks are smaller (M_pp_qk rows) — in-kernel GVA replicates
        // them inside. Block at row offset = pid * M_pp_qk.
        .q = .{ .ref = .{
            .shape = &.{ m_pp_qk, cfg.head_k_dim },
            .dtype = cfg.dtype,
            .window = mtt.ArgSpec.WindowSpec{
                .transform_returns = &.{ .{ .program_id = 0 }, .zero },
            },
        } },
        .k = .{ .ref = .{
            .shape = &.{ m_pp_qk, cfg.head_k_dim },
            .dtype = cfg.dtype,
            .window = mtt.ArgSpec.WindowSpec{
                .transform_returns = &.{ .{ .program_id = 0 }, .zero },
            },
        } },
        .v = .{ .ref = .{
            .shape = &.{ cfg.M_pp, cfg.head_v_dim },
            .dtype = cfg.dtype,
            .window = mtt.ArgSpec.WindowSpec{
                .transform_returns = &.{ .{ .program_id = 0 }, .zero },
            },
        } },
        .g = .{ .ref = .{
            .shape = &.{ cfg.M_pp, 1 },
            .dtype = .f32,
            .window = mtt.ArgSpec.WindowSpec{
                .transform_returns = &.{ .{ .program_id = 0 }, .zero },
            },
        } },
        .beta = .{ .ref = .{
            .shape = &.{ cfg.M_pp, 1 },
            .dtype = .f32,
            .window = mtt.ArgSpec.WindowSpec{
                .transform_returns = &.{ .{ .program_id = 0 }, .zero },
            },
        } },
        .s_in = .{ .ref = .{
            .shape = &.{ cfg.M_pp, cfg.head_k_dim, cfg.head_v_dim },
            .dtype = cfg.state_dtype,
            .window = mtt.ArgSpec.WindowSpec{
                .transform_returns = &.{ .{ .program_id = 0 }, .zero, .zero },
            },
        } },

        // Outputs.
        .o = .{ .ref = .{
            .shape = &.{ cfg.M_pp, cfg.head_v_dim },
            .dtype = cfg.dtype,
            .role = .output,
            .window = mtt.ArgSpec.WindowSpec{
                .transform_returns = &.{ .{ .program_id = 0 }, .zero },
            },
        } },
        .s_out = .{ .ref = .{
            .shape = &.{ cfg.M_pp, cfg.head_k_dim, cfg.head_v_dim },
            .dtype = cfg.state_dtype,
            .role = .output,
            .window = mtt.ArgSpec.WindowSpec{
                .transform_returns = &.{ .{ .program_id = 0 }, .zero, .zero },
            },
        } },
    }, &.{}, .{
        .dimension_semantics = &.{.arbitrary},   // matches Pallas's default for an unannotated grid axis
        .iteration_bounds = &.{cfg.nProgs()},
        .pallas_window_params = true,
    });

    const k = b;

    // --- Load q,k as fp32 vectors of shape [M_pp_qk, head_k]. -----------------
    var q = k.refLoad(a.q).to(.f32);
    var kk = k.refLoad(a.k).to(.f32);

    // --- L2 norm along the K axis (per row) — q completes before k starts ---
    // Pattern matches Pallas's lowering of `q * rsqrt(sum(q*q, axis=-1) + eps)`:
    //   sum (8x128 → 8) → shape_cast (8 → 8x1) → addf eps (8x1) → rsqrt (8x1)
    //   → broadcast (8x1 → 8x128) → mul.
    const acc_zero_qk = k.zeros(&.{m_pp_qk}, .f32);
    const eps_2d = k.full(&.{ m_pp_qk, 1 }, cfg.l2_eps, .f32);

    const q_sq = k.mulf(q, q);
    const q_sum_1d = k.multiReduction(.add, q_sq, acc_zero_qk, &.{1});         // [M_pp_qk]
    const q_sum_2d = k.shapeCast(q_sum_1d, &.{ m_pp_qk, 1 });
    const q_inv_2d = k.rsqrt(k.addf(q_sum_2d, eps_2d));
    const q_inv_bc = k.broadcastTo(q_inv_2d, &.{ m_pp_qk, cfg.head_k_dim });
    q = k.mulf(q, q_inv_bc);

    const k_sq = k.mulf(kk, kk);
    const k_sum_1d = k.multiReduction(.add, k_sq, acc_zero_qk, &.{1});
    const k_sum_2d = k.shapeCast(k_sum_1d, &.{ m_pp_qk, 1 });
    const k_inv_2d = k.rsqrt(k.addf(k_sum_2d, eps_2d));
    const k_inv_bc = k.broadcastTo(k_inv_2d, &.{ m_pp_qk, cfg.head_k_dim });
    kk = k.mulf(kk, k_inv_bc);

    // Apply softmax-style scale to q post-L2 (single splat constant K3).
    q = k.mulf(q, k.full(&.{ m_pp_qk, cfg.head_k_dim }, cfg.scale, .f32));

    // --- In-kernel GVA: replicate q,k along axis 0 by gva_factor. ------------
    // Pallas lowers `jnp.repeat(x, gva, axis=0)` as the broadcast triple:
    //   shape_cast (M_pp_qk, K) → (M_pp_qk, 1, K)
    //   broadcast → (M_pp_qk, gva_factor, K)
    //   shape_cast → (M_pp_qk * gva_factor = M_pp, K).
    // We emit the same triple. NOTE: do NOT additionally call `b.repeat`
    // — that would double-expand to <2*M_pp, K>.
    if (gva_factor > 1) {
        const q_3d = k.shapeCast(q, &.{ m_pp_qk, 1, cfg.head_k_dim });
        const q_bc = k.broadcastTo(q_3d, &.{ m_pp_qk, gva_factor, cfg.head_k_dim });
        q = k.shapeCast(q_bc, &.{ cfg.M_pp, cfg.head_k_dim });
        const k_3d_repeat = k.shapeCast(kk, &.{ m_pp_qk, 1, cfg.head_k_dim });
        const k_bc = k.broadcastTo(k_3d_repeat, &.{ m_pp_qk, gva_factor, cfg.head_k_dim });
        kk = k.shapeCast(k_bc, &.{ cfg.M_pp, cfg.head_k_dim });
    }
    // From here on q,kk have shape [M_pp, head_k_dim].

    // --- Load v, g, beta, s_in (matches Python's source-line-by-line order). -
    const v = k.refLoad(a.v).to(.f32);                                       // [M_pp, head_v]
    const g_2d_raw = k.refLoad(a.g);                                         // [M_pp, 1]   fp32
    // Python's `g_ref[:, 0]` drops the unit dim — shape_cast it before clip.
    const g_1d = k.shapeCast(g_2d_raw, &.{cfg.M_pp});                        // [M_pp]
    const beta_2d = k.refLoad(a.beta);                                       // [M_pp, 1]
    const s = k.refLoad(a.s_in).to(.f32);                                    // [M_pp, head_k, head_v]

    // --- g = clip(g, -clip, clip); alpha = exp(g) ----------------------------
    // jnp.clip lowers to arith.maximumf / arith.minimumf (NaN-propagating);
    // `b.maximum`/`b.minimum` default to maxnumf/minnumf — use the explicit ops.
    const clip_hi = k.full(&.{cfg.M_pp}, cfg.g_clip, .f32);
    const clip_lo = k.full(&.{cfg.M_pp}, -cfg.g_clip, .f32);
    const g_lo_clipped = k.maximumf(g_1d, clip_lo);
    const g_clipped = k.minimumf(g_lo_clipped, clip_hi);
    const alpha_1d = k.exp(g_clipped);                                       // [M_pp]

    // --- kS = matmul(k[:, None, :], s)[:, 0, :], qS likewise. ---------------
    // Per-slot batched matmul. Spelled to interleave shape_cast/matmul/cast
    // back per side, mirroring Pallas's emit order.
    const acc_zero_mhv = k.zeros(&.{ cfg.M_pp, 1, cfg.head_v_dim }, .f32);
    // output_dim_order pairs are (side, dim_idx); 0=lhs, 1=rhs.
    const dnums_batched = k.dotDimensionNumbers(
        &.{2}, &.{1}, &.{1}, &.{2},
        &.{ 0, 0, 0, 1, 1, 2 }, &.{0}, &.{0},
    );
    const k_3d = k.shapeCast(kk, &.{ cfg.M_pp, 1, cfg.head_k_dim });
    const kS_3d = k.matmulOpts(k_3d, s, acc_zero_mhv, .{ .dimension_numbers = dnums_batched });
    const kS = k.shapeCast(kS_3d, &.{ cfg.M_pp, cfg.head_v_dim });
    const q_3d = k.shapeCast(q, &.{ cfg.M_pp, 1, cfg.head_k_dim });
    const qS_3d = k.matmulOpts(q_3d, s, acc_zero_mhv, .{ .dimension_numbers = dnums_batched });
    const qS = k.shapeCast(qS_3d, &.{ cfg.M_pp, cfg.head_v_dim });

    // --- alpha (1D → 2D → broadcast), kv = alpha * kS, y_alpha = alpha * qS -
    const alpha_2d = k.shapeCast(alpha_1d, &.{ cfg.M_pp, 1 });
    const alpha_bc = k.broadcastTo(alpha_2d, &.{ cfg.M_pp, cfg.head_v_dim });
    const kv_term = k.mulf(alpha_bc, kS);
    const y_alpha = k.mulf(alpha_bc, qS);

    // --- delta = beta * (v - kv);  kq = sum(k*q, axis=-1) ------------------
    const v_minus_kv = k.subf(v, kv_term);
    const beta_bc = k.broadcastTo(beta_2d, &.{ cfg.M_pp, cfg.head_v_dim });
    const delta = k.mulf(beta_bc, v_minus_kv);

    const kq_acc_zero = k.zeros(&.{cfg.M_pp}, .f32);
    const kq_1d = k.multiReduction(.add, k.mulf(kk, q), kq_acc_zero, &.{1}); // [M_pp]
    const kq_2d = k.shapeCast(kq_1d, &.{ cfg.M_pp, 1 });
    const kq_bc = k.broadcastTo(kq_2d, &.{ cfg.M_pp, cfg.head_v_dim });
    const out = k.addf(y_alpha, k.mulf(delta, kq_bc));

    // --- s_new = alpha[:, None, None] * s + k[:, :, None] * delta[:, None, :] -
    // Python emits: shape_cast alpha 8 → 8x1x1, broadcast → 8x128x128, then mul.
    const alpha_3d_pre = k.shapeCast(alpha_1d, &.{ cfg.M_pp, 1, 1 });
    const alpha_3d = k.broadcastTo(alpha_3d_pre, &.{ cfg.M_pp, cfg.head_k_dim, cfg.head_v_dim });
    const alpha_s = k.mulf(alpha_3d, s);

    const k_outer_pre = k.shapeCast(kk, &.{ cfg.M_pp, cfg.head_k_dim, 1 });
    const delta_outer_pre = k.shapeCast(delta, &.{ cfg.M_pp, 1, cfg.head_v_dim });
    const k_outer = k.broadcastTo(k_outer_pre, &.{ cfg.M_pp, cfg.head_k_dim, cfg.head_v_dim });
    const delta_outer = k.broadcastTo(delta_outer_pre, &.{ cfg.M_pp, cfg.head_k_dim, cfg.head_v_dim });
    const k_delta = k.mulf(k_outer, delta_outer);

    const s_new = k.addf(alpha_s, k_delta);

    // --- Stores (cast back to caller dtypes). -------------------------------
    k.refStore(a.o, out.to(cfg.dtype));
    k.refStore(a.s_out, s_new.to(cfg.state_dtype));
}

// =============================================================================
// Kernel factory + sweeps
// =============================================================================

pub const Kernel = zml.kernel.mosaic_tpu.Kernel(Cfg, .{
    .name = "gdn_decode",
    .inputs = &.{ "pid", "q", "k", "v", "g", "beta", "s_in" },
    .outputs = &.{ "o", "s_out" },
    .run = run,
});

const baseline = Cfg{
    // Qwen3.5-2B decode: B=1, H_qk=16, H_v=32, K=V=128, M_pp=16.
    .batch = 1,
    .num_k_heads = 16,
    .num_v_heads = 32,
    .head_k_dim = 128,
    .head_v_dim = 128,
    .M_pp = 16,
    .dtype = .bf16,
    .state_dtype = .f32,
    .scale = 0.0883883476, // 1/sqrt(128)
};

fn override(comptime patches: anytype) Cfg {
    var c = baseline;
    inline for (std.meta.fields(@TypeOf(patches))) |f| {
        @field(c, f.name) = @field(patches, f.name);
    }
    return c;
}

pub const SWEEPS: []const harness.Sweep(Cfg) = &.{
    .{ .label = "qwen35", .cfg = baseline },
    .{ .label = "small_no_gva", .cfg = override(.{ .num_k_heads = 8, .num_v_heads = 8, .M_pp = 8 }) },
    .{ .label = "qwen35_bf16_state", .cfg = override(.{ .state_dtype = .bf16 }) },
};
