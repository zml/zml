//! GDN fused decode Mosaic-TPU kernel ("P4") — Zig DSL conversion of
//! `fused_decoding_gdn` (tpu-inference). Paired with
//! `kernels/mosaic_tpu/py/gdn_decode_fast.py` (the Pallas reference the harness
//! diffs against). Goal: emit Mosaic IR that matches what `pl.pallas_call`
//! produces for that reference, op-for-op.
//!
//! STATUS: done — harness diff is +2/-2 against the Pallas reference (constant
//! hoist-order noise; below the `ragged_paged` baseline of +9..+30). The
//! `emit_pipeline` expansion lives in `zml/kernels/mosaic_tpu/pipeline.zig`
//! (reusable; also feeds P3/P5). See PORTING_PLAYBOOK.md Part C for the
//! IR-matching methodology, especially C2 (re-emit `memref_slice`/`memref_squeeze`
//! per use — `tileBuf` below) and C3 (op-emission order within a primitive).
//!
//! Specialization: the inner kernel is hardwired for the bt8_nv128 baseline
//! (use_qk_l2norm + use_gate_in_kernel + dt_bias-present + lower_bound=None
//! softplus branch + repeat_factor>1 + V==K==num_lanes ⇒ no β/dt_bias concat).
//! The corresponding `Cfg` fields are kept for documentation; honoring them
//! at runtime would mean adding `comptime` branches in `innerKernel`.
//!
//! Known DSL pothole: `b.divFloor` assumes vector operands (uses `lhs.shape()`),
//! so the scalar `cdiv(decode_end, bt)` is hand-emitted inline below.
//!
//! `bt` (tokens per `emit_pipeline` step) is comptime-dispatched per sweep:
//! the Python's `_decode_kernel_main` loops `for i_t in range(bt)` (a Python
//! loop ⇒ fully unrolled at trace time), so the Zig `inline for (0..bt)`s here
//! need a comptime `bt` too. `cfg.bt` selects which branch of `SUPPORTED_BT`
//! is instantiated, and the dynamic `b.name = f"decoding_gdn-bt_{bt}"` lets the
//! emitted `func.func` symbol match Pallas's `f"decoding_gdn-bt_{bt}"`. The
//! sweep configs choose shape parameters so `_get_default_block_sizes_decode`
//! (Python, VMEM-driven) auto-picks the same `bt` as `cfg.bt`.
//!
//! Reference for the kernel math / structure: GDN-Google/PORTING_PLAYBOOK.md,
//! GDN-Google/README.md §4.1 + §12 (TPU memory/pipeline deep-dive), and
//! GDN-Google/tpu_inference/kernels/gdn/fused_gdn_decode_kernel.py (verbatim source).

const std = @import("std");

const mlir = @import("mlir");
const zml = @import("../../zml.zig");
const mtt = @import("kernels/mosaic_tpu/builder"); // low layer: Builder, ArgSpec, DType, …

/// Comptime-dispatch arm values for `cfg.bt`. Adding a new arm only requires
/// extending this tuple (the comptime dispatch unrolls one branch per value).
/// `_get_default_block_sizes_decode` (Python) auto-picks `bt` from VMEM, so
/// each sweep's shape parameters need to make Python pick a value in this set.
const SUPPORTED_BT = .{ @as(i64, 4), @as(i64, 8), @as(i64, 16), @as(i64, 32) };

// =============================================================================
// Config
// =============================================================================

/// Field names align with the Python harness's `build_args(cfg)` keys
/// (`kernels/mosaic_tpu/py/gdn_decode_fast.py::_decode_shapes`). The harness
/// JSON-stringifies this Cfg and hands it to the Python runner as the `cfg`
/// dict, and also passes it to `Kernel.emit(...)` for the Zig side.
pub const Cfg = struct {
    /// q/k/v/b/initial_state-output element dtype (the "compute" dtype the
    /// activations come in as; the kernel upcasts to f32 on VMEM load).
    dtype: mtt.DType = .bf16,
    /// Recurrent-state cache dtype (bf16 by default on TPU; halves HBM).
    state_dtype: mtt.DType = .bf16,
    /// q/k heads.
    num_k_heads: i64 = 16,
    /// v heads (multiple of num_k_heads → GQA; the kernel jnp.repeats q/k).
    num_v_heads: i64 = 32,
    /// key/query head dim (multiple of num_lanes = 128).
    head_k_dim: i64 = 128,
    /// value head dim (multiple of 128).
    head_v_dim: i64 = 128,
    /// Decode tokens this step (= decode requests; each is 1 token).
    num_decode_tokens: i64 = 4,
    /// Recurrent-state pool size (≥ num_decode_tokens + 1; slot 0 = null).
    num_states: i64 = 5,
    /// Lane width (TPU is 128 everywhere; the Python kernel reads
    /// `pltpu.get_tpu_info().num_lanes` at lowering time, this just sizes the
    /// per-head broadcast tiles for b / A_log / dt_bias).
    num_lanes: i64 = 128,

    // Numeric knobs (mirror the verbatim kernel; carry them over to the body).
    /// `1 / sqrt(head_k_dim)`, applied to q post-L2-norm.
    scale: f32 = 0.0883883476,
    /// L2-norm epsilon (`+ 1e-6` inside the sqrt).
    l2_eps: f32 = 1e-6,
    /// L2-normalize q,k inside the kernel (Qwen3.5 GDN: true).
    use_qk_l2norm: bool = true,
    /// Apply the gate transform `g = -exp(A_log)·softplus(g + dt_bias)` inside
    /// the kernel (Qwen3.5 GDN: true).
    use_gate_in_kernel: bool = true,
    /// `dt_bias` is present (true ⇒ the kernel adds it before softplus; this is
    /// also a `_get_default_block_sizes_decode` input that affects `bt`).
    has_dt_bias: bool = true,
    /// If non-null, use the sigmoid gate `g = lower_bound·sigmoid(a_val·g)`
    /// instead of `-a_val·softplus(g)` (some Qwen3.5 configs). null ⇒ softplus.
    lower_bound: ?f32 = null,

    /// `bt`: tokens per `emit_pipeline` step. Must match what Python's
    /// `_get_default_block_sizes_decode` picks for these shapes (VMEM-driven —
    /// roughly `vmem / (per-bt working-set)`, rounded down to power of 2). Must
    /// also appear in `SUPPORTED_BT` (the comptime-dispatch arms). The Python
    /// names the func `f"decoding_gdn-bt_{bt}"`; we mirror that at run() start.
    bt: i64 = 8,
    /// `b` (the raw beta input) is present. When false, Pallas passes
    /// `b=None` ⇒ the b_in operand is dropped from the func.func signature
    /// and the kernel uses `b_v = v_diff` directly (no sigmoid).
    has_b: bool = true,

    // --- derived ---
    pub fn nKq(self: Cfg) i64 {
        return self.num_k_heads;
    }
    pub fn nV(self: Cfg) i64 {
        return self.num_v_heads;
    }
    pub fn dK(self: Cfg) i64 {
        return self.head_k_dim;
    }
    pub fn dV(self: Cfg) i64 {
        return self.head_v_dim;
    }
    pub fn repeatFactor(self: Cfg) i64 {
        return @divExact(self.num_v_heads, self.num_k_heads);
    }
};

// =============================================================================
// Kernel body
// =============================================================================

pub fn run(b: *mtt.Builder, cfg: Cfg) mtt.FinishError!void {
    // Override the static `Kernel.name` with the cfg.bt-aware Python symbol
    // (`f"decoding_gdn-bt_{bt}"`) so the emitted `func.func` matches Pallas
    // op-for-op. `b.name` is the only field `declareArgsOpts` reads when emitting
    // `func.func`, so this must happen before the `declareArgsOpts` call below.
    const name_buf = b.arena.allocator().alloc(u8, 32) catch return error.OutOfMemory;
    b.name = std.fmt.bufPrint(name_buf, "decoding_gdn-bt_{d}", .{cfg.bt}) catch return error.OutOfMemory;

    const T = cfg.num_decode_tokens;
    const H_qk = cfg.nKq();
    const H_v = cfg.nV();
    const K = cfg.dK();
    const V = cfg.dV();
    const nl = cfg.num_lanes;
    const bt = cfg.bt;
    const ns = cfg.num_states;

    // Per the DSL convention, a `.ref`'s `.shape` is the BLOCK shape; for these
    // operands the block shape == the full HBM-array shape (a trivial window /
    // a manual-DMA `.any` operand), so no explicit `WindowSpec.block_shape`.
    //
    // Arg order MUST match the Python `pl.pallas_call`'s — see the dispatch
    // below for the variants. Optional args (b, A_log, dt_bias) follow Pallas:
    // when the Python passes `b=None` / `A_log=None` / `dt_bias=None`, the
    // corresponding func.func arg is OMITTED entirely (Pallas drops the
    // operand from the lowered signature). To match op-for-op we replay the
    // same 6 valid combinations via 6 declareArgsOpts literals + a shared body.
    const has_a_log = cfg.use_gate_in_kernel;
    const has_dt = cfg.use_gate_in_kernel and cfg.has_dt_bias;

    // 6 valid arg-signature combos (has_b × {full, no_dt, no_gate}). Each
    // branch builds its own struct literal — Zig can't conditionally include
    // fields in one literal — and forwards to the shared `runWithArgs`.
    const opts: mtt.Builder.Opts = .{ .dimension_semantics = &.{.arbitrary}, .iteration_bounds = &.{std.math.minInt(i64)}, .scalar_prefetch = 0, .scratch_operands = 3, .pallas_window_params = true };

    if (cfg.has_b) {
        if (has_a_log and has_dt) {
            const a = try b.declareArgsOpts(.{
                .pid = .{ .scalar = .i32 },
                .q = .{ .ref = .{ .shape = &.{ T, H_qk, K }, .dtype = cfg.dtype, .memory_space = .any } },
                .k = .{ .ref = .{ .shape = &.{ T, H_qk, K }, .dtype = cfg.dtype, .memory_space = .any } },
                .v = .{ .ref = .{ .shape = &.{ T, H_v, V }, .dtype = cfg.dtype, .memory_space = .any } },
                .g = .{ .ref = .{ .shape = &.{ T, H_v, K }, .dtype = .f32, .memory_space = .any } },
                .b_in = .{ .ref = .{ .shape = &.{ T, H_v, nl }, .dtype = cfg.dtype, .memory_space = .any } },
                .state_indices = .{ .ref = .{ .shape = &.{T}, .dtype = .i32, .memory_space = .smem } },
                .a_log = .{ .ref = .{ .shape = &.{ H_v, nl }, .dtype = .f32, .memory_space = .any } },
                .dt_bias = .{ .ref = .{ .shape = &.{ H_v, nl }, .dtype = .f32, .memory_space = .any } },
                .distribution = .{ .ref = .{ .shape = &.{2}, .dtype = .i32, .memory_space = .smem } },
                .initial_state = .{ .ref = .{ .shape = &.{ ns, H_v, K, V }, .dtype = cfg.state_dtype, .memory_space = .any } },
                .o = .{ .ref = .{ .shape = &.{ T, H_v, V }, .dtype = cfg.dtype, .memory_space = .any, .role = .output } },
                .state_out = .{ .ref = .{ .shape = &.{ ns, H_v, K, V }, .dtype = cfg.state_dtype, .memory_space = .any, .role = .output } },
                .h_bufs = .{ .ref = .{ .shape = &.{ 2, bt, H_v, K, V }, .dtype = cfg.state_dtype, .role = .scratch } },
                .h_load_sems = .{ .sem_array = .{ .shape = &.{2}, .kind = .dma } },
                .h_store_sems = .{ .sem_array = .{ .shape = &.{2}, .kind = .dma } },
            }, &.{}, opts);
            return runWithArgs(b, cfg, a);
        } else if (has_a_log) {
            const a = try b.declareArgsOpts(.{
                .pid = .{ .scalar = .i32 },
                .q = .{ .ref = .{ .shape = &.{ T, H_qk, K }, .dtype = cfg.dtype, .memory_space = .any } },
                .k = .{ .ref = .{ .shape = &.{ T, H_qk, K }, .dtype = cfg.dtype, .memory_space = .any } },
                .v = .{ .ref = .{ .shape = &.{ T, H_v, V }, .dtype = cfg.dtype, .memory_space = .any } },
                .g = .{ .ref = .{ .shape = &.{ T, H_v, K }, .dtype = .f32, .memory_space = .any } },
                .b_in = .{ .ref = .{ .shape = &.{ T, H_v, nl }, .dtype = cfg.dtype, .memory_space = .any } },
                .state_indices = .{ .ref = .{ .shape = &.{T}, .dtype = .i32, .memory_space = .smem } },
                .a_log = .{ .ref = .{ .shape = &.{ H_v, nl }, .dtype = .f32, .memory_space = .any } },
                .distribution = .{ .ref = .{ .shape = &.{2}, .dtype = .i32, .memory_space = .smem } },
                .initial_state = .{ .ref = .{ .shape = &.{ ns, H_v, K, V }, .dtype = cfg.state_dtype, .memory_space = .any } },
                .o = .{ .ref = .{ .shape = &.{ T, H_v, V }, .dtype = cfg.dtype, .memory_space = .any, .role = .output } },
                .state_out = .{ .ref = .{ .shape = &.{ ns, H_v, K, V }, .dtype = cfg.state_dtype, .memory_space = .any, .role = .output } },
                .h_bufs = .{ .ref = .{ .shape = &.{ 2, bt, H_v, K, V }, .dtype = cfg.state_dtype, .role = .scratch } },
                .h_load_sems = .{ .sem_array = .{ .shape = &.{2}, .kind = .dma } },
                .h_store_sems = .{ .sem_array = .{ .shape = &.{2}, .kind = .dma } },
            }, &.{}, opts);
            return runWithArgs(b, cfg, a);
        } else {
            const a = try b.declareArgsOpts(.{
                .pid = .{ .scalar = .i32 },
                .q = .{ .ref = .{ .shape = &.{ T, H_qk, K }, .dtype = cfg.dtype, .memory_space = .any } },
                .k = .{ .ref = .{ .shape = &.{ T, H_qk, K }, .dtype = cfg.dtype, .memory_space = .any } },
                .v = .{ .ref = .{ .shape = &.{ T, H_v, V }, .dtype = cfg.dtype, .memory_space = .any } },
                .g = .{ .ref = .{ .shape = &.{ T, H_v, K }, .dtype = .f32, .memory_space = .any } },
                .b_in = .{ .ref = .{ .shape = &.{ T, H_v, nl }, .dtype = cfg.dtype, .memory_space = .any } },
                .state_indices = .{ .ref = .{ .shape = &.{T}, .dtype = .i32, .memory_space = .smem } },
                .distribution = .{ .ref = .{ .shape = &.{2}, .dtype = .i32, .memory_space = .smem } },
                .initial_state = .{ .ref = .{ .shape = &.{ ns, H_v, K, V }, .dtype = cfg.state_dtype, .memory_space = .any } },
                .o = .{ .ref = .{ .shape = &.{ T, H_v, V }, .dtype = cfg.dtype, .memory_space = .any, .role = .output } },
                .state_out = .{ .ref = .{ .shape = &.{ ns, H_v, K, V }, .dtype = cfg.state_dtype, .memory_space = .any, .role = .output } },
                .h_bufs = .{ .ref = .{ .shape = &.{ 2, bt, H_v, K, V }, .dtype = cfg.state_dtype, .role = .scratch } },
                .h_load_sems = .{ .sem_array = .{ .shape = &.{2}, .kind = .dma } },
                .h_store_sems = .{ .sem_array = .{ .shape = &.{2}, .kind = .dma } },
            }, &.{}, opts);
            return runWithArgs(b, cfg, a);
        }
    } else {
        // has_b = false ⇒ drop the b_in operand entirely (Pallas does the same
        // when `b=None` is passed). 3 sub-branches mirror the has_b=true tree.
        if (has_a_log and has_dt) {
            const a = try b.declareArgsOpts(.{
                .pid = .{ .scalar = .i32 },
                .q = .{ .ref = .{ .shape = &.{ T, H_qk, K }, .dtype = cfg.dtype, .memory_space = .any } },
                .k = .{ .ref = .{ .shape = &.{ T, H_qk, K }, .dtype = cfg.dtype, .memory_space = .any } },
                .v = .{ .ref = .{ .shape = &.{ T, H_v, V }, .dtype = cfg.dtype, .memory_space = .any } },
                .g = .{ .ref = .{ .shape = &.{ T, H_v, K }, .dtype = .f32, .memory_space = .any } },
                .state_indices = .{ .ref = .{ .shape = &.{T}, .dtype = .i32, .memory_space = .smem } },
                .a_log = .{ .ref = .{ .shape = &.{ H_v, nl }, .dtype = .f32, .memory_space = .any } },
                .dt_bias = .{ .ref = .{ .shape = &.{ H_v, nl }, .dtype = .f32, .memory_space = .any } },
                .distribution = .{ .ref = .{ .shape = &.{2}, .dtype = .i32, .memory_space = .smem } },
                .initial_state = .{ .ref = .{ .shape = &.{ ns, H_v, K, V }, .dtype = cfg.state_dtype, .memory_space = .any } },
                .o = .{ .ref = .{ .shape = &.{ T, H_v, V }, .dtype = cfg.dtype, .memory_space = .any, .role = .output } },
                .state_out = .{ .ref = .{ .shape = &.{ ns, H_v, K, V }, .dtype = cfg.state_dtype, .memory_space = .any, .role = .output } },
                .h_bufs = .{ .ref = .{ .shape = &.{ 2, bt, H_v, K, V }, .dtype = cfg.state_dtype, .role = .scratch } },
                .h_load_sems = .{ .sem_array = .{ .shape = &.{2}, .kind = .dma } },
                .h_store_sems = .{ .sem_array = .{ .shape = &.{2}, .kind = .dma } },
            }, &.{}, opts);
            return runWithArgs(b, cfg, a);
        } else if (has_a_log) {
            const a = try b.declareArgsOpts(.{
                .pid = .{ .scalar = .i32 },
                .q = .{ .ref = .{ .shape = &.{ T, H_qk, K }, .dtype = cfg.dtype, .memory_space = .any } },
                .k = .{ .ref = .{ .shape = &.{ T, H_qk, K }, .dtype = cfg.dtype, .memory_space = .any } },
                .v = .{ .ref = .{ .shape = &.{ T, H_v, V }, .dtype = cfg.dtype, .memory_space = .any } },
                .g = .{ .ref = .{ .shape = &.{ T, H_v, K }, .dtype = .f32, .memory_space = .any } },
                .state_indices = .{ .ref = .{ .shape = &.{T}, .dtype = .i32, .memory_space = .smem } },
                .a_log = .{ .ref = .{ .shape = &.{ H_v, nl }, .dtype = .f32, .memory_space = .any } },
                .distribution = .{ .ref = .{ .shape = &.{2}, .dtype = .i32, .memory_space = .smem } },
                .initial_state = .{ .ref = .{ .shape = &.{ ns, H_v, K, V }, .dtype = cfg.state_dtype, .memory_space = .any } },
                .o = .{ .ref = .{ .shape = &.{ T, H_v, V }, .dtype = cfg.dtype, .memory_space = .any, .role = .output } },
                .state_out = .{ .ref = .{ .shape = &.{ ns, H_v, K, V }, .dtype = cfg.state_dtype, .memory_space = .any, .role = .output } },
                .h_bufs = .{ .ref = .{ .shape = &.{ 2, bt, H_v, K, V }, .dtype = cfg.state_dtype, .role = .scratch } },
                .h_load_sems = .{ .sem_array = .{ .shape = &.{2}, .kind = .dma } },
                .h_store_sems = .{ .sem_array = .{ .shape = &.{2}, .kind = .dma } },
            }, &.{}, opts);
            return runWithArgs(b, cfg, a);
        } else {
            const a = try b.declareArgsOpts(.{
                .pid = .{ .scalar = .i32 },
                .q = .{ .ref = .{ .shape = &.{ T, H_qk, K }, .dtype = cfg.dtype, .memory_space = .any } },
                .k = .{ .ref = .{ .shape = &.{ T, H_qk, K }, .dtype = cfg.dtype, .memory_space = .any } },
                .v = .{ .ref = .{ .shape = &.{ T, H_v, V }, .dtype = cfg.dtype, .memory_space = .any } },
                .g = .{ .ref = .{ .shape = &.{ T, H_v, K }, .dtype = .f32, .memory_space = .any } },
                .state_indices = .{ .ref = .{ .shape = &.{T}, .dtype = .i32, .memory_space = .smem } },
                .distribution = .{ .ref = .{ .shape = &.{2}, .dtype = .i32, .memory_space = .smem } },
                .initial_state = .{ .ref = .{ .shape = &.{ ns, H_v, K, V }, .dtype = cfg.state_dtype, .memory_space = .any } },
                .o = .{ .ref = .{ .shape = &.{ T, H_v, V }, .dtype = cfg.dtype, .memory_space = .any, .role = .output } },
                .state_out = .{ .ref = .{ .shape = &.{ ns, H_v, K, V }, .dtype = cfg.state_dtype, .memory_space = .any, .role = .output } },
                .h_bufs = .{ .ref = .{ .shape = &.{ 2, bt, H_v, K, V }, .dtype = cfg.state_dtype, .role = .scratch } },
                .h_load_sems = .{ .sem_array = .{ .shape = &.{2}, .kind = .dma } },
                .h_store_sems = .{ .sem_array = .{ .shape = &.{2}, .kind = .dma } },
            }, &.{}, opts);
            return runWithArgs(b, cfg, a);
        }
    }
}

/// Body of `run()` after `declareArgsOpts`. Generic over `a`'s type so the
/// 3 (has_a_log, has_dt_bias) arg-shape combos share one source. Optional
/// fields (a_log, dt_bias, eventually b_in) are accessed via `@hasField`.
fn runWithArgs(b: *mtt.Builder, cfg: Cfg, a: anytype) mtt.FinishError!void {
    const T = cfg.num_decode_tokens;
    const H_qk = cfg.nKq();
    const H_v = cfg.nV();
    const K = cfg.dK();
    const V = cfg.dV();
    const nl = cfg.num_lanes;
    const bt = cfg.bt;

    const k = b;

    // ── decode_end = distribution[0] ──  (`%0 = memref.load %arg9[%c0]`)
    const decode_end = k.scalarLoad(a.distribution, &.{k.cIndex(0)});
    // ── nb_t = pl.cdiv(decode_end, bt) = floor_div(decode_end + bt - 1, bt) ──
    //   The Python's `//` lowers to `divsi` + the floor-div sign-correction
    //   (`%1..%13` in the target; `sign(bt)` folds to 1). Hand-emitted (b.divFloor
    //   assumes vector operands — a DSL bug worth fixing later). Op order mirrors
    //   the target so the hoisted `%c{7,8,0,1}_i32` land in the same order:
    //   `addi %0,7` (births %c7_i32) → `divsi _,8` (%c8_i32) → `cmpi sgt _,0`
    //   (%c0_i32) → `cmpi ne _,1` (%c1_i32).
    const nb_t = blk: {
        const a1 = k.addi(decode_end, k.lift(@as(i32, @intCast(bt - 1)))); // decode_end + bt-1
        const q = k.divsi(a1, k.lift(@as(i32, @intCast(bt)))); // truncated quotient
        const sgn_a = k.subi(k.extui(k.cmpi(.sgt, a1, k.lift(@as(i32, 0))), .i32), k.extui(k.cmpi(.slt, a1, k.lift(@as(i32, 0))), .i32));
        const sign_diff = k.cmpi(.ne, sgn_a, k.lift(@as(i32, 1))); // sign(a) != sign(bt)=1
        const rem_nz = k.cmpi(.ne, k.remsi(a1, k.lift(@as(i32, @intCast(bt)))), k.lift(@as(i32, 0)));
        break :blk k.select(k.andi(sign_diff, rem_nz), k.subi(q, k.lift(@as(i32, 1))), q);
    };
    const c0i = k.lift(@as(i32, 0)); // i32 zero — reused by the prologue/epilogue memref slices.

    // ── Prologue: for i_t in range(bt): @pl.when(i_t < decode_end):
    //      enqueue_dma(state_out[state_indices[i_t]:+1], h_bufs[0, i_t:+1], h_load_sems[0]).start()
    //   The Python's `for i_t in range(bt)` is a Python loop ⇒ fully unrolled;
    //   mirror with `inline for`. Comptime-dispatched on `cfg.bt` (each arm in
    //   `SUPPORTED_BT` instantiates its own unroll count).
    prologue_dispatch: inline for (SUPPORTED_BT) |comptime_bt| {
        if (bt == comptime_bt) {
            inline for (0..comptime_bt) |i_t| {
                const i_const = k.lift(@as(i32, @intCast(i_t)));
                var ifb = k.openIf(k.cmpi(.sgt, decode_end, i_const)); // i_t < decode_end ⇔ decode_end > i_t
                {
                    const si = k.scalarLoad(a.state_indices, &.{k.cIndex(@intCast(i_t))});
                    const src = k.memRefSlice(a.state_out, &.{ si, c0i, c0i, c0i }, &.{ 1, H_v, K, V }, &.{});
                    const dst5 = k.memRefSlice(a.h_bufs, &.{ c0i, i_const, c0i, c0i, c0i }, &.{ 1, 1, H_v, K, V }, &.{});
                    const dst = k.memRefSqueeze(dst5, &.{ 1, H_v, K, V });
                    const sem1 = k.memRefSlice(a.h_load_sems, &.{c0i}, &.{1}, &.{});
                    const sem = k.memRefSqueeze(sem1, &.{});
                    k.enqueueDma(src, dst, sem, .{});
                    ifb.yieldThen(.{});
                }
            }
            break :prologue_dispatch;
        }
    }

    // ── `pltpu.emit_pipeline(_inner_kernel, grid=(nb_t,), in_specs=[qk,qk,v,g,b,a_log,dt_bias], out_specs=v)(...)`. ──
    //   Lowers to the `"tpu.region"() ({ allocas; scf.if(nb_t>0){ trivial sync-copies;
    //   ep_initialize_0; scf.for ...{ schedule + body }; finalize }; tpu.yield })` —
    //   emitted by `mtt.pipeline.emitPipeline` (a port of `pltpu.emit_pipeline`).
    const Pipeline = mtt.pipeline;
    var ctx = PipeCtx{
        .decode_end = decode_end,
        .state_hbm = a.state_out,
        .H_qk = H_qk,
        .H_v = H_v,
        .K = K,
        .V = V,
        .nl = nl,
        .bt = bt,
        .dtype = cfg.dtype,
        .state_dtype = cfg.state_dtype,
        .scale = cfg.scale,
        .l2_eps = cfg.l2_eps,
        .repeat_factor = cfg.repeatFactor(),
        .use_qk_l2norm = cfg.use_qk_l2norm,
        .use_gate_in_kernel = cfg.use_gate_in_kernel,
        .has_dt_bias = cfg.has_dt_bias,
        .lower_bound = cfg.lower_bound,
    };
    const bd = Pipeline.BlockDim;
    const qk_block = [_]Pipeline.BlockDim{ .{ .bounded_slice = bt }, .{ .blocked = H_qk }, .{ .blocked = K } };
    const v_block = [_]Pipeline.BlockDim{ .{ .bounded_slice = bt }, .{ .blocked = H_v }, .{ .blocked = V } };
    const g_block = [_]Pipeline.BlockDim{ .{ .bounded_slice = bt }, .{ .blocked = H_v }, .{ .blocked = K } };
    const b_block = [_]Pipeline.BlockDim{ .{ .bounded_slice = bt }, .{ .blocked = H_v }, .{ .blocked = nl } };
    _ = bd;
    const qk_spec = Pipeline.PipeSpec{ .full_shape = &.{ T, H_qk, K }, .dtype = cfg.dtype, .block_shape = &qk_block, .index_map = tokenMap, .index_map_ctx = &ctx };
    const v_spec = Pipeline.PipeSpec{ .full_shape = &.{ T, H_v, V }, .dtype = cfg.dtype, .block_shape = &v_block, .index_map = tokenMap, .index_map_ctx = &ctx };
    const g_spec = Pipeline.PipeSpec{ .full_shape = &.{ T, H_v, K }, .dtype = .f32, .block_shape = &g_block, .index_map = tokenMap, .index_map_ctx = &ctx };
    const b_spec = Pipeline.PipeSpec{ .full_shape = &.{ T, H_v, nl }, .dtype = cfg.dtype, .block_shape = &b_block, .index_map = tokenMap, .index_map_ctx = &ctx };
    // Optional specs — when the corresponding `a.X` arg isn't declared (Pallas
    // omitted that operand), pass `null` so the pipeline skips that slot.
    const a_log_spec: ?Pipeline.PipeSpec = if (@hasField(@TypeOf(a), "a_log")) .{ .full_shape = &.{ H_v, nl }, .dtype = .f32 } else null;
    const dt_bias_spec: ?Pipeline.PipeSpec = if (@hasField(@TypeOf(a), "dt_bias")) .{ .full_shape = &.{ H_v, nl }, .dtype = .f32 } else null;
    const b_in_spec: ?Pipeline.PipeSpec = if (@hasField(@TypeOf(a), "b_in")) b_spec else null;
    const in_specs = [_]?Pipeline.PipeSpec{ qk_spec, qk_spec, v_spec, g_spec, b_in_spec, a_log_spec, dt_bias_spec };
    const out_specs = [_]?Pipeline.PipeSpec{v_spec};
    const refs = [_]?mtt.Value{
        a.q, a.k, a.v, a.g,
        if (@hasField(@TypeOf(a), "b_in")) a.b_in else null,
        if (@hasField(@TypeOf(a), "a_log")) a.a_log else null,
        if (@hasField(@TypeOf(a), "dt_bias")) a.dt_bias else null,
        a.o,
    };
    const scratches = [_]mtt.Value{ a.h_bufs, a.state_indices, a.h_load_sems, a.h_store_sems };
    try Pipeline.emitPipeline(k, &refs, &scratches, .{
        .body = innerKernel,
        .body_ctx = &ctx,
        .grid = &.{nb_t},
        .in_specs = &in_specs,
        .out_specs = &out_specs,
        .trace_scopes = true,
    });

    // ── Epilogue: drain the last block's store + (if nb_t >= 2) the prev one. ──
    //   Python order: last_buf = (nb_t-1)%2 ; other_buf = nb_t%2 ; last_block_len ;
    //   first wait_dma2 ; other_block_len = where(nb_t>=2, min(bt, decode_end-(nb_t-2)*bt), 0) ;
    //   @pl.when(other_block_len>0) second wait_dma2.  (Floor-mod via floorMod2.)
    const bt_c = k.lift(@as(i32, @intCast(bt)));
    const nbm1 = k.subi(nb_t, k.lift(@as(i32, 1)));
    const last_buf = floorMod2(k, nbm1);
    const other_buf = floorMod2(k, nb_t); // Python computes this before last_block_len.
    const last_block_len = k.minsi(k.subi(decode_end, k.muli(nbm1, bt_c)), bt_c);
    {
        const sh5 = [_]i64{ 1, kDyn, H_v, K, V };
        const sh4 = [_]i64{ kDyn, H_v, K, V };
        const src5 = k.memRefSlice(a.h_bufs, &.{ last_buf, c0i, c0i, c0i, c0i }, &sh5, &.{last_block_len});
        const src = k.memRefSqueeze(src5, &sh4);
        const dst = k.memRefSlice(a.state_out, &.{ c0i, c0i, c0i, c0i }, &sh4, &.{last_block_len});
        const sem1 = k.memRefSlice(a.h_store_sems, &.{last_buf}, &.{1}, &.{});
        const sem = k.memRefSqueeze(sem1, &.{});
        k.waitDma2(sem, src, dst, .{});
    }
    //   other_block_len = jnp.where(nb_t >= 2, min(bt, decode_end-(nb_t-2)*bt), 0) — cond first.
    const ge2 = k.cmpi(.sge, nb_t, k.lift(@as(i32, 2)));
    const nbm2 = k.subi(nb_t, k.lift(@as(i32, 2)));
    const obl_raw = k.minsi(k.subi(decode_end, k.muli(nbm2, bt_c)), bt_c);
    const other_block_len = k.select(ge2, obl_raw, c0i);
    {
        var ifb = k.openIf(k.cmpi(.sgt, other_block_len, c0i));
        {
            const sh5 = [_]i64{ 1, kDyn, H_v, K, V };
            const sh4 = [_]i64{ kDyn, H_v, K, V };
            const src5 = k.memRefSlice(a.h_bufs, &.{ other_buf, c0i, c0i, c0i, c0i }, &sh5, &.{other_block_len});
            const src = k.memRefSqueeze(src5, &sh4);
            const dst = k.memRefSlice(a.state_out, &.{ c0i, c0i, c0i, c0i }, &sh4, &.{other_block_len});
            const sem1 = k.memRefSlice(a.h_store_sems, &.{other_buf}, &.{1}, &.{});
            const sem = k.memRefSqueeze(sem1, &.{});
            k.waitDma2(sem, src, dst, .{});
            ifb.yieldThen(.{});
        }
    }
}

/// MLIR `ShapedType::kDynamic` — the sentinel for a `?` dim in a memref shape.
const kDyn: i64 = std.math.minInt(i64);

/// Python-style floor-mod by 2 (`a % 2` with the `remsi` sign correction Pallas
/// emits): `m = a remsi 2; (m != 0 && m < 0) ? m + 2 : m`.
fn floorMod2(k: *mtt.Builder, a: mtt.Value) mtt.Value {
    const c0 = k.lift(@as(i32, 0));
    const c2 = k.lift(@as(i32, 2));
    const m = k.remsi(a, c2);
    const m_ne = k.cmpi(.ne, m, c0);
    const m_lt = k.cmpi(.slt, m, c0);
    return k.select(k.andi(m_lt, m_ne), k.addi(m, c2), m);
}

// =============================================================================
// emit_pipeline wiring: token_map index fn + the _inner_kernel body callback
// =============================================================================

/// Captured-by-closure state the `emit_pipeline` callbacks need (Zig has no
/// closures — pack it into a struct passed as `?*anyopaque`).
const PipeCtx = struct {
    decode_end: mtt.Value,
    state_hbm: mtt.Value, // the `state_out` HBM ref (== Python's `state_hbm`)
    H_qk: i64,
    H_v: i64,
    K: i64,
    V: i64,
    nl: i64,
    bt: i64,
    dtype: mtt.DType, // q/k/v/b/o element dtype
    state_dtype: mtt.DType, // h_bufs/state element dtype
    scale: f32,
    l2_eps: f32,
    repeat_factor: i64,
    use_qk_l2norm: bool,
    use_gate_in_kernel: bool,
    has_dt_bias: bool,
    lower_bound: ?f32,
};

/// Materialize a `RefWindow`'s current tile-buffer for use at a load/store site:
/// `slot != null` ⇒ `tpu.memref_slice buf[slot, 0…] -> 1 × tile_shape` then
/// `tpu.memref_squeeze -> tile_shape` (re-emitted per call site, matching Pallas's
/// per-use re-trace of `current_ref`); `slot == null` (trivial ref) ⇒ `buf` directly.
fn tileBuf(k: *mtt.Builder, rw: mtt.pipeline.RefWindow, tile_shape: []const i64) mtt.Value {
    const slot = rw.slot orelse return rw.buf.?;
    const a = k.arena.allocator();
    const c0i = k.lift(@as(i32, 0));
    const base = a.alloc(mtt.Value, tile_shape.len + 1) catch @panic("tileBuf: OOM");
    base[0] = slot;
    for (base[1..]) |*x| x.* = c0i;
    const rshape = a.alloc(i64, tile_shape.len + 1) catch @panic("tileBuf: OOM");
    rshape[0] = 1;
    @memcpy(rshape[1..], tile_shape);
    return k.memRefSqueeze(k.memRefSlice(rw.buf.?, base, rshape, &.{}), tile_shape);
}

/// `token_map(i) = (pl.ds(i*bt, jnp.minimum(bt, decode_end - i*bt)), 0, 0)`.
fn tokenMap(b: *mtt.Builder, indices: []const mtt.Value, ctx_: ?*anyopaque) []const mtt.pipeline.BlockIndex {
    const ctx: *PipeCtx = @ptrCast(@alignCast(ctx_.?));
    const i = indices[0];
    const bt_c = b.lift(@as(i32, @intCast(ctx.bt)));
    const t_start = i.mul(bt_c); // i * bt
    const t_size = b.minsi(b.subi(ctx.decode_end, t_start), bt_c); // min(decode_end - i*bt, bt)
    const c0 = b.lift(@as(i32, 0));
    const out = b.arena.allocator().alloc(mtt.pipeline.BlockIndex, 3) catch @panic("tokenMap: OOM");
    out[0] = .{ .slice = .{ .start = t_start, .size = t_size } };
    out[1] = .{ .scalar = c0 };
    out[2] = .{ .scalar = c0 };
    return out;
}

/// `_inner_kernel(q_ref,k_ref,v_ref,g_ref,b_ref,a_log_ref,dt_bias_ref,o_ref, h_bufs_s,
/// state_indices_s, h_load_sems_s, h_store_sems_s)` — the per-bt-block recurrent update.
/// `refs = [q,k,v,g,b,a_log,dt_bias,o]` (windowed VMEM); `scratches = [h_bufs,
/// state_indices, h_load_sems, h_store_sems]`; `grid_indices = [block_id]`.
fn innerKernel(
    b: *mtt.Builder,
    grid_indices: []const mtt.Value,
    refs: []const mtt.pipeline.RefWindow,
    scratches: []const mtt.Value,
    ctx_: ?*anyopaque,
) mtt.FinishError!void {
    const k = b;
    const ctx: *PipeCtx = @ptrCast(@alignCast(ctx_.?));
    const H_qk = ctx.H_qk;
    const H_v = ctx.H_v;
    const K = ctx.K;
    const V = ctx.V;
    const rf = ctx.repeat_factor; // H_v / H_qk (GQA)
    const bt = ctx.bt; // comptime-dispatched below via SUPPORTED_BT for the `inline for (0..bt)`s
    const c0i = k.lift(@as(i32, 0));
    const c0idx = k.cIndex(0);
    const bt_c = k.lift(@as(i32, @intCast(bt)));

    // `refs = [q,k,v,g,b,a_log,dt_bias,o]` — RefWindows (see mtt.pipeline.RefWindow):
    // q/k/v/g/b/o have `.slot` (slice `.buf[slot, …]` then squeeze, re-emitted per use);
    // a_log/dt_bias are trivial (`.slot == null` ⇒ `.buf` IS the ref).
    const q_rw = refs[0];
    const k_rw = refs[1];
    const v_rw = refs[2];
    const g_rw = refs[3];
    const b_rw = refs[4];
    // `a_log` / `dt_bias` may be absent (Pallas passes `None` when
    // `use_gate_in_kernel=False` resp. `has_dt_bias=False`). The pipeline left
    // `refs[5].buf` / `refs[6].buf` null in that case — branch on it below.
    const a_log_buf_opt: ?mtt.Value = refs[5].buf;
    const dt_bias_buf_opt: ?mtt.Value = refs[6].buf;
    const o_rw = refs[7];
    const h_bufs = scratches[0];
    const state_indices = scratches[1];
    const h_load_sems = scratches[2];
    const h_store_sems = scratches[3];
    const state_hbm = ctx.state_hbm;
    const decode_end = ctx.decode_end;

    const block_id = grid_indices[0];
    const t_start = k.muli(block_id, bt_c);
    const block_len = k.minsi(k.subi(decode_end, t_start), bt_c);
    const buf_idx = floorMod2(k, block_id);
    const next_buf_idx = floorMod2(k, k.addi(block_id, k.lift(@as(i32, 1))));

    // ── Gate setup. Computed only when `use_gate_in_kernel` (otherwise the
    //   inner loop sets `gk = g_t` directly). `a_val = exp(a_log_ref[:,0].f32)`
    //   and `dt_bias_val = dt_bias_ref[...].f32`, optionally concat'd along the
    //   last axis when `K > num_lanes` (Pallas: `jnp.concatenate([tile] * (K//nl),
    //   axis=-1)` ⇒ one `tpu.concatenate` here). `tpu.concatenate` arity cap
    //   is 16; K/nl up to 16 fits a single op.
    const a_val_opt: ?mtt.Value = if (a_log_buf_opt) |a_log_buf|
        k.exp(k.shapeCast(k.vectorLoadShape(a_log_buf, &.{ c0idx, c0idx }, &.{ H_v, 1 }), &.{H_v}))
    else
        null;
    const dt_bias_val_opt: ?mtt.Value = if (dt_bias_buf_opt) |dt_bias_buf| blk: {
        const dt_bias_tile = k.refLoad(dt_bias_buf); // [H_v, nl] f32
        break :blk if (K > ctx.nl) inner_blk: {
            const reps: usize = @intCast(@divExact(K, ctx.nl));
            var tiles_buf: [16]mtt.Value = undefined;
            for (0..reps) |i| tiles_buf[i] = dt_bias_tile;
            break :inner_blk k.concatenate(tiles_buf[0..reps], 1); // [H_v, K]
        } else dt_bias_tile; // already [H_v, K] when K == nl
    } else null;

    // ── Step 1: prefetch next bt-block's recurrent state ──
    const next_t_start = k.addi(t_start, bt_c);
    const next_block_len = k.maxsi(k.minsi(k.subi(decode_end, next_t_start), bt_c), c0i);
    prefetch_dispatch: inline for (SUPPORTED_BT) |comptime_bt| {
        if (bt == comptime_bt) {
            inline for (0..comptime_bt) |i_t| {
                const i_const = k.lift(@as(i32, @intCast(i_t)));
                var ifb = k.openIf(k.cmpi(.sgt, next_block_len, i_const)); // i_t < next_block_len
                {
                    const idx = k.scalarLoad(state_indices, &.{k.toIndex(k.addi(next_t_start, i_const))});
                    const src = k.memRefSlice(state_hbm, &.{ idx, c0i, c0i, c0i }, &.{ 1, H_v, K, V }, &.{});
                    const dst5 = k.memRefSlice(h_bufs, &.{ next_buf_idx, i_const, c0i, c0i, c0i }, &.{ 1, 1, H_v, K, V }, &.{});
                    const dst = k.memRefSqueeze(dst5, &.{ 1, H_v, K, V });
                    const sem1 = k.memRefSlice(h_load_sems, &.{next_buf_idx}, &.{1}, &.{});
                    const sem = k.memRefSqueeze(sem1, &.{});
                    k.enqueueDma(src, dst, sem, .{});
                    ifb.yieldThen(.{});
                }
            }
            break :prefetch_dispatch;
        }
    }

    // ── Step 2: wait for current bt-block's state loads ──
    {
        const sh5 = [_]i64{ 1, kDyn, H_v, K, V };
        const sh4 = [_]i64{ kDyn, H_v, K, V };
        const src = k.memRefSlice(state_hbm, &.{ c0i, c0i, c0i, c0i }, &sh4, &.{block_len});
        const dst5 = k.memRefSlice(h_bufs, &.{ buf_idx, c0i, c0i, c0i, c0i }, &sh5, &.{block_len});
        const dst = k.memRefSqueeze(dst5, &sh4);
        const sem1 = k.memRefSlice(h_load_sems, &.{buf_idx}, &.{1}, &.{});
        const sem = k.memRefSqueeze(sem1, &.{});
        k.waitDma2(sem, src, dst, .{});
    }

    // ── Step 3: compute — unrolled-×bt per-token recurrent update (§2.1). ──
    //   Mirrors py/gdn_decode_fast.py lines 326-396 for the bt8_nv128 config
    //   (use_qk_l2norm, use_gate_in_kernel, dt_bias present, lower_bound=None,
    //   b present, rf=8>1, V==num_lanes ⇒ beta no-concat, K==num_lanes ⇒ dt_bias no-concat).
    //   `dot_general(x.reshape(H_v,1,K), h_pre, (((2,),(1,)),((0,),(0,))))` → `tpu.matmul`
    //   with `#tpu.dot_dimension_numbers<[2],[1],[1],[2],[0,0,0,1,1,2],[0],[0]>`.
    //   Each load/store re-emits its `tpu.memref_slice`/`memref_squeeze` inside the
    //   per-token `scf.if` (Pallas re-traces `current_ref` per use; CSE doesn't cross regions).
    const ddn = k.dotDimensionNumbers(&.{2}, &.{1}, &.{1}, &.{2}, &.{ 0, 0, 0, 1, 1, 2 }, &.{0}, &.{0});
    const qk_tile = [_]i64{ bt, H_qk, K }; // q/k
    const v_tile = [_]i64{ bt, H_v, V }; // v/o
    const g_tile = [_]i64{ bt, H_v, K }; // g — third dim is K (≠ V when K != V)
    const b_tile_shape = [_]i64{ bt, H_v, ctx.nl }; // b — third dim is num_lanes (concat'd to V in-kernel when V > nl)
    compute_dispatch: inline for (SUPPORTED_BT) |comptime_bt| {
        if (bt == comptime_bt) {
            inline for (0..comptime_bt) |i_t| {
                const i_const = k.lift(@as(i32, @intCast(i_t)));
                const i_idx = k.cIndex(@intCast(i_t));
                var ifb = k.openIf(k.cmpi(.sgt, block_len, i_const)); // i_t < block_len
                {
            // loads (q/k bf16 [H_qk,K]; v bf16 [H_v,V]; g f32 [H_v,K]; h0 bf16 [H_v,K,V];
            // b optional [H_v,nl] bf16, gated by ctx.has_b).
            const bii = k.toIndex(buf_idx);
            const h0 = k.shapeCast(k.vectorLoadShape(h_bufs, &.{ bii, i_idx, c0idx, c0idx, c0idx }, &.{ 1, 1, H_v, K, V }), &.{ H_v, K, V }).to(.f32);
            const q_t0 = k.shapeCast(k.vectorLoadShape(tileBuf(k, q_rw, &qk_tile), &.{ i_idx, c0idx, c0idx }, &.{ 1, H_qk, K }), &.{ H_qk, K }).to(.f32);
            const k_t0 = k.shapeCast(k.vectorLoadShape(tileBuf(k, k_rw, &qk_tile), &.{ i_idx, c0idx, c0idx }, &.{ 1, H_qk, K }), &.{ H_qk, K }).to(.f32);
            const v_t = k.shapeCast(k.vectorLoadShape(tileBuf(k, v_rw, &v_tile), &.{ i_idx, c0idx, c0idx }, &.{ 1, H_v, V }), &.{ H_v, V }).to(.f32);
            const g_t = k.shapeCast(k.vectorLoadShape(tileBuf(k, g_rw, &g_tile), &.{ i_idx, c0idx, c0idx }, &.{ 1, H_v, K }), &.{ H_v, K }); // f32 already

            // beta_t (sigmoid of the optional raw beta). When b is absent the
            // kernel skips the load + sigmoid; later `b_v = v_diff` directly.
            // When V > num_lanes, Python does
            //   `jax.nn.sigmoid(jnp.concatenate([b_tile] * (V // nl), -1))`
            // — concat on the upcast f32 tensor before the sigmoid.
            const beta_t_opt: ?mtt.Value = if (b_rw.buf != null) blk: {
                const b_raw = k.shapeCast(k.vectorLoadShape(tileBuf(k, b_rw, &b_tile_shape), &.{ i_idx, c0idx, c0idx }, &.{ 1, H_v, ctx.nl }), &.{ H_v, ctx.nl }).to(.f32);
                const b_tile = if (V > ctx.nl) inner_blk: {
                    const reps: usize = @intCast(@divExact(V, ctx.nl));
                    var tiles_buf: [16]mtt.Value = undefined;
                    for (0..reps) |i| tiles_buf[i] = b_raw;
                    break :inner_blk k.concatenate(tiles_buf[0..reps], 1); // [H_v, V]
                } else b_raw; // already [H_v, V] when V == nl
                const ones_hv_v = k.full(&.{ H_v, V }, @as(f32, 1.0), .f32);
                break :blk k.divf(ones_hv_v, k.addf(k.exp(k.negf(b_tile)), ones_hv_v)); // [H_v, V]
            } else null;

            // q_t = q_t0 / sqrt(sum(q_t0², -1, keepdims) + l2_eps);  same for k.
            // Python: `if use_qk_l2norm: q_t = q_t / ...` — skip the whole
            // multiReduction/sqrt/divf chain when the flag is off. Op-emission
            // order must stay interleaved (q_norm → q_div → k_norm → k_div),
            // not (q_norm → k_norm → q_div → k_div): Pallas traces each statement
            // separately, so q_t reassignment lands before k's reduction.
            const q_n, const k_n = if (ctx.use_qk_l2norm) blk: {
                const zeros_hqk = k.zeros(&.{H_qk}, .f32);
                const eps_hqk1 = k.full(&.{ H_qk, 1 }, @as(f32, ctx.l2_eps), .f32);
                const q_norm = k.broadcastTo(k.sqrt(k.addf(k.shapeCast(k.multiReduction(.add, k.mulf(q_t0, q_t0), zeros_hqk, &.{1}), &.{ H_qk, 1 }), eps_hqk1)), &.{ H_qk, K });
                const q_div = k.divf(q_t0, q_norm);
                const k_norm = k.broadcastTo(k.sqrt(k.addf(k.shapeCast(k.multiReduction(.add, k.mulf(k_t0, k_t0), zeros_hqk, &.{1}), &.{ H_qk, 1 }), eps_hqk1)), &.{ H_qk, K });
                const k_div = k.divf(k_t0, k_norm);
                break :blk .{ q_div, k_div };
            } else .{ q_t0, k_t0 };

            // q_t *= scale.
            const scale_hqk = k.full(&.{ H_qk, K }, @as(f32, ctx.scale), .f32);
            const q_sc = k.mulf(q_n, scale_hqk); // [H_qk, K]

            // GQA repeat (H_qk → H_v, factor rf): reshape→broadcast→reshape.
            // rf > 1: keep the [H_qk, rf, K] intermediates (q3/k3) — they get
            // shape_cast different ways below (Pallas re-shape_casts the
            // broadcast result, not the [H_v, K] form, at each use site).
            // rf == 1: Python's `if repeat_factor > 1` skips both `jnp.repeat`s
            // entirely, so q_t / k_t stay [H_qk, K] == [H_v, K]. Each later
            // 3D use (matmul, outer) reshapes the 2D directly — no broadcast.
            const q_rep, const q_for_matmul = if (rf > 1) blk: {
                const q3 = k.broadcastTo(k.shapeCast(q_sc, &.{ H_qk, 1, K }), &.{ H_qk, rf, K }); // [H_qk, rf, K]
                break :blk .{ k.shapeCast(q3, &.{ H_v, K }), q3 };
            } else .{ q_sc, q_sc };
            const k_rep, const k_for_3d = if (rf > 1) blk: {
                const k3 = k.broadcastTo(k.shapeCast(k_n, &.{ H_qk, 1, K }), &.{ H_qk, rf, K }); // [H_qk, rf, K]
                break :blk .{ k.shapeCast(k3, &.{ H_v, K }), k3 };
            } else .{ k_n, k_n };

            // gk depends on `use_gate_in_kernel`:
            //   FALSE: gk = g_t (no a_log, no dt_bias, no transform).
            //   TRUE + lower_bound == None: gk = -a_val[:, None] · softplus(g_val)
            //   TRUE + lower_bound != None: gk = lower_bound · sigmoid(a_val · g_val)
            //                                  = lower_bound / (1 + exp(-(a_val · g_val)))
            // Where g_val = g_t (+ dt_bias_val if `has_dt_bias`).
            const gk = if (a_val_opt) |a_val| blk: {
                // Gate-in-kernel branch.
                const g_val = if (dt_bias_val_opt) |dt_bias_val|
                    k.addf(g_t, dt_bias_val)
                else
                    g_t; // no dt_bias ⇒ g_val = g_t directly
                if (ctx.lower_bound) |lb| {
                    // Sigmoid-gate sub-branch.
                    const a_val_2d = k.shapeCast(a_val, &.{ H_v, 1 });
                    const a_val_bc = k.broadcastTo(a_val_2d, &.{ H_v, K });
                    const ag = k.mulf(a_val_bc, g_val);
                    const zeros_hv_k_s = k.zeros(&.{ H_v, K }, .f32);
                    const neg_ag = k.subf(zeros_hv_k_s, ag); // -(a_val · g_val)
                    const ones_hv_k = k.full(&.{ H_v, K }, @as(f32, 1.0), .f32);
                    const denom = k.addf(ones_hv_k, k.exp(neg_ag));
                    const lb_const = k.full(&.{ H_v, K }, lb, .f32);
                    break :blk k.divf(lb_const, denom); // [H_v, K]
                } else {
                    // Softplus-gate sub-branch (Qwen3.5 GDN default).
                    const zeros_hv1 = k.zeros(&.{ H_v, 1 }, .f32);
                    const neg_a = k.subf(zeros_hv1, k.shapeCast(a_val, &.{ H_v, 1 })); // -a_val[:,None]  [H_v, 1]
                    // softplus(x) = logaddexp(x, 0) = max(x,0) + log1p(exp(-|x|))   (NaN-safe via select).
                    const zeros_hv_k = k.zeros(&.{ H_v, K }, .f32);
                    const sp_max = k.maximumf(g_val, zeros_hv_k);
                    const sp_nan = k.cmpf(.one, g_val, g_val);
                    const sp_x0 = k.addf(g_val, zeros_hv_k);
                    const sp_e = k.exp(k.subf(zeros_hv_k, k.absf(g_val)));
                    const sp_l1p = k.emit(mlir.Operation.make(k.ctx, "math.log1p", .{ .operands = .{ .flat = &.{sp_e.inner} }, .results = .{ .flat = &.{sp_e.type_()} }, .location = k.loc() }));
                    const softplus = k.select(sp_nan, sp_x0, k.addf(sp_max, sp_l1p)); // [H_v, K]
                    break :blk k.mulf(k.broadcastTo(neg_a, &.{ H_v, K }), softplus); // [H_v, K]
                }
            } else g_t; // no gate ⇒ gk = g_t (raw alpha)

            // h_pre = h0 * exp(gk[:,:,None]).
            const h_pre = k.mulf(h0, k.broadcastTo(k.exp(k.shapeCast(gk, &.{ H_v, K, 1 })), &.{ H_v, K, V })); // [H_v, K, V]

            // kh = dot_general(k_rep.reshape(H_v,1,K), h_pre, ...).reshape(H_v, V).
            //   rf>1: shape_cast comes off the [H_qk, rf, K] intermediate (k3);
            //   rf=1: shape_cast comes off the [H_qk, K] = [H_v, K] (k_n) directly.
            const zeros_hv1v = k.zeros(&.{ H_v, 1, V }, .f32);
            const kh = k.shapeCast(k.matmulOpts(k.shapeCast(k_for_3d, &.{ H_v, 1, K }), h_pre, zeros_hv1v, .{ .dimension_numbers = ddn }), &.{ H_v, V }); // [H_v, V]
            const v_diff = k.subf(v_t, kh); // [H_v, V]
            const b_v = if (beta_t_opt) |beta_t| k.mulf(beta_t, v_diff) else v_diff; // [H_v, V]

            // o_step1 = dot_general(q_rep.reshape(H_v,1,K), h_pre, ...).reshape(H_v, V).
            const o_step1 = k.shapeCast(k.matmulOpts(k.shapeCast(q_for_matmul, &.{ H_v, 1, K }), h_pre, zeros_hv1v, .{ .dimension_numbers = ddn }), &.{ H_v, V }); // [H_v, V]
            // qk_dot = sum(q_rep * k_rep, -1, keepdims).
            const zeros_hv = k.zeros(&.{H_v}, .f32);
            const qk_dot = k.broadcastTo(k.shapeCast(k.multiReduction(.add, k.mulf(q_rep, k_rep), zeros_hv, &.{1}), &.{ H_v, 1 }), &.{ H_v, V }); // [H_v, V]
            const o_t = k.addf(o_step1, k.mulf(qk_dot, b_v)); // [H_v, V]

            // h_new = h_pre + k_rep[:,:,None] * b_v[:,None,:]  (both shape_casts, then both broadcasts — matching Pallas).
            const k3_sc = k.shapeCast(k_for_3d, &.{ H_v, K, 1 });
            const bv_sc = k.shapeCast(b_v, &.{ H_v, 1, V });
            const k3_outer = k.broadcastTo(k3_sc, &.{ H_v, K, V });
            const bv_outer = k.broadcastTo(bv_sc, &.{ H_v, K, V });
            const h_new = k.addf(h_pre, k.mulf(k3_outer, bv_outer)); // [H_v, K, V]

            // o_ref[i_t] = o_t.astype(o_ref.dtype);  h_bufs[buf_idx, i_t] = h_new.astype(...).
            // (`.to(bf16)` ⇒ `arith.truncf`, matching Pallas; `b.truncf` would emit `tpu.truncf`.)
            const o_bf = o_t.to(ctx.dtype);
            k.vectorStoreAt(tileBuf(k, o_rw, &v_tile), k.shapeCast(o_bf, &.{ 1, H_v, V }), &.{ i_idx, c0idx, c0idx });
            k.vectorStoreAt(h_bufs, k.shapeCast(h_new.to(ctx.state_dtype), &.{ 1, 1, H_v, K, V }), &.{ bii, i_idx, c0idx, c0idx, c0idx });
            ifb.yieldThen(.{});
        }
    }
            break :compute_dispatch;
        }
    }

    // ── Step 4: wait for stores from 2 blocks ago (same buffer set) ──
    {
        const prev_t_start = k.maxsi(k.muli(k.subi(block_id, k.lift(@as(i32, 2))), bt_c), c0i);
        // `jnp.where(block_id >= 2, jnp.minimum(bt, decode_end - prev_t_start), 0)` —
        // the condition is evaluated first, then the `min` (subi/minsi).
        const ge2 = k.cmpi(.sge, block_id, k.lift(@as(i32, 2)));
        const prev_block_len_raw = k.minsi(k.subi(decode_end, prev_t_start), bt_c);
        const prev_block_len = k.select(ge2, prev_block_len_raw, c0i);
        var ifb = k.openIf(k.cmpi(.sgt, prev_block_len, c0i));
        {
            const sh5 = [_]i64{ 1, kDyn, H_v, K, V };
            const sh4 = [_]i64{ kDyn, H_v, K, V };
            const src5 = k.memRefSlice(h_bufs, &.{ buf_idx, c0i, c0i, c0i, c0i }, &sh5, &.{prev_block_len});
            const src = k.memRefSqueeze(src5, &sh4);
            const dst = k.memRefSlice(state_hbm, &.{ c0i, c0i, c0i, c0i }, &sh4, &.{prev_block_len});
            const sem1 = k.memRefSlice(h_store_sems, &.{buf_idx}, &.{1}, &.{});
            const sem = k.memRefSqueeze(sem1, &.{});
            k.waitDma2(sem, src, dst, .{});
            ifb.yieldThen(.{});
        }
    }

    // ── Step 5: start storing current bt-block's recurrent state ──
    store_dispatch: inline for (SUPPORTED_BT) |comptime_bt| {
        if (bt == comptime_bt) {
            inline for (0..comptime_bt) |i_t| {
                const i_const = k.lift(@as(i32, @intCast(i_t)));
                var ifb = k.openIf(k.cmpi(.sgt, block_len, i_const)); // i_t < block_len
                {
                    const idx = k.scalarLoad(state_indices, &.{k.toIndex(k.addi(t_start, i_const))});
                    const src5 = k.memRefSlice(h_bufs, &.{ buf_idx, i_const, c0i, c0i, c0i }, &.{ 1, 1, H_v, K, V }, &.{});
                    const src = k.memRefSqueeze(src5, &.{ 1, H_v, K, V });
                    const dst = k.memRefSlice(state_hbm, &.{ idx, c0i, c0i, c0i }, &.{ 1, H_v, K, V }, &.{});
                    const sem1 = k.memRefSlice(h_store_sems, &.{buf_idx}, &.{1}, &.{});
                    const sem = k.memRefSqueeze(sem1, &.{});
                    k.enqueueDma(src, dst, sem, .{});
                    ifb.yieldThen(.{});
                }
            }
            break :store_dispatch;
        }
    }
}

// =============================================================================
// Kernel factory + sweeps
// =============================================================================

/// "Full" config Kernel — `use_gate_in_kernel = true`, `has_dt_bias = true`,
/// `has_b = true`. This is the tpu-inference canonical config (the kernel
/// transforms a + dt_bias + A_log into the gate internally, and applies
/// sigmoid to raw `b`). 10-operand custom_call signature.
///
/// NOTE: `inputs` excludes the grid `pid` (scalar i32, injected by the TPU
/// runtime per grid step) and the output / scratch refs `declareArgsOpts`
/// declares for the IR. `initial_state` aliases `state_out` via
/// `input_output_aliases`. Scratch refs (h_bufs, h_load_sems, h_store_sems)
/// are kernel-local.
pub const KernelFull = zml.kernel.mosaic_tpu.Kernel(Cfg, .{
    .name = "decoding_gdn-bt_8",
    .inputs = &.{ "q", "k", "v", "g", "b_in", "state_indices", "a_log", "dt_bias", "distribution", "initial_state" },
    .outputs = &.{ "o", "state_out" },
    .run = run,
});

/// `use_gate_in_kernel = false` config Kernel — the gate transform is applied
/// upstream (caller passes pre-computed `g`); a_log / dt_bias are dropped from
/// the func.func signature. 8-operand custom_call signature.
///
/// llmd's GatedDeltaNet layer already computes `g = -exp(aLog) * softplus(a +
/// dt_bias)` externally (the same formula Pallas's `use_gate_in_kernel=true`
/// branch evaluates internally), so this is the natural fit for the initial
/// integration — no layer-forward refactor needed.
///
/// Cfg passed to `Kernel.call(.., .{ .cfg = ... })` MUST have
/// `use_gate_in_kernel = false` (otherwise `run()` emits the gate-arms and the
/// emitted func.func signature won't match these `inputs`).
pub const KernelNoGate = zml.kernel.mosaic_tpu.Kernel(Cfg, .{
    .name = "decoding_gdn-bt_8",
    .inputs = &.{ "q", "k", "v", "g", "b_in", "state_indices", "distribution", "initial_state" },
    .outputs = &.{ "o", "state_out" },
    .run = run,
});

/// Default alias — kept for back-compat with code that previously used the
/// `Kernel` symbol. New integrations should pick `KernelFull` or `KernelNoGate`
/// explicitly to make the cfg/operand match crystal clear.
pub const Kernel = KernelFull;
