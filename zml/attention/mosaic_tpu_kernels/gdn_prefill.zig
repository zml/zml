//! GDN chunked-prefill + recurrent-decode Mosaic-TPU kernel ("P5") — Zig DSL
//! conversion of `recurrent_scan` (tpu-inference `recurrent_scan_v2.py`). Paired
//! with `kernels/mosaic_tpu/py/gdn_prefill_scan.py` (the verbatim Pallas
//! reference the harness diffs against). Goal: emit Mosaic IR that matches
//! what `pl.pallas_call` produces for that reference, op-for-op.
//!
//! Complete and wired into llmd for Qwen3.5-2B/9B (P4 handles pure-decode buckets;
//! P5 handles prefill and mixed via the schedule table). The harness diffs against
//! the Pallas reference for IR parity; functional equivalence is validated end-to-end
//! on TPU via the llmd Qwen3.5 path.
//!
//! Body shape — four dispatch arms inside the `emit_pipeline` inner kernel,
//! gated by a 10-column preamble of schedule-table reads:
//!  - **Decode** (`decode_valid > 0`): `scf.for b = 0..BT` with per-row
//!    gather-DMA of state, mask-extract qkv row, SiLU, L2-norm + GQA + scale,
//!    NaN-safe softplus gate, unrolled per-head `tpu.matmul`, state writeback +
//!    scatter-DMA, masked output write.
//!  - **Prefill regular** (`is_transition == 0`): WY/UT chunked update — qkv
//!    silu + extract, β/gate transform, L2-norm + GQA + scale, unrolled
//!    cumsum-over-C, `S = matmul(k_beta, k^T)`, `A_inv` via unrolled
//!    block-Gaussian forward-sub (`invert_triangular_matrix`, blk=16), the
//!    6 remaining batched matmuls (u/w/attn/v_prime/term2), `h_new` writeback,
//!    `@pl.when(is_last_chunk)` state commit, masked output write.
//!  - **Prefill transition** (`is_transition > 0`): 16-unrolled per-token loop
//!    with `is_new_seq`/`seq_valid` stateful tracking, scratch ping-pong via
//!    `vector.load/where-select/vector.store`, conditional STORE / LOAD DMAs
//!    at sequence boundaries, per-token recurrent update, final state commit
//!    + gather-scatter into `recurrent_state_out[state_indices[current_r]]`.
//!  - **Stitch** (`is_transition > 0 & pid == 0 & decode_valid > 0`):
//!    `iota<split` mask + select-merge between the decode overlap and the
//!    prefill output, vector_store back to both windows.
//!
//! `pipeline.zig` extensions this kernel introduced: `PipeSpec.
//! bounded_slice_tiling` (P5's row-dim bounded slice is tiling-8, not 1 like
//! P4's token dim) + generalized `_create_bounded_slice` OOB arith for tiling
//! > 1; `memRefAlloca` / `semAllocOp` / `dmaSemRefTy` made `pub`. DSL fixes in
//! `mlir/dialects/{vector,mosaic_tpu/mosaic_tpu}.zig`:
//! `extract_strided_slice` / `insert_strided_slice` / `tpu.transpose` use
//! `ArrayAttr<IntegerAttr>` / `DenseI64ArrayAttr` (was `DenseI32ArrayAttr` —
//! verifier rejected).
//!
//! Dispatch: `num_v_heads` is comptime-dispatched over `SUPPORTED_N_V` (2..128,
//! including Qwen3.5-9B's n_v=32); `chunk_size` (multiple of 16 for the block-Gaussian
//! `invert_triangular_matrix`) is a runtime `Cfg` field. Used in production with
//! chunk_size=64 for both 2B and 9B.
//!
//! Reference for the kernel math / structure: GDN-Google/PORTING_PLAYBOOK.md
//! (Part C for the IR-match methodology), GDN-Google/IMPLEMENTATIONS.md (P5 in
//! the building-block table), GDN-Google/tpu_inference/kernels/gdn/
//! recurrent_scan_v2.py + compute_schedule_v2.py (verbatim source).

const std = @import("std");

const mlir = @import("mlir");
const zml = @import("../../zml.zig");
const mtt = @import("kernels/mosaic_tpu/builder");
const dialects = @import("mlir/dialects");
const tpu = @import("mlir/dialects/mosaic_tpu");
const scf = dialects.scf;

const Pipeline = mtt.pipeline;
const BlockIndex = Pipeline.BlockIndex;
const Value = mtt.Value;

/// Comptime-dispatch arm values for `cfg.num_v_heads` in the decode arm's
/// per-head matmul loop. Each value here generates a fully-unrolled MLIR
/// branch with exactly `N_V` `tpu.matmul` ops. The `inline for + if` pattern
/// (and the `inline switch` form below) is REQUIRED — Mosaic-TPU's MLIR can't
/// loop over heads at runtime, so each head's `vectorExtract` / `matmul` /
/// `select` must be emitted at trace time with a comptime `h` index. Adding
/// a value here just costs comptime branch budget (bumped via
/// `@setEvalBranchQuota`); it doesn't change the runtime kernel — only one
/// arm fires per `Cfg.num_v_heads`.
const SUPPORTED_N_V = .{
    @as(i64, 2), // smallest GQA test config
    @as(i64, 4),
    @as(i64, 8),
    @as(i64, 16), // Qwen3.5-2B (n_kq=16, no GQA)
    @as(i64, 32), // Qwen3.5-9B (n_kq=16, GQA repeat=2)
    @as(i64, 48), // Qwen3.5-9B
    @as(i64, 64), // larger GDN models
    @as(i64, 128), // P4 baseline / 30B-class configs
};

// =============================================================================
// Config — field names align with `py/gdn_prefill_scan.py::_prefill_shapes` cfg keys.
// =============================================================================

pub const Cfg = struct {
    /// mixed_qkv / output element dtype (activations come in as this; upcast to f32 in-kernel).
    dtype: mtt.DType = .bf16,
    /// recurrent-state cache dtype (bf16 on TPU; halves HBM).
    state_dtype: mtt.DType = .bf16,
    /// q/k heads.
    num_k_heads: i64 = 1,
    /// v heads (multiple of num_k_heads → GQA; the kernel jnp.repeats q/k).
    num_v_heads: i64 = 2,
    /// key/query head dim (multiple of num_lanes = 128).
    head_k_dim: i64 = 128,
    /// value head dim (multiple of 128).
    head_v_dim: i64 = 128,
    /// v2 chunk size; BT (decode batch) == chunk_size. Must be a multiple of 16
    /// (`invert_triangular_matrix` block_size). Drives the unrolled cumsum /
    /// triangular-inverse loops.
    chunk_size: i64 = 16,
    /// total scheduled tokens this step (>= num_seqs). Only affects array shapes
    /// and the schedule-table first dim (values never enter the IR).
    num_tokens: i64 = 50,
    /// number of requests (decode + prefill).
    num_seqs: i64 = 4,
    /// recurrent-state pool size (≥ num_seqs + 1). null → num_seqs + 1.
    num_blocks: ?i64 = null,

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
    pub fn keyDim(self: Cfg) i64 {
        return self.num_k_heads * self.head_k_dim;
    }
    pub fn dim(self: Cfg) i64 {
        return 2 * self.keyDim() + self.num_v_heads * self.head_v_dim;
    }
    pub fn numBlocks(self: Cfg) i64 {
        return self.num_blocks orelse (self.num_seqs + 1);
    }
    /// `sublanesize = (4 // itemsize) * num_sublanes`; num_sublanes = 8 on v5e.
    /// bf16/f16 → 16, f32 → 8, i8 → 32.
    pub fn sublanesize(self: Cfg) i64 {
        const itemsize: i64 = switch (self.dtype) {
            .bf16, .f16 => 2,
            .f32 => 4,
            .i8 => 1,
            else => 2,
        };
        return @divTrunc(4, itemsize) * 8;
    }
    /// `block_size = max(chunk_size, BT)`; here BT == chunk_size.
    pub fn blockSize(self: Cfg) i64 {
        return self.chunk_size;
    }
    /// `sink_offset = ceil(num_tokens / sublanesize) * sublanesize`.
    pub fn sinkOffset(self: Cfg) i64 {
        const s = self.sublanesize();
        return @divTrunc(self.num_tokens + s - 1, s) * s;
    }
    /// padded token rows = sink_offset + block_size — the first dim of the
    /// (in-kernel) padded mixed_qkv / a_padded / b_padded / output arrays.
    pub fn paddedRows(self: Cfg) i64 {
        return self.sinkOffset() + self.blockSize();
    }
    /// `safe_max_blocks = ceil(num_tokens / chunk_size) + num_seqs * 2` — first
    /// dim of the schedule table.
    pub fn safeMaxBlocks(self: Cfg) i64 {
        return @divTrunc(self.num_tokens + self.chunk_size - 1, self.chunk_size) + self.num_seqs * 2;
    }
    /// schedule-table column count = 11 + 3 * alignment, alignment = sublanesize.
    pub fn scheduleCols(self: Cfg) i64 {
        return 11 + 3 * self.sublanesize();
    }
};

// =============================================================================
// Kernel body
// =============================================================================

pub fn run(b: *mtt.Builder, cfg: Cfg) mtt.FinishError!void {
    // The decode arm's per-head matmul loop is fully unrolled at trace time;
    // expanding `SUPPORTED_N_V` to {2,4,8,16,32,64,128} pushes Zig past the
    // default 1000 backwards-branch comptime budget. Bump generously — the
    // limit is a comptime-eval throttle, no runtime cost.
    @setEvalBranchQuota(100_000);

    const k = b;

    const dim = cfg.dim();
    const n_v = cfg.nV();
    const d_k = cfg.dK();
    const d_v = cfg.dV();
    const nb = cfg.numBlocks();
    const num_seqs = cfg.num_seqs;
    const padded_rows = cfg.paddedRows();
    const smb = cfg.safeMaxBlocks();
    const sched_cols = cfg.scheduleCols();

    // `pl.pallas_call` in `recurrent_scan`:
    //   grid=(1,);  in_specs (11): mixed_qkv(HBM), recurrent_state(HBM),
    //     state_indices(SMEM), has_initial_state(SMEM), a_padded(HBM), b_padded(HBM),
    //     A_log(HBM), dt_bias(HBM), schedule_table(SMEM),
    //     decode_tokens_arr (BlockSpec (1,) idx_map ->(0,)), total_blocks_arr (same);
    //   out_specs (2): recurrent_state_out(HBM), output(HBM, (sink_offset+block_size, n_v*d_v));
    //   input_output_aliases={1:0}; scratch via run_scoped inside the body.
    //   ⇒ func args: %pid + 11 ins + 2 outs = 14.  Best-guess signature; the
    //   memory_space / windowing on decode_tokens_arr / total_blocks_arr and the
    //   window_params layout will be corrected against the baseline diff.
    const a = try k.declareArgsOpts(.{
        .pid = .{ .scalar = .i32 },

        // mixed_qkv / recurrent_state / a_padded / b_padded / A_log / dt_bias come
        // in via `pltpu.HBM` ⇒ `#tpu.memory_space<hbm>` (the kernel does its own DMAs).
        .mixed_qkv = .{ .ref = .{ .shape = &.{ padded_rows, dim }, .dtype = cfg.dtype, .memory_space = .hbm } },
        .recurrent_state = .{ .ref = .{ .shape = &.{ nb, n_v, d_k, d_v }, .dtype = cfg.state_dtype, .memory_space = .hbm } },
        // state_indices / has_initial_state / schedule_table come in via `pltpu.SMEM`
        // (read via `memref.load` in index maps + the kernel body).
        .state_indices = .{ .ref = .{ .shape = &.{num_seqs}, .dtype = .i32, .memory_space = .smem } },
        .has_initial_state = .{ .ref = .{ .shape = &.{num_seqs}, .dtype = .i32, .memory_space = .smem } },
        .a_padded = .{ .ref = .{ .shape = &.{ padded_rows, 128 }, .dtype = .f32, .memory_space = .hbm } },
        .b_padded = .{ .ref = .{ .shape = &.{ padded_rows, 128 }, .dtype = .f32, .memory_space = .hbm } },
        .a_log = .{ .ref = .{ .shape = &.{n_v}, .dtype = .f32, .memory_space = .hbm } },
        .dt_bias = .{ .ref = .{ .shape = &.{n_v}, .dtype = .f32, .memory_space = .hbm } },
        .schedule_table = .{ .ref = .{ .shape = &.{ smb, sched_cols }, .dtype = .i32, .memory_space = .smem } },
        // decode_tokens_arr / total_blocks_arr come in via `pl.BlockSpec(block_shape=(1,),
        // index_map=lambda _: (0,))` ⇒ a trivially-windowed VMEM `(1,)` ref (block ==
        // operand shape, identity index map) ⇒ Buffered(1) ⇒ `pipeline_mode=synchronous`.
        .decode_tokens_arr = .{ .ref = .{ .shape = &.{1}, .dtype = .i32, .window = mtt.ArgSpec.WindowSpec{ .block_shape = &.{1}, .transform_returns = &.{.zero} } } },
        .total_blocks_arr = .{ .ref = .{ .shape = &.{1}, .dtype = .i32, .window = mtt.ArgSpec.WindowSpec{ .block_shape = &.{1}, .transform_returns = &.{.zero} } } },

        // Outputs (Pallas `out_shape = [recurrent_state, output_padded]`) — both `pltpu.HBM`.
        .recurrent_state_out = .{ .ref = .{ .shape = &.{ nb, n_v, d_k, d_v }, .dtype = cfg.state_dtype, .memory_space = .hbm, .role = .output } },
        .output = .{ .ref = .{ .shape = &.{ padded_rows, n_v * d_v }, .dtype = cfg.dtype, .memory_space = .hbm, .role = .output } },
    }, &.{}, .{
        .dimension_semantics = &.{.arbitrary},
        // grid=(1,) is a static bound.
        .iteration_bounds = &.{1},
        .scalar_prefetch = 0,
        .scratch_operands = 0,
        .pallas_window_params = true,
    });
    const C = cfg.chunk_size; // == BT
    const sink_off = cfg.sinkOffset();
    const align16 = cfg.sublanesize(); // `alignment` arg of compute_schedule_table_v2 / pl.multiple_of
    const out_cols = n_v * d_v;
    const bt = C; // decode-batch size (== chunk_size)

    // ── decode_tokens = decode_tokens_arr[0];  total_blocks = total_blocks_arr[0] ──
    //   A `(1,)` VMEM ref read at `[0]` lowers to `vector.load … : vector<1xi32>`
    //   then `vector.extract … [0]` (`%0..%3` in the target).
    const decode_tokens = k.vectorExtract(k.refLoad(a.decode_tokens_arr), &.{0});
    const total_blocks = k.vectorExtract(k.refLoad(a.total_blocks_arr), &.{0});

    // Per-block ctx threaded to the inner kernel + the index maps.
    var ctx = PipeCtx{
        .schedule_table = a.schedule_table,
        .state_indices = a.state_indices,
        .has_initial_state = a.has_initial_state,
        .recurrent_state_in = a.recurrent_state,
        .recurrent_state_out = a.recurrent_state_out,
        .decode_tokens = decode_tokens,
        .n_kq = cfg.nKq(),
        .n_v = n_v,
        .d_k = d_k,
        .d_v = d_v,
        .C = C,
        .bt = bt,
        .num_seqs = num_seqs,
        .sink_offset = sink_off,
        .alignment = align16,
        .dtype = cfg.dtype,
        .state_dtype = cfg.state_dtype,
        // run_scoped scratches — filled below once allocated.
        .prefill_scratch = undefined,
        .decode_state_scratch = undefined,
        .state_commit_scratch = undefined,
        .decode_output_scratch = undefined,
        .decode_read_sems = undefined,
        .decode_write_sem = undefined,
        .prefill_sem = undefined,
    };

    // ── `pl.run_scoped(_run_with_scratch, VMEM(...)×4, DMA-sem(1), DMA-sem(1), DMA-sem(2))` ──
    //   ⇒ `"tpu.region"() ({ %alloca×4 ; %sem×3 ; <emit_pipeline> ; tpu.yield })`.
    const vmem = mtt.MemorySpace.vmem.attribute(k.ctx);
    const f32_ty = mtt.DType.f32.toMlir(k.ctx);
    const region_block = mlir.Block.init(&.{}, &.{});
    k.pushBlock(region_block);
    {
        // run_scoped allocations, in the order `pl.run_scoped` lists them:
        // prefill_scratch (2,n_v,d_k,d_v) f32 (double-buffered) → decode_state_scratch
        // (1,n_v,d_k,d_v) f32 → state_commit_scratch (1,n_v,d_k,d_v) state_dtype →
        // decode_output_scratch (bt,n_v*d_v) dtype → decode_read_sems DMA(1) →
        // decode_write_sem DMA(1) → prefill_sem DMA(2).
        ctx.prefill_scratch = Pipeline.memRefAlloca(k, mlir.Type.memRef(f32_ty, &.{ 2, n_v, d_k, d_v }, null, vmem));
        ctx.decode_state_scratch = Pipeline.memRefAlloca(k, mlir.Type.memRef(f32_ty, &.{ 1, n_v, d_k, d_v }, null, vmem));
        ctx.state_commit_scratch = Pipeline.memRefAlloca(k, mlir.Type.memRef(cfg.state_dtype.toMlir(k.ctx), &.{ 1, n_v, d_k, d_v }, null, vmem));
        ctx.decode_output_scratch = Pipeline.memRefAlloca(k, mlir.Type.memRef(cfg.dtype.toMlir(k.ctx), &.{ bt, out_cols }, null, vmem));
        ctx.decode_read_sems = Pipeline.semAllocOp(k, Pipeline.dmaSemRefTy(k, 1));
        ctx.decode_write_sem = Pipeline.semAllocOp(k, Pipeline.dmaSemRefTy(k, 1));
        ctx.prefill_sem = Pipeline.semAllocOp(k, Pipeline.dmaSemRefTy(k, 2));

        // ── `pltpu.emit_pipeline(inner_kernel, grid=(total_blocks,), in_specs, out_specs)(...)`. ──
        //   8 pipelined inputs (mixed_qkv prefill+decode windows, a_raw prefill+decode,
        //   b_raw prefill+decode, a_log trivial, dt_bias trivial) + 2 outputs (output
        //   prefill+decode windows). Index maps = `get_qkv_index_map_v2` (reads
        //   schedule_table[step, valid_col / offset_col], `pl.multiple_of` 16,
        //   `where(valid>0, offset, sink_offset)`); the BoundedSlice dim's tiling is 8.
        const qkv_block = [_]Pipeline.BlockDim{ .{ .bounded_slice = C }, .{ .blocked = dim } };
        const raw_block = [_]Pipeline.BlockDim{ .{ .bounded_slice = C }, .{ .blocked = 128 } };
        const out_block = [_]Pipeline.BlockDim{ .{ .bounded_slice = C }, .{ .blocked = out_cols } };
        const mixed_p = Pipeline.PipeSpec{ .full_shape = &.{ padded_rows, dim }, .dtype = cfg.dtype, .block_shape = &qkv_block, .index_map = prefillIdxMap, .index_map_ctx = &ctx, .bounded_slice_tiling = 8 };
        const mixed_d = Pipeline.PipeSpec{ .full_shape = &.{ padded_rows, dim }, .dtype = cfg.dtype, .block_shape = &qkv_block, .index_map = decodeIdxMap, .index_map_ctx = &ctx, .bounded_slice_tiling = 8 };
        const a_p = Pipeline.PipeSpec{ .full_shape = &.{ padded_rows, 128 }, .dtype = .f32, .block_shape = &raw_block, .index_map = prefillIdxMap, .index_map_ctx = &ctx, .bounded_slice_tiling = 8 };
        const a_d = Pipeline.PipeSpec{ .full_shape = &.{ padded_rows, 128 }, .dtype = .f32, .block_shape = &raw_block, .index_map = decodeIdxMap, .index_map_ctx = &ctx, .bounded_slice_tiling = 8 };
        const b_p = Pipeline.PipeSpec{ .full_shape = &.{ padded_rows, 128 }, .dtype = .f32, .block_shape = &raw_block, .index_map = prefillIdxMap, .index_map_ctx = &ctx, .bounded_slice_tiling = 8 };
        const b_d = Pipeline.PipeSpec{ .full_shape = &.{ padded_rows, 128 }, .dtype = .f32, .block_shape = &raw_block, .index_map = decodeIdxMap, .index_map_ctx = &ctx, .bounded_slice_tiling = 8 };
        const a_log_s = Pipeline.PipeSpec{ .full_shape = &.{n_v}, .dtype = .f32 }; // trivial window
        const dt_bias_s = Pipeline.PipeSpec{ .full_shape = &.{n_v}, .dtype = .f32 };
        const out_p = Pipeline.PipeSpec{ .full_shape = &.{ padded_rows, out_cols }, .dtype = cfg.dtype, .block_shape = &out_block, .index_map = prefillIdxMap, .index_map_ctx = &ctx, .bounded_slice_tiling = 8 };
        const out_d = Pipeline.PipeSpec{ .full_shape = &.{ padded_rows, out_cols }, .dtype = cfg.dtype, .block_shape = &out_block, .index_map = decodeIdxMap, .index_map_ctx = &ctx, .bounded_slice_tiling = 8 };

        const in_specs = [_]?Pipeline.PipeSpec{ mixed_p, mixed_d, a_p, a_d, b_p, b_d, a_log_s, dt_bias_s };
        const out_specs = [_]?Pipeline.PipeSpec{ out_p, out_d };
        const refs = [_]?Value{ a.mixed_qkv, a.mixed_qkv, a.a_padded, a.a_padded, a.b_padded, a.b_padded, a.a_log, a.dt_bias, a.output, a.output };
        const scratches = [_]Value{ a.schedule_table, a.state_indices, a.has_initial_state };
        try Pipeline.emitPipeline(k, &refs, &scratches, .{
            .body = innerKernel,
            .body_ctx = &ctx,
            .grid = &.{total_blocks},
            .in_specs = &in_specs,
            .out_specs = &out_specs,
            .trace_scopes = true,
        });
    }
    _ = tpu.yield(k.ctx, &.{}, k.loc()).appendTo(region_block);
    k.popBlock();
    _ = tpu.region(k.ctx, region_block, &.{}, k.loc()).appendTo(k.currentBlock());
}

// =============================================================================
// emit_pipeline glue — index maps + the per-block ctx + the (stub) inner kernel.
// =============================================================================

/// Threaded to the index maps and `innerKernel`. (Zig has no closures — this is
/// `functools.partial`'s captured kwargs, by hand.)
const PipeCtx = struct {
    schedule_table: Value, // SMEM (safe_max_blocks, 11+3*alignment) i32
    state_indices: Value, // SMEM (num_seqs,) i32
    has_initial_state: Value, // SMEM (num_seqs,) i32
    recurrent_state_in: Value, // HBM (num_blocks, n_v, d_k, d_v)
    recurrent_state_out: Value, // HBM "
    decode_tokens: Value, // i32 scalar = distribution[0]
    // run_scoped scratches (VMEM allocas + DMA sem pools)
    prefill_scratch: Value, // (2, n_v, d_k, d_v) f32  (double-buffered)
    decode_state_scratch: Value, // (1, n_v, d_k, d_v) f32
    state_commit_scratch: Value, // (1, n_v, d_k, d_v) state_dtype
    decode_output_scratch: Value, // (bt, n_v*d_v) dtype
    decode_read_sems: Value, // memref<1 x dma_sem, sem_mem>
    decode_write_sem: Value, // memref<1 x dma_sem, sem_mem>
    prefill_sem: Value, // memref<2 x dma_sem, sem_mem>
    n_kq: i64,
    n_v: i64,
    d_k: i64,
    d_v: i64,
    C: i64,
    bt: i64,
    num_seqs: i64,
    sink_offset: i64,
    alignment: i64, // = sublanesize (16 for bf16)
    dtype: mtt.DType,
    state_dtype: mtt.DType,
};

/// `get_qkv_index_map_v2(step, schedule_table, valid_col, offset_col, ...,
///  alignment, block_size, sink_offset)` ⇒ returns `(pl.ds(safe_offset, block_size), 0)`:
///   `offset   = pl.multiple_of(schedule_table[step, offset_col], alignment)`
///   `safe_off = pl.multiple_of(jnp.where(schedule_table[step, valid_col] > 0, offset, sink_offset), alignment)`
/// emitted op order (matching JAX tracing): load(valid) → load(offset) →
/// assume_multiple(offset) → cmpi sgt(valid, 0) → select → assume_multiple → ...
fn qkvIdxMap(b: *mtt.Builder, indices: []const Value, ctx_: ?*anyopaque, valid_col: i64, offset_col: i64) []const BlockIndex {
    const ctx: *PipeCtx = @ptrCast(@alignCast(ctx_.?));
    const step_i = b.toIndex(indices[0]);
    const valid = b.scalarLoad(ctx.schedule_table, &.{ step_i, b.cIndex(valid_col) });
    const offset_raw = b.scalarLoad(ctx.schedule_table, &.{ step_i, b.cIndex(offset_col) });
    const offset = b.emit(tpu.assume_multiple(b.ctx, offset_raw.inner, @intCast(ctx.alignment), b.loc()));
    const safe_pre = b.select(b.cmpi(.sgt, valid, b.lift(@as(i32, 0))), offset, b.lift(@as(i32, @intCast(ctx.sink_offset))));
    const safe = b.emit(tpu.assume_multiple(b.ctx, safe_pre.inner, @intCast(ctx.alignment), b.loc()));
    const out = b.arena.allocator().alloc(BlockIndex, 2) catch unreachable;
    out[0] = .{ .slice = .{ .start = safe, .size = b.lift(@as(i32, @intCast(ctx.C))) } };
    out[1] = .{ .scalar = b.lift(@as(i32, 0)) }; // dim 1 (the `0` in `(pl.ds(...), 0)`) — folds 0*dim → 0
    return out;
}

/// prefill BlockSpec index map: valid_col=0 (prefill_valid), offset_col=1 (block_offset).
fn prefillIdxMap(b: *mtt.Builder, indices: []const Value, ctx_: ?*anyopaque) []const BlockIndex {
    return qkvIdxMap(b, indices, ctx_, 0, 1);
}
/// decode BlockSpec index map: valid_col=4 (decode_valid), offset_col=5 (decode_offset).
fn decodeIdxMap(b: *mtt.Builder, indices: []const Value, ctx_: ?*anyopaque) []const BlockIndex {
    return qkvIdxMap(b, indices, ctx_, 4, 5);
}

/// `inner_kernel(*ep_refs, *ep_scratches, recurrent_state_in, recurrent_state_out,
/// + run_scoped scratches via ctx)` — called once per `emit_pipeline` step. The
/// step's schedule-table row determines which of four paths run (each emits a
/// `scf.if` regardless of runtime predicate, so the IR is the union):
///   - `@pl.when(decode_valid > 0)` decode batch
///   - `@pl.when(prefill_valid > 0)` ⇒ `lax.cond(is_transition, transition, regular)`
///   - `lax.cond(needs_stitching, do_stitch, noop)` (transition + first_block + decode_valid)
/// Bodies are partially-filled stubs as of 2026-05-13.
fn innerKernel(b: *mtt.Builder, grid_indices: []const Value, refs: []const Pipeline.RefWindow, scratches: []const Value, ctx_: ?*anyopaque) mtt.FinishError!void {
    // The decode arm dispatches over `SUPPORTED_N_V = {2,4,8,16,32,64,128}` and
    // the prefill arm unrolls the `invert_triangular_matrix` block-Gaussian
    // forward-sub (16 inner iters × 4 outer blocks). Both blow past Zig's
    // default 1000 backwards-branch comptime budget. Bump generously — the
    // limit is a comptime-eval throttle, no runtime cost.
    @setEvalBranchQuota(200_000);

    const ctx: *PipeCtx = @ptrCast(@alignCast(ctx_.?));
    const k = b;
    // refs: [mixed_qkv prefill, mixed_qkv decode, a_raw prefill, a_raw decode,
    //        b_raw prefill, b_raw decode, a_log (trivial), dt_bias (trivial),
    //        output prefill, output decode].
    const sched = scratches[0];
    const has_init_ref = scratches[2];

    const step = grid_indices[0]; // i32
    const step_i = k.toIndex(step); // CSEs with the outer-loop's `%225 = index_cast(grid_idx)` if hoist allows.
    const c0i = k.lift(@as(i32, 0));

    // ── Schedule-table reads (10 cols), in `inner_kernel`'s source order:
    //     col 0 prefill_valid, col 2 prefill_req_id, col 4 decode_valid, col 5 decode_offset,
    //     col 6 decode_req_id, col 7 decode_count, col 1 prefill_offset, col 10 is_transition,
    //     col 8 is_last_chunk, col 9 is_first_chunk.
    const prefill_valid = k.scalarLoad(sched, &.{ step_i, k.cIndex(0) });
    const prefill_req_id = k.scalarLoad(sched, &.{ step_i, k.cIndex(2) });
    const decode_valid = k.scalarLoad(sched, &.{ step_i, k.cIndex(4) });
    const decode_offset = k.scalarLoad(sched, &.{ step_i, k.cIndex(5) });
    const decode_req_id = k.scalarLoad(sched, &.{ step_i, k.cIndex(6) });
    const decode_count = k.scalarLoad(sched, &.{ step_i, k.cIndex(7) });
    const prefill_offset = k.scalarLoad(sched, &.{ step_i, k.cIndex(1) });
    const is_transition = k.scalarLoad(sched, &.{ step_i, k.cIndex(10) });
    const is_last_chunk = k.scalarLoad(sched, &.{ step_i, k.cIndex(8) });
    const is_first_chunk = k.scalarLoad(sched, &.{ step_i, k.cIndex(9) });

    // ── (1) Decode path: `@pl.when(decode_valid > 0)`. ──
    //   `fori_loop(0, BT, process_decode, None)` — a real `scf.for` (not unrolled).
    //   Inside: `@pl.when(b < decode_count)`: gather state → state_commit → upcast to
    //   decode_state f32; mask-extract row b of decode_qkv/a_raw/b_raw; SiLU; L2-norm
    //   q,k; GQA-repeat; ×scale; gate; per-head `pl.dot` × n_v unrolled;
    //   gather-store new state via state_commit; mask-merge out into decode_output_scratch.
    //   After the loop: mask `arange(BT) < decode_count` and write decode_output_scratch
    //   → decode_output_ref.
    {
        var when = k.openIf(k.cmpi(.sgt, decode_valid, c0i));
        {
            const i32_ty = k.scalarTy(.i32);
            const for_body = mlir.Block.init(&.{i32_ty}, &.{k.loc()});
            k.pushBlock(for_body);
            {
                const b_iv: Value = .{ .inner = for_body.argument(0), .kernel = k };
                var when_b = k.openIf(k.cmpi(.slt, b_iv, decode_count));
                {
                    // do_work():
                    //   safe_req_id = min(decode_req_id + b, num_seqs - 1)
                    //   target_idx = state_indices[safe_req_id]
                    //   sync_copy(recurrent_state_in[ds(target_idx,1)] → state_commit_scratch, sem=decode_read_sems[0])
                    //   decode_state_scratch[0:1] = state_commit_scratch[...].astype(f32)
                    const safe_req_id = k.minsi(k.addi(decode_req_id, b_iv), k.lift(@as(i32, @intCast(ctx.num_seqs - 1))));
                    const safe_idx = k.toIndex(safe_req_id);
                    const target_idx = k.scalarLoad(scratches[1], &.{safe_idx}); // state_indices[safe_req_id]
                    const state_slice = k.memRefSlice(ctx.recurrent_state_in, &.{ target_idx, c0i, c0i, c0i }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v }, &.{});
                    // decode_read_sems is `memref<1 x dma_sem>` — squeezes directly to scalar `!dma_sem` (no slice needed).
                    const sem = k.memRefSqueeze(ctx.decode_read_sems, &.{});
                    k.enqueueDma(state_slice, ctx.state_commit_scratch, sem, .{});
                    k.waitDma2(sem, state_slice, ctx.state_commit_scratch, .{});
                    const c0idx = k.cIndex(0);
                    const loaded = k.vectorLoadShape(ctx.state_commit_scratch, &.{ c0idx, c0idx, c0idx, c0idx }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });
                    const upcast = loaded.to(.f32);
                    k.vectorStoreAt(ctx.decode_state_scratch, upcast, &.{ c0idx, c0idx, c0idx, c0idx });

                    // b_aligned = (b // sublanesize) * sublanesize — emitted as the
                    // explicit floor-div sign-fixup (`divsi` + `sign(b) != 1` check) +
                    // `muli`, matching `lax.floor_div`'s lowering.
                    const sl = k.lift(@as(i32, @intCast(ctx.alignment))); // sublanesize = 16
                    const q = k.divsi(b_iv, sl);
                    const b_pos = k.extui(k.cmpi(.sgt, b_iv, c0i), .i32);
                    const b_neg = k.extui(k.cmpi(.slt, b_iv, c0i), .i32);
                    const sign_b = k.subi(b_pos, b_neg);
                    const sign_diff = k.cmpi(.ne, sign_b, k.lift(@as(i32, 1)));
                    const rem_b = k.remsi(b_iv, sl);
                    const rem_nz = k.cmpi(.ne, rem_b, c0i);
                    const needs_fixup = k.andi(sign_diff, rem_nz);
                    const q_minus1 = k.subi(q, k.lift(@as(i32, 1)));
                    const floor_q = k.select(needs_fixup, q_minus1, q);
                    const b_aligned = k.muli(floor_q, sl);

                    // decode_qkv_ref[ds(b_aligned, sublanesize), :] — re-emit the
                    // memref_slice/squeeze of refs[1] here (per-use-site, see
                    // pallas-ir-matching-gotchas C2).
                    const dim_v: i64 = 2 * ctx.n_kq * ctx.d_k + ctx.n_v * ctx.d_v; // == 512 in `small`
                    const dec_qkv_w = refs[1];
                    const dec_qkv_slot = dec_qkv_w.slot.?;
                    const dec_qkv_tile = k.memRefSqueeze(
                        k.memRefSlice(dec_qkv_w.buf.?, &.{ dec_qkv_slot, c0i, c0i }, &.{ 1, ctx.bt, dim_v }, &.{}),
                        &.{ ctx.bt, dim_v },
                    );
                    // Pallas: `decode_qkv_ref[pl.ds(b_aligned, sublanesize), :]` — reads
                    //   `sublanesize` rows (= `alignment`), NOT `bt` rows.  For chunk_size
                    //   == sublanesize (==16) they coincide, but for chunk_size > 16 the
                    //   read must still be sublanesize-tall.
                    const b_aligned_idx = k.toIndex(b_aligned);
                    const qkv_blk_bf = k.vectorLoadShape(dec_qkv_tile, &.{ b_aligned_idx, c0idx }, &.{ ctx.alignment, dim_v });
                    const qkv_blk = qkv_blk_bf.to(.f32);

                    // mask = (arange(sublanesize) == (b % sublanesize)).astype(qkv_blk.dtype)[:, None]
                    //   via tpu.iota along dim 1 of shape (1, sublanesize) → shape_cast to (sublanesize,)
                    //   then compare against broadcast of floor_mod(b, sublanesize) → uitofp to f32
                    //   → shape_cast to (sublanesize, 1) → broadcast to (sublanesize, dim).
                    const iota_2d = k.iota(&.{ 1, ctx.alignment }, .i32, &.{1});
                    const iota_1d = k.shapeCast(iota_2d, &.{ctx.alignment});
                    const rem_neg = k.cmpi(.slt, rem_b, c0i);
                    const rem_needs_fixup = k.andi(rem_neg, rem_nz);
                    const rem_plus = k.addi(rem_b, sl);
                    const floor_mod_b = k.select(rem_needs_fixup, rem_plus, rem_b);
                    const fm_bc = k.broadcastTo(floor_mod_b, &.{ctx.alignment});
                    const mask_i1 = k.cmpi(.eq, iota_1d, fm_bc);
                    const mask_f = mask_i1.to(.f32);
                    const mask_2d = k.shapeCast(mask_f, &.{ ctx.alignment, 1 });
                    const mask_full = k.broadcastTo(mask_2d, &.{ ctx.alignment, dim_v });
                    const qkv_masked = k.mulf(qkv_blk, mask_full);
                    const dim_zeros = k.zeros(&.{dim_v}, .f32);
                    const qkv_summed = k.multiReduction(.add, qkv_masked, dim_zeros, &.{0});
                    const qkv_row = k.shapeCast(qkv_summed, &.{ 1, dim_v });

                    // Fused SiLU on the (1, dim) row: silu(x) = x * sigmoid(x).
                    const ones_1xdim = k.full(&.{ 1, dim_v }, @as(f32, 1.0), .f32);
                    const neg_x = k.negf(qkv_row);
                    const exp_neg = k.exp(neg_x);
                    const denom = k.addf(exp_neg, ones_1xdim);
                    const sig = k.divf(ones_1xdim, denom);
                    const silu = k.mulf(qkv_row, sig);

                    // Extract q [0:key_dim], k [key_dim:2*key_dim], v [2*key_dim:dim] from
                    // the silu'd row; reshape v to (n_v, d_v).
                    const key_dim = ctx.n_kq * ctx.d_k;
                    const v_cols = ctx.n_v * ctx.d_v;
                    const q_ty = k.vectorTy(&.{ 1, key_dim }, .f32);
                    const k_ty = k.vectorTy(&.{ 1, key_dim }, .f32);
                    const v_ty = k.vectorTy(&.{ 1, v_cols }, .f32);
                    const dialects_v = dialects.vector;
                    // Pallas interleaves extract+shape_cast per row (q → q_resh, k → k_resh,
                    //   v → v_resh) — emit in that order so the per-row reshape lands at
                    //   Pallas's diff position.
                    const q_row = k.emit(dialects_v.extract_strided_slice(k.ctx, silu.inner, .{ .offsets = &.{ 0, 0 }, .sizes = &.{ 1, key_dim }, .strides = &.{ 1, 1 } }, q_ty, k.loc()));
                    const q_resh = k.shapeCast(q_row, &.{ ctx.n_kq, ctx.d_k });
                    const k_row = k.emit(dialects_v.extract_strided_slice(k.ctx, silu.inner, .{ .offsets = &.{ 0, key_dim }, .sizes = &.{ 1, key_dim }, .strides = &.{ 1, 1 } }, k_ty, k.loc()));
                    const k_resh = k.shapeCast(k_row, &.{ ctx.n_kq, ctx.d_k });
                    const v_row = k.emit(dialects_v.extract_strided_slice(k.ctx, silu.inner, .{ .offsets = &.{ 0, 2 * key_dim }, .sizes = &.{ 1, v_cols }, .strides = &.{ 1, 1 } }, v_ty, k.loc()));
                    const v_resh = k.shapeCast(v_row, &.{ ctx.n_v, ctx.d_v });

                    // L2-normalize q and k (eps=1e-6, axis=-1, keepdims=True).  q/k come
                    //   in flattened as (1, key_dim) where key_dim = n_kq*d_k; reshape to
                    //   (n_kq, d_k), reduce over d_k, divide.  For n_kq=1 this collapses
                    //   to the (1, d_k) flat form; for n_kq>1 it normalizes per-head.
                    const eps_v = k.full(&.{ ctx.n_kq, 1 }, @as(f32, 1e-6), .f32);
                    const sum_zero = k.zeros(&.{ctx.n_kq}, .f32);
                    const q_sq = k.mulf(q_resh, q_resh);
                    const q_sumsq = k.multiReduction(.add, q_sq, sum_zero, &.{1});
                    const q_sumsq_2d = k.shapeCast(q_sumsq, &.{ ctx.n_kq, 1 });
                    const q_norm_pre = k.addf(q_sumsq_2d, eps_v);
                    const q_norm_sqrt = k.sqrt(q_norm_pre);
                    const q_norm = k.broadcastTo(q_norm_sqrt, &.{ ctx.n_kq, ctx.d_k });
                    const q_normed_kq = k.divf(q_resh, q_norm);
                    const k_sq = k.mulf(k_resh, k_resh);
                    const k_sumsq = k.multiReduction(.add, k_sq, sum_zero, &.{1});
                    const k_sumsq_2d = k.shapeCast(k_sumsq, &.{ ctx.n_kq, 1 });
                    const k_norm_pre = k.addf(k_sumsq_2d, eps_v);
                    const k_norm_sqrt = k.sqrt(k_norm_pre);
                    const k_norm = k.broadcastTo(k_norm_sqrt, &.{ ctx.n_kq, ctx.d_k });
                    const k_normed_kq = k.divf(k_resh, k_norm);

                    // GQA repeat (rf = n_v / n_kq) of q,k; then × scale.
                    //   (n_kq, d_k) → shape_cast (n_kq, 1, d_k) → broadcast (n_kq, rf, d_k)
                    //   → shape_cast (n_v, d_k).  For n_kq=1, rf=n_v and the first
                    //   shape_cast is a no-op.
                    const scale_f32: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(ctx.d_k)));
                    const scale_v = k.full(&.{ ctx.n_v, ctx.d_k }, scale_f32, .f32);
                    const rep_factor: i64 = @divExact(ctx.n_v, ctx.n_kq);
                    const q_resh4 = k.shapeCast(q_normed_kq, &.{ ctx.n_kq, 1, ctx.d_k });
                    const q_resh4_bc = k.broadcastTo(q_resh4, &.{ ctx.n_kq, rep_factor, ctx.d_k });
                    const q_rep = k.shapeCast(q_resh4_bc, &.{ ctx.n_v, ctx.d_k });
                    const k_resh4 = k.shapeCast(k_normed_kq, &.{ ctx.n_kq, 1, ctx.d_k });
                    const k_resh4_bc = k.broadcastTo(k_resh4, &.{ ctx.n_kq, rep_factor, ctx.d_k });
                    const k_normed = k.shapeCast(k_resh4_bc, &.{ ctx.n_v, ctx.d_k });
                    const q_scaled = k.mulf(q_rep, scale_v);

                    // Read decode_a_raw_ref[ds(b_aligned, sl), :] and decode_b_raw_ref[ds(b_aligned, sl), :]
                    //   (the two scalar gate-input rows; (sublanesize, 128) f32 each).
                    //   Pallas's `pl.ds(b_aligned, sublanesize)` reads `alignment` rows, NOT
                    //   `bt` rows — they coincide only when chunk_size == sublanesize.
                    const a_raw_w = refs[3]; // a_raw decode window
                    const a_raw_tile = k.memRefSqueeze(
                        k.memRefSlice(a_raw_w.buf.?, &.{ a_raw_w.slot.?, c0i, c0i }, &.{ 1, ctx.bt, 128 }, &.{}),
                        &.{ ctx.bt, 128 },
                    );
                    const a_raw_blk = k.vectorLoadShape(a_raw_tile, &.{ b_aligned_idx, c0idx }, &.{ ctx.alignment, 128 });
                    const b_raw_w = refs[5]; // b_raw decode window
                    const b_raw_tile = k.memRefSqueeze(
                        k.memRefSlice(b_raw_w.buf.?, &.{ b_raw_w.slot.?, c0i, c0i }, &.{ 1, ctx.bt, 128 }, &.{}),
                        &.{ ctx.bt, 128 },
                    );
                    const b_raw_blk = k.vectorLoadShape(b_raw_tile, &.{ b_aligned_idx, c0idx }, &.{ ctx.alignment, 128 });
                    // Reuse mask_2d (shape (sublanesize, 1)) broadcast to (sublanesize, 128) — but
                    // mask_2d is reconstructed because vector.broadcast was emitted once into
                    // (sublanesize, dim); we need a separate broadcast to (sublanesize, 128).
                    const mask_128 = k.broadcastTo(mask_2d, &.{ ctx.alignment, 128 });
                    const lane_zeros = k.zeros(&.{128}, .f32);
                    const a_raw_masked = k.mulf(a_raw_blk, mask_128);
                    const a_raw_summed = k.multiReduction(.add, a_raw_masked, lane_zeros, &.{0});
                    const curr_g_slice = k.shapeCast(a_raw_summed, &.{ 1, 128 });
                    const b_raw_masked = k.mulf(b_raw_blk, mask_128);
                    const b_raw_summed = k.multiReduction(.add, b_raw_masked, lane_zeros, &.{0});
                    const curr_beta_slice = k.shapeCast(b_raw_summed, &.{ 1, 128 });

                    // a_raw_new = curr_g_slice[:, :n_v].reshape(n_v); b_raw_new = same.
                    const nv_2d_ty = k.vectorTy(&.{ 1, ctx.n_v }, .f32);
                    const a_raw_pre = k.emit(dialects_v.extract_strided_slice(k.ctx, curr_g_slice.inner, .{ .offsets = &.{ 0, 0 }, .sizes = &.{ 1, ctx.n_v }, .strides = &.{ 1, 1 } }, nv_2d_ty, k.loc()));
                    const a_raw_new = k.shapeCast(a_raw_pre, &.{ctx.n_v});
                    const b_raw_pre = k.emit(dialects_v.extract_strided_slice(k.ctx, curr_beta_slice.inner, .{ .offsets = &.{ 0, 0 }, .sizes = &.{ 1, ctx.n_v }, .strides = &.{ 1, 1 } }, nv_2d_ty, k.loc()));
                    const b_raw_new = k.shapeCast(b_raw_pre, &.{ctx.n_v});

                    // curr_beta = sigmoid(b_raw_new): 1 / (1 + exp(-b)).
                    const ones_nv = k.full(&.{ctx.n_v}, @as(f32, 1.0), .f32);
                    const neg_b = k.negf(b_raw_new);
                    const exp_neg_b = k.exp(neg_b);
                    const denom_b = k.addf(exp_neg_b, ones_nv);
                    const curr_beta = k.divf(ones_nv, denom_b);

                    // a_val = exp(A_log)  — A_log lives in refs[6].buf (trivial window, (n_v,) f32).
                    const a_log_buf = refs[6].buf.?;
                    const a_val = k.refLoad(a_log_buf);
                    const a_val_exp = k.exp(a_val);
                    const zeros_nv = k.zeros(&.{ctx.n_v}, .f32);
                    const neg_a_val = k.subf(zeros_nv, a_val_exp); // = -exp(A_log)

                    // dt_bias load + a_plus_dt = a_raw_new + dt_bias.
                    const dt_bias_buf = refs[7].buf.?;
                    const dt_bias_v = k.refLoad(dt_bias_buf);
                    const a_plus_dt = k.addf(a_raw_new, dt_bias_v);

                    // NaN-safe softplus = (NaN ? a_plus_dt + 0 : max(x, 0) + log1p(exp(-|x|))).
                    const sp_max = k.maximumf(a_plus_dt, zeros_nv);
                    const sp_nan = k.cmpf(.one, a_plus_dt, a_plus_dt); // x != x → NaN
                    const sp_xp0 = k.addf(a_plus_dt, zeros_nv);
                    const sp_absx = k.absf(a_plus_dt);
                    const sp_negabsx = k.subf(zeros_nv, sp_absx);
                    const sp_exp = k.exp(sp_negabsx);
                    const sp_l1p_ty = sp_exp.type_();
                    const sp_l1p = k.emit(mlir.Operation.make(k.ctx, "math.log1p", .{
                        .operands = .{ .flat = &.{sp_exp.inner} },
                        .results = .{ .flat = &.{sp_l1p_ty} },
                        .location = k.loc(),
                    }));
                    const sp_sum = k.addf(sp_max, sp_l1p);
                    const softplus_v = k.select(sp_nan, sp_xp0, sp_sum);
                    const gk = k.mulf(neg_a_val, softplus_v);
                    const m100 = k.full(&.{ctx.n_v}, @as(f32, -100.0), .f32);
                    const gk_clamped = k.maximumf(gk, m100);
                    const decay = k.exp(gk_clamped);

                    // current_state = decode_state_scratch[0] : (1, n_v, d_k, d_v) f32
                    const current_state = k.vectorLoadShape(ctx.decode_state_scratch, &.{ c0idx, c0idx, c0idx, c0idx }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });

                    // ── Per-head pl.dot loop (UNROLLED, n_v = 2 in this sweep) ──
                    //   For each h: k_h · state_h, v_diff = v_h - decay_h·k_state, v_new = β_h·v_diff,
                    //   q_h · state_h, q_k = sum(q_h·k_h, -1), decay_q_state, out_h = decay_q_state +
                    //   q_k·v_new, k_v_new = k_h^T·v_new (outer, 128×128), decay_state = state·decay_h,
                    //   new_state_h = decay_state + k_v_new. The `where(isinf, 0, …)` defensive
                    //   guards are emitted as `cmpf oeq(absf(x), +Inf)` + `arith.select`.
                    const acc_2d = k.zeros(&.{ 1, ctx.d_v }, .f32); // K24 in target
                    const inf_2d = k.full(&.{ 1, ctx.d_v }, std.math.inf(f32), .f32); // K25
                    const acc_outer = k.zeros(&.{ ctx.d_k, ctx.d_v }, .f32); // K26
                    const inf_outer = k.full(&.{ ctx.d_k, ctx.d_v }, std.math.inf(f32), .f32); // K27
                    const sum1_zero = k.zeros(&.{1}, .f32); // K17 reuse
                    // dim_numbers for 1×d_k @ d_k×d_v → 1×d_v (the k/q · state matmuls).
                    const ddn_qkv = k.dotDimensionNumbers(
                        &.{1}, // lhs_contracting (col)
                        &.{0}, // rhs_contracting (row)
                        &.{0}, // lhs_non_contracting (row → out dim 0)
                        &.{1}, // rhs_non_contracting (col → out dim 1)
                        &.{ 0, 0, 1, 1 },
                        &.{},
                        &.{},
                    );
                    // dim_numbers for 1×d_k @ 1×d_v → d_k×d_v (the k^T·v_new outer product).
                    const ddn_outer = k.dotDimensionNumbers(
                        &.{0}, // contract on row of lhs
                        &.{0}, // contract on row of rhs
                        &.{1}, // lhs col
                        &.{1}, // rhs col
                        &.{ 0, 1, 1, 1 },
                        &.{},
                        &.{},
                    );
                    const mm_opts = mtt.MatmulOpts{ .precision = .fp32, .dimension_numbers = ddn_qkv };
                    const mm_opts_outer = mtt.MatmulOpts{ .precision = .fp32, .dimension_numbers = ddn_outer };

                    // Per-head matmul loop: Python's `for h in range(n_v)` ⇒ fully
                    //   unrolled at trace time, so the Zig mirror must dispatch through a
                    //   comptime construct. `cfg.num_v_heads` is a Cfg field (comptime-known
                    //   at the emit boundary), but the Pipeline ctx passes it as runtime
                    //   `ctx.n_v`. We use `inline for` over `SUPPORTED_N_V` so each branch
                    //   gets a comptime-known `N_V` for the `[N_V]Value` array and the
                    //   inner unrolled head loop. The matching `if (ctx.n_v == ...)` picks
                    //   exactly one branch at runtime; the others compile to dead code that
                    //   the canonicalizer prunes.
                    const out = decode_dispatch: {
                        inline for (SUPPORTED_N_V) |comptime_n_v| {
                            if (ctx.n_v == comptime_n_v) {
                                const N_V: usize = @intCast(comptime_n_v);
                                var out_h_list: [N_V]Value = undefined;
                                var ns_h_list: [N_V]Value = undefined;
                                inline for (0..N_V) |h_| {
                                    const h: i64 = @intCast(h_);
                                    const q_h_pre = k.vectorExtract(q_scaled, &.{h});
                                    const q_h = k.shapeCast(q_h_pre, &.{ 1, ctx.d_k });
                                    const k_h_pre = k.vectorExtract(k_normed, &.{h});
                                    const k_h = k.shapeCast(k_h_pre, &.{ 1, ctx.d_k });
                                    const v_h_pre = k.vectorExtract(v_resh, &.{h});
                                    const v_h = k.shapeCast(v_h_pre, &.{ 1, ctx.d_v });
                                    const state_h = k.vectorExtract(current_state, &.{ 0, h });

                                    // k_state_h = matmul k_h, state_h, acc=zeros<1, d_v>
                                    const k_state_h = k.matmulOpts(k_h, state_h, acc_2d, mm_opts);
                                    const k_state_abs = k.absf(k_state_h);
                                    const isinf_k = k.cmpf(.oeq, k_state_abs, inf_2d);
                                    const decay_h = k.vectorExtract(decay, &.{h});
                                    const decay_h_bc = k.broadcastTo(decay_h, &.{ 1, ctx.d_v });
                                    const decay_k_state = k.mulf(decay_h_bc, k_state_h);
                                    const safe_dks = k.select(isinf_k, acc_2d, decay_k_state);
                                    const v_diff_h = k.subf(v_h, safe_dks);
                                    const beta_h = k.vectorExtract(curr_beta, &.{h});
                                    const beta_h_bc = k.broadcastTo(beta_h, &.{ 1, ctx.d_v });
                                    const v_new_h = k.mulf(beta_h_bc, v_diff_h);

                                    const q_state_h = k.matmulOpts(q_h, state_h, acc_2d, mm_opts);
                                    const q_dot_k = k.mulf(q_h, k_h);
                                    const q_k_h = k.multiReduction(.add, q_dot_k, sum1_zero, &.{1});
                                    const q_state_abs = k.absf(q_state_h);
                                    const isinf_q = k.cmpf(.oeq, q_state_abs, inf_2d);
                                    const decay_q_state = k.mulf(decay_h_bc, q_state_h);
                                    const safe_dqs = k.select(isinf_q, acc_2d, decay_q_state);
                                    const q_k_h_bc = k.broadcastTo(q_k_h, &.{ 1, ctx.d_v });
                                    const qkv_new = k.mulf(q_k_h_bc, v_new_h);
                                    const out_h = k.addf(safe_dqs, qkv_new);
                                    out_h_list[h_] = out_h;

                                    // k_v_new_h = matmul(k_h^T, v_new_h) — outer product (d_k, d_v).
                                    const k_v_new_h = k.matmulOpts(k_h, v_new_h, acc_outer, mm_opts_outer);
                                    const state_abs = k.absf(state_h);
                                    const isinf_state = k.cmpf(.oeq, state_abs, inf_outer);
                                    const decay_h_outer = k.broadcastTo(decay_h, &.{ ctx.d_k, ctx.d_v });
                                    const decay_state = k.mulf(state_h, decay_h_outer);
                                    const safe_ds = k.select(isinf_state, acc_outer, decay_state);
                                    const new_state_h = k.addf(safe_ds, k_v_new_h);
                                    ns_h_list[h_] = new_state_h;
                                }
                                // Pallas caps `tpu.concatenate` arity at 16 operands; for
                                //   n_v > 16 it emits `ceil(n_v/16)` 16-wide concats then a
                                //   final merging concat.  Mirror so n_v>16 sweeps match.
                                const out_concat = blk_oc: {
                                    if (N_V <= 16) break :blk_oc k.concatenate(out_h_list[0..], 0);
                                    var groups: [8]Value = undefined;
                                    var n_groups: usize = 0;
                                    var ig: usize = 0;
                                    while (ig < N_V) : (ig += 16) {
                                        const end_g = @min(ig + 16, N_V);
                                        groups[n_groups] = k.concatenate(out_h_list[ig..end_g], 0);
                                        n_groups += 1;
                                    }
                                    break :blk_oc k.concatenate(groups[0..n_groups], 0);
                                };
                                // Reshape each new_state_h (d_k, d_v) → (1, d_k, d_v), concat in dim 0 → (n_v, d_k, d_v),
                                // shape_cast → (1, n_v, d_k, d_v) for the store.
                                var ns_3d: [N_V]Value = undefined;
                                inline for (0..N_V) |h_| {
                                    ns_3d[h_] = k.shapeCast(ns_h_list[h_], &.{ 1, ctx.d_k, ctx.d_v });
                                }
                                const ns_cat = blk_ns: {
                                    if (N_V <= 16) break :blk_ns k.concatenate(ns_3d[0..], 0);
                                    var groups: [8]Value = undefined;
                                    var n_groups: usize = 0;
                                    var ig: usize = 0;
                                    while (ig < N_V) : (ig += 16) {
                                        const end_g = @min(ig + 16, N_V);
                                        groups[n_groups] = k.concatenate(ns_3d[ig..end_g], 0);
                                        n_groups += 1;
                                    }
                                    break :blk_ns k.concatenate(groups[0..n_groups], 0);
                                };
                                const ns_4d = k.shapeCast(ns_cat, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });
                                k.vectorStoreAt(ctx.decode_state_scratch, ns_4d, &.{ c0idx, c0idx, c0idx, c0idx });
                                break :decode_dispatch out_concat;
                            }
                        }
                        @panic("decode_dispatch: unsupported n_v (add it to the comptime tuple)");
                    };

                    // ── Output mask-merge into decode_output_scratch ──
                    //   current_output = decode_output_scratch[...]
                    //   mask = (arange(BT) == b).astype(dtype)[:, None]
                    //   new_output = where(mask, out.reshape(1, n_v*d_v), current_output)
                    //   decode_output_scratch[...] = new_output.astype(dtype)
                    const cur_out_bf = k.vectorLoadShape(ctx.decode_output_scratch, &.{ c0idx, c0idx }, &.{ ctx.bt, v_cols });
                    // BT-sized iota for the per-token select.  `iota_1d` above is
                    //   alignment-sized; for chunk_size > sublanesize they differ.
                    //   Pallas emits the iota+shape_cast BEFORE the b broadcast.
                    const bt_iota_2d_local = k.iota(&.{ 1, ctx.bt }, .i32, &.{1});
                    const bt_iota_1d_local = k.shapeCast(bt_iota_2d_local, &.{ctx.bt});
                    const b_bc_bt = k.broadcastTo(b_iv, &.{ctx.bt});
                    const bt_mask_i1 = k.cmpi(.eq, bt_iota_1d_local, b_bc_bt);
                    const bt_mask_bf = bt_mask_i1.to(ctx.dtype);
                    const bt_mask_2d_bf = k.shapeCast(bt_mask_bf, &.{ ctx.bt, 1 });
                    const out_flat = k.shapeCast(out, &.{ 1, v_cols });
                    const zero_bt_bf = k.zeros(&.{ ctx.bt, 1 }, ctx.dtype);
                    const bt_mask_i1_again = k.cmpf(.one, bt_mask_2d_bf, zero_bt_bf); // mask_bf != 0 → i1
                    const cur_out_f32 = cur_out_bf.to(.f32);
                    const mask_bc = k.broadcastTo(bt_mask_i1_again, &.{ ctx.bt, v_cols });
                    const out_bc = k.broadcastTo(out_flat, &.{ ctx.bt, v_cols });
                    const new_out_f32 = k.select(mask_bc, out_bc, cur_out_f32);
                    const new_out_bf = new_out_f32.to(ctx.dtype);
                    k.vectorStoreAt(ctx.decode_output_scratch, new_out_bf, &.{ c0idx, c0idx });

                    // ── Gather-DMA out: state_commit_scratch ← decode_state_scratch.truncf →
                    //                    recurrent_state_out[ds(target_idx, 1)] via decode_write_sem[0]. ──
                    const ds_loaded = k.vectorLoadShape(ctx.decode_state_scratch, &.{ c0idx, c0idx, c0idx, c0idx }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });
                    const ds_3d = k.shapeCast(ds_loaded, &.{ ctx.n_v, ctx.d_k, ctx.d_v });
                    const ds_trunc = ds_3d.to(ctx.state_dtype);
                    const ds_4d_back = k.shapeCast(ds_trunc, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });
                    k.vectorStoreAt(ctx.state_commit_scratch, ds_4d_back, &.{ c0idx, c0idx, c0idx, c0idx });
                    const dst_slice = k.memRefSlice(ctx.recurrent_state_out, &.{ target_idx, c0i, c0i, c0i }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v }, &.{});
                    const w_sem = k.memRefSqueeze(ctx.decode_write_sem, &.{});
                    k.enqueueDma(ctx.state_commit_scratch, dst_slice, w_sem, .{});
                    k.waitDma2(w_sem, ctx.state_commit_scratch, dst_slice, .{});
                    when_b.yieldThen(.{});
                }
                _ = scf.yield(k.ctx, &.{}, k.loc()).appendTo(for_body);
            }
            k.popBlock();
            const bt_val = k.lift(@as(i32, @intCast(ctx.bt)));
            const c1_i32 = k.lift(@as(i32, 1));
            const for_op = scf.for_(k.ctx, c0i.inner, bt_val.inner, c1_i32.inner, &.{}, for_body, .{}, k.loc());
            _ = for_op.appendTo(k.currentBlock());

            // ── Post-loop: mask `arange(BT) < decode_count` → multiply into
            //   decode_output_scratch → write to decode_output_ref (= refs[9]'s window). ──
            const post_iota_2d = k.iota(&.{ 1, ctx.bt }, .i32, &.{1});
            const post_iota = k.shapeCast(post_iota_2d, &.{ctx.bt});
            const dc_bc = k.broadcastTo(decode_count, &.{ctx.bt});
            const post_mask_i1 = k.cmpi(.slt, post_iota, dc_bc);
            const post_mask_bf = post_mask_i1.to(ctx.dtype);
            const post_mask_2d = k.shapeCast(post_mask_bf, &.{ ctx.bt, 1 });
            const out_buf = k.vectorLoadShape(ctx.decode_output_scratch, &.{ k.cIndex(0), k.cIndex(0) }, &.{ ctx.bt, ctx.n_v * ctx.d_v });
            const post_mask_full = k.broadcastTo(post_mask_2d, &.{ ctx.bt, ctx.n_v * ctx.d_v });
            const out_masked = k.mulf(out_buf, post_mask_full);
            // Write to refs[9]'s window — `vector_store output_decode_buf[slot, 0, 0]`,
            // shape-cast (bt, n_v*d_v) → (1, bt, n_v*d_v).
            const out_dec_w = refs[9];
            const out_dec_slot = out_dec_w.slot.?;
            const out_dec_idx = k.toIndex(out_dec_slot);
            const out_masked_3d = k.shapeCast(out_masked, &.{ 1, ctx.bt, ctx.n_v * ctx.d_v });
            k.vectorStoreAt(out_dec_w.buf.?, out_masked_3d, &.{ out_dec_idx, k.cIndex(0), k.cIndex(0) });
            when.yieldThen(.{});
        }
    }

    // ── (2) Prefill path: `@pl.when(prefill_valid > 0)` ⇒ `lax.cond(is_transition>0,
    //   process_transition_prefill, process_regular_prefill)`. ──
    {
        var when = k.openIf(k.cmpi(.sgt, prefill_valid, c0i));
        {
            // prefill_slot = prefill_req_id % 2  (lax.rem floor-mod sign-fixup).
            const c2 = k.lift(@as(i32, 2));
            const rem_pr = k.remsi(prefill_req_id, c2);
            const pr_nz = k.cmpi(.ne, rem_pr, c0i);
            const pr_neg = k.cmpi(.slt, rem_pr, c0i);
            const pr_needs = k.andi(pr_neg, pr_nz);
            const rem_pr_plus = k.addi(rem_pr, c2);
            const prefill_slot = k.select(pr_needs, rem_pr_plus, rem_pr);

            // Re-load is_transition inside the prefill scf.if (JAX emits the schedule
            // load again because CSE doesn't cross scf.if regions and the lax.cond
            // body re-traces the predicate). `lax.cond(is_tr > 0, transition, regular)`
            // lowers to a single `scf.if cond { ... } else { ... }` — match that shape
            // with `openIfElse` (one scf.if), not two consecutive `openIf`s.
            const is_tr_inner = k.scalarLoad(sched, &.{ step_i, k.cIndex(10) });
            var when_dispatch = k.openIfElse(k.cmpi(.sgt, is_tr_inner, c0i), .{});
            {
                // ── process_transition_prefill setup ──
                //   Same qkv silu / extract / a/b_raw transpose / sigmoid β / gate g
                //   / L2-norm / GQA / transpose / scale as regular, but only on the
                //   first C_trans = sublanesize rows (== C in this config since
                //   chunk_size == sublanesize).
                const C_trans: i64 = ctx.alignment; // sublanesize, = 16 for bf16
                const dim_tr: i64 = 2 * ctx.n_kq * ctx.d_k + ctx.n_v * ctx.d_v;
                const key_dim_tr = ctx.n_kq * ctx.d_k;
                const v_cols_tr = ctx.n_v * ctx.d_v;
                const qkv_pre_w_tr = refs[0];
                const qkv_tr_tile = k.memRefSqueeze(
                    k.memRefSlice(qkv_pre_w_tr.buf.?, &.{ qkv_pre_w_tr.slot.?, c0i, c0i }, &.{ 1, ctx.C, dim_tr }, &.{}),
                    &.{ ctx.C, dim_tr },
                );
                const qkv_tr_bf = k.vectorLoadShape(qkv_tr_tile, &.{ k.cIndex(0), k.cIndex(0) }, &.{ C_trans, dim_tr });
                const qkv_tr_f = qkv_tr_bf.to(.f32);
                const ones_tr = k.full(&.{ C_trans, dim_tr }, @as(f32, 1.0), .f32);
                const neg_tr = k.negf(qkv_tr_f);
                const exp_neg_tr = k.exp(neg_tr);
                const denom_tr = k.addf(exp_neg_tr, ones_tr);
                const sig_tr = k.divf(ones_tr, denom_tr);
                const silu_tr = k.mulf(qkv_tr_f, sig_tr);
                const q_ty_tr = k.vectorTy(&.{ C_trans, key_dim_tr }, .f32);
                const v_ty_tr = k.vectorTy(&.{ C_trans, v_cols_tr }, .f32);
                const q_tr = k.emit(dialects.vector.extract_strided_slice(k.ctx, silu_tr.inner, .{ .offsets = &.{ 0, 0 }, .sizes = &.{ C_trans, key_dim_tr }, .strides = &.{ 1, 1 } }, q_ty_tr, k.loc()));
                const k_tr = k.emit(dialects.vector.extract_strided_slice(k.ctx, silu_tr.inner, .{ .offsets = &.{ 0, key_dim_tr }, .sizes = &.{ C_trans, key_dim_tr }, .strides = &.{ 1, 1 } }, q_ty_tr, k.loc()));
                const v_tr = k.emit(dialects.vector.extract_strided_slice(k.ctx, silu_tr.inner, .{ .offsets = &.{ 0, 2 * key_dim_tr }, .sizes = &.{ C_trans, v_cols_tr }, .strides = &.{ 1, 1 } }, v_ty_tr, k.loc()));

                // a/b_raw + transpose + sigmoid β + gate g (clamp ≥ -100) — same as regular.
                const a_p_w_tr = refs[2];
                const a_p_idx_tr = k.toIndex(a_p_w_tr.slot.?);
                const a_p_pre_tr = k.vectorLoadShape(a_p_w_tr.buf.?, &.{ a_p_idx_tr, k.cIndex(0), k.cIndex(0) }, &.{ 1, ctx.C, 128 });
                const a_p_2d_tr = k.shapeCast(a_p_pre_tr, &.{ ctx.C, 128 });
                const b_p_w_tr = refs[4];
                const b_p_idx_tr = k.toIndex(b_p_w_tr.slot.?);
                const b_p_pre_tr = k.vectorLoadShape(b_p_w_tr.buf.?, &.{ b_p_idx_tr, k.cIndex(0), k.cIndex(0) }, &.{ 1, ctx.C, 128 });
                const b_p_2d_tr = k.shapeCast(b_p_pre_tr, &.{ ctx.C, 128 });
                const cnv_ty_tr = k.vectorTy(&.{ C_trans, ctx.n_v }, .f32);
                const a_p_sl_tr = k.emit(dialects.vector.extract_strided_slice(k.ctx, a_p_2d_tr.inner, .{ .offsets = &.{ 0, 0 }, .sizes = &.{ C_trans, ctx.n_v }, .strides = &.{ 1, 1 } }, cnv_ty_tr, k.loc()));
                const a_raw_proc_tr = k.transpose(a_p_sl_tr, &.{ 1, 0 });
                const b_p_sl_tr = k.emit(dialects.vector.extract_strided_slice(k.ctx, b_p_2d_tr.inner, .{ .offsets = &.{ 0, 0 }, .sizes = &.{ C_trans, ctx.n_v }, .strides = &.{ 1, 1 } }, cnv_ty_tr, k.loc()));
                const b_raw_proc_tr = k.transpose(b_p_sl_tr, &.{ 1, 0 });
                const ones_nvCtr = k.full(&.{ ctx.n_v, C_trans }, @as(f32, 1.0), .f32);
                const neg_b_tr = k.negf(b_raw_proc_tr);
                const exp_neg_b_tr = k.exp(neg_b_tr);
                const denom_b_tr = k.addf(exp_neg_b_tr, ones_nvCtr);
                const beta_tr = k.divf(ones_nvCtr, denom_b_tr);
                const a_log_tr = k.refLoad(refs[6].buf.?);
                const a_log_2d_tr = k.shapeCast(a_log_tr, &.{ ctx.n_v, 1 });
                const a_log_exp_tr = k.exp(a_log_2d_tr);
                const zeros_nv1_tr = k.zeros(&.{ ctx.n_v, 1 }, .f32);
                const neg_aval_tr = k.subf(zeros_nv1_tr, a_log_exp_tr);
                const dt_bias_tr = k.refLoad(refs[7].buf.?);
                const dt_2d_tr = k.shapeCast(dt_bias_tr, &.{ ctx.n_v, 1 });
                const dt_bc_tr = k.broadcastTo(dt_2d_tr, &.{ ctx.n_v, C_trans });
                const apdt_tr = k.addf(a_raw_proc_tr, dt_bc_tr);
                const zeros_nvCtr2 = k.zeros(&.{ ctx.n_v, C_trans }, .f32);
                const sp_max_tr = k.maximumf(apdt_tr, zeros_nvCtr2);
                const sp_nan_tr = k.cmpf(.one, apdt_tr, apdt_tr);
                const sp_xp0_tr = k.addf(apdt_tr, zeros_nvCtr2);
                const sp_abs_tr = k.absf(apdt_tr);
                const sp_neg_abs_tr = k.subf(zeros_nvCtr2, sp_abs_tr);
                const sp_e_tr = k.exp(sp_neg_abs_tr);
                const sp_l1p_tr = k.emit(mlir.Operation.make(k.ctx, "math.log1p", .{
                    .operands = .{ .flat = &.{sp_e_tr.inner} },
                    .results = .{ .flat = &.{sp_e_tr.type_()} },
                    .location = k.loc(),
                }));
                const sp_sum_tr = k.addf(sp_max_tr, sp_l1p_tr);
                const softplus_tr = k.select(sp_nan_tr, sp_xp0_tr, sp_sum_tr);
                const neg_aval_bc_tr = k.broadcastTo(neg_aval_tr, &.{ ctx.n_v, C_trans });
                const gk_tr = k.mulf(neg_aval_bc_tr, softplus_tr);
                const m100_tr = k.full(&.{ ctx.n_v, C_trans }, @as(f32, -100.0), .f32);
                const g_chunk_tr = k.maximumf(gk_tr, m100_tr);

                // q, k, v reshape + L2norm + GQA + transpose to (n_v, C_trans, d_k/v) + scale (q only).
                const q_3d_tr = k.shapeCast(q_tr, &.{ C_trans, ctx.n_kq, ctx.d_k });
                const k_3d_tr = k.shapeCast(k_tr, &.{ C_trans, ctx.n_kq, ctx.d_k });
                const v_3d_tr = k.shapeCast(v_tr, &.{ C_trans, ctx.n_v, ctx.d_v });
                const eps_tr = k.full(&.{ C_trans, ctx.n_kq, 1 }, @as(f32, 1e-6), .f32);
                const sum0_tr = k.zeros(&.{ C_trans, ctx.n_kq }, .f32);
                const q_sq_tr = k.mulf(q_3d_tr, q_3d_tr);
                const q_sumsq_tr = k.multiReduction(.add, q_sq_tr, sum0_tr, &.{2});
                const q_sumsq_3d_tr = k.shapeCast(q_sumsq_tr, &.{ C_trans, ctx.n_kq, 1 });
                const q_pre_tr = k.addf(q_sumsq_3d_tr, eps_tr);
                const q_sqrt_tr = k.sqrt(q_pre_tr);
                const q_nrm_tr = k.broadcastTo(q_sqrt_tr, &.{ C_trans, ctx.n_kq, ctx.d_k });
                const q_n3_tr = k.divf(q_3d_tr, q_nrm_tr);
                const k_sq_tr = k.mulf(k_3d_tr, k_3d_tr);
                const k_sumsq_tr = k.multiReduction(.add, k_sq_tr, sum0_tr, &.{2});
                const k_sumsq_3d_tr = k.shapeCast(k_sumsq_tr, &.{ C_trans, ctx.n_kq, 1 });
                const k_pre_tr = k.addf(k_sumsq_3d_tr, eps_tr);
                const k_sqrt_tr = k.sqrt(k_pre_tr);
                const k_nrm_tr = k.broadcastTo(k_sqrt_tr, &.{ C_trans, ctx.n_kq, ctx.d_k });
                const k_n3_tr = k.divf(k_3d_tr, k_nrm_tr);
                const q_n4_tr = k.shapeCast(q_n3_tr, &.{ C_trans, ctx.n_kq, 1, ctx.d_k });
                const q_n4_bc_tr = k.broadcastTo(q_n4_tr, &.{ C_trans, ctx.n_kq, @divExact(ctx.n_v, ctx.n_kq), ctx.d_k });
                const q_rep_tr = k.shapeCast(q_n4_bc_tr, &.{ C_trans, ctx.n_v, ctx.d_k });
                const k_n4_tr = k.shapeCast(k_n3_tr, &.{ C_trans, ctx.n_kq, 1, ctx.d_k });
                const k_n4_bc_tr = k.broadcastTo(k_n4_tr, &.{ C_trans, ctx.n_kq, @divExact(ctx.n_v, ctx.n_kq), ctx.d_k });
                const k_rep_tr = k.shapeCast(k_n4_bc_tr, &.{ C_trans, ctx.n_v, ctx.d_k });
                const q_tr_t = k.transpose(q_rep_tr, &.{ 1, 0, 2 });
                const k_tr_t = k.transpose(k_rep_tr, &.{ 1, 0, 2 });
                const v_tr_t = k.transpose(v_3d_tr, &.{ 1, 0, 2 });
                const scale_tr_f32: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(ctx.d_k)));
                const scale_tr_v = k.full(&.{ ctx.n_v, C_trans, ctx.d_k }, scale_tr_f32, .f32);
                const q_sc_tr = k.mulf(q_tr_t, scale_tr_v);

                // ── 16-unrolled per-token loop (transition path) ──
                //   For each token i ∈ [0, sublanesize):
                //     read t_req / t_is_first / t_is_last
                //     compute is_new_seq, sequence_valid updates
                //     c_slot = current_r % 2; ping-pong prefill_scratch[0/1]
                //     conditional store of current_r's state (if prefill & new seq)
                //     conditional load of t_req's state (if t_is_first & t_has_init)
                //     per-token recurrent update; write output; update sequence_valid
                const C_trans_u: usize = @intCast(C_trans);
                // Initial state setup: first_req_id, first_is_first, first_has_init.
                //   Pallas emits the floor-mod chain BEFORE the index_cast + has_init
                //   load so the rem result is available at the start of the predicate
                //   computation; mirror that order to avoid SSA cascade through the
                //   transition setup. `first_slot = lax.rem(first_req_id, 2)` = JAX's
                //   signed `remsi` + sign-fixup `select((r != 0) & (r < 0), r + m, r)`.
                const first_req_id = k.scalarLoad(sched, &.{ step_i, k.cIndex(11) });
                const first_is_first = k.scalarLoad(sched, &.{ step_i, k.cIndex(11 + C_trans) });
                const c2_fr = k.lift(@as(i32, 2));
                const fr_rem = k.remsi(first_req_id, c2_fr);
                const fr_nz = k.cmpi(.ne, fr_rem, c0i);
                const fr_neg = k.cmpi(.slt, fr_rem, c0i);
                const fr_needs = k.andi(fr_neg, fr_nz);
                const fr_plus = k.addi(fr_rem, c2_fr);
                const first_slot_rem = k.select(fr_needs, fr_plus, fr_rem);
                const first_req_idx = k.toIndex(first_req_id);
                const first_has_init = k.scalarLoad(has_init_ref, &.{first_req_idx});
                // Conditional gather-DMA: if first_is_first & first_has_init, load
                // recurrent_state_in[ds(state_indices[first_req_id], 1)] → state_commit
                // → prefill_scratch[first_slot] (upcast to f32).
                const first_is_first_gt = k.cmpi(.sgt, first_is_first, c0i);
                const first_has_init_gt = k.cmpi(.sgt, first_has_init, c0i);
                const first_load_pred = k.andi(first_is_first_gt, first_has_init_gt);
                var when_first_load = k.openIf(first_load_pred);
                {
                    const f_state_idx = k.scalarLoad(scratches[1], &.{first_req_idx});
                    const f_src = k.memRefSlice(ctx.recurrent_state_in, &.{ f_state_idx, c0i, c0i, c0i }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v }, &.{});
                    const f_sem_sl = k.memRefSlice(ctx.prefill_sem, &.{first_slot_rem}, &.{1}, &.{});
                    const f_sem = k.memRefSqueeze(f_sem_sl, &.{});
                    k.enqueueDma(f_src, ctx.state_commit_scratch, f_sem, .{});
                    k.waitDma2(f_sem, f_src, ctx.state_commit_scratch, .{});
                    // Upcast: prefill_scratch[first_slot] = state_commit_scratch[0].astype(f32)
                    //   shape_cast 4D→3D, extf 3D, shape_cast 3D→4D, store (Pallas pattern).
                    const f_loaded_4d = k.vectorLoadShape(ctx.state_commit_scratch, &.{ k.cIndex(0), k.cIndex(0), k.cIndex(0), k.cIndex(0) }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });
                    const f_loaded_3d = k.shapeCast(f_loaded_4d, &.{ ctx.n_v, ctx.d_k, ctx.d_v });
                    const f_upcast_3d = f_loaded_3d.to(.f32);
                    const f_slot_idx = k.toIndex(first_slot_rem);
                    const f_upcast_4d = k.shapeCast(f_upcast_3d, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });
                    k.vectorStoreAt(ctx.prefill_scratch, f_upcast_4d, &.{ f_slot_idx, k.cIndex(0), k.cIndex(0), k.cIndex(0) });
                    when_first_load.yieldThen(.{});
                }
                // Initial h: read prefill_scratch[first_slot] (dynamic index — Pallas
                //   uses a single `index_cast + vector.load[first_slot_idx, ...]` here),
                //   then `where(first_is_first & !first_has_init, zeros, h)` to zero-out
                //   if no prior state.
                const first_slot_idx_init = k.toIndex(first_slot_rem);
                var h_tr = k.shapeCast(
                    k.vectorLoadShape(ctx.prefill_scratch, &.{ first_slot_idx_init, k.cIndex(0), k.cIndex(0), k.cIndex(0) }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v }),
                    &.{ ctx.n_v, ctx.d_k, ctx.d_v },
                );
                const first_no_init = k.cmpi(.eq, first_has_init, c0i);
                const first_zero_pred = k.andi(first_is_first_gt, first_no_init);
                const h_zero_init = k.zeros(&.{ ctx.n_v, ctx.d_k, ctx.d_v }, .f32);
                h_tr = k.select(first_zero_pred, h_zero_init, h_tr);
                var current_r_v = first_req_id;
                // c_slot iter-carry: at iter 0, c_slot = first_slot.  At iter i>0,
                //   c_slot at iter i = current_r % 2 = t_req_at_(i-1) % 2 = t_slot_at_(i-1).
                //   Pallas reuses the previous iter's t_slot (already a full floor-mod
                //   chain value) as the current iter's c_slot, eliminating a per-iter remui.
                var c_slot_iter: Value = first_slot_rem;
                // sequence_valid: initial True (i1).
                var seq_valid: Value = k.liftAs(@as(i32, 1), .i1);
                const out_pre_w_tr = refs[8];
                // Per-iter inline emits the destination slice + squeeze; do not hoist.
                inline for (0..16) |i_| {
                    if (i_ >= C_trans_u) break;
                    // t_req = schedule_table[step, 11 + i]
                    const t_req = k.scalarLoad(sched, &.{ step_i, k.cIndex(11 + @as(i64, @intCast(i_))) });
                    const t_is_first = k.scalarLoad(sched, &.{ step_i, k.cIndex(11 + C_trans + @as(i64, @intCast(i_))) });
                    const t_is_last = k.scalarLoad(sched, &.{ step_i, k.cIndex(11 + 2 * C_trans + @as(i64, @intCast(i_))) });
                    // sequence_valid stub: combine t_is_first / t_is_last / t_req predicates
                    // so the schedule reads + column-index constants survive canonicalize.
                    // `t_is_last_gt = (t_is_last > 0)` — Pallas emits this `cmpi sgt`
                    //   just-in-time before its `select` use; computed below.

                    // is_new_seq = t_req != current_r (using current_r_v from prior iteration).
                    const is_new_seq = k.cmpi(.ne, t_req, current_r_v);
                    // sequence_valid = where(is_new_seq, True, seq_valid)
                    const true_i1 = k.liftAs(@as(i32, 1), .i1);
                    seq_valid = k.select(is_new_seq, true_i1, seq_valid);
                    // is_decode_token = t_req < decode_tokens
                    const is_decode_tok = k.cmpi(.slt, t_req, ctx.decode_tokens);
                    const false_i1 = k.liftAs(@as(i32, 0), .i1);
                    seq_valid = k.select(is_decode_tok, false_i1, seq_valid);

                    // `should_write = (current_r >= decode_tokens) & is_new_seq` — Pallas
                    //   emits the `cmpi sge + andi` pair right before `scf.if`, not eagerly
                    //   up here.  Computed below, next to the conditional STORE.

                    // Scratch ping-pong: c_slot = current_r % 2.  Use the carried
                    //   `c_slot_iter` value (= prev iter's t_slot, or first_slot at iter 0).
                    //   Pallas interleaves cmpi/select/store per slot — match that order.
                    const c_slot_rem = c_slot_iter;
                    const h0_pp = k.shapeCast(
                        k.vectorLoadShape(ctx.prefill_scratch, &.{ k.cIndex(0), k.cIndex(0), k.cIndex(0), k.cIndex(0) }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v }),
                        &.{ ctx.n_v, ctx.d_k, ctx.d_v },
                    );
                    const h1_pp = k.shapeCast(
                        k.vectorLoadShape(ctx.prefill_scratch, &.{ k.cIndex(1), k.cIndex(0), k.cIndex(0), k.cIndex(0) }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v }),
                        &.{ ctx.n_v, ctx.d_k, ctx.d_v },
                    );
                    const c_eq_0 = k.cmpi(.eq, c_slot_rem, c0i);
                    const ps0_new = k.select(c_eq_0, h_tr, h0_pp);
                    k.vectorStoreAt(ctx.prefill_scratch, k.shapeCast(ps0_new, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v }), &.{ k.cIndex(0), k.cIndex(0), k.cIndex(0), k.cIndex(0) });
                    const c_eq_1 = k.cmpi(.eq, c_slot_rem, k.lift(@as(i32, 1)));
                    const ps1_new = k.select(c_eq_1, h_tr, h1_pp);
                    k.vectorStoreAt(ctx.prefill_scratch, k.shapeCast(ps1_new, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v }), &.{ k.cIndex(1), k.cIndex(0), k.cIndex(0), k.cIndex(0) });

                    // state_commit_scratch[0] = prefill_scratch[c_slot].astype(state_dtype).
                    //   Pallas: dynamic load with toIndex(c_slot), shape_cast 4D→3D before
                    //   truncf, shape_cast 3D→4D after, store.
                    const c_slot_idx = k.toIndex(c_slot_rem);
                    const cs_loaded_4d = k.vectorLoadShape(ctx.prefill_scratch, &.{ c_slot_idx, k.cIndex(0), k.cIndex(0), k.cIndex(0) }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });
                    const cs_loaded_3d = k.shapeCast(cs_loaded_4d, &.{ ctx.n_v, ctx.d_k, ctx.d_v });
                    const cs_trunc_3d = cs_loaded_3d.to(ctx.state_dtype);
                    const cs_trunc_4d = k.shapeCast(cs_trunc_3d, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });
                    k.vectorStoreAt(ctx.state_commit_scratch, cs_trunc_4d, &.{ k.cIndex(0), k.cIndex(0), k.cIndex(0), k.cIndex(0) });

                    // Conditional STORE: if (current_r >= decode_tokens) & is_new_seq, write
                    // state_commit_scratch → recurrent_state_out[ds(state_indices[current_r], 1)]
                    // via prefill_sem[c_slot].  Compute `should_write` here to match Pallas's
                    // emission position (cmpi sge + andi land right before scf.if).
                    const cur_is_prefill = k.cmpi(.sge, current_r_v, ctx.decode_tokens);
                    const should_write = k.andi(cur_is_prefill, is_new_seq);
                    var when_store_t = k.openIf(should_write);
                    {
                        const cur_idx = k.toIndex(current_r_v);
                        const cur_state_idx = k.scalarLoad(scratches[1], &.{cur_idx});
                        const cur_dst = k.memRefSlice(ctx.recurrent_state_out, &.{ cur_state_idx, c0i, c0i, c0i }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v }, &.{});
                        const c_sem_sl = k.memRefSlice(ctx.prefill_sem, &.{c_slot_rem}, &.{1}, &.{});
                        const c_sem = k.memRefSqueeze(c_sem_sl, &.{});
                        k.enqueueDma(ctx.state_commit_scratch, cur_dst, c_sem, .{});
                        k.waitDma2(c_sem, ctx.state_commit_scratch, cur_dst, .{});
                        when_store_t.yieldThen(.{});
                    }

                    // t_slot = lax.rem(t_req, 2) — JAX's signed `remsi` + sign-fixup
                    //   `select((r != 0) & (r < 0), r + m, r)`.  Pallas emits this floor-mod
                    //   chain BEFORE the conditional LOAD so the sem index is available
                    //   ahead of the predicate computation.  Same pattern as `first_slot_rem`.
                    const c2_t = k.lift(@as(i32, 2));
                    const t_rem_pre = k.remsi(t_req, c2_t);
                    const t_nz = k.cmpi(.ne, t_rem_pre, c0i);
                    const t_neg = k.cmpi(.slt, t_rem_pre, c0i);
                    const t_needs = k.andi(t_neg, t_nz);
                    const t_plus = k.addi(t_rem_pre, c2_t);
                    const t_slot = k.select(t_needs, t_plus, t_rem_pre);

                    // Conditional LOAD: if t_is_first > 0 & t_has_init > 0, load
                    // recurrent_state_in[state_indices[t_req]] → state_commit_scratch and
                    // upcast it into prefill_scratch[t_slot] so next iter's new_h re-read
                    // sees the fresh init state.  Pallas emits t_is_first_gt close to the
                    // andi (after t_has_init load) — match that ordering.
                    const t_req_idx = k.toIndex(t_req);
                    const t_has_init = k.scalarLoad(has_init_ref, &.{t_req_idx});
                    const t_has_init_gt = k.cmpi(.sgt, t_has_init, c0i);
                    const t_is_first_gt = k.cmpi(.sgt, t_is_first, c0i);
                    const load_pred = k.andi(t_is_first_gt, t_has_init_gt);
                    var when_load_t = k.openIf(load_pred);
                    {
                        const t_state_idx = k.scalarLoad(scratches[1], &.{t_req_idx});
                        const t_src = k.memRefSlice(ctx.recurrent_state_in, &.{ t_state_idx, c0i, c0i, c0i }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v }, &.{});
                        const t_sem_sl = k.memRefSlice(ctx.prefill_sem, &.{t_slot}, &.{1}, &.{});
                        const t_sem = k.memRefSqueeze(t_sem_sl, &.{});
                        k.enqueueDma(t_src, ctx.state_commit_scratch, t_sem, .{});
                        k.waitDma2(t_sem, t_src, ctx.state_commit_scratch, .{});

                        // Upcast: prefill_scratch[t_slot] = state_commit_scratch[0].astype(f32)
                        //   shape_cast 4D→3D, extf 3D, shape_cast 3D→4D, store.
                        const t_loaded_4d = k.vectorLoadShape(ctx.state_commit_scratch, &.{ k.cIndex(0), k.cIndex(0), k.cIndex(0), k.cIndex(0) }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });
                        const t_loaded_3d = k.shapeCast(t_loaded_4d, &.{ ctx.n_v, ctx.d_k, ctx.d_v });
                        const t_extf_3d = t_loaded_3d.to(.f32);
                        const t_slot_idx = k.toIndex(t_slot);
                        const t_extf_4d = k.shapeCast(t_extf_3d, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });
                        k.vectorStoreAt(ctx.prefill_scratch, t_extf_4d, &.{ t_slot_idx, k.cIndex(0), k.cIndex(0), k.cIndex(0) });
                        when_load_t.yieldThen(.{});
                    }

                    // new_h = where(t_slot == 0, prefill_scratch[0], prefill_scratch[1])
                    //   gated by zero-pred = t_is_first & !t_has_init (use zeros when a
                    //   new sequence starts with no prior state).  If the conditional LOAD
                    //   fired, prefill_scratch[t_slot] now holds the freshly upcast init
                    //   state; otherwise it still holds the previous-iter's h.
                    const h0_n = k.shapeCast(
                        k.vectorLoadShape(ctx.prefill_scratch, &.{ k.cIndex(0), k.cIndex(0), k.cIndex(0), k.cIndex(0) }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v }),
                        &.{ ctx.n_v, ctx.d_k, ctx.d_v },
                    );
                    const h1_n = k.shapeCast(
                        k.vectorLoadShape(ctx.prefill_scratch, &.{ k.cIndex(1), k.cIndex(0), k.cIndex(0), k.cIndex(0) }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v }),
                        &.{ ctx.n_v, ctx.d_k, ctx.d_v },
                    );
                    const t_eq_0 = k.cmpi(.eq, t_slot, c0i);
                    const base_new_h = k.select(t_eq_0, h0_n, h1_n);
                    const t_no_init = k.cmpi(.eq, t_has_init, c0i);
                    const zero_pred = k.andi(t_is_first_gt, t_no_init);
                    const zeros_h_3d = k.zeros(&.{ ctx.n_v, ctx.d_k, ctx.d_v }, .f32);
                    const new_h = k.select(zero_pred, zeros_h_3d, base_new_h);

                    // current_r = t_req for next iteration; c_slot at next iter = this
                    //   iter's t_slot (Pallas carries this value instead of recomputing).
                    current_r_v = t_req;
                    c_slot_iter = t_slot;
                    // Save previous iter_arg h (h_tr) for the seq_valid preserve path,
                    // then rebind h_tr to the re-read new_h for use in the recurrence body.
                    const h_prev_iter = h_tr;
                    h_tr = new_h;
                    // Per-token recurrent update:
                    //   k_i = k_tr_t[:, i, :]   (n_v, d_k)
                    //   v_i = v_tr_t[:, i, :]   (n_v, d_v)
                    //   g_i = g_chunk_tr[:, i]  (n_v,)
                    //   beta_i = beta_tr[:, i]  (n_v,)
                    //   q_i = q_sc_tr[:, i, :]  (n_v, d_k)
                    //   decay = exp(g_i)[:, None]            (n_v, 1)
                    //   k_state = sum(k_i[..., None] * h, axis=1)   (n_v, d_v)
                    //   v_diff = v_i - decay * k_state
                    //   v_new = beta_i[:, None] * v_diff
                    //   q_state = sum(q_i[..., None] * h, axis=1)
                    //   q_k = sum(q_i * k_i, axis=-1, keepdims=True)
                    //   out_i = decay * q_state + q_k * v_new
                    //   k_v_new = k_i[..., None] * v_new[:, None, :]
                    //   h_new = h * decay[..., None] + k_v_new
                    //   h = sequence_valid ? h_new : h   (the seq-valid guard stubbed)
                    //   out_i = sequence_valid ? out_i : 0
                    const cnv_ty_tok = k.vectorTy(&.{ ctx.n_v, 1, ctx.d_k }, .f32);
                    const k_i_pre = k.emit(dialects.vector.extract_strided_slice(k.ctx, k_tr_t.inner, .{ .offsets = &.{ 0, @intCast(i_), 0 }, .sizes = &.{ ctx.n_v, 1, ctx.d_k }, .strides = &.{ 1, 1, 1 } }, cnv_ty_tok, k.loc()));
                    const k_i = k.shapeCast(k_i_pre, &.{ ctx.n_v, ctx.d_k });
                    const cnv_ty_tok_v = k.vectorTy(&.{ ctx.n_v, 1, ctx.d_v }, .f32);
                    const v_i_pre = k.emit(dialects.vector.extract_strided_slice(k.ctx, v_tr_t.inner, .{ .offsets = &.{ 0, @intCast(i_), 0 }, .sizes = &.{ ctx.n_v, 1, ctx.d_v }, .strides = &.{ 1, 1, 1 } }, cnv_ty_tok_v, k.loc()));
                    const v_i = k.shapeCast(v_i_pre, &.{ ctx.n_v, ctx.d_v });
                    const cnv_ty_tok_1 = k.vectorTy(&.{ ctx.n_v, 1 }, .f32);
                    const g_i_pre = k.emit(dialects.vector.extract_strided_slice(k.ctx, g_chunk_tr.inner, .{ .offsets = &.{ 0, @intCast(i_) }, .sizes = &.{ ctx.n_v, 1 }, .strides = &.{ 1, 1 } }, cnv_ty_tok_1, k.loc()));
                    // Pallas emits the (n_v,1)→(n_v,) flatten BETWEEN the g_i and beta_i
                    //   extracts so it lives right next to the math.exp input.  Mirror that.
                    const g_i_flat = k.shapeCast(g_i_pre, &.{ctx.n_v});
                    const beta_i_pre = k.emit(dialects.vector.extract_strided_slice(k.ctx, beta_tr.inner, .{ .offsets = &.{ 0, @intCast(i_) }, .sizes = &.{ ctx.n_v, 1 }, .strides = &.{ 1, 1 } }, cnv_ty_tok_1, k.loc()));
                    const q_i_pre = k.emit(dialects.vector.extract_strided_slice(k.ctx, q_sc_tr.inner, .{ .offsets = &.{ 0, @intCast(i_), 0 }, .sizes = &.{ ctx.n_v, 1, ctx.d_k }, .strides = &.{ 1, 1, 1 } }, cnv_ty_tok, k.loc()));
                    const q_i = k.shapeCast(q_i_pre, &.{ ctx.n_v, ctx.d_k });
                    const decay_flat = k.exp(g_i_flat);
                    const decay_tok = k.shapeCast(decay_flat, &.{ ctx.n_v, 1 });
                    // Pallas builds k_i_3d directly from k_i_pre (2x1x128 → 2x128x1) rather
                    //   than going through the flat k_i (2x128).
                    const k_i_3d = k.shapeCast(k_i_pre, &.{ ctx.n_v, ctx.d_k, 1 });
                    const k_i_bc = k.broadcastTo(k_i_3d, &.{ ctx.n_v, ctx.d_k, ctx.d_v });
                    const kh = k.mulf(k_i_bc, h_tr);
                    const zeros_nvdv_t = k.zeros(&.{ ctx.n_v, ctx.d_v }, .f32);
                    const k_state = k.multiReduction(.add, kh, zeros_nvdv_t, &.{1});
                    // Pallas emits broadcast(decay_tok, [n_v,d_v]) RIGHT AFTER the k_state
                    // multi_reduction (interleaved per-mulf), not eagerly upfront.
                    const decay_dv = k.broadcastTo(decay_tok, &.{ ctx.n_v, ctx.d_v });
                    const dks = k.mulf(decay_dv, k_state);
                    const v_diff_t = k.subf(v_i, dks);
                    const beta_dv = k.broadcastTo(beta_i_pre, &.{ ctx.n_v, ctx.d_v });
                    const v_new_t = k.mulf(beta_dv, v_diff_t);
                    const q_i_3d = k.shapeCast(q_i_pre, &.{ ctx.n_v, ctx.d_k, 1 });
                    const q_i_bc = k.broadcastTo(q_i_3d, &.{ ctx.n_v, ctx.d_k, ctx.d_v });
                    const qh = k.mulf(q_i_bc, h_tr);
                    const q_state = k.multiReduction(.add, qh, zeros_nvdv_t, &.{1});
                    const q_dot_k_tok = k.mulf(q_i, k_i);
                    const zeros_nv_t = k.zeros(&.{ctx.n_v}, .f32);
                    const q_k_tok = k.multiReduction(.add, q_dot_k_tok, zeros_nv_t, &.{1});
                    const q_k_tok_2d = k.shapeCast(q_k_tok, &.{ ctx.n_v, 1 });
                    // Pallas: dqs mulf BEFORE q_k_tok_bc broadcast (interleaved).
                    const dqs = k.mulf(decay_dv, q_state);
                    const q_k_tok_bc = k.broadcastTo(q_k_tok_2d, &.{ ctx.n_v, ctx.d_v });
                    const out_tok_part = k.mulf(q_k_tok_bc, v_new_t);
                    const out_tok = k.addf(dqs, out_tok_part);
                    const v_new_3d = k.shapeCast(v_new_t, &.{ ctx.n_v, 1, ctx.d_v });
                    const v_new_bc = k.broadcastTo(v_new_3d, &.{ ctx.n_v, ctx.d_k, ctx.d_v });
                    const k_v_new_t = k.mulf(k_i_bc, v_new_bc);
                    // decay_3d uses the FLAT decay (Pallas: %693 → vector<2x1x1xf32>).
                    const decay_3d = k.shapeCast(decay_flat, &.{ ctx.n_v, 1, 1 });
                    const decay_bc_3d = k.broadcastTo(decay_3d, &.{ ctx.n_v, ctx.d_k, ctx.d_v });
                    const h_dec = k.mulf(h_tr, decay_bc_3d);
                    const h_new_t = k.addf(h_dec, k_v_new_t);
                    // h = where(sequence_valid, h_new_t, h_prev_iter) — preserve the
                    // PREVIOUS iter_arg h (not the re-read new_h) when seq_valid is false.
                    h_tr = k.select(seq_valid, h_new_t, h_prev_iter);
                    // out_i = where(sequence_valid, out_tok, 0)
                    const out_tok_gated_native = k.select(seq_valid, out_tok, zeros_nvdv_t);
                    // sequence_valid = where(t_is_last > 0, False, sequence_valid).
                    //   Compute t_is_last_gt just-in-time here to match Pallas's emission.
                    const t_is_last_gt = k.cmpi(.sgt, t_is_last, c0i);
                    seq_valid = k.select(t_is_last_gt, false_i1, seq_valid);
                    // Write per-token output: prefill_output_ref[i, :] = out_tok.reshape(n_v*d_v).astype(dtype).
                    //   No additional seq_pred gating — the seq_valid-based gate above
                    //   already zeros the row, and Pallas's lowering writes the gated
                    //   value directly without the extra (1, n_v*d_v) where chain.
                    // Pallas does the shape transform around truncf as: shape_cast 2x128→256
                    //   (flat), truncf 256, shape_cast 256→1x256 (back to 1×N for store).
                    //   The destination memref slice + squeeze is emitted PER iter (not
                    //   hoisted) — JAX's `pl.store` per-token lowering produces this inline.
                    const out_flat_2d = k.shapeCast(out_tok_gated_native, &.{ctx.n_v * ctx.d_v});
                    const out_flat_dt = out_flat_2d.to(ctx.dtype);
                    const out_slice_t = k.memRefSlice(out_pre_w_tr.buf.?, &.{ out_pre_w_tr.slot.?, c0i, c0i }, &.{ 1, ctx.bt, ctx.n_v * ctx.d_v }, &.{});
                    const out_squeeze_t = k.memRefSqueeze(out_slice_t, &.{ ctx.bt, ctx.n_v * ctx.d_v });
                    const out_flat_dtype = k.shapeCast(out_flat_dt, &.{ 1, ctx.n_v * ctx.d_v });
                    k.vectorStoreAt(out_squeeze_t, out_flat_dtype, &.{ k.cIndex(@intCast(i_)), k.cIndex(0) });
                }
                // ── After-loop: final state commit ──
                //   final_slot = current_r % 2; prefill_scratch[final_slot] = h_tr;
                //   state_commit_scratch[0] = h_tr.astype(state_dtype);
                //   if (current_r >= decode_tokens): gather-DMA state_commit → recurrent_state_out
                //                                    [ds(state_indices[current_r], 1)]
                //                                    via prefill_sem[final_slot].
                //   Reuse the carried `c_slot_iter` (= iter 15's t_slot, a floor-mod
                //   chain result) as final_slot; emit toIndex for the prefill_scratch
                //   store target so Pallas's `index_cast` lands at the same diff position.
                //   prefill_scratch[final_slot] = h_tr (NOT slot 0 — semantic correctness
                //   AND IR match with Pallas).
                const final_slot = c_slot_iter;
                const final_slot_idx = k.toIndex(final_slot);
                const h_tr_4d = k.shapeCast(h_tr, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });
                k.vectorStoreAt(ctx.prefill_scratch, h_tr_4d, &.{ final_slot_idx, k.cIndex(0), k.cIndex(0), k.cIndex(0) });
                const h_tr_trunc = h_tr.to(ctx.state_dtype);
                const h_tr_trunc_4d = k.shapeCast(h_tr_trunc, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });
                k.vectorStoreAt(ctx.state_commit_scratch, h_tr_trunc_4d, &.{ k.cIndex(0), k.cIndex(0), k.cIndex(0), k.cIndex(0) });
                const final_is_prefill = k.cmpi(.sge, current_r_v, ctx.decode_tokens);
                var when_final_write = k.openIf(final_is_prefill);
                {
                    const final_idx = k.toIndex(current_r_v);
                    const final_state_idx = k.scalarLoad(scratches[1], &.{final_idx});
                    const final_dst = k.memRefSlice(ctx.recurrent_state_out, &.{ final_state_idx, c0i, c0i, c0i }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v }, &.{});
                    const final_sem_sl = k.memRefSlice(ctx.prefill_sem, &.{final_slot}, &.{1}, &.{});
                    const final_sem = k.memRefSqueeze(final_sem_sl, &.{});
                    k.enqueueDma(ctx.state_commit_scratch, final_dst, final_sem, .{});
                    k.waitDma2(final_sem, ctx.state_commit_scratch, final_dst, .{});
                    when_final_write.yieldThen(.{});
                }
                when_dispatch.yieldThen(.{});
            }
            // process_regular_prefill — `lax.cond` else branch (is_transition == 0).
            {
                // @pl.when(is_first_chunk > 0): init state.
                //   lax.cond(has_init > 0, load_from_hbm, zero_state)
                //   load_from_hbm: state_idx = state_indices[prefill_req_id];
                //                  sync_copy recurrent_state_in[ds(state_idx,1)] → state_commit;
                //                  prefill_scratch[prefill_slot] = state_commit[0].astype(f32);
                //   zero_state:    prefill_scratch[prefill_slot] = zeros<n_v,d_k,d_v xf32>;
                var when_first = k.openIf(k.cmpi(.sgt, is_first_chunk, c0i));
                {
                    const pr_idx = k.toIndex(prefill_req_id);
                    const has_init = k.scalarLoad(has_init_ref, &.{pr_idx});
                    // lax.cond(has_init > 0, load_from_hbm, zero_state) → one scf.if/else.
                    var when_has_init = k.openIfElse(k.cmpi(.sgt, has_init, c0i), .{});
                    {
                        const state_idx = k.scalarLoad(scratches[1], &.{pr_idx});
                        const src = k.memRefSlice(ctx.recurrent_state_in, &.{ state_idx, c0i, c0i, c0i }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v }, &.{});
                        // sem = prefill_sem[prefill_slot] — uses i32 directly; Pallas
                        //   emits `toIndex(prefill_slot)` later (after extf) for the store.
                        const sem_slice = k.memRefSlice(ctx.prefill_sem, &.{prefill_slot}, &.{1}, &.{});
                        const sem = k.memRefSqueeze(sem_slice, &.{});
                        k.enqueueDma(src, ctx.state_commit_scratch, sem, .{});
                        k.waitDma2(sem, src, ctx.state_commit_scratch, .{});
                        // prefill_scratch[prefill_slot] = state_commit_scratch[0].astype(f32)
                        //   Pallas: vector.load 4D → shape_cast 4D→3D → extf 3D → index_cast
                        //   → shape_cast 3D→4D → store.
                        const loaded_4d = k.vectorLoadShape(ctx.state_commit_scratch, &.{ k.cIndex(0), k.cIndex(0), k.cIndex(0), k.cIndex(0) }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });
                        const loaded_3d = k.shapeCast(loaded_4d, &.{ ctx.n_v, ctx.d_k, ctx.d_v });
                        const upcast_3d = loaded_3d.to(.f32);
                        const slot_idx = k.toIndex(prefill_slot);
                        const upcast_4d = k.shapeCast(upcast_3d, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });
                        k.vectorStoreAt(ctx.prefill_scratch, upcast_4d, &.{ slot_idx, k.cIndex(0), k.cIndex(0), k.cIndex(0) });
                        when_has_init.yieldThen(.{});
                    }
                    {
                        const slot_idx2 = k.toIndex(prefill_slot);
                        const zeros_state = k.zeros(&.{ 1, ctx.n_v, ctx.d_k, ctx.d_v }, .f32);
                        k.vectorStoreAt(ctx.prefill_scratch, zeros_state, &.{ slot_idx2, k.cIndex(0), k.cIndex(0), k.cIndex(0) });
                        when_has_init.yieldElse(.{});
                    }
                    when_first.yieldThen(.{});
                }

                // ── qkv_chunk = prefill_qkv_ref[...].astype(f32); SiLU; split q/k/v ──
                const dim_v2: i64 = 2 * ctx.n_kq * ctx.d_k + ctx.n_v * ctx.d_v;
                const key_dim2 = ctx.n_kq * ctx.d_k;
                const v_cols2 = ctx.n_v * ctx.d_v;
                const qkv_pre_w = refs[0]; // prefill mixed_qkv window
                // Pallas reads via dynamic `vector.load src[slot_idx, 0, 0]` (returning
                //   vector<1xCxdim_v2>) + shape_cast to (C, dim_v2) — no memref_slice +
                //   squeeze pair.  Mirror that to keep the IR identical.
                const qkv_pre_slot_idx = k.toIndex(qkv_pre_w.slot.?);
                const qkv_chunk_3d_bf = k.vectorLoadShape(qkv_pre_w.buf.?, &.{ qkv_pre_slot_idx, k.cIndex(0), k.cIndex(0) }, &.{ 1, ctx.C, dim_v2 });
                const qkv_chunk_bf = k.shapeCast(qkv_chunk_3d_bf, &.{ ctx.C, dim_v2 });
                const qkv_chunk_f = qkv_chunk_bf.to(.f32);
                // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
                const ones_C_dim = k.full(&.{ ctx.C, dim_v2 }, @as(f32, 1.0), .f32);
                const neg_qkv = k.negf(qkv_chunk_f);
                const exp_neg_qkv = k.exp(neg_qkv);
                const denom_qkv = k.addf(exp_neg_qkv, ones_C_dim);
                const sig_qkv = k.divf(ones_C_dim, denom_qkv);
                const silu_qkv = k.mulf(qkv_chunk_f, sig_qkv);
                // q,k,v = silu_qkv[:, :key_dim] / [key_dim:2*key_dim] / [2*key_dim:]
                const q_ty_pf = k.vectorTy(&.{ ctx.C, key_dim2 }, .f32);
                const k_ty_pf = k.vectorTy(&.{ ctx.C, key_dim2 }, .f32);
                const v_ty_pf = k.vectorTy(&.{ ctx.C, v_cols2 }, .f32);
                const q_chunk = k.emit(dialects.vector.extract_strided_slice(k.ctx, silu_qkv.inner, .{ .offsets = &.{ 0, 0 }, .sizes = &.{ ctx.C, key_dim2 }, .strides = &.{ 1, 1 } }, q_ty_pf, k.loc()));
                const k_chunk = k.emit(dialects.vector.extract_strided_slice(k.ctx, silu_qkv.inner, .{ .offsets = &.{ 0, key_dim2 }, .sizes = &.{ ctx.C, key_dim2 }, .strides = &.{ 1, 1 } }, k_ty_pf, k.loc()));
                const v_chunk = k.emit(dialects.vector.extract_strided_slice(k.ctx, silu_qkv.inner, .{ .offsets = &.{ 0, 2 * key_dim2 }, .sizes = &.{ ctx.C, v_cols2 }, .strides = &.{ 1, 1 } }, v_ty_pf, k.loc()));

                // ── a_raw / b_raw read + transpose to (n_v, C) ──
                //   Direct `vector.load buf[slot, 0, 0]` returning vector<1xCx128xf32>,
                //   shape_cast → (C, 128), extract_strided_slice [0:n_v] → (C, n_v),
                //   `tpu.transpose [1, 0]` → (n_v, C). Pattern matches the target's
                //   single vector.load (no memref_slice+squeeze needed when no dynamic
                //   row indexing).
                const a_p_w = refs[2]; // a_raw prefill window
                const a_p_slot_idx = k.toIndex(a_p_w.slot.?);
                const a_p_pre = k.vectorLoadShape(a_p_w.buf.?, &.{ a_p_slot_idx, k.cIndex(0), k.cIndex(0) }, &.{ 1, ctx.C, 128 });
                const a_p_2d = k.shapeCast(a_p_pre, &.{ ctx.C, 128 });
                const b_p_w = refs[4]; // b_raw prefill window
                const b_p_slot_idx = k.toIndex(b_p_w.slot.?);
                const b_p_pre = k.vectorLoadShape(b_p_w.buf.?, &.{ b_p_slot_idx, k.cIndex(0), k.cIndex(0) }, &.{ 1, ctx.C, 128 });
                const b_p_2d = k.shapeCast(b_p_pre, &.{ ctx.C, 128 });
                const cnv_ty = k.vectorTy(&.{ ctx.C, ctx.n_v }, .f32);
                const a_p_sl = k.emit(dialects.vector.extract_strided_slice(k.ctx, a_p_2d.inner, .{ .offsets = &.{ 0, 0 }, .sizes = &.{ ctx.C, ctx.n_v }, .strides = &.{ 1, 1 } }, cnv_ty, k.loc()));
                const a_raw_proc = k.transpose(a_p_sl, &.{ 1, 0 }); // (n_v, C)
                const b_p_sl = k.emit(dialects.vector.extract_strided_slice(k.ctx, b_p_2d.inner, .{ .offsets = &.{ 0, 0 }, .sizes = &.{ ctx.C, ctx.n_v }, .strides = &.{ 1, 1 } }, cnv_ty, k.loc()));
                const b_raw_proc = k.transpose(b_p_sl, &.{ 1, 0 });

                // ── sigmoid β = 1 / (1 + exp(-b_raw_proc)) ──
                const ones_nvC = k.full(&.{ ctx.n_v, ctx.C }, @as(f32, 1.0), .f32);
                const neg_b_pf = k.negf(b_raw_proc);
                const exp_neg_b_pf = k.exp(neg_b_pf);
                const denom_pf = k.addf(exp_neg_b_pf, ones_nvC);
                const beta_pf = k.divf(ones_nvC, denom_pf);

                // ── gate g = -exp(A_log) · softplus(a_raw_proc + dt_bias), clamp ≥ -100 ──
                //   A_log (n_v,) → shape_cast (n_v, 1); exp; -_ via subf zeros<n_v,1>.
                const a_log_pf = k.refLoad(refs[6].buf.?);
                const a_log_2d = k.shapeCast(a_log_pf, &.{ ctx.n_v, 1 });
                const a_log_exp_pf = k.exp(a_log_2d);
                const zeros_nv1 = k.zeros(&.{ ctx.n_v, 1 }, .f32);
                const neg_aval_pf = k.subf(zeros_nv1, a_log_exp_pf);
                const dt_bias_pf = k.refLoad(refs[7].buf.?);
                const dt_bias_2d_pf = k.shapeCast(dt_bias_pf, &.{ ctx.n_v, 1 });
                const dt_bc_pf = k.broadcastTo(dt_bias_2d_pf, &.{ ctx.n_v, ctx.C });
                const apdt_pf = k.addf(a_raw_proc, dt_bc_pf);
                const zeros_nvC = k.zeros(&.{ ctx.n_v, ctx.C }, .f32);
                const sp_max_pf = k.maximumf(apdt_pf, zeros_nvC);
                const sp_nan_pf = k.cmpf(.one, apdt_pf, apdt_pf);
                const sp_xp0_pf = k.addf(apdt_pf, zeros_nvC);
                const sp_abs_pf = k.absf(apdt_pf);
                const sp_neg_abs_pf = k.subf(zeros_nvC, sp_abs_pf);
                const sp_e_pf = k.exp(sp_neg_abs_pf);
                const sp_l1p_pf = k.emit(mlir.Operation.make(k.ctx, "math.log1p", .{
                    .operands = .{ .flat = &.{sp_e_pf.inner} },
                    .results = .{ .flat = &.{sp_e_pf.type_()} },
                    .location = k.loc(),
                }));
                const sp_sum_pf = k.addf(sp_max_pf, sp_l1p_pf);
                const softplus_pf = k.select(sp_nan_pf, sp_xp0_pf, sp_sum_pf);
                const neg_aval_bc_pf = k.broadcastTo(neg_aval_pf, &.{ ctx.n_v, ctx.C });
                const gk_pf = k.mulf(neg_aval_bc_pf, softplus_pf);
                const m100_pf = k.full(&.{ ctx.n_v, ctx.C }, @as(f32, -100.0), .f32);
                const g_pf_raw = k.maximumf(gk_pf, m100_pf);

                // ── prefill_count mask: zero-out trailing padding rows of q/k/v/g/β. ──
                //   `prefill_count = schedule[step, 3]; mask_float = (arange(C) < prefill_count).astype(q.dtype)`
                //   then `q = where(mask_float[:, None] > 0, q, 0)` etc. Pallas reuses the
                //   `(C,) f32 mask_float` across q/k/v (via `mask_float[:, None]` row broadcast)
                //   and g/β (via `mask_float[None, :]` col broadcast). We mirror exactly.
                const prefill_count_in = k.scalarLoad(sched, &.{ step_i, k.cIndex(3) });
                // Pallas's mask construction interleaves with the selects: build the
                // col mask up-front (for q, k), do the q/k selects, THEN build the row
                // mask (for g, β), select g, only then go back to broadcast the col
                // mask wider for v, select v, β (py-canon.mlir:3936-3948).
                const pf_iota_2d = k.iota(&.{ 1, ctx.C }, .i32, &.{1});
                const pf_iota_C = k.shapeCast(pf_iota_2d, &.{ctx.C});
                const pf_pc_bc = k.broadcastTo(prefill_count_in, &.{ctx.C});
                const pf_mask_i1 = k.cmpi(.slt, pf_iota_C, pf_pc_bc); // (C,) i1
                const pf_mask_f = pf_mask_i1.to(.f32); // (C,) f32 — mask_float
                const pf_mask_col = k.shapeCast(pf_mask_f, &.{ ctx.C, 1 }); // (C, 1) f32 — mask_float[:, None]
                const zeros_C1 = k.zeros(&.{ ctx.C, 1 }, .f32);
                const pf_mask_col_pos = k.cmpf(.ogt, pf_mask_col, zeros_C1); // (C, 1) i1
                // q, k: (C, key_dim) — col-broadcast mask
                const pf_mask_q_bc = k.broadcastTo(pf_mask_col_pos, &.{ ctx.C, key_dim2 });
                const zeros_C_kd = k.zeros(&.{ ctx.C, key_dim2 }, .f32);
                const q_masked = k.select(pf_mask_q_bc, q_chunk, zeros_C_kd);
                const k_masked = k.select(pf_mask_q_bc, k_chunk, zeros_C_kd);
                // Row mask, built lazily *after* the q/k selects — matches Pallas.
                const pf_mask_row = k.shapeCast(pf_mask_f, &.{ 1, ctx.C }); // (1, C) f32 — mask_float[None, :]
                const zeros_1C = k.zeros(&.{ 1, ctx.C }, .f32);
                const pf_mask_row_pos = k.cmpf(.ogt, pf_mask_row, zeros_1C); // (1, C) i1
                const pf_mask_g_bc = k.broadcastTo(pf_mask_row_pos, &.{ ctx.n_v, ctx.C });
                const g_pf = k.select(pf_mask_g_bc, g_pf_raw, zeros_nvC);
                // v: (C, v_cols) — col-broadcast mask widened (Pallas does this AFTER g).
                const pf_mask_v_bc = k.broadcastTo(pf_mask_col_pos, &.{ ctx.C, v_cols2 });
                const zeros_C_vc = k.zeros(&.{ ctx.C, v_cols2 }, .f32);
                const v_masked = k.select(pf_mask_v_bc, v_chunk, zeros_C_vc);
                const beta_masked = k.select(pf_mask_g_bc, beta_pf, zeros_nvC);

                // ── reshape + L2-norm + GQA repeat + transpose + scale ──
                //   q: (C, key_dim) → (C, n_kq, d_k) → L2-norm axis=-1 → GQA repeat axis=1
                //     → transpose [1, 0, 2] → (n_v, C, d_k) → ×scale
                //   k: same flow (no scale)
                //   v: (C, v_cols) → (C, n_v, d_v) → transpose [1, 0, 2] → (n_v, C, d_v)
                const q_3d = k.shapeCast(q_masked, &.{ ctx.C, ctx.n_kq, ctx.d_k });
                const k_3d = k.shapeCast(k_masked, &.{ ctx.C, ctx.n_kq, ctx.d_k });
                const v_3d = k.shapeCast(v_masked, &.{ ctx.C, ctx.n_v, ctx.d_v });
                // L2-norm q
                const eps_C_nkq_1 = k.full(&.{ ctx.C, ctx.n_kq, 1 }, @as(f32, 1e-6), .f32);
                const sum0_C_nkq = k.zeros(&.{ ctx.C, ctx.n_kq }, .f32);
                const q_sq3 = k.mulf(q_3d, q_3d);
                const q_sumsq3 = k.multiReduction(.add, q_sq3, sum0_C_nkq, &.{2});
                const q_sumsq3_3d = k.shapeCast(q_sumsq3, &.{ ctx.C, ctx.n_kq, 1 });
                const q_pre3 = k.addf(q_sumsq3_3d, eps_C_nkq_1);
                const q_sqrt3 = k.sqrt(q_pre3);
                const q_nrm3 = k.broadcastTo(q_sqrt3, &.{ ctx.C, ctx.n_kq, ctx.d_k });
                const q_n3 = k.divf(q_3d, q_nrm3);
                const k_sq3 = k.mulf(k_3d, k_3d);
                const k_sumsq3 = k.multiReduction(.add, k_sq3, sum0_C_nkq, &.{2});
                const k_sumsq3_3d = k.shapeCast(k_sumsq3, &.{ ctx.C, ctx.n_kq, 1 });
                const k_pre3 = k.addf(k_sumsq3_3d, eps_C_nkq_1);
                const k_sqrt3 = k.sqrt(k_pre3);
                const k_nrm3 = k.broadcastTo(k_sqrt3, &.{ ctx.C, ctx.n_kq, ctx.d_k });
                const k_n3 = k.divf(k_3d, k_nrm3);
                // GQA repeat: (C, n_kq, d_k) → shape_cast (C, n_kq, 1, d_k) → broadcast (C, n_kq, rf, d_k)
                //   → shape_cast (C, n_v, d_k). For n_kq=1, rf=n_v.
                const q_n4 = k.shapeCast(q_n3, &.{ ctx.C, ctx.n_kq, 1, ctx.d_k });
                const q_n4_bc = k.broadcastTo(q_n4, &.{ ctx.C, ctx.n_kq, @divExact(ctx.n_v, ctx.n_kq), ctx.d_k });
                const q_rep_pf = k.shapeCast(q_n4_bc, &.{ ctx.C, ctx.n_v, ctx.d_k });
                const k_n4 = k.shapeCast(k_n3, &.{ ctx.C, ctx.n_kq, 1, ctx.d_k });
                const k_n4_bc = k.broadcastTo(k_n4, &.{ ctx.C, ctx.n_kq, @divExact(ctx.n_v, ctx.n_kq), ctx.d_k });
                const k_rep_pf = k.shapeCast(k_n4_bc, &.{ ctx.C, ctx.n_v, ctx.d_k });
                // Transpose to (n_v, C, _).
                const q_pf = k.transpose(q_rep_pf, &.{ 1, 0, 2 });
                const k_pf = k.transpose(k_rep_pf, &.{ 1, 0, 2 });
                const v_pf = k.transpose(v_3d, &.{ 1, 0, 2 });
                // Scale q.
                const scale_pf_f32: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(ctx.d_k)));
                const scale_pf_v = k.full(&.{ ctx.n_v, ctx.C, ctx.d_k }, scale_pf_f32, .f32);
                const q_sc_pf = k.mulf(q_pf, scale_pf_v);

                // ── cumsum g over C (UNROLLED Python loop) ──
                //   g_cumsum[h, i] = sum_{j=0..i} g[h, j].
                //   Pallas's lowering: keep the running sum FLAT `(n_v,)` through the
                //   16-iter chain, then shape_cast each running snapshot to `(n_v, 1)`
                //   *after* the loop and concatenate along dim 1. Doing the shape_cast
                //   inside the loop (one per iter) emits 16 extra `vector.shape_cast`
                //   ops vs Pallas, which cascades through the rest of the body.
                const C_const: usize = @intCast(ctx.C);
                var cumsum_running = k.zeros(&.{ctx.n_v}, .f32);
                var cumsum_flat: [128]Value = undefined;
                std.debug.assert(C_const <= cumsum_flat.len);
                const cumsum_ty_1 = k.vectorTy(&.{ ctx.n_v, 1 }, .f32);
                // Cap is `chunk_size_max = 128` (`invert_triangular_matrix` block size 16 ×
                //   max num_blocks = 8); the runtime `break` stops emission at C_const.
                inline for (0..128) |i_| {
                    if (i_ >= C_const) break;
                    const slice = k.emit(dialects.vector.extract_strided_slice(k.ctx, g_pf.inner, .{ .offsets = &.{ 0, @intCast(i_) }, .sizes = &.{ ctx.n_v, 1 }, .strides = &.{ 1, 1 } }, cumsum_ty_1, k.loc()));
                    const slice_1d = k.shapeCast(slice, &.{ctx.n_v});
                    cumsum_running = k.addf(cumsum_running, slice_1d);
                    cumsum_flat[i_] = cumsum_running; // (n_v,) — defer the shape_cast
                }
                // shape_cast each (n_v,) to (n_v, 1), then concat along dim 1 → (n_v, C).
                var cumsum_cols: [128]Value = undefined;
                inline for (0..128) |i_| {
                    if (i_ >= C_const) break;
                    cumsum_cols[i_] = k.shapeCast(cumsum_flat[i_], &.{ ctx.n_v, 1 });
                }
                // Pallas's `tpu.concatenate` lowering caps the arity at 16 operands; for
                //   chunk_size > 16 it emits `ceil(C/16)` 16-wide concats then a final
                //   merging concat.  Mirror that exactly so chunk_size>16 sweeps match.
                const g_cumsum = blk_gcum: {
                    if (C_const <= 16) break :blk_gcum k.concatenate(cumsum_cols[0..C_const], 1);
                    var groups: [8]Value = undefined;
                    var n_groups: usize = 0;
                    var ig: usize = 0;
                    while (ig < C_const) : (ig += 16) {
                        const end_g = @min(ig + 16, C_const);
                        groups[n_groups] = k.concatenate(cumsum_cols[ig..end_g], 1);
                        n_groups += 1;
                    }
                    break :blk_gcum k.concatenate(groups[0..n_groups], 1);
                };

                // ── k_beta = k_pf * beta[..., None] → (n_v, C, d_k) ──
                const beta_3d = k.shapeCast(beta_masked, &.{ ctx.n_v, ctx.C, 1 });
                const beta_bc_dk = k.broadcastTo(beta_3d, &.{ ctx.n_v, ctx.C, ctx.d_k });
                const k_beta = k.mulf(k_pf, beta_bc_dk);

                // ── S = matmul(k_beta, k_pf^T) — batched (n_v, C, C); dim_numbers
                //   <[2], [1], [1], [2], [0, 0, 0, 1, 1, 2], [0], [0]>; precision=fp32. ──
                const ddn_b = k.dotDimensionNumbers(
                    &.{2},
                    &.{1},
                    &.{1},
                    &.{2},
                    &.{ 0, 0, 0, 1, 1, 2 },
                    &.{0},
                    &.{0},
                );
                const mm_b_opts = mtt.MatmulOpts{ .precision = .fp32, .dimension_numbers = ddn_b };
                const k_T = k.transpose(k_pf, &.{ 0, 2, 1 }); // (n_v, d_k, C)
                const S_zero = k.zeros(&.{ ctx.n_v, ctx.C, ctx.C }, .f32);
                const S_raw = k.matmulOpts(k_beta, k_T, S_zero, mm_b_opts);

                // ── g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :] ──
                //     g_cumsum: (n_v, C). g_diff: (n_v, C, C).
                // Pallas emits both shape_casts BEFORE either broadcast (paired) — match.
                const g_cumsum_col = k.shapeCast(g_cumsum, &.{ ctx.n_v, ctx.C, 1 });
                const g_cumsum_row = k.shapeCast(g_cumsum, &.{ ctx.n_v, 1, ctx.C });
                const g_cumsum_col_bc = k.broadcastTo(g_cumsum_col, &.{ ctx.n_v, ctx.C, ctx.C });
                const g_cumsum_row_bc = k.broadcastTo(g_cumsum_row, &.{ ctx.n_v, ctx.C, ctx.C });
                const g_diff = k.subf(g_cumsum_col_bc, g_cumsum_row_bc);
                const zeros_CC = k.zeros(&.{ ctx.n_v, ctx.C, ctx.C }, .f32);
                // Pallas reuses the earlier `(1, C) i32` iota from the prefill_count
                // mask via shape_cast + broadcast — `i_iota` becomes shape_cast→(C,1)
                // then broadcast→(C,C); `j_iota` is a direct broadcast of (1,C)→(C,C).
                // Emitting two fresh `iota (C, C)` ops costs ~50 lines of cascade.
                const j_iota_col_pre = k.shapeCast(pf_iota_2d, &.{ ctx.C, 1 });
                const i_iota = k.broadcastTo(j_iota_col_pre, &.{ ctx.C, ctx.C });
                const j_iota = k.broadcastTo(pf_iota_2d, &.{ ctx.C, ctx.C });
                // tri_mask via the (1, C, C) detour Pallas uses (uitofp → shape_cast →
                //   cmpf ogt zeros → broadcast → select).  Pallas's lowering interleaves
                //   minimumf for g_diff_safe AFTER the uitofp; mirror that op order.
                const tri_mask_i1_2d = k.cmpi(.sgt, i_iota, j_iota); // (C, C) i1
                const tri_mask_f_2d = k.arithUitofp(tri_mask_i1_2d, .f32); // (C, C) f32
                const g_diff_safe = k.emit(mlir.Operation.make(k.ctx, "arith.minimumf", .{
                    .operands = .{ .flat = &.{ g_diff.inner, zeros_CC.inner } },
                    .results = .{ .flat = &.{g_diff.type_()} },
                    .location = k.loc(),
                }));
                const tri_mask_3d_f = k.shapeCast(tri_mask_f_2d, &.{ 1, ctx.C, ctx.C });
                const tri_zeros_1CC = k.zeros(&.{ 1, ctx.C, ctx.C }, .f32);
                const tri_mask_3d_i1 = k.cmpf(.ogt, tri_mask_3d_f, tri_zeros_1CC);
                const S_expgd = k.mulf(S_raw, k.exp(g_diff_safe));
                const tri_mask_bc = k.broadcastTo(tri_mask_3d_i1, &.{ ctx.n_v, ctx.C, ctx.C });
                const S_masked = k.select(tri_mask_bc, S_expgd, zeros_CC);

                // ── S_q = matmul(q_sc, k_pf^T) → (n_v, C, C) — same dim_numbers as S ──
                const S_q_raw = k.matmulOpts(q_sc_pf, k_T, S_zero, mm_b_opts);
                // Pallas's `g_diff_Sq` chain uses mul-add through (1, C, C) intermediates
                //   instead of a select.  Logic:
                //     mask_f (C,C) = uitofp(triq_mask_i1)
                //     mask_nvCC = broadcast(mask_f, [n_v, C, C])
                //     g_diff_masked = mulf(g_diff_safe, mask_nvCC)         // (n_v, C, C)
                //     mask_1CC = shape_cast(mask_f, [1, C, C])
                //     neg_inf_1CC = mulf(subf(ones_1CC, mask_1CC), -1e30)  // (1, C, C)
                //     neg_inf_nvCC = broadcast(neg_inf_1CC, [n_v, C, C])
                //     g_diff_Sq = addf(g_diff_masked, neg_inf_nvCC)
                //   S_q_masked = mulf(S_q_expgd, mask_nvCC) — reuses the broadcast above.
                const triq_mask_i1_2d = k.cmpi(.sge, i_iota, j_iota);
                const triq_mask_f_2d = k.arithUitofp(triq_mask_i1_2d, .f32);
                const triq_mask_nvCC_f = k.broadcastTo(triq_mask_f_2d, &.{ ctx.n_v, ctx.C, ctx.C });
                const g_diff_masked = k.mulf(g_diff_safe, triq_mask_nvCC_f);
                const triq_mask_1CC_f = k.shapeCast(triq_mask_f_2d, &.{ 1, ctx.C, ctx.C });
                const ones_1CC = k.full(&.{ 1, ctx.C, ctx.C }, @as(f32, 1.0), .f32);
                const ones_minus_mask_1CC = k.subf(ones_1CC, triq_mask_1CC_f);
                const neg_inf_1CC = k.full(&.{ 1, ctx.C, ctx.C }, @as(f32, -1.0e30), .f32);
                const neg_inf_part_1CC = k.mulf(ones_minus_mask_1CC, neg_inf_1CC);
                const neg_inf_part_nvCC = k.broadcastTo(neg_inf_part_1CC, &.{ ctx.n_v, ctx.C, ctx.C });
                const g_diff_Sq = k.addf(g_diff_masked, neg_inf_part_nvCC);
                const S_q_expgd = k.mulf(S_q_raw, k.exp(g_diff_Sq));
                const S_q_masked = k.mulf(S_q_expgd, triq_mask_nvCC_f);

                // ── I_plus_S = eye(C) + S_masked  (n_v, C, C) ──
                //     Pallas emits fresh (C, C) iotas for the eye mask (rather than
                //     reusing the prefill_count `pf_iota_2d` via shape_cast/broadcast),
                //     then broadcasts the (C, C) f32 eye directly to (n_v, C, C) with
                //     no `(1, C, C)` intermediate.
                const iota_eye_0 = k.iota(&.{ ctx.C, ctx.C }, .i32, &.{0});
                const iota_eye_1 = k.iota(&.{ ctx.C, ctx.C }, .i32, &.{1});
                const eye_mask_i1 = k.cmpi(.eq, iota_eye_0, iota_eye_1);
                const eye_f = eye_mask_i1.to(.f32); // (C, C) f32
                const eye_bc = k.broadcastTo(eye_f, &.{ ctx.n_v, ctx.C, ctx.C });
                const I_plus_S = k.addf(eye_bc, S_masked);

                // ── A_inv = invert_triangular_matrix(I_plus_S, block_size=16) ──
                //   Block-Gaussian inverse over `num_blocks = C / 16` blocks.  For each
                //   block b ∈ [0, num_blocks):
                //     start, end = b*16, (b+1)*16
                //     e_block = eye_f[start:end, :]  (16, C), broadcast → (n_v, 16, C)
                //     if b == 0: target_b = e_block
                //     else:      interaction_A = I_plus_S[:, start:end, :start]
                //                solved_x = concat(x_blocks[0..b], axis=1)
                //                prev_sum = matmul(interaction_A, solved_x)
                //                target_b = e_block - prev_sum
                //     x_block = local_forward_sub(I_plus_S[:, start:end, start:end], target_b)
                //   For chunk_size=16, num_blocks=1 and the cross-block branch is skipped,
                //     producing the original single-block forward-sub pattern.  For C=64,
                //     num_blocks=4 (4 cross-block matmuls + 64 forward-sub rows).
                const block_size_const: usize = 16;
                const max_num_blocks: usize = 8; // chunk_size up to 128
                const N_const: usize = @intCast(ctx.C);
                const num_blocks_const: usize = N_const / block_size_const;
                std.debug.assert(N_const % block_size_const == 0);
                std.debug.assert(num_blocks_const <= max_num_blocks);
                const eye_f_NN = k.arithUitofp(eye_mask_i1, .f32); // (C, C) f32 — Pallas's %746-equivalent
                const zeros_nvC_fs = k.zeros(&.{ ctx.n_v, ctx.C }, .f32);
                var x_blocks: [max_num_blocks]Value = undefined;
                inline for (0..max_num_blocks) |block_idx| {
                    if (block_idx >= num_blocks_const) break;
                    const start: i64 = @intCast(block_idx * block_size_const);

                    // Block-0 with num_blocks==1 keeps the original optimized pattern
                    //   (`vectorExtract eye_f[i] + broadcast` per row); for chunk_size > 16
                    //   or block_idx > 0 we build an explicit `target_b: (n_v, 16, C)` and
                    //   extract rows via `extract_strided_slice`.
                    if (num_blocks_const == 1) {
                        var x_blocks_3d: [16]Value = undefined;
                        const b_0_init = k.vectorExtract(eye_f_NN, &.{0});
                        var b_i_bc = k.broadcastTo(b_0_init, &.{ ctx.n_v, ctx.C });
                        inline for (0..block_size_const) |i_| {
                            var x_i = b_i_bc;
                            if (i_ > 0) {
                                const i_int: i64 = @intCast(i_);
                                const stacked_x = if (i_ == 1) x_blocks_3d[0] else k.concatenate(x_blocks_3d[0..i_], 1);
                                const a_row_ty = k.vectorTy(&.{ ctx.n_v, 1, i_int }, .f32);
                                const a_row = k.emit(dialects.vector.extract_strided_slice(k.ctx, I_plus_S.inner, .{ .offsets = &.{ 0, i_int, 0 }, .sizes = &.{ ctx.n_v, 1, i_int }, .strides = &.{ 1, 1, 1 } }, a_row_ty, k.loc()));
                                const a_T = k.shapeCast(a_row, &.{ ctx.n_v, i_int, 1 });
                                const a_bc = k.broadcastTo(a_T, &.{ ctx.n_v, i_int, ctx.C });
                                const term = k.mulf(a_bc, stacked_x);
                                const prev_sum = k.multiReduction(.add, term, zeros_nvC_fs, &.{1});
                                x_i = k.subf(b_i_bc, prev_sum);
                            }
                            if (i_ + 1 < block_size_const) {
                                const b_next = k.vectorExtract(eye_f_NN, &.{@as(i64, @intCast(i_ + 1))});
                                b_i_bc = k.broadcastTo(b_next, &.{ ctx.n_v, ctx.C });
                            }
                            x_blocks_3d[i_] = k.shapeCast(x_i, &.{ ctx.n_v, 1, ctx.C });
                        }
                        x_blocks[0] = k.concatenate(x_blocks_3d[0..], 1);
                        break;
                    }

                    // num_blocks > 1: build target_b explicitly.
                    const eblock_ty = k.vectorTy(&.{ block_size_const, ctx.C }, .f32);
                    const e_block = k.emit(dialects.vector.extract_strided_slice(k.ctx, eye_f_NN.inner, .{ .offsets = &.{ start, 0 }, .sizes = &.{ block_size_const, ctx.C }, .strides = &.{ 1, 1 } }, eblock_ty, k.loc()));
                    const e_block_3d = k.shapeCast(e_block, &.{ 1, block_size_const, ctx.C });
                    const e_block_nv = k.broadcastTo(e_block_3d, &.{ ctx.n_v, block_size_const, ctx.C });
                    var target_b: Value = e_block_nv;
                    if (block_idx > 0) {
                        const ia_ty = k.vectorTy(&.{ ctx.n_v, block_size_const, start }, .f32);
                        const interaction_A = k.emit(dialects.vector.extract_strided_slice(k.ctx, I_plus_S.inner, .{ .offsets = &.{ 0, start, 0 }, .sizes = &.{ ctx.n_v, block_size_const, start }, .strides = &.{ 1, 1, 1 } }, ia_ty, k.loc()));
                        const solved_x = if (block_idx == 1) x_blocks[0] else k.concatenate(x_blocks[0..block_idx], 1);
                        const ps_zero = k.zeros(&.{ ctx.n_v, block_size_const, ctx.C }, .f32);
                        const prev_sum_x = k.matmulOpts(interaction_A, solved_x, ps_zero, mm_b_opts);
                        target_b = k.subf(e_block_nv, prev_sum_x);
                    }

                    var x_rows_3d: [16]Value = undefined;
                    const b_row_ty = k.vectorTy(&.{ ctx.n_v, 1, ctx.C }, .f32);
                    // Pre-emit b_0 (initial row); Pallas emits shape_cast(b_3d → b) for
                    //   iter i+1 BETWEEN the extract for iter i+1 and the shape_cast of
                    //   iter i's x_i — mirror by carrying b_i_2d across iters.
                    const b_0_3d = k.emit(dialects.vector.extract_strided_slice(k.ctx, target_b.inner, .{ .offsets = &.{ 0, 0, 0 }, .sizes = &.{ ctx.n_v, 1, ctx.C }, .strides = &.{ 1, 1, 1 } }, b_row_ty, k.loc()));
                    var b_i_2d = k.shapeCast(b_0_3d, &.{ ctx.n_v, ctx.C });
                    inline for (0..block_size_const) |i_| {
                        var x_i = b_i_2d;
                        if (i_ > 0) {
                            const i_int: i64 = @intCast(i_);
                            const stacked_x = if (i_ == 1) x_rows_3d[0] else k.concatenate(x_rows_3d[0..i_], 1);
                            const a_row_ty = k.vectorTy(&.{ ctx.n_v, 1, i_int }, .f32);
                            const a_row = k.emit(dialects.vector.extract_strided_slice(k.ctx, I_plus_S.inner, .{ .offsets = &.{ 0, start + i_int, start }, .sizes = &.{ ctx.n_v, 1, i_int }, .strides = &.{ 1, 1, 1 } }, a_row_ty, k.loc()));
                            const a_T = k.shapeCast(a_row, &.{ ctx.n_v, i_int, 1 });
                            const a_bc = k.broadcastTo(a_T, &.{ ctx.n_v, i_int, ctx.C });
                            const term = k.mulf(a_bc, stacked_x);
                            const prev_sum_in = k.multiReduction(.add, term, zeros_nvC_fs, &.{1});
                            x_i = k.subf(b_i_2d, prev_sum_in);
                        }
                        // Pallas emits NEXT iter's `extract + shape_cast b_i_3d → b_i`
                        //   AHEAD of this iter's `shape_cast x_i → 3D`.
                        if (i_ + 1 < block_size_const) {
                            const next_b_3d = k.emit(dialects.vector.extract_strided_slice(k.ctx, target_b.inner, .{ .offsets = &.{ 0, @as(i64, @intCast(i_ + 1)), 0 }, .sizes = &.{ ctx.n_v, 1, ctx.C }, .strides = &.{ 1, 1, 1 } }, b_row_ty, k.loc()));
                            b_i_2d = k.shapeCast(next_b_3d, &.{ ctx.n_v, ctx.C });
                        }
                        x_rows_3d[i_] = k.shapeCast(x_i, &.{ ctx.n_v, 1, ctx.C });
                    }
                    x_blocks[block_idx] = k.concatenate(x_rows_3d[0..], 1);
                }
                const A_inv = if (num_blocks_const == 1) x_blocks[0] else k.concatenate(x_blocks[0..num_blocks_const], 1);

                // ── Now the remaining matmul chain. ──
                //   v_beta = v_pf * beta[..., None]  (n_v, C, d_v)
                //   u = matmul(A_inv, v_beta)        (n_v, C, d_v)
                //   k_beta_g = k_beta * exp(g_cumsum)[..., None]  (n_v, C, d_k)
                //   w = matmul(A_inv, k_beta_g)      (n_v, C, d_k)
                //   q_g = q_sc * exp(g_cumsum)[..., None]
                //   current_state = prefill_scratch[slot]  (n_v, d_k, d_v)
                //   attn_inter = matmul(q_g, current_state)
                //   v_prime = matmul(w, current_state)
                //   v_new = u - v_prime
                //   term2 = matmul(S_q_masked, v_new)
                //   o_c = attn_inter + term2
                //   g_i_last_exp = exp(g_cumsum[..., -1])[..., None, None]
                //   g_diff_exp_state = exp(g_cumsum[..., -1, None] - g_cumsum)[..., None]
                //   k_i_g_diff = k_pf * g_diff_exp_state
                //   update_term = matmul(k_i_g_diff^T, v_new)
                //   h_new = current_state * g_i_last_exp + update_term
                const beta_bc_dv = k.broadcastTo(beta_3d, &.{ ctx.n_v, ctx.C, ctx.d_v });
                const v_beta = k.mulf(v_pf, beta_bc_dv);
                const u_zero = k.zeros(&.{ ctx.n_v, ctx.C, ctx.d_v }, .f32);
                const u = k.matmulOpts(A_inv, v_beta, u_zero, mm_b_opts);
                const exp_gcumsum = k.exp(g_cumsum);
                const exp_gcumsum_dk = k.broadcastTo(k.shapeCast(exp_gcumsum, &.{ ctx.n_v, ctx.C, 1 }), &.{ ctx.n_v, ctx.C, ctx.d_k });
                const k_beta_g = k.mulf(k_beta, exp_gcumsum_dk);
                const w_zero = k.zeros(&.{ ctx.n_v, ctx.C, ctx.d_k }, .f32);
                const w = k.matmulOpts(A_inv, k_beta_g, w_zero, mm_b_opts);
                const q_g = k.mulf(q_sc_pf, exp_gcumsum_dk);
                const slot_idx_pf = k.toIndex(prefill_slot);
                const cs_loaded = k.vectorLoadShape(ctx.prefill_scratch, &.{ slot_idx_pf, k.cIndex(0), k.cIndex(0), k.cIndex(0) }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });
                const current_state_pf = k.shapeCast(cs_loaded, &.{ ctx.n_v, ctx.d_k, ctx.d_v });
                const attn_inter = k.matmulOpts(q_g, current_state_pf, u_zero, mm_b_opts);
                const v_prime = k.matmulOpts(w, current_state_pf, u_zero, mm_b_opts);
                const v_new_pf = k.subf(u, v_prime);
                const term2 = k.matmulOpts(S_q_masked, v_new_pf, u_zero, mm_b_opts);
                const o_c = k.addf(attn_inter, term2);

                // g_i_last_exp: exp(g_cumsum[:, -1])[:, None, None] → (n_v, 1, 1).
                //   Pallas shape_casts (n_v,1)→(n_v,1,1) directly (no flatten detour) and
                //   exp on the 3D shape; the broadcast to (n_v, d_k, d_v) is emitted later
                //   just-in-time before its mulf.
                const last_col_ty = k.vectorTy(&.{ ctx.n_v, 1 }, .f32);
                const g_last = k.emit(dialects.vector.extract_strided_slice(k.ctx, g_cumsum.inner, .{ .offsets = &.{ 0, @intCast(@as(i64, ctx.C - 1)) }, .sizes = &.{ ctx.n_v, 1 }, .strides = &.{ 1, 1 } }, last_col_ty, k.loc()));
                const g_last_3d = k.shapeCast(g_last, &.{ ctx.n_v, 1, 1 });
                const g_last_exp_3d = k.exp(g_last_3d);
                // g_diff_exp_state = exp(g_cumsum[:, -1, None] - g_cumsum)[..., None] (n_v, C, 1)
                const g_last_bc = k.broadcastTo(g_last, &.{ ctx.n_v, ctx.C });
                const g_diff_state = k.subf(g_last_bc, g_cumsum);
                const g_diff_state_exp = k.exp(g_diff_state);
                const g_diff_state_3d = k.shapeCast(g_diff_state_exp, &.{ ctx.n_v, ctx.C, 1 });
                const g_diff_state_bc = k.broadcastTo(g_diff_state_3d, &.{ ctx.n_v, ctx.C, ctx.d_k });
                const k_i_g_diff = k.mulf(k_pf, g_diff_state_bc);
                // update_term = matmul(k_i_g_diff^T, v_new) — k_i_g_diff is (n_v, C, d_k);
                //   transpose to (n_v, d_k, C) then matmul against v_new (n_v, C, d_v) → (n_v, d_k, d_v).
                const kigd_T = k.transpose(k_i_g_diff, &.{ 0, 2, 1 });
                const update_zero = k.zeros(&.{ ctx.n_v, ctx.d_k, ctx.d_v }, .f32);
                const update_term = k.matmulOpts(kigd_T, v_new_pf, update_zero, mm_b_opts);
                // Pallas broadcasts g_last_exp_3d (n_v,1,1) to (n_v,d_k,d_v) just-in-time
                //   before the mulf — emit it here.
                const g_last_exp_dkdv = k.broadcastTo(g_last_exp_3d, &.{ ctx.n_v, ctx.d_k, ctx.d_v });
                const cs_scaled = k.mulf(current_state_pf, g_last_exp_dkdv);
                const h_new = k.addf(cs_scaled, update_term);
                const h_new_4d = k.shapeCast(h_new, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });
                k.vectorStoreAt(ctx.prefill_scratch, h_new_4d, &.{ slot_idx_pf, k.cIndex(0), k.cIndex(0), k.cIndex(0) });
                // The output is now `o_c` (n_v, C, d_v); shape-cast/transpose to (C, n_v*d_v) for the output write below.
                const o_c_pf = o_c;

                // Note: the full chunked WY/UT compute middle (a/b_raw transpose +
                // sigmoid β + gate g + clamp; reshape + L2norm + GQA + transpose +
                // ×scale; cumsum-over-C UNROLLED; k_beta; S = matmul; g_diff; S_q;
                // I+S → A_inv via invert_triangular_matrix(blk=16) UNROLLED;
                // u/w/q_g/attn_inter/v_prime/v_new/term2/k_i_g_diff/update_term/h_new) is
                // implemented above; the comment block below was the original spec and is
                // retained as a structural cross-reference to Pallas's process_regular_prefill.
                //
                // Reference (Pallas's process_regular_prefill, in source order — implemented above):
                //   a_raw/b_raw transpose + sigmoid β + gate g (clamp ≥ -100); mask
                //   `arange(C) < prefill_count`; reshape (C, n_kq, d_k) + L2norm + GQA +
                //   transpose (H, C, D) + ×scale; cumsum g over C (UNROLLED); k_beta;
                //   S = matmul(k_beta, k^T, HIGHEST); g_diff masks; S_q = matmul(q, k^T);
                //   I_plus_S = eye(C) + S; A_inv = invert_triangular_matrix(blk=16) —
                //   block-Gaussian (1 block × 16 forward_sub for C=16); u = A_inv·v_beta;
                //   w = A_inv·(k_beta·exp(g_cumsum)); q_g = q·exp(g_cumsum); current_state =
                //   prefill_scratch[slot]; attn_inter = q_g·state; v_prime = w·state;
                //   v_new = u - v_prime; term2 = S_q·v_new; o_c = attn_inter + term2;
                //   g_i_last_exp + g_diff_exp_state + k_i_g_diff; update_term =
                // k_i_g_diff^T·v_new; h_new = state·g_i_last_exp + update_term;
                // prefill_scratch[slot] = h_new.

                // @pl.when(is_last_chunk > 0): commit state to HBM.
                var when_last = k.openIf(k.cmpi(.sgt, is_last_chunk, c0i));
                {
                    const slot_idx_l = k.toIndex(prefill_slot);
                    const ps_loaded = k.vectorLoadShape(ctx.prefill_scratch, &.{ slot_idx_l, k.cIndex(0), k.cIndex(0), k.cIndex(0) }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });
                    const ps_3d = k.shapeCast(ps_loaded, &.{ ctx.n_v, ctx.d_k, ctx.d_v });
                    const ps_trunc = ps_3d.to(ctx.state_dtype);
                    const ps_4d = k.shapeCast(ps_trunc, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v });
                    k.vectorStoreAt(ctx.state_commit_scratch, ps_4d, &.{ k.cIndex(0), k.cIndex(0), k.cIndex(0), k.cIndex(0) });
                    const pr_idx2 = k.toIndex(prefill_req_id);
                    const state_idx_l = k.scalarLoad(scratches[1], &.{pr_idx2});
                    const dst = k.memRefSlice(ctx.recurrent_state_out, &.{ state_idx_l, c0i, c0i, c0i }, &.{ 1, ctx.n_v, ctx.d_k, ctx.d_v }, &.{});
                    const sem_slice2 = k.memRefSlice(ctx.prefill_sem, &.{prefill_slot}, &.{1}, &.{});
                    const sem2 = k.memRefSqueeze(sem_slice2, &.{});
                    k.enqueueDma(ctx.state_commit_scratch, dst, sem2, .{});
                    k.waitDma2(sem2, ctx.state_commit_scratch, dst, .{});
                    when_last.yieldThen(.{});
                }

                // ── Prefill output write: o_c.transpose(1, 0, 2).reshape(C, n_v*d_v) * mask
                //   → refs[8] (prefill output window). Stub o_c here from stub_summed (n_v, d_v)
                //   broadcast to (n_v, C, d_v); the real o_c is `attn_inter + S_q·v_new`. ──
                const oc_t = k.transpose(o_c_pf, &.{ 1, 0, 2 }); // (n_v, C, d_v) → (C, n_v, d_v)
                const oc_flat = k.shapeCast(oc_t, &.{ ctx.C, ctx.n_v * ctx.d_v });
                // Pallas reads `prefill_count` AFTER the transpose+shape_cast, just before
                //   its mask use — match that emission position.
                const prefill_count = k.scalarLoad(sched, &.{ step_i, k.cIndex(3) });
                const oc_iota_2d = k.iota(&.{ 1, ctx.C }, .i32, &.{1});
                const oc_iota = k.shapeCast(oc_iota_2d, &.{ctx.C});
                const pc_bc = k.broadcastTo(prefill_count, &.{ctx.C});
                const oc_mask_i1 = k.cmpi(.slt, oc_iota, pc_bc);
                const oc_mask_f = oc_mask_i1.to(.f32);
                const oc_mask_2d = k.shapeCast(oc_mask_f, &.{ ctx.C, 1 });
                const oc_mask_full = k.broadcastTo(oc_mask_2d, &.{ ctx.C, ctx.n_v * ctx.d_v });
                const oc_masked_f32 = k.mulf(oc_flat, oc_mask_full);
                const oc_masked = oc_masked_f32.to(ctx.dtype);
                const out_pre_w2 = refs[8];
                const out_pre_slot_idx = k.toIndex(out_pre_w2.slot.?);
                const oc_3d = k.shapeCast(oc_masked, &.{ 1, ctx.C, ctx.n_v * ctx.d_v });
                k.vectorStoreAt(out_pre_w2.buf.?, oc_3d, &.{ out_pre_slot_idx, k.cIndex(0), k.cIndex(0) });

                when_dispatch.yieldElse(.{});
            }
            _ = ctx.prefill_scratch;
            _ = ctx.prefill_sem;
            when.yieldThen(.{});
        }
    }

    // ── (3) Stitch path: `lax.cond((is_transition > 0) & (program_id(0) == 0) &
    //   (decode_valid > 0), do_stitch, noop)`. ──
    {
        const is_first_block = k.cmpi(.eq, step, c0i);
        const tr_gt0 = k.cmpi(.sgt, is_transition, c0i);
        const dv_gt0 = k.cmpi(.sgt, decode_valid, c0i);
        const needs_stitching = k.andi(k.andi(tr_gt0, is_first_block), dv_gt0);
        var when = k.openIf(needs_stitching);
        {
            // do_stitch:
            //   local_start = prefill_offset - decode_offset
            //   local_split = decode_tokens - prefill_offset
            //   safe_local_start = pl.multiple_of(local_start, sublanesize)
            //   decode_overlap = decode_output_ref[ds(safe_local_start, sl), :]
            //   prefill_arr    = prefill_output_ref[ds(0, sl), :]
            //   iota = arange(sl) < local_split
            //   merged = where(iota<local_split, decode_overlap, prefill_arr)
            //   decode_output_ref[ds(safe_local_start, sl), :] = merged
            //   prefill_output_ref[ds(0, sl), :]                = merged
            const local_start = k.subi(prefill_offset, decode_offset);
            const local_split = k.subi(ctx.decode_tokens, prefill_offset);
            const safe_local_start = k.emit(tpu.assume_multiple(k.ctx, local_start.inner, @intCast(ctx.alignment), k.loc()));

            // Pallas interleaves: slice+squeeze for the FIRST buffer (out_dec → decode_overlap),
            //   then `index_cast` for sls_idx, then `vector.load`; then slice+squeeze for the
            //   SECOND buffer (out_pre → prefill_arr), then its `vector.load`.  Mine had been
            //   pre-emitting both slice/squeeze pairs and the index_cast up front, then both
            //   loads — match Pallas's interleaved order.
            const out_dec_w2 = refs[9];
            const out_dec_tile = k.memRefSqueeze(
                k.memRefSlice(out_dec_w2.buf.?, &.{ out_dec_w2.slot.?, c0i, c0i }, &.{ 1, ctx.bt, ctx.n_v * ctx.d_v }, &.{}),
                &.{ ctx.bt, ctx.n_v * ctx.d_v },
            );
            const sls_idx = k.toIndex(safe_local_start);
            const decode_overlap = k.vectorLoadShape(out_dec_tile, &.{ sls_idx, k.cIndex(0) }, &.{ ctx.alignment, ctx.n_v * ctx.d_v });
            const out_pre_w = refs[8];
            const out_pre_tile = k.memRefSqueeze(
                k.memRefSlice(out_pre_w.buf.?, &.{ out_pre_w.slot.?, c0i, c0i }, &.{ 1, ctx.bt, ctx.n_v * ctx.d_v }, &.{}),
                &.{ ctx.bt, ctx.n_v * ctx.d_v },
            );
            const prefill_arr = k.vectorLoadShape(out_pre_tile, &.{ k.cIndex(0), k.cIndex(0) }, &.{ ctx.alignment, ctx.n_v * ctx.d_v });
            // mask: iota<sl> < local_split, broadcast to (sl, n_v*d_v).
            //   Pallas emits the iota as (1, sl) along dim 1 + shape_cast to (sl,);
            //   then i1→i32 extui + shape_cast + cmpi ne instead of a direct
            //   shape_cast(i1).  Mirror that to keep the stitch IR identical.
            const stitch_iota_2d = k.iota(&.{ 1, ctx.alignment }, .i32, &.{1}); // (1, sl) iota along dim 1
            const stitch_iota = k.shapeCast(stitch_iota_2d, &.{ctx.alignment});
            const ls_bc = k.broadcastTo(local_split, &.{ctx.alignment});
            const stitch_mask_i1_flat = k.cmpi(.slt, stitch_iota, ls_bc);
            const stitch_mask_i32_flat = k.extui(stitch_mask_i1_flat, .i32);
            const stitch_mask_i32_2d = k.shapeCast(stitch_mask_i32_flat, &.{ ctx.alignment, 1 });
            const stitch_zeros_2d = k.zeros(&.{ ctx.alignment, 1 }, .i32);
            const stitch_mask_2d = k.cmpi(.ne, stitch_mask_i32_2d, stitch_zeros_2d);
            const stitch_mask_full = k.broadcastTo(stitch_mask_2d, &.{ ctx.alignment, ctx.n_v * ctx.d_v });
            const merged = k.select(stitch_mask_full, decode_overlap, prefill_arr);
            k.vectorStoreAt(out_dec_tile, merged, &.{ sls_idx, k.cIndex(0) });
            k.vectorStoreAt(out_pre_tile, merged, &.{ k.cIndex(0), k.cIndex(0) });
            when.yieldThen(.{});
        }
    }
}

// =============================================================================
// Kernel factory + sweeps
// =============================================================================

pub const Kernel = zml.kernel.mosaic_tpu.Kernel(Cfg, .{
    // The lowered `func.func` symbol — `name="recurrent_scan"` matches
    // Pallas's pallas_call name.
    //
    // NOTE: `inputs` excludes the grid `pid` (scalar i32, injected by the TPU
    // runtime per grid step) and the output / scratch refs that
    // `declareArgsOpts` declares for the IR. `recurrent_state_out` is aliased
    // to `recurrent_state` via `input_output_aliases`.
    .name = "recurrent_scan",
    .inputs = &.{
        "mixed_qkv",      "recurrent_state",   "state_indices",    "has_initial_state",
        "a_padded",       "b_padded",          "a_log",            "dt_bias",
        "schedule_table", "decode_tokens_arr", "total_blocks_arr",
    },
    .outputs = &.{ "recurrent_state_out", "output" },
    .run = run,
});
