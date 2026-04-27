//! Faithful Mosaic-TPU port of JAX Pallas's
//! `ragged_paged_attention.kernel.ragged_paged_attention` (`platforms/tpu/kernel.py`),
//! parameterised on every shape/dtype/tuning knob the production caller
//! needs to vary.
//!
//! Use through `mosaic_tpu.Kernel(Cfg, .{...}).call(...)` (see
//! `zml/attention/mosaic_tpu_kernels/ragged_attention.zig`). The body
//! lives in `run(b, cfg)` below; the kernel factory in `zml/kernel.zig`
//! handles the `Builder.open` / `Builder.finishOpts` lifecycle.

const std = @import("std");

const mlir = @import("mlir");
const mtt = @import("kernels/mosaic_tpu/builder");

pub const Cfg = struct {
    // Shapes that vary at compile-of-kernel time.
    num_q_tokens: i64,
    num_q_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    total_num_pages: i64,
    page_size: i64,
    max_num_seqs: i64,
    pages_per_seq: i64,

    // Block-level tuning knobs.
    num_kv_pages_per_block: i64,
    num_queries_per_block: i64,

    // Element types.
    q_dtype: mtt.DType = .bf16,
    kv_dtype: mtt.DType = .bf16,

    // Numeric knobs.
    sm_scale: f32,
    mask_value: f32 = -2.38197633e+38,

    // Optional metadata threaded into the `tpu_custom_call` backend_config.
    sliding_window: ?i64 = null,
    vmem_limit_bytes: ?i64 = null,

    // Derived shape helpers — keeps the body readable.
    pub fn numCombinedKvHeads(self: Cfg) i64 {
        return 2 * self.num_kv_heads;
    }
    pub fn numQHeadsPerKvHead(self: Cfg) i64 {
        return @divExact(self.num_q_heads, self.num_kv_heads);
    }
    pub fn numQHeadsPerBlk(self: Cfg) i64 {
        return self.num_q_heads;
    }
    pub fn numCombinedKvHeadsPerBlk(self: Cfg) i64 {
        return self.numCombinedKvHeads();
    }
    pub fn numKvHeadsPerBlk(self: Cfg) i64 {
        return @divExact(self.numCombinedKvHeadsPerBlk(), 2);
    }
    pub fn numQPerBlk(self: Cfg) i64 {
        return self.num_queries_per_block;
    }
    pub fn numKvPerBlk(self: Cfg) i64 {
        return self.num_kv_pages_per_block * self.page_size;
    }
    pub fn numHeadsBlks(self: Cfg) i64 {
        return @divExact(self.num_q_heads, self.numQHeadsPerBlk());
    }
    pub fn numQBlks(self: Cfg) i64 {
        return @divExact(self.num_q_tokens, self.numQPerBlk());
    }
    pub fn flashQRows(self: Cfg) i64 {
        return self.numQPerBlk() * self.numQHeadsPerKvHead();
    }
    pub fn flashKRows(self: Cfg) i64 {
        return self.numKvPerBlk();
    }

    /// Pallas's `get_dtype_packing(dtype) = 32 // bitwidth(dtype)`. Mirrors
    /// `kernel.py:get_dtype_packing` exactly.
    pub fn kvPackingFactor(self: Cfg) i64 {
        return switch (self.kv_dtype) {
            .i32, .f32 => 1,
            .i16, .f16, .bf16 => 2,
            .i8, .f8e4m3fn, .f8e5m2 => 4,
            else => std.debug.panic("ragged_paged: unsupported KV dtype {s}", .{@tagName(self.kv_dtype)}),
        };
    }
    /// `kv_load_step = max(1, kv_packing // 2)` — number of (k, v) pairs the
    /// caller unpacks from a single `b` strided-load (Pallas's outer-loop step).
    pub fn kvLoadStep(self: Cfg) i64 {
        const p = self.kvPackingFactor();
        return @max(@as(i64, 1), @divFloor(p, 2));
    }
    pub fn kvFlatRows(self: Cfg) i64 {
        return self.num_kv_pages_per_block * self.page_size * self.numCombinedKvHeadsPerBlk();
    }
    pub fn kvFlatI32Rows(self: Cfg) i64 {
        return @divExact(self.kvFlatRows(), self.kvPackingFactor());
    }
};

/// Pallas-style `pltpu.store(ref, val, mask=jnp.logical_and(iota >= start,
/// iota < end))` for the m/l refs (group=num_q_heads_per_kv_head):
/// `iota = floor_div(iota_dim0, group)`. `kv_head_idx_idx` indexes the
/// leading `num_kv_heads_per_blk` axis of `ref`.
fn maskedStoreLM(
    k: *mtt.Builder,
    ref: mtt.Value,
    val: mtt.Value,
    kv_head_idx_idx: mtt.Value,
    start: mtt.Value,
    end: mtt.Value,
    group: i64,
) void {
    const shape = val.shape().constSlice();
    const iota = k.iota(shape, .i32, &.{0});
    const iota_div = k.divFloor(iota, @as(i32, @intCast(group)));
    const ge = k.cmpi(.sge, iota_div, k.broadcastTo(start, shape));
    const lt = k.cmpi(.slt, iota_div, k.broadcastTo(end, shape));
    const mask = k.andi(ge, lt);
    const slot_shape = [_]i64{ 1, shape[0], shape[1] };
    const val_3d = k.shapeCast(val, &slot_shape);
    const mask_3d = k.shapeCast(mask, &slot_shape);
    k.refStoreOpts(ref, val_3d, &.{ kv_head_idx_idx, k.cIndex(0), k.cIndex(0) }, .{ .mask = mask_3d, .strides = &.{} });
}

/// `pltpu.store(acc_ref, val, mask=…)` — group=1 so no floor-div.
fn maskedStoreAcc(
    k: *mtt.Builder,
    ref: mtt.Value,
    val: mtt.Value,
    q_head_offset_idx: mtt.Value,
    start: mtt.Value,
    end: mtt.Value,
) void {
    const shape = val.shape().constSlice();
    const iota = k.iota(shape, .i32, &.{0});
    const ge = k.cmpi(.sge, iota, k.broadcastTo(start, shape));
    const lt = k.cmpi(.slt, iota, k.broadcastTo(end, shape));
    const mask = k.andi(ge, lt);
    k.refStoreOpts(ref, val, &.{ k.cIndex(0), q_head_offset_idx, k.cIndex(0) }, .{ .mask = mask, .strides = &.{} });
}

/// `MultiPageAsyncCopyDescriptor.__init__` — `tpu.memref_slice → squeeze
/// → enqueue_dma` chain that Pallas emits for one async page-copy.
fn createKvAsyncCopyDescriptors(
    k: *mtt.Builder,
    cfg: Cfg,
    a: anytype,
    heads_blk_idx: mtt.Value, // i32
    seq_idx: mtt.Value, // i32
    kv_blk_idx: mtt.Value, // i32 — the block-relative page index
    buf_idx: mtt.Value, // i32
) void {
    const start_kv_page_idx = k.muli(kv_blk_idx, k.lift(@as(i32, @intCast(cfg.num_kv_pages_per_block))));
    const kv_at_seq = k.scalarLoad(a.kv_lens, &.{k.toIndex(seq_idx)});
    const end_kv_page_idx = k.minsi(
        k.divsi(k.addi(kv_at_seq, k.lift(@as(i32, @intCast(cfg.page_size - 1)))), k.lift(@as(i32, @intCast(cfg.page_size)))),
        k.lift(@as(i32, @intCast(cfg.pages_per_seq))),
    );
    const heads_start = k.muli(heads_blk_idx, k.lift(@as(i32, @intCast(cfg.numCombinedKvHeadsPerBlk()))));

    const within = k.cmpi(.slt, start_kv_page_idx, end_kv_page_idx);
    const page_idx_safe = k.select(within, start_kv_page_idx, k.lift(@as(i32, 0)));
    const page_table = k.scalarLoad(a.page_indices, &.{ k.toIndex(seq_idx), k.toIndex(page_idx_safe) });

    const src0 = k.memRefSlice(a.kv_pages, &.{ k.lift(@as(i32, 0)), k.lift(@as(i32, 0)), heads_start, k.lift(@as(i32, 0)) }, &.{ cfg.total_num_pages, cfg.page_size, cfg.numCombinedKvHeadsPerBlk(), cfg.head_dim }, &.{});
    const src1 = k.memRefSlice(src0, &.{ page_table, k.lift(@as(i32, 0)), k.lift(@as(i32, 0)), k.lift(@as(i32, 0)) }, &.{ 1, cfg.page_size, cfg.numCombinedKvHeadsPerBlk(), cfg.head_dim }, &.{});
    const src = k.memRefSqueeze(src1, &.{ cfg.page_size, cfg.numCombinedKvHeadsPerBlk(), cfg.head_dim });

    const dst0 = k.memRefSlice(a.kv_bufs, &.{ buf_idx, k.lift(@as(i32, 0)), k.lift(@as(i32, 0)), k.lift(@as(i32, 0)), k.lift(@as(i32, 0)) }, &.{ 1, cfg.num_kv_pages_per_block, cfg.page_size, cfg.numCombinedKvHeadsPerBlk(), cfg.head_dim }, &.{});
    const dst1 = k.memRefSqueeze(dst0, &.{ cfg.num_kv_pages_per_block, cfg.page_size, cfg.numCombinedKvHeadsPerBlk(), cfg.head_dim });
    const dst = k.memRefSqueeze(dst1, &.{ cfg.page_size, cfg.numCombinedKvHeadsPerBlk(), cfg.head_dim });

    const sem_slice = k.memRefSlice(a.sems, &.{buf_idx}, &.{1}, &.{});
    const sem = k.memRefSqueeze(sem_slice, &.{});

    k.enqueueDma(src, dst, sem, .{});
}

fn waitKvDma(
    k: *mtt.Builder,
    cfg: Cfg,
    a: anytype,
    heads_blk_idx: mtt.Value,
    seq_idx: mtt.Value,
    kv_blk_idx: mtt.Value,
    buf_idx: mtt.Value,
) void {
    const start_kv_page_idx = k.muli(kv_blk_idx, k.lift(@as(i32, @intCast(cfg.num_kv_pages_per_block))));
    const kv_at_seq = k.scalarLoad(a.kv_lens, &.{k.toIndex(seq_idx)});
    const end_kv_page_idx = k.minsi(
        k.divsi(k.addi(kv_at_seq, k.lift(@as(i32, @intCast(cfg.page_size - 1)))), k.lift(@as(i32, @intCast(cfg.page_size)))),
        k.lift(@as(i32, @intCast(cfg.pages_per_seq))),
    );
    const heads_start = k.muli(heads_blk_idx, k.lift(@as(i32, @intCast(cfg.numCombinedKvHeadsPerBlk()))));
    const within = k.cmpi(.slt, start_kv_page_idx, end_kv_page_idx);
    const page_idx_safe = k.select(within, start_kv_page_idx, k.lift(@as(i32, 0)));
    const page_table = k.scalarLoad(a.page_indices, &.{ k.toIndex(seq_idx), k.toIndex(page_idx_safe) });

    const src0 = k.memRefSlice(a.kv_pages, &.{ k.lift(@as(i32, 0)), k.lift(@as(i32, 0)), heads_start, k.lift(@as(i32, 0)) }, &.{ cfg.total_num_pages, cfg.page_size, cfg.numCombinedKvHeadsPerBlk(), cfg.head_dim }, &.{});
    const src1 = k.memRefSlice(src0, &.{ page_table, k.lift(@as(i32, 0)), k.lift(@as(i32, 0)), k.lift(@as(i32, 0)) }, &.{ 1, cfg.page_size, cfg.numCombinedKvHeadsPerBlk(), cfg.head_dim }, &.{});
    const src = k.memRefSqueeze(src1, &.{ cfg.page_size, cfg.numCombinedKvHeadsPerBlk(), cfg.head_dim });

    const dst0 = k.memRefSlice(a.kv_bufs, &.{ buf_idx, k.lift(@as(i32, 0)), k.lift(@as(i32, 0)), k.lift(@as(i32, 0)), k.lift(@as(i32, 0)) }, &.{ 1, cfg.num_kv_pages_per_block, cfg.page_size, cfg.numCombinedKvHeadsPerBlk(), cfg.head_dim }, &.{});
    const dst1 = k.memRefSqueeze(dst0, &.{ cfg.num_kv_pages_per_block, cfg.page_size, cfg.numCombinedKvHeadsPerBlk(), cfg.head_dim });
    const dst = k.memRefSqueeze(dst1, &.{ cfg.page_size, cfg.numCombinedKvHeadsPerBlk(), cfg.head_dim });

    const sem_slice = k.memRefSlice(a.sems, &.{buf_idx}, &.{1}, &.{});
    const sem = k.memRefSqueeze(sem_slice, &.{});

    k.waitDma2(sem, src, dst, .{});
}

const KvPair = struct { k: mtt.Value, v: mtt.Value };

/// Mirrors Pallas's `b_ref[b_start::b_step, :]` lowering:
///   - `b_step == 1` → `vector.load` (the slice is contiguous).
///   - `last_dim ≤ 128` and `b_step > 1` → `tpu.strided_load`.
///   - `last_dim > 128` → per-row `vector.load` + `tpu.concatenate`
///     (`jnp.stack([ref[start + i*step, :] for i in range(num_rows)])`).
fn loadStridedOrStack(
    k: *mtt.Builder,
    ref: mtt.Value,
    start_row: i64,
    stride: i32,
    out_rows: i64,
    out_last_dim: i64,
) mtt.Value {
    if (out_last_dim <= 128) {
        if (stride == 1) {
            return k.vectorLoadShape(ref, &.{ k.cIndex(start_row), k.cIndex(0) }, &.{ out_rows, out_last_dim });
        }
        return k.stridedLoadShape(ref, &.{ k.cIndex(start_row), k.cIndex(0) }, &.{ stride, 1 }, &.{ out_rows, out_last_dim });
    }
    const rows_count: usize = @intCast(out_rows);
    const rows = k.arena.allocator().alloc(mtt.Value, rows_count) catch @panic("loadStridedOrStack OOM");
    for (0..rows_count) |i| {
        const row_idx: i64 = start_row + @as(i64, @intCast(i)) * @as(i64, stride);
        rows[i] = k.vectorLoadShape(ref, &.{ k.cIndex(row_idx), k.cIndex(0) }, &.{ 1, out_last_dim });
    }
    return k.concatenate(rows, 0);
}

/// Pallas's `broadcast_to_shape(arr, shape)` — emit `tpu.concatenate` of
/// `shape[1] / arr.shape[1]` copies when expanding axis 1 (e.g.,
/// `<flash_q, 128>` → `<flash_q, head_dim>` for `head_dim > 128`).
fn broadcastToShape(k: *mtt.Builder, arr: mtt.Value, shape: []const i64) mtt.Value {
    const arr_shape = arr.shape();
    std.debug.assert(arr_shape.len == 2 and shape.len == 2);
    if (arr_shape.get(1) == shape[1]) return arr;
    std.debug.assert(arr_shape.get(0) == shape[0]);
    std.debug.assert(@mod(shape[1], arr_shape.get(1)) == 0);
    const repeats: usize = @intCast(@divExact(shape[1], arr_shape.get(1)));
    const sources = k.arena.allocator().alloc(mtt.Value, repeats) catch @panic("broadcastToShape OOM");
    for (sources) |*s| s.* = arr;
    return k.concatenate(sources, 1);
}

/// `strided_load_kv(ref, kv_head_chunk_idx * 2, num_combined_kv_heads_per_blk)`
/// — full Pallas parity.
///
/// - `packing == 1` (i32/f32): two strided loads in original dtype.
/// - `packing == 2` bf16: special-case shli + and + tpu.bitcast(f32) + truncf.
/// - `packing == 2` f16/i16, `packing == 4` (i8/u8/f8): generic shrui + trunci
///   + tpu.bitcast unpack producing `kv_load_step` (k, v) pairs.
fn stridedLoadKv(
    k: *mtt.Builder,
    cfg: Cfg,
    a: anytype,
    buf_idx: mtt.Value,
    kv_head_chunk_idx: i64,
) []const KvPair {
    const packing = cfg.kvPackingFactor();
    const start: i64 = kv_head_chunk_idx * 2;
    const step: i32 = @intCast(cfg.numCombinedKvHeadsPerBlk());

    const dst0 = k.memRefSlice(a.kv_bufs, &.{ buf_idx, k.lift(@as(i32, 0)), k.lift(@as(i32, 0)), k.lift(@as(i32, 0)), k.lift(@as(i32, 0)) }, &.{ 1, cfg.num_kv_pages_per_block, cfg.page_size, cfg.numCombinedKvHeadsPerBlk(), cfg.head_dim }, &.{});
    const dst1 = k.memRefSqueeze(dst0, &.{ cfg.num_kv_pages_per_block, cfg.page_size, cfg.numCombinedKvHeadsPerBlk(), cfg.head_dim });
    const flat = k.memRefReshape(dst1, &.{ cfg.kvFlatRows(), cfg.head_dim });

    if (packing == 1) {
        const k_v = loadStridedOrStack(k, flat, start, step, cfg.flashKRows(), cfg.head_dim);
        const v_v = loadStridedOrStack(k, flat, start + 1, step, cfg.flashKRows(), cfg.head_dim);
        const result = k.arena.allocator().alloc(KvPair, 1) catch @panic("stridedLoadKv OOM");
        result[0] = .{ .k = k_v, .v = v_v };
        return result;
    }

    std.debug.assert(packing == 2 or packing == 4 or packing == 8);
    std.debug.assert(@mod(step, packing) == 0);

    const ref_i32 = k.memRefBitcastShape(flat, &.{ cfg.kvFlatI32Rows(), cfg.head_dim }, .i32);
    const b_stride: i32 = @intCast(@divExact(cfg.numCombinedKvHeadsPerBlk(), packing));
    const b_start: i64 = @divExact(start, packing);
    const b = loadStridedOrStack(k, ref_i32, b_start, b_stride, cfg.flashKRows(), cfg.head_dim);

    const splat_shape = &[_]i64{ cfg.flashKRows(), cfg.head_dim };

    if (cfg.kv_dtype == .bf16) {
        const bk = k.shli(b, k.splat(@as(i32, 16), splat_shape, .i32));
        const bv = k.andi(b, k.splat(@as(i32, @bitCast(@as(u32, 0xFFFF0000))), splat_shape, .i32));
        const k_v = k.arithTruncf(k.bitcastTo(bk, .f32), cfg.kv_dtype);
        const v_v = k.arithTruncf(k.bitcastTo(bv, .f32), cfg.kv_dtype);
        const result = k.arena.allocator().alloc(KvPair, 1) catch @panic("stridedLoadKv OOM");
        result[0] = .{ .k = k_v, .v = v_v };
        return result;
    }

    const bitwidth: i64 = @divExact(@as(i64, 32), packing);
    const trunc_dtype: mtt.DType = switch (bitwidth) {
        16 => .i16,
        8 => .i8,
        else => std.debug.panic("ragged_paged: unsupported packing bitwidth {d}", .{bitwidth}),
    };
    const kv_load_step: usize = @intCast(cfg.kvLoadStep());
    const result = k.arena.allocator().alloc(KvPair, kv_load_step) catch @panic("stridedLoadKv OOM");
    var i: usize = 0;
    while (i < kv_load_step) : (i += 1) {
        const shift_k: i32 = @intCast(@as(i64, @intCast(i)) * 2 * bitwidth);
        const shift_v: i32 = @intCast((@as(i64, @intCast(i)) * 2 + 1) * bitwidth);
        const bk = k.shrui(b, k.splat(shift_k, splat_shape, .i32));
        const bv = k.shrui(b, k.splat(shift_v, splat_shape, .i32));
        const bk_trunc = k.trunci(bk, trunc_dtype);
        const bv_trunc = k.trunci(bv, trunc_dtype);
        result[i] = .{
            .k = k.bitcastTo(bk_trunc, cfg.kv_dtype),
            .v = k.bitcastTo(bv_trunc, cfg.kv_dtype),
        };
    }
    return result;
}

/// `flash_attention(q, k, v, ...)` — one online-softmax step over `num_kv_per_blk`
/// keys.
fn flashAttention(
    k: *mtt.Builder,
    cfg: Cfg,
    a: anytype,
    q: mtt.Value,
    kv_k: mtt.Value,
    v: mtt.Value,
    kv_head_idx: i64,
    q_head_idx: i64,
    kv_blk_idx: mtt.Value,
    kv_len: mtt.Value,
    q_start: mtt.Value,
    q_end: mtt.Value,
    q_len: mtt.Value,
    q_len_start: mtt.Value,
) void {
    const flash_q: i64 = cfg.flashQRows();
    const flash_k: i64 = cfg.flashKRows();
    const head_dim: i64 = cfg.head_dim;
    const num_q_per_blk: i64 = cfg.numQPerBlk();
    const num_q_heads_per_kv_head: i64 = cfg.numQHeadsPerKvHead();

    const kv_len_start = k.muli(kv_blk_idx, k.lift(@as(i32, @intCast(cfg.numKvPerBlk()))));

    const kv_mask = k.cmpi(
        .slt,
        k.iota(&.{ flash_k, head_dim }, .i32, &.{0}),
        k.broadcastTo(k.subi(kv_len, kv_len_start), &.{ flash_k, head_dim }),
    );
    const k_zeroed = k.select(kv_mask, kv_k.to(.f32), k.zeros(&.{ flash_k, head_dim }, .f32)).to(cfg.kv_dtype);
    const v_zeroed = k.select(kv_mask, v.to(.f32), k.zeros(&.{ flash_k, head_dim }, .f32)).to(cfg.kv_dtype);

    k.traceStart(10, "nd,md->nm");
    const qk_raw = k.matmulOpts(
        q,
        k_zeroed,
        k.zeros(&.{ flash_q, flash_k }, .f32),
        .{ .dimension_numbers = k.dotDimensionNumbers(&.{1}, &.{1}, &.{0}, &.{0}, &.{ 0, 0, 1, 0 }, &.{}, &.{}) },
    );
    k.traceStop();
    const qk = k.mulf(qk_raw, k.splat(cfg.sm_scale, &.{ flash_q, flash_k }, .f32));

    const store_start = k.maxsi(k.subi(q_start, q_len_start), k.lift(@as(i32, 0)));
    const store_end = k.minsi(k.subi(q_end, q_len_start), k.lift(@as(i32, @intCast(num_q_per_blk))));

    const row_offset = k.subi(k.addi(k.subi(kv_len, q_len), q_len_start), q_start);
    const row_iota = k.iota(&.{ flash_q, flash_k }, .i32, &.{0});
    const row_q = k.divFloor(row_iota, @as(i32, @intCast(num_q_heads_per_kv_head)));
    const row_ids = k.addi(k.broadcastTo(row_offset, &.{ flash_q, flash_k }), row_q);
    const col_ids = k.addi(
        k.broadcastTo(kv_len_start, &.{ flash_q, flash_k }),
        k.iota(&.{ flash_q, flash_k }, .i32, &.{1}),
    );
    const causal_mask = k.cmpi(.slt, row_ids, col_ids);
    const qk_masked = k.addf(
        qk,
        k.select(causal_mask, k.splat(cfg.mask_value, &.{ flash_q, flash_k }, .f32), k.zeros(&.{ flash_q, flash_k }, .f32)),
    );

    // jnp.max → maximumf (NaN-propagating).
    const m_curr_v = k.multiReduction(.maximumf, qk_masked, k.splat(@as(f32, -std.math.inf(f32)), &.{flash_q}, .f32), &.{1});
    const m_curr_2d = k.shapeCast(m_curr_v, &.{ flash_q, 1 });
    const s_curr = k.exp(k.subf(qk_masked, k.broadcastTo(m_curr_2d, &.{ flash_q, flash_k })));

    const qkv = k.matmulOpts(
        s_curr,
        v_zeroed,
        k.zeros(&.{ flash_q, head_dim }, .f32),
        .{ .dimension_numbers = k.dotDimensionNumbers(&.{1}, &.{0}, &.{0}, &.{1}, &.{ 0, 0, 1, 1 }, &.{}, &.{}) },
    );

    // Pallas hardcodes the m/l-ref trailing dim to 128 (the TPU lane width)
    // regardless of `head_dim`. Broadcast m/l values to `head_dim` only
    // immediately before mixing with `qkv` / `o_curr`.
    const lm_lane: i64 = 128;

    const m_curr_bc = k.broadcastTo(m_curr_2d, &.{ flash_q, lm_lane });
    const l_curr_v = k.multiReduction(.add, s_curr, k.zeros(&.{flash_q}, .f32), &.{1});
    const l_curr_bc = k.broadcastTo(k.shapeCast(l_curr_v, &.{ flash_q, 1 }), &.{ flash_q, lm_lane });

    const head_l_ref_idx = k.cIndex(kv_head_idx);
    const m_prev_load = k.shapeCast(
        k.vectorLoadShape(a.m_ref, &.{ head_l_ref_idx, k.cIndex(0), k.cIndex(0) }, &.{ 1, flash_q, lm_lane }),
        &.{ flash_q, lm_lane },
    );
    const m_prev = k.select(k.cmpi(.eq, kv_blk_idx, k.lift(@as(i32, 0))), k.splat(@as(f32, -std.math.inf(f32)), &.{ flash_q, lm_lane }, .f32), m_prev_load);
    const l_prev_load = k.shapeCast(
        k.vectorLoadShape(a.l_ref, &.{ head_l_ref_idx, k.cIndex(0), k.cIndex(0) }, &.{ 1, flash_q, lm_lane }),
        &.{ flash_q, lm_lane },
    );
    const l_prev = k.select(k.cmpi(.eq, kv_blk_idx, k.lift(@as(i32, 0))), k.zeros(&.{ flash_q, lm_lane }, .f32), l_prev_load);

    const m_next = k.maximumf(m_prev, m_curr_bc);
    maskedStoreLM(k, a.m_ref, m_next, head_l_ref_idx, store_start, store_end, num_q_heads_per_kv_head);

    const alpha = k.exp(k.subf(m_prev, m_next));
    const beta = k.exp(k.subf(m_curr_bc, m_next));
    const l_alpha = k.mulf(alpha, l_prev);
    const l_next = k.addf(l_alpha, k.mulf(beta, l_curr_bc));
    const l_next_safe = k.select(
        k.cmpf(.oeq, l_next, k.zeros(&.{ flash_q, lm_lane }, .f32)),
        k.splat(@as(f32, 1.0), &.{ flash_q, lm_lane }, .f32),
        l_next,
    );
    maskedStoreLM(k, a.l_ref, l_next_safe, head_l_ref_idx, store_start, store_end, num_q_heads_per_kv_head);

    const q_head_idx_idx = k.cIndex(q_head_idx);
    const o_curr_3d = k.vectorLoadShape(a.acc_ref, &.{ k.cIndex(0), q_head_idx_idx, k.cIndex(0) }, &.{ num_q_per_blk, num_q_heads_per_kv_head, head_dim });
    const o_curr_pre = k.select(
        k.cmpi(.eq, kv_blk_idx, k.lift(@as(i32, 0))),
        k.zeros(&.{ num_q_per_blk, num_q_heads_per_kv_head, head_dim }, .f32),
        o_curr_3d,
    );
    const o_curr = k.shapeCast(o_curr_pre, &.{ flash_q, head_dim });
    const l_alpha_bc = broadcastToShape(k, l_alpha, &.{ flash_q, head_dim });
    const beta_bc = broadcastToShape(k, beta, &.{ flash_q, head_dim });
    const l_next_safe_bc = broadcastToShape(k, l_next_safe, &.{ flash_q, head_dim });
    const out = k.divf(k.addf(k.mulf(l_alpha_bc, o_curr), k.mulf(beta_bc, qkv)), l_next_safe_bc);
    const out_3d = k.shapeCast(out, &.{ num_q_per_blk, num_q_heads_per_kv_head, head_dim });
    maskedStoreAcc(k, a.acc_ref, out_3d, q_head_idx_idx, store_start, store_end);
}

/// Emit the kernel body into a pre-opened `mtt.Builder`. The factory in
/// `zml/kernel.zig` opens the builder, calls this `run`, and finalizes the
/// IR with `b.finishOpts(&.{}, .{ .canonicalize = true })`.
pub fn run(b: *mtt.Builder, cfg: Cfg) mtt.FinishError!void {
    const q_transform: []const mtt.ArgSpec.TransformReturn = &.{
        .{ .program_id = 1 },
        .{ .program_id = 0 },
        .zero,
    };

    const a = try b.declareArgsOpts(.{
        .heads_blk_idx = .{ .scalar = .i32 },
        .q_blk_idx = .{ .scalar = .i32 },

        .kv_lens = .{ .ref = .{ .shape = &.{cfg.max_num_seqs}, .dtype = .i32, .memory_space = .smem, .role = .scalar_prefetch } },
        .page_indices = .{ .ref = .{ .shape = &.{ cfg.max_num_seqs, cfg.pages_per_seq }, .dtype = .i32, .memory_space = .smem, .role = .scalar_prefetch } },
        .cu_q_lens = .{ .ref = .{ .shape = &.{cfg.max_num_seqs + 1}, .dtype = .i32, .memory_space = .smem, .role = .scalar_prefetch } },
        .seq_buf_idx = .{ .ref = .{ .shape = &.{2}, .dtype = .i32, .memory_space = .smem, .role = .scalar_prefetch } },
        .num_seqs = .{ .ref = .{ .shape = &.{1}, .dtype = .i32, .memory_space = .smem, .role = .scalar_prefetch } },

        .q = .{ .ref = .{
            .shape = &.{ cfg.numQPerBlk(), cfg.numQHeadsPerBlk(), cfg.head_dim },
            .dtype = cfg.q_dtype,
            .window = mtt.ArgSpec.WindowSpec{ .transform_returns = q_transform },
        } },
        .kv_pages = .{ .ref = .{
            .shape = &.{ cfg.total_num_pages, cfg.page_size, cfg.numCombinedKvHeads(), cfg.head_dim },
            .dtype = cfg.kv_dtype,
            .memory_space = .any,
        } },
        .o = .{ .ref = .{
            .shape = &.{ cfg.numQPerBlk(), cfg.numQHeadsPerBlk(), cfg.head_dim },
            .dtype = cfg.q_dtype,
            .role = .output,
            .window = mtt.ArgSpec.WindowSpec{ .transform_returns = q_transform },
        } },

        .kv_bufs = .{ .ref = .{
            .shape = &.{ 2, cfg.num_kv_pages_per_block, cfg.page_size, cfg.numCombinedKvHeadsPerBlk(), cfg.head_dim },
            .dtype = cfg.kv_dtype,
            .role = .scratch,
        } },
        .sems = .{ .sem_array = .{ .shape = &.{2}, .kind = .dma } },
        // Pallas hardcodes the trailing dim to 128 (TPU lane width); see
        // `kernel.py:lm_scratch`.
        .l_ref = .{ .ref = .{ .shape = &.{ cfg.numKvHeadsPerBlk(), cfg.flashQRows(), 128 }, .dtype = .f32, .role = .scratch } },
        .m_ref = .{ .ref = .{ .shape = &.{ cfg.numKvHeadsPerBlk(), cfg.flashQRows(), 128 }, .dtype = .f32, .role = .scratch } },
        .acc_ref = .{ .ref = .{ .shape = &.{ cfg.numQPerBlk(), cfg.numQHeadsPerBlk(), cfg.head_dim }, .dtype = .f32, .role = .scratch } },
    }, &.{}, .{
        .dimension_semantics = &.{ .arbitrary, .arbitrary },
        .iteration_bounds = &.{ cfg.numHeadsBlks(), cfg.numQBlks() },
        .scalar_prefetch = 5,
        .scratch_operands = 5,
        .pallas_window_params = true,
    });

    const k = b;

    const num_seqs = k.scalarLoad(a.num_seqs, &.{k.cIndex(0)});
    const init_seq_idx = k.scalarLoad(a.seq_buf_idx, &.{k.cIndex(0)});
    const init_buf_idx = k.scalarLoad(a.seq_buf_idx, &.{k.cIndex(1)});
    const q_len_start = k.muli(a.q_blk_idx, k.lift(@as(i32, @intCast(cfg.numQPerBlk()))));
    const q_len_end = k.addi(q_len_start, k.lift(@as(i32, @intCast(cfg.numQPerBlk()))));

    {
        var i = k.openIf(k.cmpi(.eq, k.addi(a.heads_blk_idx, a.q_blk_idx), k.lift(@as(i32, 0))));
        createKvAsyncCopyDescriptors(k, cfg, a, a.heads_blk_idx, init_seq_idx, k.lift(@as(i32, 0)), init_buf_idx);
        i.yieldThen(.{});
    }

    var q_loop = k.openWhile(.{ k.lift(@as(i32, 0)), init_seq_idx, init_buf_idx }, .{ k.scalarTy(.i32), k.scalarTy(.i32), k.scalarTy(.i32) });
    {
        const done = q_loop.before_carried[0];
        const cur_seq_idx = q_loop.before_carried[1];
        const cur_buf_idx = q_loop.before_carried[2];
        const cu_at_num_seqs = k.scalarLoad(a.cu_q_lens, &.{k.toIndex(num_seqs)});
        const should_run = k.andi(
            k.cmpi(.slt, q_len_start, cu_at_num_seqs),
            k.cmpi(.slt, cur_seq_idx, num_seqs),
        );
        const cond = k.andi(k.cmpi(.eq, done, k.lift(@as(i32, 0))), should_run);
        q_loop.yieldBefore(cond, .{ done, cur_seq_idx, cur_buf_idx });
    }
    {
        const done = q_loop.after_carried[0];
        const cur_seq_idx = q_loop.after_carried[1];
        const cur_buf_idx = q_loop.after_carried[2];

        const q_start = k.scalarLoad(a.cu_q_lens, &.{k.toIndex(cur_seq_idx)});
        const q_end = k.scalarLoad(a.cu_q_lens, &.{k.toIndex(k.addi(cur_seq_idx, k.lift(@as(i32, 1))))});
        const q_len = k.subi(q_end, q_start);
        const kv_len = k.scalarLoad(a.kv_lens, &.{k.toIndex(cur_seq_idx)});

        var kv_loop = k.openWhile(.{ k.lift(@as(i32, 0)), cur_buf_idx }, .{ k.scalarTy(.i32), k.scalarTy(.i32) });
        {
            const kv_blk = kv_loop.before_carried[0];
            const buf = kv_loop.before_carried[1];
            const cond = k.cmpi(.slt, k.muli(kv_blk, k.lift(@as(i32, @intCast(cfg.numKvPerBlk())))), kv_len);
            kv_loop.yieldBefore(cond, .{ kv_blk, buf });
        }
        {
            const kv_blk = kv_loop.after_carried[0];
            const cur_buf = kv_loop.after_carried[1];

            const next_kv_blk_raw = k.addi(kv_blk, k.lift(@as(i32, 1)));
            const is_last_kv = k.cmpi(.sge, k.muli(next_kv_blk_raw, k.lift(@as(i32, @intCast(cfg.numKvPerBlk())))), kv_len);
            const next_kv_blk = k.select(is_last_kv, k.lift(@as(i32, 0)), next_kv_blk_raw);
            const cur_seq_end = k.cmpi(.sle, q_end, q_len_end);
            const next_seq_a = k.select(cur_seq_end, k.addi(cur_seq_idx, k.lift(@as(i32, 1))), cur_seq_idx);
            const next_seq_b = k.select(is_last_kv, next_seq_a, cur_seq_idx);
            const is_last_seq = k.cmpi(.eq, next_seq_b, num_seqs);
            const next_seq_idx = k.select(is_last_seq, k.lift(@as(i32, 0)), next_seq_b);
            const next_heads_blk_idx = k.select(is_last_seq, k.addi(a.heads_blk_idx, k.lift(@as(i32, 1))), a.heads_blk_idx);
            const next_buf = k.extui(k.cmpi(.eq, cur_buf, k.lift(@as(i32, 0))), .i32);

            {
                var ii = k.openIf(k.cmpi(.slt, next_heads_blk_idx, k.lift(@as(i32, @intCast(cfg.numHeadsBlks())))));
                createKvAsyncCopyDescriptors(k, cfg, a, next_heads_blk_idx, next_seq_idx, next_kv_blk, next_buf);
                ii.yieldThen(.{});
            }

            waitKvDma(k, cfg, a, a.heads_blk_idx, cur_seq_idx, kv_blk, cur_buf);

            // for kv_head_chunk_idx in range(0, num_kv_heads_per_blk, kv_load_step) — unrolled at IR-emit time.
            const num_kv_heads_per_blk: usize = @intCast(cfg.numKvHeadsPerBlk());
            const kv_load_step: usize = @intCast(cfg.kvLoadStep());
            var kv_head_chunk_idx: usize = 0;
            while (kv_head_chunk_idx < num_kv_heads_per_blk) : (kv_head_chunk_idx += kv_load_step) {
                const kv_pairs = stridedLoadKv(k, cfg, a, cur_buf, @intCast(kv_head_chunk_idx));
                var step_idx: usize = 0;
                while (step_idx < kv_load_step) : (step_idx += 1) {
                    const kv = kv_pairs[step_idx];
                    const kv_head_idx: i64 = @intCast(kv_head_chunk_idx + step_idx);
                    const q_head_idx = kv_head_idx * cfg.numQHeadsPerKvHead();

                    const q_block = k.vectorLoadShape(a.q, &.{ k.cIndex(0), k.cIndex(q_head_idx), k.cIndex(0) }, &.{ cfg.numQPerBlk(), cfg.numQHeadsPerKvHead(), cfg.head_dim });
                    const q_2d = k.shapeCast(q_block, &.{ cfg.flashQRows(), cfg.head_dim });
                    flashAttention(k, cfg, a, q_2d, kv.k, kv.v, kv_head_idx, q_head_idx, kv_blk, kv_len, q_start, q_end, q_len, q_len_start);
                }
            }

            kv_loop.yieldAfter(.{ k.addi(kv_blk, k.lift(@as(i32, 1))), next_buf });
        }
        const next_buf_idx = kv_loop.results[1];

        const next_seq_idx_outer = k.select(k.cmpi(.sle, q_end, q_len_end), k.addi(cur_seq_idx, k.lift(@as(i32, 1))), cur_seq_idx);
        const next_done = k.select(k.cmpi(.slt, q_end, q_len_end), done, k.lift(@as(i32, 1)));
        q_loop.yieldAfter(.{ next_done, next_seq_idx_outer, next_buf_idx });
    }
    const seq_idx_out = q_loop.results[1];
    const buf_idx_out = q_loop.results[2];

    k.scalarStore(a.seq_buf_idx, k.select(k.cmpi(.slt, seq_idx_out, num_seqs), seq_idx_out, k.lift(@as(i32, 0))), &.{k.cIndex(0)});
    k.scalarStore(a.seq_buf_idx, buf_idx_out, &.{k.cIndex(1)});

    k.vectorStoreAt(a.o, k.arithTruncf(k.refLoad(a.acc_ref), cfg.q_dtype), &.{ k.cIndex(0), k.cIndex(0), k.cIndex(0) });
}
