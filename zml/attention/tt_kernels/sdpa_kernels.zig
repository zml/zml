//! The device kernels of tt-metal's flash-attention prefill
//! (`transformer/sdpa/device/kernels/`, tt-metal d04395ed) in the TTKernel
//! DSL: `reader_interleaved`, `writer_interleaved` and `sdpa.cpp`'s standard
//! compute path (`sdpa_standard` of `compute_common.hpp`), for the
//! configuration the port runs: causal, K/V tensors or a paged cache from a
//! chunk start read at runtime, the lightweight causal mask, zigzag Q
//! scheduling; no provided masks, sinks, MLA, windows, ring or K/V chain
//! forwarding.
//!
//! The (batch, head, Q chunk) space is split over the cores; each core's
//! compute runs the online softmax over the K chunks up to its Q chunk's
//! diagonal, the writer stamping nothing but the two mask tiles and
//! writing the normalized chunks.

const std = @import("std");

const zml = @import("../../zml.zig");
const dsl = zml.kernel.ttkernel.dsl;
const llk = @import("llk.zig");
const sdpa = @import("sdpa_llk.zig");
const Builder = dsl.Builder;
const Value = dsl.Value;
const Cb = dsl.CircularBuffer;

pub const Config = struct {
    nqh: u32,
    nkh: u32,
    nvh: u32,
    /// K tiles, padded to the chunks.
    skt: u32,
    valid_sqt: u32,
    valid_skt: u32,
    dht: u32,
    vdht: u32,
    sq_chunk_t: u32,
    q_num_chunks: u32,
    sk_chunk_t: u32,
    /// A paged cache through a page table, from the chunk start.
    chunked: bool,
    block_size_t: u32,
    page_table_page_bytes: u32,
    tile_bytes: u32,
    /// The identity scale buffer's index, for the reduce scaler helper.
    cb_identity_scale: u32,
    zigzag: bool,
    scale_fp32: u32,
    identity_scalar_packed: u32,
    // compute
    qk_subblock_w: u32,
    qk_subblock_h: u32,
    qk_in0_num_subblocks: u32,
    qk_in1_num_subblocks: u32,
    out_subblock_w: u32,
    out_subblock_h: u32,
    out_in0_num_subblocks: u32,
    out_in1_num_subblocks: u32,
    dst_size: u32,
    stats_granularity: u32,
    sub_exp_granularity: u32,
    dht_granularity: u32,
    exp_approx: bool,
};

/// `linear_to_zigzag` / `remap_q_index` of `q_chunk_remapping.hpp`: within
/// each head, Q chunks in the order first, last, second, second last...
fn remapQ(k: *Builder, linear: Value, c: Config) Value {
    if (!c.zigzag) return linear;
    const head = linear.div(c.q_num_chunks);
    const pos = linear.rem(c.q_num_chunks);
    const half = pos.div(2);
    const q_chunk = k.select(pos.rem(2).eq(0), half, k.int(c.q_num_chunks - 1).sub(half));
    return head.mul(c.q_num_chunks).add(q_chunk);
}

const GlobalQ = struct { nb: Value, nq: Value, q_chunk: Value };

/// `decompose_global_q_index`: (batch, head, Q chunk) of a global Q index.
fn decomposeQ(k: *Builder, idx: Value, c: Config) GlobalQ {
    const r = remapQ(k, idx, c);
    return .{
        .nb = r.div(c.nqh * c.q_num_chunks),
        .nq = r.div(c.q_num_chunks).rem(c.nqh),
        .q_chunk = r.rem(c.q_num_chunks),
    };
}

/// The chunk start's Q chunks offset (`chunk_start_idx / q_chunk_size`).
fn qChunkOffset(start: Value, c: Config) Value {
    return start.div(c.sq_chunk_t * 32);
}

// =============================================================================
// Reader
// =============================================================================

/// Reads `rows` x `cols` tiles from tile `first` on, row-major, into the
/// `dst_rows` x `dst_cols` block at L1 `base`, transposed (column-major)
/// or not, the rest zero (`read_chunk_with_padding`'s placement). `tile_of`
/// gives a row's first tile.
fn readPadded(k: *Builder, c: Config, acc: Value, base: Value, rows: Value, cols: u32, dst_rows: u32, transpose: bool, tile_of: anytype) void {
    const noc = k.kernelNoc();
    const tb = c.tile_bytes;
    const outer: u32 = if (transpose) tb else cols * tb;
    const inner: u32 = if (transpose) tb * dst_rows else tb;
    var row_loop = k.openFor(0, rows, 1, .{});
    {
        const row = row_loop.iv.toI32();
        const first = tile_of.at(k, row);
        var col_loop, const col = llk.range(k, cols);
        k.op(.noc_async_read_tile, .{ first.add(col), acc, base.add(row.mul(outer)).add(col.mul(inner)), noc });
        col_loop.yield(.{});
    }
    row_loop.yield(.{});
    // Padding rows.
    var pad_loop = k.openFor(rows, dst_rows, 1, .{});
    {
        const row = pad_loop.iv.toI32();
        var col_loop, const col = llk.range(k, cols);
        sdpa.fillTile(k, base.add(row.mul(outer)).add(col.mul(inner)), tb, 0, 0);
        col_loop.yield(.{});
    }
    pad_loop.yield(.{});
    k.op(.noc_async_read_barrier, .{noc});
}

/// Rows of a tensor laid out `[batch, heads, rows, cols]` in tiles.
const DenseRows = struct {
    first: Value,
    cols: u32,

    fn at(self: DenseRows, k: *Builder, row: Value) Value {
        _ = k;
        return self.first.add(row.mul(self.cols));
    }
};

/// Rows of a head of the paged cache from sequence tile `start` on
/// (`virtual_seq_tile_id_to_physical_tile_id`).
const PagedRows = struct {
    c: Config,
    page_table: Value,
    start: Value,
    head: Value,
    heads: u32,
    cols: u32,

    fn at(self: PagedRows, k: *Builder, row: Value) Value {
        const seq = self.start.add(row);
        const block = k.loadL1(self.page_table, seq.div(self.c.block_size_t), 32);
        const bs = self.c.block_size_t * self.cols;
        return block.mul(self.heads * bs).add(self.head.mul(bs)).add(seq.rem(self.c.block_size_t).mul(self.cols));
    }
};

/// `reader_interleaved.cpp`: compile-time args the q, k, v (and chunked:
/// page table and the compute and writer chunk start) buffers, then the
/// accessor args of q, k, v (and chunked: the page table and the chunk
/// start). Runtime args as tt-metal's.
pub fn reader(k: *Builder, c: Config) void {
    const cb_q = k.circularBufferOf(0, .bf16, 1);
    const cb_k = k.circularBufferOf(1, .bf16, 1);
    const cb_v = k.circularBufferOf(2, .bf16, 1);
    const num_cbs: u32 = if (c.chunked) 6 else 3;
    const noc = k.kernelNoc();
    const tb = c.tile_bytes;
    const q_addr, const k_addr, const v_addr = .{ k.runtimeArg(0), k.runtimeArg(1), k.runtimeArg(2) };
    const page_table_addr, const chunk_start_addr = .{ k.runtimeArg(4), k.runtimeArg(6) };
    const read_offset = k.runtimeArg(10);
    const global_q_start, const global_q_count = .{ k.runtimeArg(11), k.runtimeArg(12) };

    const q_args = k.tensorAccessorArgs(num_cbs, 0);
    const k_args = k.nextTensorAccessorArgs(q_args);
    const v_args = k.nextTensorAccessorArgs(k_args);
    const q_acc = k.tensorAccessor(q_args, q_addr, tb);
    const k_acc = k.tensorAccessor(k_args, k_addr, tb);
    const v_acc = k.tensorAccessor(v_args, v_addr, tb);

    // Chunked: the chunk start, for compute and the writer, and its Q
    // chunks offset; the keys up to the chunk's end.
    var q_offset = k.int(0);
    var valid_skt_bound = k.int(c.valid_skt);
    var cb_page_table: Cb = undefined;
    var page_table_args: Value = undefined;
    if (c.chunked) {
        cb_page_table = k.circularBufferOf(3, .si32, 1);
        const cb_start_compute = k.circularBufferOf(4, .si32, 1);
        const cb_start_writer = k.circularBufferOf(5, .si32, 1);
        page_table_args = k.nextTensorAccessorArgs(v_args);
        const start_acc = k.tensorAccessor(k.nextTensorAccessorArgs(page_table_args), chunk_start_addr, 4);
        k.op(.cb_reserve_back, .{ cb_start_compute, 1 });
        const l1 = k.op(.get_write_ptr, .{cb_start_compute});
        k.op(.noc_async_read_tile, .{ 0, start_acc, l1, noc });
        k.op(.noc_async_read_barrier, .{noc});
        const start = k.loadL1(k.l1Ptr(l1, 32), 0, 32);
        k.op(.cb_push_back, .{ cb_start_compute, 1 });
        k.op(.cb_reserve_back, .{ cb_start_writer, 1 });
        k.op(.noc_async_read_tile, .{ 0, start_acc, k.op(.get_write_ptr, .{cb_start_writer}), noc });
        k.op(.noc_async_read_barrier, .{noc});
        k.op(.cb_push_back, .{ cb_start_writer, 1 });
        q_offset = qChunkOffset(start, c);
        valid_skt_bound = q_offset.mul(c.sq_chunk_t).add(c.valid_sqt).min(c.valid_skt);
    }

    const q_heads_per_k = c.nqh / c.nkh;
    const q_heads_per_v = c.nqh / c.nvh;
    const q_num_subblocks = c.sq_chunk_t / c.qk_subblock_h;
    const q_subblock_push = q_num_subblocks > 1;

    var q_loop = k.openFor(0, global_q_count, 1, .{});
    {
        const iter = q_loop.iv.toI32();
        const g = decomposeQ(k, global_q_start.add(iter), c);
        if (c.chunked) {
            // A new batch's page table, the previous one popped.
            const prev_nb = decomposeQ(k, global_q_start.add(iter).sub(1), c).nb;
            const first = iter.eq(0);
            var if_new = k.openIf(first.bitOr(g.nb.ne(prev_nb)));
            {
                var if_pop = k.openIf(first.eq(false));
                k.op(.cb_pop_front, .{ cb_page_table, 1 });
                if_pop.yieldThen(.{});
                k.op(.cb_reserve_back, .{ cb_page_table, 1 });
                const acc = k.tensorAccessor(page_table_args, page_table_addr, c.page_table_page_bytes);
                k.op(.noc_async_read_tile, .{ g.nb, acc, k.op(.get_write_ptr, .{cb_page_table}), noc });
                k.op(.noc_async_read_barrier, .{noc});
                k.op(.cb_push_back, .{ cb_page_table, 1 });
            }
            if_new.yieldThen(.{});
        }

        // This Q chunk's valid rows.
        const q_row_start = g.q_chunk.mul(c.sq_chunk_t).min(c.valid_sqt);
        const q_rows = q_row_start.add(c.sq_chunk_t).min(c.valid_sqt).sub(q_row_start);
        const q_first = g.nb.mul(c.nqh).add(g.nq).mul(c.valid_sqt).add(read_offset).add(q_row_start).mul(c.dht);
        if (!q_subblock_push) {
            k.op(.cb_reserve_back, .{ cb_q, c.sq_chunk_t * c.dht });
            readPadded(k, c, q_acc, k.op(.get_write_ptr, .{cb_q}), q_rows, c.dht, c.sq_chunk_t, false, DenseRows{ .first = q_first, .cols = c.dht });
            k.op(.cb_push_back, .{ cb_q, c.sq_chunk_t * c.dht });
        }

        // The K chunks up to the diagonal.
        const q_low = g.q_chunk.add(q_offset).mul(c.sq_chunk_t);
        const q_high = q_low.add(c.sq_chunk_t).min(c.skt);
        const k_chunks = q_high.add(c.sk_chunk_t - 1).div(c.sk_chunk_t);
        const k_head = g.nq.div(q_heads_per_k);
        const v_head = g.nq.div(q_heads_per_v);
        const page_table = if (c.chunked) k.l1Ptr(k.op(.get_read_ptr, .{cb_page_table}), 32) else undefined;
        var k_loop = k.openFor(0, k_chunks, 1, .{});
        {
            const k_chunk = k_loop.iv.toI32();
            const kv_row_start = k_chunk.mul(c.sk_chunk_t).min(valid_skt_bound);
            const kv_rows = kv_row_start.add(c.sk_chunk_t).min(valid_skt_bound).sub(kv_row_start);

            k.op(.cb_reserve_back, .{ cb_k, c.sk_chunk_t * c.dht });
            const k_l1 = k.op(.get_write_ptr, .{cb_k});
            if (c.chunked) {
                readPadded(k, c, k_acc, k_l1, kv_rows, c.dht, c.sk_chunk_t, true, PagedRows{ .c = c, .page_table = page_table, .start = k_chunk.mul(c.sk_chunk_t), .head = k_head, .heads = c.nkh, .cols = c.dht });
            } else {
                const first = g.nb.mul(c.nkh).add(k_head).mul(c.valid_skt).add(kv_row_start).mul(c.dht);
                readPadded(k, c, k_acc, k_l1, kv_rows, c.dht, c.sk_chunk_t, true, DenseRows{ .first = first, .cols = c.dht });
            }
            k.op(.cb_push_back, .{ cb_k, c.sk_chunk_t * c.dht });

            // Q a subblock at a time, along the first K chunk.
            if (q_subblock_push) {
                var if_first = k.openIf(k_chunk.eq(0));
                {
                    var sb_loop, const sb = llk.range(k, q_num_subblocks);
                    {
                        const sb_row = sb.mul(c.qk_subblock_h);
                        const rows = q_rows.sub(sb_row).max(0).min(c.qk_subblock_h);
                        k.op(.cb_reserve_back, .{ cb_q, c.qk_subblock_h * c.dht });
                        readPadded(k, c, q_acc, k.op(.get_write_ptr, .{cb_q}), rows, c.dht, c.qk_subblock_h, false, DenseRows{ .first = q_first.add(sb_row.mul(c.dht)), .cols = c.dht });
                        k.op(.cb_push_back, .{ cb_q, c.qk_subblock_h * c.dht });
                    }
                    sb_loop.yield(.{});
                }
                if_first.yieldThen(.{});
            }

            k.op(.cb_reserve_back, .{ cb_v, c.sk_chunk_t * c.vdht });
            const v_l1 = k.op(.get_write_ptr, .{cb_v});
            if (c.chunked) {
                readPadded(k, c, v_acc, v_l1, kv_rows, c.vdht, c.sk_chunk_t, false, PagedRows{ .c = c, .page_table = page_table, .start = k_chunk.mul(c.sk_chunk_t), .head = v_head, .heads = c.nvh, .cols = c.vdht });
            } else {
                const first = g.nb.mul(c.nvh).add(v_head).mul(c.valid_skt).add(kv_row_start).mul(c.vdht);
                readPadded(k, c, v_acc, v_l1, kv_rows, c.vdht, c.sk_chunk_t, false, DenseRows{ .first = first, .cols = c.vdht });
            }
            k.op(.cb_push_back, .{ cb_v, c.sk_chunk_t * c.vdht });
        }
        k_loop.yield(.{});
    }
    q_loop.yield(.{});
    if (c.chunked) {
        var if_any = k.openIf(global_q_count.gt(0));
        k.op(.cb_pop_front, .{ cb_page_table, 1 });
        if_any.yieldThen(.{});
    }
}

// =============================================================================
// Writer
// =============================================================================

const neg_inf_bf16x2: u32 = 0xFF80FF80;

/// `fill_causal_diagonal_tile_bf16`: zero on and below the diagonal, -inf
/// above, in tile `tile` of L1 `base`.
fn causalDiagonalTile(k: *Builder, base: Value, tile_bytes: u32) void {
    sdpa.fillTile(k, base, tile_bytes, 0, 0);
    const words = k.l1Ptr(base, 32);
    // Face 1 (rows 0..16, columns 16..32): -inf.
    var f1_loop, const w = llk.range(k, 128);
    k.storeL1(words, k.int(128).add(w), neg_inf_bf16x2, 32);
    f1_loop.yield(.{});
    // Faces 0 and 3: row r valid up to column r.
    for ([_]u32{ 0, 3 }) |face| {
        var row_loop, const row = llk.range(k, 15);
        {
            const row_base = row.mul(8).add(face * 128);
            const first_masked = row.add(1);
            var odd = k.openIf(first_masked.rem(2).ne(0));
            k.storeL1(words, row_base.add(first_masked.div(2)), 0xFF800000, 32);
            odd.yieldThen(.{});
            var word_loop = k.openFor(first_masked.add(1).div(2), 8, 1, .{});
            k.storeL1(words, row_base.add(word_loop.iv.toI32()), neg_inf_bf16x2, 32);
            word_loop.yield(.{});
        }
        row_loop.yield(.{});
    }
}

const reduce_helpers_h = "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp";

/// `writer_interleaved.cpp`: compile-time args the mask, identity scale,
/// column identity and output (and chunked: chunk start) buffers, then the
/// output's accessor args. Runtime args as tt-metal's.
pub fn writer(k: *Builder, c: Config) void {
    const cb_mask = k.circularBufferOf(0, .bf16, 1);
    _ = k.circularBufferOf(1, .bf16, 1);
    const cb_col_identity = k.circularBufferOf(2, .bf16, 1);
    const cb_out = k.circularBufferOf(3, .bf16, 1);
    const num_cbs: u32 = if (c.chunked) 5 else 4;
    const noc = k.kernelNoc();
    const tb = c.tile_bytes;
    const out_addr = k.runtimeArg(0);
    const use_chunk_start = k.runtimeArg(3);
    const write_offset = k.runtimeArg(5);
    const global_q_start, const global_q_count = .{ k.runtimeArg(8), k.runtimeArg(9) };
    const out_acc = k.tensorAccessor(k.tensorAccessorArgs(num_cbs, 0), out_addr, tb);

    // Compute's constant tiles: the reduce scaler, the column identity and
    // the lightweight causal mask (-inf, diagonal).
    k.call(reduce_helpers_h, "dataflow_kernel_lib::calculate_and_prepare_reduce_scaler", .{ c.cb_identity_scale, "ckernel::PoolType::MAX", "ckernel::ReduceDim::REDUCE_ROW", "dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR" }, .{});
    sdpa.bcastColScalar(k, cb_col_identity, c.identity_scalar_packed);
    k.op(.cb_reserve_back, .{ cb_mask, 2 });
    const mask_l1 = k.op(.get_write_ptr, .{cb_mask});
    sdpa.fillTile(k, mask_l1, tb, 0, neg_inf_bf16x2);
    causalDiagonalTile(k, mask_l1.add(tb), tb);
    k.op(.cb_push_back, .{ cb_mask, 2 });

    if (c.chunked) {
        const cb_start = k.circularBufferOf(4, .si32, 1);
        var if_start = k.openIf(use_chunk_start.ne(0));
        k.op(.cb_wait_front, .{ cb_start, 1 });
        k.op(.cb_pop_front, .{ cb_start, 1 });
        if_start.yieldThen(.{});
    }

    const out_chunk = c.sq_chunk_t * c.vdht;
    var q_loop = k.openFor(0, global_q_count, 1, .{});
    {
        const g = decomposeQ(k, global_q_start.add(q_loop.iv.toI32()), c);
        const row_start = g.q_chunk.mul(c.sq_chunk_t).min(c.valid_sqt);
        const rows = row_start.add(c.sq_chunk_t).min(c.valid_sqt).sub(row_start);
        const first = g.nb.mul(c.nqh).add(g.nq).mul(c.valid_sqt).add(write_offset).add(row_start).mul(c.vdht);
        // `write_block`: the chunk's valid rows.
        k.op(.cb_wait_front, .{ cb_out, out_chunk });
        const l1 = k.op(.get_read_ptr, .{cb_out});
        var t_loop = k.openFor(0, rows.mul(c.vdht), 1, .{});
        {
            const t = t_loop.iv.toI32();
            k.op(.noc_async_write_tile, .{ first.add(t), out_acc, l1.add(t.mul(tb)), noc });
        }
        t_loop.yield(.{});
        k.op(.noc_async_write_barrier, .{noc});
        k.op(.cb_pop_front, .{ cb_out, out_chunk });
    }
    q_loop.yield(.{});
}

// =============================================================================
// Compute
// =============================================================================

/// The circular buffers of `compute`, its compile-time args in order.
pub const ComputeCbs = enum { q, k, v, mask, identity_scale, col_identity, out, qk_im, out_im_a, out_im_b, max_a, max_b, sum_a, sum_b, exp_max_diff, chunk_start };

/// `sdpa.cpp` on its standard path (`sdpa_standard`): compile-time args the
/// `ComputeCbs` (chunk start when chunked). Runtime args as tt-metal's.
pub fn compute(k: *Builder, c: Config) void {
    const f32_dt: dsl.program.DataType = .f32;
    const q = k.circularBufferOf(0, .bf16, 1);
    const kk = k.circularBufferOf(1, .bf16, 1);
    const v = k.circularBufferOf(2, .bf16, 1);
    const mask = k.circularBufferOf(3, .bf16, 1);
    const scale = k.circularBufferOf(4, .bf16, 1);
    const col_identity = k.circularBufferOf(5, .bf16, 1);
    const out = k.circularBufferOf(6, .bf16, 1);
    const qk_im = k.circularBufferOf(7, f32_dt, 1);
    const out_a = k.circularBufferOf(8, .bf16, 1);
    const out_b = k.circularBufferOf(9, .bf16, 1);
    const max_a = k.circularBufferOf(10, .bf16, 1);
    const max_b = k.circularBufferOf(11, .bf16, 1);
    const sum_a = k.circularBufferOf(12, f32_dt, 1);
    const sum_b = k.circularBufferOf(13, f32_dt, 1);
    const exp_max_diff = k.circularBufferOf(14, .bf16, 1);
    const use_chunk_start = k.runtimeArg(2);
    const global_q_start, const global_q_count = .{ k.runtimeArg(5), k.runtimeArg(6) };
    const sq = c.sq_chunk_t;
    const sk = c.sk_chunk_t;
    const qk_tiles = sq * sk;

    k.call("api/compute/compute_kernel_hw_startup.h", "compute_kernel_hw_startup", .{"SrcOrder::Reverse"}, .{ q, kk, out });
    k.call("api/compute/matmul.h", "matmul_init", .{}, .{ q, kk });

    // Chunked: the chunk start, read by all three threads
    // (`read_tile_value`), as a Q chunks offset.
    var q_offset = k.int(0);
    if (c.chunked) {
        const cb_start = k.circularBufferOf(15, .si32, 1);
        var read = k.openIfElse(use_chunk_start.ne(0), .{k.int(0).type_()});
        {
            llk.wait(k, cb_start, 1);
            const start = k.callValue("api/compute/cb_api.h", "read_tile_value", .{}, .{ cb_start, 0, 0 }, .int(k.ctx, .i32));
            llk.pop(k, cb_start, 1);
            read.yieldThen(.{qChunkOffset(start, c)});
        }
        read.yieldElse(.{k.int(0)});
        q_offset = read.results[0];
    }
    llk.wait(k, mask, 2);

    const qk: sdpa.MatmulBlocks = .{
        .m = sq,
        .n = k.int(sk),
        .k = k.int(c.dht),
        .in0_num_subblocks = c.qk_in0_num_subblocks,
        .in1_num_subblocks = k.int(c.qk_in1_num_subblocks),
        .in0_block_w = k.int(c.dht),
        .subblock_h = c.qk_subblock_h,
        .subblock_w = k.int(c.qk_subblock_w),
        .transpose = true,
    };
    const qkv: sdpa.MatmulBlocks = .{
        .m = sq,
        .n = k.int(c.vdht),
        .k = k.int(sk),
        .in0_num_subblocks = c.out_in0_num_subblocks,
        .in1_num_subblocks = k.int(c.out_in1_num_subblocks),
        .in0_block_w = k.int(sk),
        .subblock_h = c.out_subblock_h,
        .subblock_w = k.int(c.out_subblock_w),
        .transpose = false,
    };

    var q_loop = k.openFor(0, global_q_count, 1, .{});
    {
        const linear = global_q_start.add(q_loop.iv.toI32());
        const q_chunk = remapQ(k, linear, c).rem(c.q_num_chunks).add(q_offset);
        const q_start_tile = q_chunk.mul(sq);
        const q_high = q_start_tile.add(sq).min(c.skt);
        const k_chunks = q_high.add(sk - 1).div(sk);

        // The running (max, sum, output) ping-pong between the A and B
        // buffers: chunk p writes B when p is even, A when odd.
        var k_loop = k.openFor(0, k_chunks, 1, .{});
        {
            const p = k_loop.iv.toI32();
            const even = p.rem(2).eq(0);
            const cur_max = k.selectCb(even, max_b, max_a);
            const prev_max = k.selectCb(even, max_a, max_b);
            const cur_sum = k.selectCb(even, sum_b, sum_a);
            const prev_sum = k.selectCb(even, sum_a, sum_b);
            const cur_out = k.selectCb(even, out_b, out_a);
            const prev_out = k.selectCb(even, out_a, out_b);

            // QK = Q @ K^T, plus the causal mask where the chunk reaches
            // the diagonal.
            llk.reconfigDataFormat(k, kk, q);
            k.op(.pack_reconfig_data_format, .{qk_im});
            sdpa.matmulBlocks(k, q, kk, qk_im, qk, null, mask, mask);
            const k_start = p.mul(sk);
            var if_mask = k.openIf(q_start_tile.lt(k_start.add(sk)));
            {
                llk.reconfigDataFormat(k, qk_im, mask);
                // Back to reserved, to stamp the mask in place.
                llk.wait(k, qk_im, qk_tiles);
                llk.pop(k, qk_im, qk_tiles);
                llk.reserve(k, qk_im, qk_tiles);
                sdpa.causalMaskLightweight(k, mask, 0, 1, qk_im, q_start_tile, k_start, sq, sk, c.dst_size);
                llk.push(k, qk_im, qk_tiles);
            }
            if_mask.yieldThen(.{});

            // The row maxima (with the previous ones), QK = exp((QK - max)
            // * scale) and its partial row sums.
            llk.reconfigDataFormat(k, qk_im, scale);
            sdpa.reduceRows(k, .max, qk_im, scale, cur_max, prev_max, sq, sk, p.gt(0), .c);
            sdpa.subExpBlockBcastColsInplace(k, qk_im, cur_max, cur_sum, sq, sk, c.scale_fp32, .rc, c.sub_exp_granularity);

            // OUT = QK @ V.
            llk.reconfigDataFormat(k, v, qk_im);
            k.op(.pack_reconfig_data_format, .{cur_out});
            sdpa.matmulBlocks(k, qk_im, v, cur_out, qkv, null, mask, mask);
            llk.pop(k, qk_im, qk_tiles);
            llk.reconfigDataFormat(k, prev_max, cur_max);

            // Later chunks: the previous sum and output rescaled to the new
            // maximum and added.
            var if_later = k.openIf(p.gt(0));
            {
                sdpa.subExpBlock(k, prev_max, cur_max, exp_max_diff, sq, c.scale_fp32, c.exp_approx);
                llk.pop(k, prev_max, sq);
                sdpa.mulTilesBcastColsInplace(k, prev_sum, exp_max_diff, sq);
                sdpa.addBlockInplace(k, cur_sum, prev_sum, sq, true);
                sdpa.mulBlockBcastColsBlocked(k, prev_out, exp_max_diff, cur_out, sq, c.vdht, c.dht_granularity, true);
            }
            if_later.yieldThen(.{});
        }
        k_loop.yield(.{});

        // The last chunk's buffers: B after an odd count, A after an even.
        const odd = k_chunks.rem(2).eq(1);
        const last_sum = k.selectCb(odd, sum_b, sum_a);
        const last_max = k.selectCb(odd, max_b, max_a);
        const last_out = k.selectCb(odd, out_b, out_a);
        // The sums within the tiles, their reciprocals, the normalized
        // output.
        sdpa.matmulReduce(k, col_identity, last_sum, sq, c.stats_granularity);
        sdpa.recipBlockInplace(k, last_sum, sq);
        k.op(.pack_reconfig_data_format, .{out});
        sdpa.mulBlockBcastColsBlocked(k, last_out, last_sum, out, sq, c.vdht, c.dht_granularity, false);
        llk.pop(k, last_max, sq);
        llk.pop(k, q, sq * c.dht);
    }
    q_loop.yield(.{});
}
