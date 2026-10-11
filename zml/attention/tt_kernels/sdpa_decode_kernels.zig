//! The device kernels of tt-metal's flash decode
//! (`transformer/sdpa_decode/device/kernels/`, tt-metal d04395ed) in the
//! TTKernel DSL, for the configuration the port runs: causal, positions in a
//! tensor, Q in DRAM, a paged cache or not; no attention mask, sink, sliding
//! window or block padding, K neither multicast nor reused, full tiles.
//!
//! The cores of a head split its key chunks; their partial results meet in
//! a tree, each core waiting for its children's (output, max, sum), merging
//! them, and sending its own to its parent, the root writing the output.

const std = @import("std");

const zml = @import("../../zml.zig");
const dsl = zml.kernel.ttkernel.dsl;
const llk = @import("llk.zig");
const sdpa = @import("sdpa_llk.zig");
const Builder = dsl.Builder;
const Value = dsl.Value;
const Cb = dsl.CircularBuffer;

/// Circular buffer indices of the program.
pub const cb = struct {
    pub const q = 0;
    pub const k = 1;
    pub const v = 2;
    pub const mask = 3;
    pub const identity_scale = 5;
    pub const m_in = 6;
    pub const l_in = 7;
    pub const writer_cur_pos = 8;
    pub const page_table = 9;
    pub const col_identity = 11;
    pub const zero = 12;
    pub const compute_cur_pos = 15;
    pub const out_o = 16;
    pub const out_m = 17;
    pub const out_l = 18;
    pub const intermed = 19;
    pub const out_final = 20;
    pub const prev_sum_2 = 21;
    pub const exp_max_diff_2 = 22;
    pub const out_acc_2 = 23;
    pub const qk_im = 24;
    pub const out_im = 25;
    pub const out_acc = 26;
    pub const max_1 = 27;
    pub const max_2 = 28;
    pub const sum_1 = 29;
    pub const sum_2 = 30;
    pub const exp_max_diff = 31;
};

pub const Config = struct {
    pnht: u32,
    st: u32,
    dht: u32,
    vdht: u32,
    /// Tiles per key chunk, 0: chosen per position, up to
    /// `max_dynamic_chunk_size`.
    sk_chunk_t: u32,
    max_dynamic_chunk_size: u32,
    num_cores_per_head: u32,
    num_heads_per_core: u32,
    num_kv_heads: u32,
    num_q_heads: u32,
    paged: bool,
    block_size_t: u32,
    bkv: u32,
    tile_bytes: u32,
    cur_pos_page_bytes: u32,
    page_table_page_bytes: u32,
    scale_fp32: u32,
    identity_scalar_packed: u32,
    num_tree_reduction_rounds: u32,
    // compute
    out_subblock_w: u32,
    out_subblock_h: u32,
    out_in0_num_subblocks: u32,
    out_in1_num_subblocks: u32,
    qk_subblock_w: u32,
    qk_subblock_h: u32,
    qk_in0_num_subblocks: u32,
    qk_in1_num_subblocks: u32,
    exp_approx: bool,
    dht_granularity: u32,
    sub_exp_granularity: ?u32,

    fn outChunkTiles(c: Config) u32 {
        return c.pnht * c.vdht;
    }
};

const no_core = -1;

fn runtimeArgs(comptime n: usize, k: *Builder, first: u32) [n]Value {
    var out: [n]Value = undefined;
    for (&out, 0..) |*v, i| v.* = k.runtimeArg(first + @as(u32, @intCast(i)));
    return out;
}

/// `virtual_seq_tile_id_to_physical_tile_id`: the first tile of a sequence
/// tile row of `head` in the paged cache.
fn pagedTile(k: *Builder, c: Config, page_table: Value, seq_tile: Value, head: Value, wt: u32) Value {
    const block = k.loadL1(page_table, seq_tile.div(c.block_size_t), 32);
    return block.mul(c.num_kv_heads * c.block_size_t * wt)
        .add(head.mul(c.block_size_t * wt))
        .add(seq_tile.rem(c.block_size_t).mul(wt));
}

// =============================================================================
// Reader
// =============================================================================

/// `reader_decode_all.cpp`: compile-time args the q, k, v, writer and
/// compute position and (paged) page table buffers, then the accessor args
/// of q, k, v, the positions and the page table. Runtime args as tt-metal's.
pub fn reader(k: *Builder, c: Config) void {
    const cb_q = k.circularBufferOf(0, .bf16, 1);
    const cb_k = k.circularBufferOf(1, .bf16, 1);
    const cb_v = k.circularBufferOf(2, .bf16, 1);
    const cb_writer_pos = k.circularBufferOf(3, .si32, 1);
    const cb_compute_pos = k.circularBufferOf(4, .si32, 1);
    const cb_page_table = if (c.paged) k.circularBufferOf(5, .si32, 1) else undefined;
    const num_cbs: u32 = if (c.paged) 6 else 5;
    const noc = k.kernelNoc();
    const q_addr, const k_addr, const v_addr, const pos_addr, const page_table_addr = runtimeArgs(5, k, 0);
    const cur_head_group, const cur_batch, const core_num_in_reduce = runtimeArgs(3, k, 10);
    const tb = c.tile_bytes;

    const q_args = k.tensorAccessorArgs(num_cbs, 0);
    const k_args = k.nextTensorAccessorArgs(q_args);
    const v_args = k.nextTensorAccessorArgs(k_args);
    const pos_args = k.nextTensorAccessorArgs(v_args);

    // Idle cores have no tensor.
    var active = k.openIf(q_addr.ne(0));
    {
        // The positions, for this kernel, the writer and compute.
        k.op(.cb_reserve_back, .{ cb_writer_pos, 1 });
        const writer_pos = k.op(.get_write_ptr, .{cb_writer_pos});
        k.op(.noc_async_read_tile, .{ 0, k.tensorAccessor(pos_args, pos_addr, c.cur_pos_page_bytes), writer_pos, noc });
        k.op(.noc_async_read_barrier, .{noc});
        k.op(.cb_reserve_back, .{ cb_compute_pos, 1 });
        llk.localCopy(k, writer_pos, k.op(.get_write_ptr, .{cb_compute_pos}), c.cur_pos_page_bytes);
        k.op(.cb_push_back, .{ cb_writer_pos, 1 });
        k.op(.cb_push_back, .{ cb_compute_pos, 1 });
        const cur_pos = k.loadL1(k.l1Ptr(writer_pos, 32), cur_batch, 32);

        var valid = k.openIf(cur_pos.ne(no_core));
        {
            const sk = sdpa.dynamicChunkTiles(k, cur_pos, c.sk_chunk_t, c.max_dynamic_chunk_size);
            const work = sdpa.workloadForCore(k, cur_pos, core_num_in_reduce, c.num_cores_per_head, sk.mul(32));
            var has_work = k.openIf(work.start.ne(work.end));
            {
                // This user's queries.
                const q_tiles = c.pnht * c.dht;
                k.op(.cb_reserve_back, .{ cb_q, q_tiles });
                const q_l1 = k.op(.get_write_ptr, .{cb_q});
                const q_acc = k.tensorAccessor(q_args, q_addr, tb);
                var q_loop, const t = llk.range(k, q_tiles);
                k.op(.noc_async_read_tile, .{ cur_batch.mul(q_tiles).add(t), q_acc, q_l1.add(t.mul(tb)), noc });
                q_loop.yield(.{});
                k.op(.noc_async_read_barrier, .{noc});
                k.op(.cb_push_back, .{ cb_q, q_tiles });

                const k_acc = k.tensorAccessor(k_args, k_addr, tb);
                const v_acc = k.tensorAccessor(v_args, v_addr, tb);
                const page_table = if (c.paged) blk: {
                    k.op(.cb_reserve_back, .{ cb_page_table, 1 });
                    const l1 = k.op(.get_write_ptr, .{cb_page_table});
                    const acc = k.tensorAccessor(k.nextTensorAccessorArgs(pos_args), page_table_addr, c.page_table_page_bytes);
                    k.op(.noc_async_read_tile, .{ cur_batch, acc, l1, noc });
                    k.op(.noc_async_read_barrier, .{noc});
                    k.op(.cb_push_back, .{ cb_page_table, 1 });
                    break :blk k.l1Ptr(l1, 32);
                } else undefined;

                var heads, const h = llk.range(k, c.num_heads_per_core);
                {
                    const head = cur_head_group.mul(c.num_heads_per_core).add(h);
                    const k_base = cur_batch.rem(c.bkv).mul(c.num_kv_heads).add(head).mul(c.st * c.dht);
                    const v_base = cur_batch.rem(c.bkv).mul(c.num_kv_heads).add(head).mul(c.st * c.vdht);
                    var chunks = k.openFor(work.start, work.end, 1, .{});
                    {
                        const chunk = chunks.iv.toI32();
                        const row0 = chunk.mul(sk);
                        // K transposed: [dht, sk] tiles.
                        k.op(.cb_reserve_back, .{ cb_k, sk.mul(c.dht) });
                        const k_l1 = k.op(.get_write_ptr, .{cb_k});
                        var rows, const row = llk.range(k, sk);
                        {
                            const first = if (c.paged) pagedTile(k, c, page_table, row0.add(row), head, c.dht) else k_base.add(row0.add(row).mul(c.dht));
                            var cols, const col = llk.range(k, c.dht);
                            k.op(.noc_async_read_tile, .{ first.add(col), k_acc, k_l1.add(col.mul(sk).add(row).mul(tb)), noc });
                            cols.yield(.{});
                        }
                        rows.yield(.{});
                        k.op(.noc_async_read_barrier, .{noc});
                        k.op(.cb_push_back, .{ cb_k, sk.mul(c.dht) });

                        k.op(.cb_reserve_back, .{ cb_v, sk.mul(c.vdht) });
                        const v_l1 = k.op(.get_write_ptr, .{cb_v});
                        var v_rows, const v_row = llk.range(k, sk);
                        {
                            const first = if (c.paged) pagedTile(k, c, page_table, row0.add(v_row), head, c.vdht) else v_base.add(row0.add(v_row).mul(c.vdht));
                            var cols, const col = llk.range(k, c.vdht);
                            k.op(.noc_async_read_tile, .{ first.add(col), v_acc, v_l1.add(v_row.mul(c.vdht).add(col).mul(tb)), noc });
                            cols.yield(.{});
                        }
                        v_rows.yield(.{});
                        k.op(.noc_async_read_barrier, .{noc});
                        k.op(.cb_push_back, .{ cb_v, sk.mul(c.vdht) });
                    }
                    chunks.yield(.{});
                }
                heads.yield(.{});
            }
            has_work.yieldThen(.{});
        }
        valid.yieldThen(.{});
    }
    active.yieldThen(.{});
}

// =============================================================================
// Writer
// =============================================================================

const neg_inf_bf16x2: u32 = 0xFF80FF80;

fn fillTile(k: *Builder, c: Config, base: Value, tile: anytype, word: u32) void {
    sdpa.fillTile(k, base, c.tile_bytes, tile, word);
}

/// `fill_tile_partial`: tile `tile` -inf right of column `pos`, zero up to
/// it (bf16 faces of 16x16).
fn fillTilePartial(k: *Builder, c: Config, base: Value, tile: Value, pos: Value) void {
    fillTile(k, c, base, tile, 0);
    var any = k.openIf(pos.ne(31));
    {
        const tile_l1 = base.add(tile.mul(c.tile_bytes));
        const p16 = k.l1Ptr(tile_l1, 16);
        const p32 = k.l1Ptr(tile_l1, 32);
        // Left faces wholly -inf if the column is in the left half.
        const face_start = k.select(pos.lt(15), 0, 1);
        var whole = k.openIf(face_start.eq(0));
        {
            var faces, const f = llk.range(k, 2);
            {
                var words, const w = llk.range(k, 128);
                k.storeL1(p32, f.mul(2).add(1).mul(128).add(w), neg_inf_bf16x2, 32);
                words.yield(.{});
            }
            faces.yield(.{});
        }
        whole.yieldThen(.{});
        const fill_pos = pos.add(1).rem(16);
        const odd = fill_pos.rem(2).eq(1);
        const fill_word = fill_pos.add(1).shr(1);
        var faces, const m = llk.range(k, 2);
        {
            const face = face_start.add(m.mul(2));
            var rows, const r = llk.range(k, 16);
            {
                var if_odd = k.openIf(odd);
                k.storeL1(p16, face.mul(256).add(fill_pos).add(r.mul(16)), neg_inf_bf16x2 >> 16, 16);
                if_odd.yieldThen(.{});
                var words = k.openFor(fill_word, 8, 1, .{});
                k.storeL1(p32, face.mul(128).add(words.iv.toI32()).add(r.mul(8)), neg_inf_bf16x2, 32);
                words.yield(.{});
            }
            rows.yield(.{});
        }
        faces.yield(.{});
    }
    any.yieldThen(.{});
}

/// `generate_mask` (causal): per head row, the key tiles before the
/// position's zero, its tile partial, the ones after -inf.
fn generateCausalMask(k: *Builder, c: Config, cb_mask: Cb, sk: Value, cur_pos: Value) void {
    const total = sk.mul(c.pnht);
    const in_chunk = cur_pos.rem(sk.mul(32));
    const pos_t = in_chunk.div(32);
    const pos_in_tile = in_chunk.rem(32);
    k.op(.cb_reserve_back, .{ cb_mask, total });
    const base = k.op(.get_read_ptr, .{cb_mask});
    const tb = c.tile_bytes;
    var tiles, const i = llk.range(k, sk);
    {
        var before = k.openIfElse(i.lt(pos_t), .{});
        {
            var first = k.openIfElse(i.eq(0), .{});
            fillTile(k, c, base, i, 0);
            first.yieldThen(.{});
            llk.localCopy(k, base, base.add(i.mul(tb)), tb);
            first.yieldElse(.{});
        }
        before.yieldThen(.{});
        {
            var at = k.openIfElse(i.eq(pos_t), .{});
            fillTilePartial(k, c, base, i, pos_in_tile);
            at.yieldThen(.{});
            {
                var next = k.openIfElse(i.eq(pos_t.add(1)), .{});
                fillTile(k, c, base, i, neg_inf_bf16x2);
                next.yieldThen(.{});
                llk.localCopy(k, base.add(pos_t.add(1).mul(tb)), base.add(i.mul(tb)), tb);
                next.yieldElse(.{});
            }
            at.yieldElse(.{});
        }
        before.yieldElse(.{});
        // The other head rows: copies.
        var rows = k.openFor(1, c.pnht, 1, .{});
        llk.localCopy(k, base.add(i.mul(tb)), base.add(rows.iv.toI32().mul(sk).add(i).mul(tb)), tb);
        rows.yield(.{});
    }
    tiles.yield(.{});
    k.op(.cb_push_back, .{ cb_mask, total });
}

const reduce_helpers_h = "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp";
const l1_helpers_h = "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp";
const dataflow_h = "api/dataflow/dataflow_api.h";

/// `writer_decode_all.cpp`: compile-time args the mask, identity scale,
/// m_in, l_in, position, column identity, zero, out_o, out_m, out_l,
/// intermediate and output buffers, the reducer semaphore, then the
/// output's accessor args. Runtime args as tt-metal's.
pub fn writer(k: *Builder, c: Config) void {
    const cb_mask = k.circularBufferOf(0, .bf16, 1);
    _ = k.circularBufferOf(1, .bf16, 1);
    const cb_m_in = k.circularBufferOf(2, .bf16, 1);
    const cb_l_in = k.circularBufferOf(3, .bf16, 1);
    const cb_pos = k.circularBufferOf(4, .si32, 1);
    const cb_col_identity = k.circularBufferOf(5, .bf16, 1);
    _ = k.circularBufferOf(6, .bf16, 1);
    const cb_out_o = k.circularBufferOf(7, .bf16, 1);
    const cb_out_m = k.circularBufferOf(8, .bf16, 1);
    const cb_out_l = k.circularBufferOf(9, .bf16, 1);
    const cb_intermed = k.circularBufferOf(10, .bf16, 1);
    const cb_out = k.circularBufferOf(11, .bf16, 1);
    const reducer_sem = llk.Semaphore.init(k, 12);
    const noc = k.kernelNoc();
    const out_addr = k.runtimeArg(0);
    const cur_head_group, const cur_batch, const core_num_in_reduce = runtimeArgs(3, k, 5);
    const is_tree_root, const parent, const send_at_round = runtimeArgs(3, k, 10);
    const tb = c.tile_bytes;
    const out_chunk = c.outChunkTiles();
    const group_xs = 22;
    const group_ys = 22 + c.num_cores_per_head;

    var active = k.openIf(out_addr.ne(0));
    {
        k.op(.cb_wait_front, .{ cb_pos, 1 });
        const cur_pos = k.loadL1(k.l1Ptr(k.op(.get_read_ptr, .{cb_pos}), 32), cur_batch, 32);
        k.op(.cb_pop_front, .{ cb_pos, 1 });
        var valid = k.openIf(cur_pos.ne(no_core));
        {
            const sk = sdpa.dynamicChunkTiles(k, cur_pos, c.sk_chunk_t, c.max_dynamic_chunk_size);
            const work = sdpa.workloadForCore(k, cur_pos, core_num_in_reduce, c.num_cores_per_head, sk.mul(32));
            var has_work = k.openIf(work.start.ne(work.end));
            {
                // Compute's constant tiles and the causal mask.
                k.call(reduce_helpers_h, "dataflow_kernel_lib::calculate_and_prepare_reduce_scaler", .{ cb.identity_scale, "ckernel::PoolType::MAX", "ckernel::ReduceDim::REDUCE_ROW", "dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR" }, .{});
                k.call(l1_helpers_h, "dataflow_kernel_lib::prepare_zero_tile", .{cb.zero}, .{});
                sdpa.bcastColScalar(k, cb_col_identity, c.identity_scalar_packed);
                generateCausalMask(k, c, cb_mask, sk, cur_pos);

                const out_acc = k.tensorAccessor(k.tensorAccessorArgs(13, 0), out_addr, tb);
                k.op(.noc_async_write_barrier, .{noc});

                var heads, const h = llk.range(k, c.num_heads_per_core);
                {
                    const head = cur_head_group.mul(c.num_heads_per_core).add(h);
                    // The children's (l, m, o), in their rounds' slots of
                    // the intermediate buffer.
                    var rounds, const round = llk.range(k, c.num_tree_reduction_rounds);
                    {
                        const child = k.runtimeArgAt(round.add(16));
                        var if_child = k.openIf(child.ne(no_core).bitAnd(child.lt(work.num_chunks)));
                        {
                            var spin = k.openWhile(.{}, .{});
                            k.call(dataflow_h, "invalidate_l1_cache", .{}, .{});
                            spin.yieldBefore(reducer_sem.value().shr(round.mul(4)).bitAnd(0xF).lt(1), .{});
                            spin.yieldAfter(.{});
                            const slot = k.op(.get_read_ptr, .{cb_intermed}).add(round.mul((out_chunk + 2 * c.pnht) * tb));
                            for ([_]struct { Cb, u32, u32 }{ .{ cb_l_in, c.pnht, 0 }, .{ cb_m_in, c.pnht, c.pnht }, .{ cb_out_o, out_chunk, 2 * c.pnht } }) |part| {
                                k.op(.cb_reserve_back, .{ part[0], part[1] });
                                llk.localCopy(k, slot.add(part[2] * tb), k.op(.get_write_ptr, .{part[0]}), part[1] * tb);
                                k.op(.cb_push_back, .{ part[0], part[1] });
                            }
                        }
                        if_child.yieldThen(.{});
                    }
                    rounds.yield(.{});

                    // A child sends its (l, m, o) to its parent.
                    var if_send = k.openIf(is_tree_root.eq(0).bitAnd(parent.ne(no_core)));
                    {
                        k.op(.cb_wait_front, .{ cb_out_o, out_chunk });
                        k.op(.cb_wait_front, .{ cb_out_m, c.pnht });
                        k.op(.cb_wait_front, .{ cb_out_l, c.pnht });
                        const px = k.runtimeArgAt(parent.add(group_xs)).toIndex();
                        const py = k.runtimeArgAt(parent.add(group_ys)).toIndex();
                        const slot = k.op(.get_write_ptr, .{cb_intermed}).add(send_at_round.mul((out_chunk + 2 * c.pnht) * tb));
                        remoteWrite(k, k.op(.get_read_ptr, .{cb_out_l}), px, py, slot, c.pnht * tb);
                        remoteWrite(k, k.op(.get_read_ptr, .{cb_out_m}), px, py, slot.add(c.pnht * tb), c.pnht * tb);
                        remoteWrite(k, k.op(.get_read_ptr, .{cb_out_o}), px, py, slot.add(2 * c.pnht * tb), out_chunk * tb);
                        k.op(.noc_async_write_barrier, .{noc});
                        // Round r is the parent's semaphore nibble r.
                        reducer_sem.up(px, py, k.int(1).shl(send_at_round.mul(4)));
                        k.op(.cb_pop_front, .{ cb_out_o, out_chunk });
                        k.op(.cb_pop_front, .{ cb_out_m, c.pnht });
                        k.op(.cb_pop_front, .{ cb_out_l, c.pnht });
                        k.op(.noc_async_atomic_barrier, .{noc});
                    }
                    if_send.yieldThen(.{});

                    // The root writes the output.
                    var if_root = k.openIf(is_tree_root.ne(0));
                    {
                        const first_tile = cur_batch.mul(out_chunk);
                        k.op(.cb_wait_front, .{ cb_out, out_chunk });
                        k.op(.noc_async_writes_flushed, .{noc});
                        const l1 = k.op(.get_read_ptr, .{cb_out});
                        if (c.num_kv_heads > 1) {
                            writePartialTiles(k, c, l1, out_acc, first_tile, head);
                        } else {
                            var tiles, const t = llk.range(k, out_chunk);
                            k.op(.noc_async_write_tile, .{ first_tile.add(t), out_acc, l1.add(t.mul(tb)), noc });
                            tiles.yield(.{});
                        }
                        k.op(.noc_async_write_barrier, .{noc});
                        k.op(.cb_pop_front, .{ cb_out, out_chunk });
                    }
                    if_root.yieldThen(.{});
                }
                heads.yield(.{});
            }
            has_work.yieldThen(.{});
        }
        valid.yieldThen(.{});
    }
    active.yieldThen(.{});
}

/// Writes `size` bytes at L1 `src` to L1 `addr` of core `(x, y)`.
fn remoteWrite(k: *Builder, src: Value, x: Value, y: Value, addr: Value, size: u32) void {
    const noc = k.kernelNoc();
    const dst = k.op(.get_noc_addr, .{ x, y, addr, noc });
    k.call(dataflow_h, "noc_async_write", .{}, .{ src, dst, size, noc });
}

/// `write_partial_tiles_to_memory`: of each output tile, the rows of this
/// key-value head's query heads (two face lines each).
fn writePartialTiles(k: *Builder, c: Config, l1: Value, out_acc: Value, first_tile: Value, head: Value) void {
    const tb = c.tile_bytes;
    const num_hidden = c.vdht;
    const heads_to_write = c.num_q_heads / c.num_kv_heads;
    const face_line = 16 * 2;
    const noc = k.kernelNoc();
    var hidden_loop, const hidden = llk.range(k, num_hidden);
    {
        var heads_loop, const hh = llk.range(k, heads_to_write);
        {
            const row = head.mul(heads_to_write).add(hh);
            const tile_row = row.rem(32);
            const head_tile = row.div(32);
            const offset = k.select(tile_row.lt(16), tile_row, tile_row.add(16)).mul(face_line);
            const index = head_tile.mul(num_hidden).add(hidden);
            const src = l1.add(index.mul(tb)).add(offset);
            const page = first_tile.add(index);
            for ([_]u32{ 0, 256 * 2 }) |face| {
                const dst = k.op(.tensor_accessor_get_noc_addr, .{ out_acc, page, offset.add(face), noc });
                k.call(dataflow_h, "noc_async_write", .{}, .{ src.add(face), dst, face_line, noc });
            }
        }
        heads_loop.yield(.{});
    }
    hidden_loop.yield(.{});
}

// =============================================================================
// Compute
// =============================================================================

pub const compute_cbs = [_]u32{ cb.q, cb.k, cb.v, cb.mask, cb.identity_scale, cb.m_in, cb.l_in, cb.zero, cb.compute_cur_pos, cb.out_o, cb.out_m, cb.out_l, cb.out_final, cb.prev_sum_2, cb.exp_max_diff_2, cb.out_acc_2, cb.qk_im, cb.out_im, cb.out_acc, cb.max_1, cb.max_2, cb.sum_1, cb.sum_2, cb.exp_max_diff };

/// `sdpa_flash_decode.cpp`: compile-time args the `compute_cbs`. Runtime
/// args as tt-metal's.
pub fn compute(k: *Builder, c: Config) void {
    var cbs: [compute_cbs.len]Cb = undefined;
    for (&cbs, 0..) |*x, i| x.* = k.circularBufferOf(@intCast(i), .bf16, 1);
    const q, const kk, const v, const mask, const scale, const m_in, const l_in, const zero, const cur_pos_cb, const out_o, const out_m, const out_l, const out_final, const prev_sum_2, const exp_max_diff_2, const out_acc_2, const qk_im, const out_im, const out_acc, const max_1, const max_2, const sum_1, const sum_2, const exp_max_diff = cbs;
    const do_reduce, _, _, const cur_batch, const core_num_in_reduce = runtimeArgs(5, k, 0);
    const is_tree_root, const parent = runtimeArgs(2, k, 7);
    const sq = c.pnht;
    const out_chunk = c.outChunkTiles();
    const q_tiles = sq * c.dht;
    const mode: sdpa.VectorMode = .rc;
    const dynamic = c.sk_chunk_t == 0;

    // Idle cores: 65.
    var active = k.openIf(do_reduce.ne(65));
    {
        llk.wait(k, cur_pos_cb, 1);
        const cur_pos = k.callValue("api/compute/cb_api.h", "read_tile_value", .{}, .{ cur_pos_cb, 0, cur_batch }, .int(k.ctx, .i32));
        llk.pop(k, cur_pos_cb, 1);
        var valid = k.openIf(cur_pos.ne(no_core));
        {
            const sk = sdpa.dynamicChunkTiles(k, cur_pos, c.sk_chunk_t, c.max_dynamic_chunk_size);
            const work = sdpa.workloadForCore(k, cur_pos, core_num_in_reduce, c.num_cores_per_head, sk.mul(32));
            var has_work = k.openIf(work.start.ne(work.end));
            {
                k.call("api/compute/compute_kernel_hw_startup.h", "compute_kernel_hw_startup", .{"SrcOrder::Reverse"}, .{ q, kk, qk_im });
                k.call("api/compute/matmul.h", "matmul_init", .{}, .{ q, kk });
                llk.wait(k, q, q_tiles);
                const qk_tiles = sk.mul(sq);
                const qk: sdpa.MatmulBlocks = .{
                    .m = sq,
                    .n = sk,
                    .k = k.int(c.dht),
                    .in0_num_subblocks = if (dynamic) 1 else c.qk_in0_num_subblocks,
                    .in1_num_subblocks = if (dynamic) k.int(1) else k.int(c.qk_in1_num_subblocks),
                    .in0_block_w = k.int(c.dht),
                    .subblock_h = if (dynamic) 1 else c.qk_subblock_h,
                    .subblock_w = if (dynamic) sk else k.int(c.qk_subblock_w),
                    .transpose = true,
                };
                const out: sdpa.MatmulBlocks = .{
                    .m = sq,
                    .n = k.int(c.vdht),
                    .k = sk,
                    .in0_num_subblocks = c.out_in0_num_subblocks,
                    .in1_num_subblocks = k.int(c.out_in1_num_subblocks),
                    .in0_block_w = sk,
                    .subblock_h = c.out_subblock_h,
                    .subblock_w = k.int(c.out_subblock_w),
                    .transpose = false,
                };
                const cur_max, const prev_max, const cur_sum, const prev_sum = .{ max_1, max_2, sum_1, sum_2 };

                var heads, _ = llk.range(k, c.num_heads_per_core);
                {
                    var chunks = k.openFor(work.start, work.end, 1, .{});
                    {
                        const chunk = chunks.iv.toI32();
                        const first = chunk.eq(work.start);
                        const last = chunk.eq(work.end.sub(1));
                        // The first chunk's output accumulates in place.
                        const out_mm = k.selectCb(first, out_acc, out_im);

                        llk.reconfigDataFormat(k, kk, q);
                        k.op(.pack_reconfig_data_format, .{qk_im});
                        // Causal mask on the reducer's last chunk: fused in
                        // the matmul with dynamic chunks, added after it
                        // otherwise.
                        const masked = last.bitAnd(do_reduce.ne(0));
                        sdpa.matmulBlocks(k, q, kk, qk_im, qk, if (dynamic) masked else null, mask, zero);
                        if (!dynamic) {
                            var if_mask = k.openIf(masked);
                            llk.reconfigDataFormat(k, qk_im, mask);
                            sdpa.addBlockInplace(k, qk_im, mask, qk_tiles, false);
                            if_mask.yieldThen(.{});
                        }

                        llk.reconfigDataFormat(k, qk_im, scale);
                        k.op(.pack_reconfig_data_format, .{cur_max});
                        sdpa.reduceRows(k, .max, qk_im, scale, cur_max, prev_max, sq, sk, chunk.gt(work.start), mode);
                        llk.reconfigDataFormat(k, qk_im, cur_max);
                        k.op(.pack_reconfig_data_format, .{qk_im});
                        sdpa.subExpBlockBcastColsInplace(k, qk_im, cur_max, null, sq, sk, c.scale_fp32, mode, c.sub_exp_granularity);
                        llk.wait(k, qk_im, qk_tiles);
                        llk.reconfigDataFormat(k, qk_im, scale);
                        k.op(.pack_reconfig_data_format, .{cur_sum});
                        sdpa.reduceRows(k, .sum, qk_im, scale, cur_sum, cur_sum, sq, sk, k.lift(false), mode);
                        llk.reconfigDataFormat(k, v, qk_im);
                        k.op(.pack_reconfig_data_format, .{out_im});
                        sdpa.matmulBlocks(k, qk_im, v, out_mm, out, null, mask, zero);
                        llk.reconfigDataFormatSrca(k, out_im);
                        llk.pop(k, qk_im, qk_tiles);

                        // Later chunks rescale what came before to the new
                        // maximum, then add.
                        var if_later = k.openIf(first.eq(false));
                        {
                            llk.reconfigDataFormat(k, prev_max, cur_max);
                            k.op(.pack_reconfig_data_format, .{exp_max_diff});
                            sdpa.subExpBlock(k, prev_max, cur_max, exp_max_diff, sq, c.scale_fp32, c.exp_approx);
                            llk.pop(k, prev_max, sq);
                            sdpa.mulBlockInplace(k, prev_sum, exp_max_diff, sq);
                            llk.reconfigDataFormat(k, out_acc, exp_max_diff);
                            k.op(.pack_reconfig_data_format, .{out_acc});
                            sdpa.mulBlockBcastCols(k, out_acc, exp_max_diff, out_acc, sq, c.vdht);
                            llk.reconfigDataFormat(k, cur_sum, prev_sum);
                            k.op(.pack_reconfig_data_format, .{cur_sum});
                            sdpa.addBlockInplace(k, cur_sum, prev_sum, sq, true);
                            llk.reconfigDataFormat(k, out_acc, out_im);
                            k.op(.pack_reconfig_data_format, .{out_acc});
                            sdpa.addBlockInplace(k, out_acc, out_im, out_chunk, true);
                        }
                        if_later.yieldThen(.{});
                        llk.reconfigDataFormat(k, cur_max, cur_max);
                        k.op(.pack_reconfig_data_format, .{prev_max});
                        sdpa.moveBlock(k, cur_max, prev_max, sq, true);
                        sdpa.moveBlock(k, cur_sum, prev_sum, sq, true);
                    }
                    chunks.yield(.{});

                    // The children's partial results, round by round.
                    var rounds, const round = llk.range(k, c.num_tree_reduction_rounds);
                    {
                        const child = k.runtimeArgAt(round.add(12));
                        var if_child = k.openIf(child.ne(no_core).bitAnd(child.lt(work.num_chunks)));
                        {
                            sdpa.moveBlock(k, l_in, prev_sum_2, sq, true);
                            sdpa.correctionBlock(k, m_in, prev_sum_2, cur_max, prev_max, cur_sum, prev_sum, exp_max_diff, exp_max_diff_2, sq, c.scale_fp32, c.exp_approx, mode);
                            sdpa.moveBlock(k, out_o, out_acc_2, out_chunk, true);
                            sdpa.mulBlockBcastColsInplace(k, out_acc, exp_max_diff, sq, c.vdht, c.dht_granularity);
                            sdpa.mulBlockBcastColsInplace(k, out_acc_2, exp_max_diff_2, sq, c.vdht, c.dht_granularity);
                            sdpa.addBlockInplace(k, out_acc, out_acc_2, out_chunk, true);
                            llk.pop(k, prev_max, sq);
                            llk.pop(k, m_in, sq);
                            sdpa.moveBlock(k, cur_max, prev_max, sq, true);
                            sdpa.moveBlock(k, cur_sum, prev_sum, sq, true);
                        }
                        if_child.yieldThen(.{});
                    }
                    rounds.yield(.{});

                    // The root normalizes; the others pass on.
                    var if_root = k.openIfElse(is_tree_root.ne(0), .{});
                    {
                        llk.reconfigDataFormat(k, prev_sum, prev_sum);
                        k.op(.pack_reconfig_data_format, .{prev_sum});
                        sdpa.recipBlockInplace(k, prev_sum, sq);
                        llk.reconfigDataFormat(k, out_acc, prev_sum);
                        k.op(.pack_reconfig_data_format, .{out_acc});
                        sdpa.mulBlockBcastColsInplace(k, out_acc, prev_sum, sq, c.vdht, c.dht_granularity);
                        k.op(.pack_reconfig_data_format, .{out_final});
                        llk.pop(k, prev_max, sq);
                        sdpa.moveBlock(k, out_acc, out_final, out_chunk, true);
                    }
                    if_root.yieldThen(.{});
                    {
                        var if_parent = k.openIf(parent.ne(no_core));
                        sdpa.moveBlock(k, out_acc, out_o, out_chunk, true);
                        sdpa.moveBlock(k, prev_max, out_m, sq, true);
                        sdpa.moveBlock(k, prev_sum, out_l, sq, true);
                        if_parent.yieldThen(.{});
                    }
                    if_root.yieldElse(.{});
                }
                heads.yield(.{});
                llk.pop(k, q, q_tiles);
            }
            has_work.yieldThen(.{});
        }
        valid.yieldThen(.{});
    }
    active.yieldThen(.{});
}
