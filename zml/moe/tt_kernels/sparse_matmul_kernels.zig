//! The device kernels of tt-metal's sparse matmul (`ttnn::sparse_matmul`,
//! `matmul/device/kernels/`, tt-metal d04395ed) in the TTKernel DSL:
//! `reader_bmm_tile_layout_in0_sender_padding.cpp`,
//! `reader_bmm_tile_layout_in0_receiver.cpp`,
//! `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` and
//! `bmm_large_block_zm_fused_bias_activation.cpp`, for the sparse program:
//! interleaved operands, batches told valid by the in0 reader (no `nnz`),
//! no bias, activation, untilize, packer accumulation or sharding; `in1`
//! optionally transposed (`IN1_TRANSPOSE_TILE`).

const std = @import("std");

const zml = @import("../../zml.zig");
const dsl = zml.kernel.ttkernel.dsl;
const llk = @import("../../attention/tt_kernels/llk.zig");
const Builder = dsl.Builder;
const Value = dsl.Value;
const Cb = dsl.CircularBuffer;

/// Circular buffer indices of the program.
pub const cb = struct {
    pub const in0 = 0;
    pub const in1 = 1;
    pub const out = 4;
    pub const intermed0 = 5;
    pub const sparsity0 = 6;
    pub const sparsity1 = 7;
};

/// The `IGNORE_BATCH` state of the in0 receiver semaphore.
const ignore_batch = 2;

/// The compile-time arguments of the four kernels, as the factory computes
/// them.
pub const Config = struct {
    // in0
    in0_stride_w: u32,
    in0_stride_h: u32,
    in0_next_inner_dim_block_stride: u32,
    in0_next_h_dim_block_stride: u32,
    in0_block_w: u32,
    in0_block_h: u32,
    in0_block_num_tiles: u32,
    in0_aligned_tile_bytes: u32,
    num_blocks_inner_dim: u32,
    num_blocks_w_dim: u32,
    num_blocks_h_dim: u32,
    /// Receivers of the in0 blocks; none: no multicast.
    in0_mcast_num_dests: u32,
    in0_mcast_num_cores: u32,
    mt_kt: u32,
    /// Batches of `a` the reader walks, sparsity rows.
    batch_a: u32,
    /// Sparsity entries per row: experts.
    batch_b: u32,
    sparsity_page_bytes: u32,
    /// `a` has one batch for all of a row's experts.
    bcast_a: bool,
    // in1 and output
    in1_stride_w: u32,
    in1_stride_h: u32,
    in1_next_block_stride: u32,
    in1_next_w_dim_block_stride: u32,
    in1_block_w: u32,
    in1_block_h: u32,
    in1_block_num_tiles: u32,
    in1_aligned_tile_bytes: u32,
    kt_nt: u32,
    out_stride_w: u32,
    out_stride_h: u32,
    out_next_subblock_stride_w: u32,
    out_next_subblock_stride_h: u32,
    out_next_w_dim_block_stride: u32,
    out_next_h_dim_block_stride: u32,
    out_subblock_w: u32,
    out_subblock_h: u32,
    out_subblock_tile_count: u32,
    mt_nt: u32,
    // compute
    in0_num_subblocks: u32,
    in0_subblock_num_tiles: u32,
    in1_num_subblocks: u32,
    in1_per_core_w: u32,
    /// Sparsity entries, valid or not: the batches compute is told about.
    batch_compute: u32,
    transpose_b: bool,
    fp32_dest_acc_en: bool,
};

fn runtimeArgs(comptime n: usize, k: *Builder, first: u32) [n]Value {
    var out: [n]Value = undefined;
    for (&out, 0..) |*v, i| v.* = k.runtimeArg(first + @as(u32, @intCast(i)));
    return out;
}

fn nocCoords(comptime n: usize, k: *Builder, first: u32) [n]Value {
    var out = runtimeArgs(n, k, first);
    for (&out) |*v| v.* = v.toIndex();
    return out;
}

fn sel(k: *Builder, cond: Value, a: anytype, b: anytype) Value {
    return k.select(cond, a, b);
}

/// Posts a batch's validity to the three compute threads.
fn postValidity(k: *Builder, v: Value) void {
    llk.mailboxWrite(k, .unpack, v);
    llk.mailboxWrite(k, .math, v);
    llk.mailboxWrite(k, .pack, v);
}

/// in0 sender: compile-time args `cb.in0`, `cb.sparsity0`, the sender and
/// receiver semaphores, then the accessor args of `a` and the sparsity.
/// Runtime args: `a`'s address, first tile, the multicast rectangle as the
/// kernel's NoC walks it, the rows of the last block, the sparsity's
/// address.
pub fn in0Sender(k: *Builder, c: Config) void {
    const cb_in0 = k.circularBufferOf(0, .bf16, 1);
    const cb_sparsity = k.circularBufferOf(1, .bf16, 1);
    const sender = llk.Semaphore.init(k, 2);
    const receiver = llk.Semaphore.init(k, 3);
    const in0_addr, const start_tile = runtimeArgs(2, k, 0);
    // NoC coordinates are unsigned: `index`.
    const x0, const y0, const x1, const y1 = nocCoords(4, k, 2);
    const last_block_h, const sparsity_addr = runtimeArgs(2, k, 6);
    const noc = k.kernelNoc();
    const mcast = c.in0_mcast_num_dests > 0;

    const tile_bytes = k.op(.get_tile_size, .{cb_in0});
    const in0_args = k.tensorAccessorArgs(4, 0);
    const s0 = k.tensorAccessor(in0_args, in0_addr, tile_bytes);
    const s_sparsity = k.tensorAccessor(k.nextTensorAccessorArgs(in0_args), sparsity_addr, c.sparsity_page_bytes);

    if (mcast) receiver.set(llk.valid);
    k.op(.cb_reserve_back, .{ cb_sparsity, 1 });
    const sparsity_l1 = k.op(.get_write_ptr, .{cb_sparsity});
    const sparsity = k.l1Ptr(sparsity_l1, 16);
    const in0_start = k.variable(start_tile);

    var rows, const b = llk.range(k, c.batch_a);
    {
        k.op(.noc_async_read_tile, .{ b, s_sparsity, sparsity_l1, noc });
        k.op(.noc_async_read_barrier, .{noc});
        var experts, const e = llk.range(k, c.batch_b);
        {
            const is_valid = sel(k, k.loadL1(sparsity, e, 16).ne(0), 1, 0);
            if (mcast) {
                sender.wait(c.in0_mcast_num_dests);
                sender.set(0);
                receiver.set(sel(k, is_valid.ne(0), llk.valid, ignore_batch));
                receiver.setMulticast(x0, y0, x1, y1, c.in0_mcast_num_cores);
                k.op(.noc_async_writes_flushed, .{noc});
                receiver.set(llk.valid);
            }
            postValidity(k, is_valid);

            var if_valid = k.openIf(is_valid);
            {
                var bh_loop, const bh = llk.range(k, c.num_blocks_h_dim);
                {
                    const h_tile = in0_start.get().add(bh.mul(c.in0_next_h_dim_block_stride));
                    var bw_loop, _ = llk.range(k, c.num_blocks_w_dim);
                    {
                        var blocks, const block = llk.range(k, c.num_blocks_inner_dim);
                        {
                            k.op(.cb_reserve_back, .{ cb_in0, c.in0_block_num_tiles });
                            const l1 = k.op(.get_write_ptr, .{cb_in0});
                            const row_tile = h_tile.add(block.mul(c.in0_next_inner_dim_block_stride));
                            var h_loop, const h = llk.range(k, c.in0_block_h);
                            {
                                // The last block's rows past `last_block_h` are padding.
                                var if_row = k.openIf(bh.lt(c.num_blocks_h_dim - 1).bitOr(h.lt(last_block_h)));
                                {
                                    var w_loop, const w = llk.range(k, c.in0_block_w);
                                    const tile = row_tile.add(h.mul(c.in0_stride_h)).add(w.mul(c.in0_stride_w));
                                    const offset = h.mul(c.in0_block_w).add(w).mul(c.in0_aligned_tile_bytes);
                                    k.op(.noc_async_read_tile, .{ tile, s0, l1.add(offset), noc });
                                    w_loop.yield(.{});
                                }
                                if_row.yieldThen(.{});
                            }
                            h_loop.yield(.{});
                            k.op(.noc_async_read_barrier, .{noc});
                            if (mcast) {
                                sender.wait(c.in0_mcast_num_dests);
                                sender.set(0);
                                k.op(.noc_async_write_multicast, .{ l1, c.in0_block_num_tiles * c.in0_aligned_tile_bytes, c.in0_mcast_num_cores, x0, y0, x1, y1, l1, noc, true });
                                k.op(.noc_async_writes_flushed, .{noc});
                                receiver.setMulticast(x0, y0, x1, y1, c.in0_mcast_num_cores);
                            }
                            k.op(.cb_push_back, .{ cb_in0, c.in0_block_num_tiles });
                        }
                        blocks.yield(.{});
                    }
                    bw_loop.yield(.{});
                }
                bh_loop.yield(.{});
            }
            if_valid.yieldThen(.{});
            // A batch of `a` per expert, consumed or skipped.
            if (!c.bcast_a) in0_start.set(in0_start.get().add(c.mt_kt));
        }
        experts.yield(.{});
        if (c.bcast_a) in0_start.set(in0_start.get().add(c.mt_kt));
    }
    rows.yield(.{});
    k.op(.noc_async_write_barrier, .{noc});
    k.op(.cb_push_back, .{ cb_sparsity, 1 });
    k.op(.cb_wait_front, .{ cb_sparsity, 1 });
    k.op(.cb_pop_front, .{ cb_sparsity, 1 });
}

/// in0 receiver: compile-time args `cb.in0`, the sender and receiver
/// semaphores. Runtime args: the sender's NoC coordinates.
pub fn in0Receiver(k: *Builder, c: Config) void {
    const cb_in0 = k.circularBufferOf(0, .bf16, 1);
    const sender = llk.Semaphore.init(k, 1);
    const receiver = llk.Semaphore.init(k, 2);
    const sender_x, const sender_y = nocCoords(2, k, 0);

    var batches, _ = llk.range(k, c.batch_compute);
    {
        receiver.set(llk.invalid);
        sender.up(sender_x, sender_y, 1);
        receiver.waitMin(llk.valid);
        const is_valid = sel(k, receiver.value().eq(llk.valid), 1, 0);
        postValidity(k, is_valid);
        var if_valid = k.openIf(is_valid);
        {
            var bh_loop, _ = llk.range(k, c.num_blocks_h_dim * c.num_blocks_w_dim * c.num_blocks_inner_dim);
            k.op(.cb_reserve_back, .{ cb_in0, c.in0_block_num_tiles });
            receiver.set(llk.invalid);
            sender.up(sender_x, sender_y, 1);
            receiver.wait(llk.valid);
            k.op(.cb_push_back, .{ cb_in0, c.in0_block_num_tiles });
            bh_loop.yield(.{});
        }
        if_valid.yieldThen(.{});
    }
    batches.yield(.{});
    k.op(.noc_async_atomic_barrier, .{k.kernelNoc()});
}

/// in1 reader and output writer: compile-time args `cb.in1`, `cb.out`,
/// `cb.sparsity1`, then the accessor args of `b`, the sparsity and the
/// output. Runtime args as tt-metal's (`in1_tensor_addr`, ...).
pub fn in1ReaderWriter(k: *Builder, c: Config) void {
    const cb_in1 = k.circularBufferOf(0, .bf16, 1);
    const cb_out = k.circularBufferOf(1, .bf16, 1);
    const cb_sparsity = k.circularBufferOf(2, .bf16, 1);
    const in1_addr, const in1_start = runtimeArgs(2, k, 0);
    const sparsity_addr, const out_addr, const out_start_tile, const last_block_w = runtimeArgs(4, k, 6);
    const out_num_nonzero_subblocks_h, const out_last_subblock_h, const padded_block_tiles_h_skip = runtimeArgs(3, k, 10);
    const out_num_nonzero_subblocks_w, const out_last_num_nonzero_subblocks_w, const out_last_subblock_w = runtimeArgs(3, k, 13);
    const padded_subblock_tiles_addr_skip, const padded_block_tiles_w_skip = runtimeArgs(2, k, 16);
    const last_num_blocks_w_dim = k.runtimeArg(20);
    const noc = k.kernelNoc();

    const in1_tile_bytes = k.op(.get_tile_size, .{cb_in1});
    const out_tile_bytes = k.op(.get_tile_size, .{cb_out});
    const in1_args = k.tensorAccessorArgs(3, 0);
    const sparsity_args = k.nextTensorAccessorArgs(in1_args);
    const s1 = k.tensorAccessor(in1_args, in1_addr, in1_tile_bytes);
    const s_sparsity = k.tensorAccessor(sparsity_args, sparsity_addr, c.sparsity_page_bytes);
    const s_out = k.tensorAccessor(k.nextTensorAccessorArgs(sparsity_args), out_addr, out_tile_bytes);

    k.op(.cb_reserve_back, .{ cb_sparsity, 1 });
    const sparsity_l1 = k.op(.get_write_ptr, .{cb_sparsity});
    const sparsity = k.l1Ptr(sparsity_l1, 16);
    const out_start = k.variable(out_start_tile);

    var rows, const b = llk.range(k, c.batch_a);
    {
        k.op(.noc_async_read_tile, .{ b, s_sparsity, sparsity_l1, noc });
        k.op(.noc_async_read_barrier, .{noc});
        var experts, const e = llk.range(k, c.batch_b);
        {
            // `b` is the experts' weights, the same for every row.
            const in1_batch = in1_start.add(e.mul(c.kt_nt));
            var if_valid = k.openIf(k.loadL1(sparsity, e, 16).ne(0));
            {
                var bh_loop, const bh = llk.range(k, c.num_blocks_h_dim);
                {
                    const out_h = out_start.get().add(bh.mul(c.out_next_h_dim_block_stride));
                    var bw_loop, const bw = llk.range(k, c.num_blocks_w_dim);
                    {
                        const in1_w = in1_batch.add(bw.mul(c.in1_next_w_dim_block_stride));
                        const out_w = out_h.add(bw.mul(c.out_next_w_dim_block_stride));
                        readIn1Blocks(k, c, cb_in1, s1, in1_w, bw, last_block_w);

                        const blocks_w = sel(k, bw.ge(last_num_blocks_w_dim.sub(1)), last_num_blocks_w_dim, c.num_blocks_w_dim);
                        const last_bw = bw.eq(blocks_w.sub(1));
                        const subblocks_w = sel(k, last_bw, out_last_num_nonzero_subblocks_w, out_num_nonzero_subblocks_w);
                        var sbh_loop, const sbh = llk.range(k, out_num_nonzero_subblocks_h);
                        {
                            var sbw_loop, const sbw = llk.range(k, subblocks_w);
                            {
                                const last_h = bh.eq(c.num_blocks_h_dim - 1).bitAnd(sbh.eq(out_num_nonzero_subblocks_h.sub(1)));
                                const last_w = last_bw.bitAnd(sbw.eq(subblocks_w.sub(1)));
                                const rows_n = sel(k, last_h, out_last_subblock_h, c.out_subblock_h);
                                const cols_n = sel(k, last_w, out_last_subblock_w, c.out_subblock_w);
                                const skip = sel(k, last_w, padded_subblock_tiles_addr_skip, 0);
                                const first = out_w.add(sbh.mul(c.out_next_subblock_stride_h)).add(sbw.mul(c.out_next_subblock_stride_w));
                                writeSubblock(k, c, cb_out, s_out, first, rows_n, cols_n, skip, bw.lt(blocks_w), out_tile_bytes);
                            }
                            sbw_loop.yield(.{});
                            var if_last_w = k.openIf(last_bw);
                            k.op(.cb_wait_front, .{ cb_out, padded_block_tiles_w_skip });
                            k.op(.cb_pop_front, .{ cb_out, padded_block_tiles_w_skip });
                            if_last_w.yieldThen(.{});
                        }
                        sbh_loop.yield(.{});
                        var if_last_h = k.openIf(bh.eq(c.num_blocks_h_dim - 1));
                        k.op(.cb_wait_front, .{ cb_out, padded_block_tiles_h_skip });
                        k.op(.cb_pop_front, .{ cb_out, padded_block_tiles_h_skip });
                        if_last_h.yieldThen(.{});
                    }
                    bw_loop.yield(.{});
                }
                bh_loop.yield(.{});
            }
            if_valid.yieldThen(.{});
            // The expanded output keeps a place for every expert.
            out_start.set(out_start.get().add(c.mt_nt));
        }
        experts.yield(.{});
    }
    rows.yield(.{});
    k.op(.noc_async_write_barrier, .{noc});
}

/// The `num_blocks_inner_dim` blocks of `in1` of an output column block.
fn readIn1Blocks(k: *Builder, c: Config, cb_in1: Cb, s1: Value, first: Value, bw: Value, last_block_w: Value) void {
    const noc = k.kernelNoc();
    var blocks, const block = llk.range(k, c.num_blocks_inner_dim);
    {
        k.op(.cb_reserve_back, .{ cb_in1, c.in1_block_num_tiles });
        const l1 = k.op(.get_write_ptr, .{cb_in1});
        const row_tile = first.add(block.mul(c.in1_next_block_stride));
        var h_loop, const h = llk.range(k, c.in1_block_h);
        {
            var w_loop, const w = llk.range(k, c.in1_block_w);
            {
                // The last block's columns past `last_block_w` are padding.
                var if_col = k.openIf(bw.lt(c.num_blocks_w_dim - 1).bitOr(w.lt(last_block_w)));
                const tile = row_tile.add(h.mul(c.in1_stride_h)).add(w.mul(c.in1_stride_w));
                const offset = h.mul(c.in1_block_w).add(w).mul(c.in1_aligned_tile_bytes);
                k.op(.noc_async_read_tile, .{ tile, s1, l1.add(offset), noc });
                if_col.yieldThen(.{});
            }
            w_loop.yield(.{});
        }
        h_loop.yield(.{});
        k.op(.noc_async_read_barrier, .{noc});
        k.op(.cb_push_back, .{ cb_in1, c.in1_block_num_tiles });
    }
    blocks.yield(.{});
}

/// Writes an output subblock of `rows_n` by `cols_n` tiles from `cb_out`,
/// its rows `skip` bytes of padding apart, if `write`.
fn writeSubblock(k: *Builder, c: Config, cb_out: Cb, s_out: Value, first: Value, rows_n: Value, cols_n: Value, skip: Value, write: Value, tile_bytes: Value) void {
    const noc = k.kernelNoc();
    k.op(.cb_wait_front, .{ cb_out, c.out_subblock_tile_count });
    const l1 = k.op(.get_read_ptr, .{cb_out});
    const row_bytes = cols_n.mul(tile_bytes).add(skip);
    var h_loop, const h = llk.range(k, rows_n);
    {
        var w_loop, const w = llk.range(k, cols_n);
        {
            var if_write = k.openIf(write);
            const tile = first.add(h.mul(c.out_stride_h)).add(w.mul(c.out_stride_w));
            k.op(.noc_async_write_tile, .{ tile, s_out, l1.add(h.mul(row_bytes)).add(w.mul(tile_bytes)), noc });
            if_write.yieldThen(.{});
        }
        w_loop.yield(.{});
    }
    h_loop.yield(.{});
    k.op(.noc_async_write_barrier, .{noc});
    k.op(.cb_pop_front, .{ cb_out, c.out_subblock_tile_count });
}

const matmul_h = "api/compute/matmul.h";

fn matmulBlockInit(k: *Builder, c: Config, in0: Cb, in1: Cb) void {
    k.call(matmul_h, "matmul_block_init", .{}, .{ in0, in1, @as(u32, @intFromBool(c.transpose_b)), c.out_subblock_w, c.out_subblock_h, c.in0_block_w });
}

/// Compute: compile-time args `cb.in0`, `cb.in1`, `cb.out`, `cb.intermed0`.
pub fn compute(k: *Builder, c: Config) void {
    const in0 = k.circularBufferOf(0, .bf16, 1);
    const in1 = k.circularBufferOf(1, .bf16, 1);
    const out = k.circularBufferOf(2, .bf16, 1);
    const partials = k.circularBufferOf(3, .bf16, 1);
    const out_subblock_num_tiles = c.out_subblock_h * c.out_subblock_w;
    const spill = c.num_blocks_inner_dim > 1;
    const several = c.batch_compute > 1 or c.num_blocks_h_dim > 1 or c.num_blocks_w_dim > 1;

    k.call("api/compute/compute_kernel_hw_startup.h", "compute_kernel_hw_startup", .{"SrcOrder::Reverse"}, .{ in0, in1, partials });
    matmulBlockInit(k, c, in0, in1);
    var batches, _ = llk.range(k, c.batch_compute);
    {
        var if_valid = k.openIf(llk.mailboxRead(k, .brisc));
        {
            var bhw, _ = llk.range(k, c.num_blocks_h_dim * c.num_blocks_w_dim);
            {
                if (several) k.op(.pack_reconfig_data_format, .{partials});
                var blocks, const block = llk.range(k, c.num_blocks_inner_dim);
                {
                    const last_out = block.eq(c.num_blocks_inner_dim - 1);
                    llk.wait(k, in0, c.in0_block_num_tiles);
                    llk.wait(k, in1, c.in1_block_num_tiles);
                    var sb0, const in0_subblock = llk.range(k, c.in0_num_subblocks);
                    {
                        var sb1, const in1_subblock = llk.range(k, c.in1_num_subblocks);
                        {
                            k.op(.tile_regs_acquire, .{});
                            // The partials of the previous blocks back in the registers.
                            if (spill) {
                                var if_reload = k.openIf(block.gt(0));
                                reloadPartials(k, c, in0, in1, partials, out_subblock_num_tiles);
                                if_reload.yieldThen(.{});
                            }
                            var inner, const i = llk.range(k, c.in0_block_w);
                            k.call(matmul_h, "matmul_block", .{}, .{
                                in0,
                                in1,
                                in0_subblock.mul(c.in0_subblock_num_tiles).add(i),
                                in1_subblock.mul(c.out_subblock_w).add(i.mul(c.in1_per_core_w)),
                                0,
                                @as(u32, @intFromBool(c.transpose_b)),
                                c.out_subblock_w,
                                c.out_subblock_h,
                                c.in0_block_w,
                            });
                            inner.yield(.{});
                            k.op(.tile_regs_commit, .{});
                            var if_last = k.openIfElse(last_out, .{});
                            {
                                packSubblock(k, out, out_subblock_num_tiles, c.fp32_dest_acc_en);
                            }
                            if_last.yieldThen(.{});
                            {
                                packSubblock(k, partials, out_subblock_num_tiles, false);
                            }
                            if_last.yieldElse(.{});
                        }
                        sb1.yield(.{});
                    }
                    sb0.yield(.{});
                    llk.pop(k, in0, c.in0_block_num_tiles);
                    llk.pop(k, in1, c.in1_block_num_tiles);
                }
                blocks.yield(.{});
                if (several) {
                    llk.reconfigDataFormatSrca2(k, partials, in1);
                    matmulBlockInit(k, c, in0, in1);
                }
            }
            bhw.yield(.{});
        }
        if_valid.yieldThen(.{});
    }
    batches.yield(.{});
}

/// `reload_from_cb_to_dst`: the spilled partials of a subblock back into
/// the destination registers.
fn reloadPartials(k: *Builder, c: Config, in0: Cb, in1: Cb, partials: Cb, n: u32) void {
    llk.reconfigDataFormatSrca2(k, in1, partials);
    llk.copyInit(k, partials);
    llk.wait(k, partials, n);
    k.call("api/compute/tile_move_copy.h", "copy_block", .{}, .{ partials, 0, 0, n });
    llk.pop(k, partials, n);
    llk.reconfigDataFormatSrca2(k, partials, in1);
    matmulBlockInit(k, c, in0, in1);
}

fn packSubblock(k: *Builder, o: Cb, n: u32, reconfig: bool) void {
    llk.reserve(k, o, n);
    k.op(.tile_regs_wait, .{});
    if (reconfig) k.op(.pack_reconfig_data_format, .{o});
    k.call("api/compute/pack.h", "pack_block", .{}, .{ 0, o, n });
    k.op(.tile_regs_release, .{});
    llk.push(k, o, n);
}
