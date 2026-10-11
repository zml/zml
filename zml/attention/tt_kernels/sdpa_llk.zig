//! The block operations of tt-metal's SDPA compute kernels
//! (`transformer/sdpa/device/kernels/compute/compute_common.hpp`, tt-metal
//! d04395ed) in the TTKernel DSL, and the work split of
//! `sdpa_decode/device/kernels/rt_args_common.hpp`. The SFPU routines only
//! the math thread declares are called on it, as `MATH((...))` does.

const std = @import("std");

const zml = @import("../../zml.zig");
const dsl = zml.kernel.ttkernel.dsl;
const llk = @import("llk.zig");
const Builder = dsl.Builder;
const Value = dsl.Value;
const Cb = dsl.CircularBuffer;
const verbatim = dsl.verbatim;

/// Declares the math-thread SFPU routines of the SDPA kernels.
const compute_common_h = "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp";
const binary_max_h = "api/compute/binary_max_min.h";
const reduce_h = "api/compute/reduce.h";
const exp_h = "api/compute/eltwise_unary/exp.h";
const matmul_h = "api/compute/matmul.h";
const pack_h = "api/compute/pack.h";
const reconfig_h = "api/compute/reconfig_data_format.h";
const compute_api_h = "api/compute/compute_kernel_api.h";

pub const VectorMode = enum {
    none,
    r,
    c,
    rc,

    pub fn cpp(self: VectorMode) []const u8 {
        return switch (self) {
            .none => "VectorMode::None",
            .r => "VectorMode::R",
            .c => "VectorMode::C",
            .rc => "VectorMode::RC",
        };
    }
};

/// The top 16 bits of an f32: its bf16.
fn bf16Bits(f32_bits: u32) u32 {
    return f32_bits >> 16;
}

fn range(k: *Builder, n: anytype) struct { dsl.ForScope(0), Value } {
    return llk.range(k, n);
}

/// `move_block<pop_in>`: `n` tiles of `in` appended to `out`.
pub fn moveBlock(k: *Builder, in: Cb, out: Cb, n: anytype, comptime pop_in: bool) void {
    llk.copyInit(k, in);
    llk.wait(k, in, n);
    llk.reserve(k, out, n);
    var loop, const i = range(k, n);
    {
        k.op(.tile_regs_acquire, .{});
        k.op(.copy_tile, .{ in, i, 0 });
        k.op(.tile_regs_commit, .{});
        k.op(.tile_regs_wait, .{});
        k.op(.pack_tile, .{ 0, out, 0 });
        k.op(.tile_regs_release, .{});
        llk.push(k, out, 1);
    }
    loop.yield(.{});
    if (pop_in) llk.pop(k, in, n);
}

/// `add_block_inplace<pop_in1>`: `in0 += in1` over `n` tiles.
pub fn addBlockInplace(k: *Builder, in0: Cb, in1: Cb, n: anytype, comptime pop_in1: bool) void {
    llk.reconfigDataFormat(k, in0, in1);
    k.op(.pack_reconfig_data_format, .{in0});
    llk.binaryInit(k, .add, in0, in1);
    llk.wait(k, in0, n);
    llk.wait(k, in1, n);
    var loop, const i = range(k, n);
    {
        k.op(.tile_regs_acquire, .{});
        k.op(.add_tiles, .{ in0, in1, i, i, 0 });
        k.op(.tile_regs_commit, .{});
        k.op(.tile_regs_wait, .{});
        k.op(.pack_tile, .{ 0, in0, 0 });
        k.op(.tile_regs_release, .{});
    }
    loop.yield(.{});
    llk.pop(k, in0, n);
    if (pop_in1) llk.pop(k, in1, n);
    llk.reserve(k, in0, n);
    llk.push(k, in0, n);
}

/// `mul_block_inplace`: `in0[i] *= in1[i]` over `n` tiles.
pub fn mulBlockInplace(k: *Builder, in0: Cb, in1: Cb, n: anytype) void {
    llk.binaryInit(k, .mul, in0, in1);
    llk.wait(k, in0, n);
    llk.wait(k, in1, n);
    var loop, const i = range(k, n);
    {
        k.call(compute_api_h, "invalidate_l1_cache", .{}, .{});
        k.op(.tile_regs_acquire, .{});
        k.op(.mul_tiles, .{ in0, in1, 0, i, 0 });
        k.op(.tile_regs_commit, .{});
        llk.pop(k, in0, 1);
        llk.reserve(k, in0, 1);
        k.op(.tile_regs_wait, .{});
        k.op(.pack_tile, .{ 0, in0, 0 });
        k.op(.tile_regs_release, .{});
        llk.push(k, in0, 1);
    }
    loop.yield(.{});
}


/// `mul_block_bcast_cols_inplace<rows, cols>`: each row of tiles of `in0`
/// times its column tile of `in1`, `dht_granularity` tiles at a time
/// (`DHT_GRANULARITY`).
pub fn mulBlockBcastColsInplace(k: *Builder, in0: Cb, in1: Cb, rows: u32, cols: u32, dht_granularity: u32) void {
    const dst_tiles = @min(cols, dht_granularity);
    const granularity = if (cols >= dht_granularity) cols / dht_granularity else 1;
    llk.reconfigDataFormat(k, in0, in1);
    llk.bcastInit(k, .mul, .cols, in0, in1);
    k.op(.pack_reconfig_data_format, .{in0});
    llk.wait(k, in0, rows * cols);
    llk.wait(k, in1, rows);
    var rows_loop, const i = range(k, rows);
    {
        var g_loop, _ = range(k, granularity);
        {
            k.op(.tile_regs_acquire, .{});
            var j_loop, const j = range(k, dst_tiles);
            llk.bcastTiles(k, .mul, .cols, in0, in1, j, i, j);
            j_loop.yield(.{});
            k.op(.tile_regs_commit, .{});
            llk.pop(k, in0, dst_tiles);
            llk.reserve(k, in0, dst_tiles);
            k.op(.tile_regs_wait, .{});
            var p_loop, const p = range(k, dst_tiles);
            k.op(.pack_tile, .{ p, in0, 0 });
            p_loop.yield(.{});
            llk.push(k, in0, dst_tiles);
            k.op(.tile_regs_release, .{});
        }
        g_loop.yield(.{});
    }
    rows_loop.yield(.{});
    llk.pop(k, in1, rows);
    k.call(reconfig_h, "reconfig_data_format_srcb", .{}, .{in0});
}

/// `mul_block_bcast_cols<rows, cols, true, false>`: `out` gets each tile of
/// `in0` times its row's column tile of `in1`, `in0` popped as it goes.
pub fn mulBlockBcastCols(k: *Builder, in0: Cb, in1: Cb, out: Cb, rows: u32, cols: u32) void {
    llk.reconfigDataFormat(k, in0, in1);
    k.op(.pack_reconfig_data_format, .{out});
    llk.bcastInit(k, .mul, .cols, in0, in1);
    llk.wait(k, in0, rows * cols);
    llk.wait(k, in1, rows);
    var rows_loop, const i = range(k, rows);
    {
        var cols_loop, _ = range(k, cols);
        {
            k.op(.tile_regs_acquire, .{});
            llk.bcastTiles(k, .mul, .cols, in0, in1, 0, i, 0);
            k.op(.tile_regs_commit, .{});
            llk.pop(k, in0, 1);
            llk.reserve(k, out, 1);
            k.op(.tile_regs_wait, .{});
            k.op(.pack_tile, .{ 0, out, 0 });
            k.op(.tile_regs_release, .{});
            llk.push(k, out, 1);
        }
        cols_loop.yield(.{});
    }
    rows_loop.yield(.{});
    llk.pop(k, in1, rows);
}

/// The packer's L1 accumulation: packed tiles added to what is in L1
/// (`llk_pack_reconfig_l1_acc`).
fn l1Accumulate(k: *Builder, on: bool) void {
    k.callOn(.pack, pack_h, "llk_pack_reconfig_l1_acc", .{}, .{@as(u32, @intFromBool(on))});
}

/// `mul_block_bcast_cols<rows, cols, false, accumulate>`: `out` gets (or,
/// accumulating, is added) each tile of `in0` times its row's column tile
/// of `in1`, `dht_granularity` tiles at a time (`DHT_GRANULARITY`). An
/// accumulated `out` holds exactly `rows * cols` tiles, already pushed.
pub fn mulBlockBcastColsBlocked(k: *Builder, in0: Cb, in1: Cb, out: Cb, rows: u32, cols: u32, dht_granularity: u32, accumulate: bool) void {
    const n = rows * cols;
    const dst_tiles = @min(cols, dht_granularity);
    const granularity = if (cols >= dht_granularity) cols / dht_granularity else 1;
    llk.reconfigDataFormat(k, in0, in1);
    k.op(.pack_reconfig_data_format, .{out});
    llk.bcastInit(k, .mul, .cols, in0, in1);
    llk.wait(k, in0, n);
    llk.wait(k, in1, rows);
    l1Accumulate(k, accumulate);
    if (!accumulate) llk.reserve(k, out, n);
    var rows_loop, const i = range(k, rows);
    {
        var g_loop, const u = range(k, granularity);
        {
            const first = i.mul(cols).add(u.mul(dst_tiles));
            k.op(.tile_regs_acquire, .{});
            var j_loop, const j = range(k, dst_tiles);
            llk.bcastTiles(k, .mul, .cols, in0, in1, first.add(j), i, j);
            j_loop.yield(.{});
            k.op(.tile_regs_commit, .{});
            k.op(.tile_regs_wait, .{});
            var p_loop, const p = range(k, dst_tiles);
            k.op(.pack_tile, .{ p, out, 0 });
            p_loop.yield(.{});
            k.op(.tile_regs_release, .{});
        }
        g_loop.yield(.{});
    }
    rows_loop.yield(.{});
    llk.pop(k, in1, rows);
    llk.pop(k, in0, n);
    if (accumulate) {
        l1Accumulate(k, false);
        llk.pop(k, out, n);
        llk.reserve(k, out, n);
    }
    llk.push(k, out, n);
}

/// `mul_tiles_bcast_cols_inplace`: `in0[i] *= in1[i]` broadcast across
/// columns, `n` tiles.
pub fn mulTilesBcastColsInplace(k: *Builder, in0: Cb, in1: Cb, n: u32) void {
    llk.reconfigDataFormat(k, in0, in1);
    llk.bcastInit(k, .mul, .cols, in0, in1);
    k.op(.pack_reconfig_data_format, .{in0});
    llk.wait(k, in0, n);
    llk.wait(k, in1, n);
    var loop, const i = range(k, n);
    {
        k.op(.tile_regs_acquire, .{});
        llk.bcastTiles(k, .mul, .cols, in0, in1, 0, i, 0);
        k.op(.tile_regs_commit, .{});
        llk.pop(k, in0, 1);
        llk.reserve(k, in0, 1);
        k.op(.tile_regs_wait, .{});
        k.op(.pack_tile, .{ 0, in0, 0 });
        k.op(.tile_regs_release, .{});
        llk.push(k, in0, 1);
    }
    loop.yield(.{});
}

/// `matmul_reduce<m>`: each of the `m` tiles of `out` summed within its
/// rows (times the column identity `in1`) in place, `stats_granularity`
/// tiles at a time (`STATS_GRANULARITY`). `out` holds exactly `m` tiles.
pub fn matmulReduce(k: *Builder, in1: Cb, out: Cb, m: u32, stats_granularity: u32) void {
    const h = stats_granularity;
    llk.reconfigDataFormat(k, in1, out);
    k.call(matmul_h, "matmul_block_init", .{}, .{ out, in1, 0, 1, h, 1 });
    k.op(.pack_reconfig_data_format, .{out});
    llk.wait(k, in1, 1);
    llk.wait(k, out, m);
    var loop, _ = range(k, m / h);
    {
        k.op(.tile_regs_acquire, .{});
        k.call(matmul_h, "matmul_block", .{}, .{ out, in1, 0, 0, 0, 0, 1, h, 1 });
        k.op(.tile_regs_commit, .{});
        llk.pop(k, out, h);
        k.op(.tile_regs_wait, .{});
        var p_loop, const p = range(k, h);
        k.op(.pack_tile, .{ p, out, 0 });
        p_loop.yield(.{});
        k.op(.tile_regs_release, .{});
        llk.push(k, out, h);
    }
    loop.yield(.{});
}

/// `stamp_tile_range_l1_acc<dst_batch>`: tile `src_tile` of `src` added
/// onto tiles `offset..offset + count` of `out`, `dst_batch` at a time.
fn stampTileRange(k: *Builder, src: Cb, src_tile: u32, out: Cb, offset: Value, count: Value, dst_batch: u32) void {
    var batches = k.openFor(0, count, dst_batch, .{});
    {
        const base = batches.iv.toI32();
        const batch = count.sub(base).min(dst_batch);
        k.op(.tile_regs_acquire, .{});
        var c_loop, const i = range(k, batch);
        k.op(.copy_tile, .{ src, src_tile, i });
        c_loop.yield(.{});
        k.op(.tile_regs_commit, .{});
        k.op(.tile_regs_wait, .{});
        var p_loop, const p = range(k, batch);
        k.op(.pack_tile, .{ p, out, offset.add(base).add(p), true });
        p_loop.yield(.{});
        k.op(.tile_regs_release, .{});
    }
    batches.yield(.{});
}

/// `apply_causal_mask_lightweight<dst_batch>`: the causal mask added onto
/// the `rows` x `cols` tiles of `out` (queries from tile `q_start`, keys
/// from tile `k_start`) from two tiles of `mask`, -inf (`neginf`) and the
/// diagonal (`diag`): -inf right of the diagonal tile, nothing left of it.
pub fn causalMaskLightweight(k: *Builder, mask: Cb, neginf: u32, diag: u32, out: Cb, q_start: Value, k_start: Value, rows: u32, cols: u32, dst_batch: u32) void {
    llk.reconfigDataFormatSrca(k, mask);
    k.op(.pack_reconfig_data_format, .{out});
    llk.copyInit(k, mask);
    l1Accumulate(k, true);
    var rows_loop, const row = range(k, rows);
    {
        const row_base = row.mul(cols);
        const q_pos = q_start.add(row);
        // Above the diagonal: all -inf.
        var above = k.openIfElse(q_pos.lt(k_start), .{});
        stampTileRange(k, mask, neginf, out, row_base, k.int(cols), dst_batch);
        above.yieldThen(.{});
        {
            const diag_col = q_pos.sub(k_start);
            var crosses = k.openIf(diag_col.lt(cols));
            {
                k.op(.tile_regs_acquire, .{});
                k.op(.copy_tile, .{ mask, diag, 0 });
                k.op(.tile_regs_commit, .{});
                k.op(.tile_regs_wait, .{});
                k.op(.pack_tile, .{ 0, out, row_base.add(diag_col), true });
                k.op(.tile_regs_release, .{});
                const right = diag_col.add(1);
                var has_right = k.openIf(right.lt(cols));
                stampTileRange(k, mask, neginf, out, row_base.add(right), k.int(cols).sub(right), dst_batch);
                has_right.yieldThen(.{});
            }
            crosses.yieldThen(.{});
        }
        above.yieldElse(.{});
    }
    rows_loop.yield(.{});
    l1Accumulate(k, false);
}

/// `recip_block_inplace`: `in = 1 / in` over its first column, `n` tiles.
pub fn recipBlockInplace(k: *Builder, in: Cb, n: anytype) void {
    llk.reconfigDataFormatSrca(k, in);
    llk.copyInit(k, in);
    k.op(.recip_tile_init, .{});
    k.op(.pack_reconfig_data_format, .{in});
    llk.wait(k, in, n);
    var loop, const i = range(k, n);
    {
        k.op(.tile_regs_acquire, .{});
        k.op(.copy_tile, .{ in, i, 0 });
        k.callOn(.math, compute_common_h, "recip_tile_first_column", .{}, .{0});
        k.op(.tile_regs_commit, .{});
        k.op(.tile_regs_wait, .{});
        k.op(.pack_tile, .{ 0, in, 0 });
        k.op(.tile_regs_release, .{});
    }
    loop.yield(.{});
    llk.pop(k, in, n);
    llk.reserve(k, in, n);
    llk.push(k, in, n);
}

/// `sub_exp_block<scale>`: `out = exp((in0 - in1) * scale)` over the first
/// column, `n` tiles.
pub fn subExpBlock(k: *Builder, in0: Cb, in1: Cb, out: Cb, n: anytype, scale_fp32: u32, approx: bool) void {
    llk.binaryInit(k, .sub, in0, in1);
    k.call(exp_h, "exp_tile_init", .{approx}, .{});
    llk.wait(k, in0, n);
    llk.wait(k, in1, n);
    llk.reserve(k, out, n);
    var loop, const i = range(k, n);
    {
        k.call(compute_api_h, "invalidate_l1_cache", .{}, .{});
        k.op(.tile_regs_acquire, .{});
        k.op(.sub_tiles, .{ in0, in1, i, i, 0 });
        k.callOn(.math, compute_common_h, "exp_tile_first_column", .{ approx, bf16Bits(scale_fp32) }, .{0});
        k.op(.tile_regs_commit, .{});
        k.op(.tile_regs_wait, .{});
        k.op(.pack_tile, .{ 0, out, 0 });
        k.op(.tile_regs_release, .{});
        llk.push(k, out, 1);
    }
    loop.yield(.{});
}

pub const Pool = enum { max, sum };

/// `reduce_c<pool, REDUCE_ROW, in0, scale, rows, vector_mode>(out, prev,
/// cols, do_eltwise_max)`: each row of `cols` tiles of `in0` reduced to a
/// column tile of `out`, the maximum with `prev`'s if `do_eltwise_max`.
pub fn reduceRows(k: *Builder, comptime pool: Pool, in0: Cb, scale: Cb, out: Cb, prev: Cb, rows: u32, cols: anytype, do_eltwise_max: Value, mode: VectorMode) void {
    const pool_cpp = switch (pool) {
        .max => "PoolType::MAX",
        .sum => "PoolType::SUM",
    };
    llk.wait(k, scale, 1);
    llk.wait(k, in0, k.lift(cols).mul(rows));
    llk.reserve(k, out, rows);
    k.op(.pack_reconfig_data_format, .{out});
    k.call(binary_max_h, "binary_max_tile_init", .{}, .{});
    var rows_loop, const i = range(k, rows);
    {
        llk.reconfigDataFormatSrca(k, in0);
        k.op(.tile_regs_acquire, .{});
        k.call(reduce_h, "reduce_init", .{ pool_cpp, "ReduceDim::REDUCE_ROW" }, .{ in0, scale, out });
        var cols_loop, const j = range(k, cols);
        k.call(reduce_h, "reduce_tile", .{ pool_cpp, "ReduceDim::REDUCE_ROW" }, .{ in0, scale, i.mul(k.lift(cols)).add(j), 0, 0 });
        cols_loop.yield(.{});
        k.call(reduce_h, "reduce_uninit", .{}, .{});
        var if_max = k.openIf(do_eltwise_max);
        {
            llk.reconfigDataFormatSrca(k, prev);
            llk.copyInit(k, prev);
            k.op(.copy_tile, .{ prev, i, 1 });
            k.call(binary_max_h, "binary_max_tile", .{}, .{ 0, 1, 0, verbatim(mode.cpp()) });
        }
        if_max.yieldThen(.{});
        k.op(.tile_regs_commit, .{});
        k.op(.tile_regs_wait, .{});
        k.op(.pack_tile, .{ 0, out, 0 });
        k.op(.tile_regs_release, .{});
    }
    rows_loop.yield(.{});
    llk.push(k, out, rows);
}

/// `sub_exp_block_bcast_cols_inplace<in0, rows, scale, true, do_reduce,
/// mode>`: `in0 = exp((in0 - in1 broadcast across columns) * scale)`, rows
/// of `cols` tiles, negative results packed as zero, and with a `reduce`
/// buffer its tiles summed per row into it (L1 accumulation; the sums
/// within the tiles are left to `matmulReduce`). `sub_exp_granularity`
/// tiles at a time (`SUB_EXP_GRANULARITY`), or whole rows.
pub fn subExpBlockBcastColsInplace(k: *Builder, in0: Cb, in1: Cb, reduce: ?Cb, rows: u32, cols: anytype, scale_fp32: u32, mode: VectorMode, sub_exp_granularity: ?u32) void {
    llk.reconfigDataFormat(k, in0, in1);
    llk.bcastInit(k, .sub, .cols, in0, in1);
    k.call(exp_h, "exp_tile_init", .{ true, scale_fp32, "InputClamping::None" }, .{});
    k.callOn(.pack, pack_h, "llk_pack_relu_config", .{}, .{verbatim("ReluConfig::zero()")});
    const cols_v = k.lift(cols);
    llk.wait(k, in0, cols_v.mul(rows));
    llk.wait(k, in1, rows);
    if (reduce) |r| llk.reserve(k, r, rows);
    const dst_tiles: Value, const granularity: Value = if (sub_exp_granularity) |g|
        .{ k.select(cols_v.lt(g), cols_v, g), k.select(cols_v.ge(g), cols_v.div(g), 1) }
    else
        .{ cols_v, k.int(1) };
    const iterations: u32 = if (mode == .rc) 32 else 8;
    const exp_mode: VectorMode = if (mode == .rc) .none else mode;
    var rows_loop, const i = range(k, rows);
    {
        var g_loop, const u = range(k, granularity);
        {
            k.op(.tile_regs_acquire, .{});
            var j_loop, const j = range(k, dst_tiles);
            llk.bcastTiles(k, .sub, .cols, in0, in1, j, i, j);
            k.call(exp_h, "exp_tile", .{ true, false, "InputClamping::None", iterations }, .{ j, verbatim(exp_mode.cpp()) });
            j_loop.yield(.{});
            k.op(.tile_regs_commit, .{});
            llk.pop(k, in0, dst_tiles);
            llk.reserve(k, in0, dst_tiles);
            k.op(.tile_regs_wait, .{});
            k.op(.pack_reconfig_data_format, .{in0});
            var p_loop, const p = range(k, dst_tiles);
            k.op(.pack_tile, .{ p, in0, 0 });
            p_loop.yield(.{});
            llk.push(k, in0, dst_tiles);
            if (reduce) |r| {
                // The row's tiles summed onto its tile of `r`: the first
                // packed as is, the others accumulated.
                k.op(.pack_reconfig_data_format, .{r});
                var if_later = k.openIf(u.gt(0));
                l1Accumulate(k, true);
                if_later.yieldThen(.{});
                var r_loop, const q = range(k, dst_tiles);
                {
                    k.op(.pack_tile, .{ q, r, i, true });
                    var if_first = k.openIf(u.eq(0).bitAnd(q.eq(0)));
                    l1Accumulate(k, true);
                    if_first.yieldThen(.{});
                }
                r_loop.yield(.{});
            }
            k.op(.tile_regs_release, .{});
            if (reduce != null) l1Accumulate(k, false);
        }
        g_loop.yield(.{});
    }
    rows_loop.yield(.{});
    if (reduce) |r| llk.push(k, r, rows);
    k.callOn(.pack, pack_h, "llk_pack_relu_config", .{}, .{verbatim("ReluConfig::none()")});
}

/// `correction_block<scale, mode>`: merges a worker's (max, sum) into the
/// running ones: the new max, the two rescaling factors, the new sum.
pub fn correctionBlock(
    k: *Builder,
    worker_max: Cb,
    worker_sum: Cb,
    cur_max: Cb,
    prev_max: Cb,
    cur_sum: Cb,
    prev_sum: Cb,
    exp_max_diff: Cb,
    exp_max_diff_2: Cb,
    n: u32,
    scale_fp32: u32,
    approx: bool,
    mode: VectorMode,
) void {
    llk.wait(k, worker_max, n);
    llk.wait(k, worker_sum, n);
    llk.wait(k, prev_max, n);
    llk.wait(k, prev_sum, n);
    llk.reserve(k, cur_max, n);
    llk.reserve(k, cur_sum, n);
    llk.reserve(k, exp_max_diff, n);
    llk.reserve(k, exp_max_diff_2, n);
    var loop, const i = range(k, n);
    {
        k.op(.tile_regs_acquire, .{});
        llk.copyInit(k, worker_max);
        k.call(exp_h, "exp_tile_init", .{approx}, .{});
        k.op(.copy_tile, .{ prev_max, i, 0 });
        k.op(.copy_tile, .{ worker_max, i, 1 });
        k.op(.copy_tile, .{ prev_sum, i, 3 });
        k.op(.copy_tile, .{ worker_sum, i, 4 });
        k.callOn(.math, compute_common_h, "fused_max_sub_exp_add_tile", .{mode.cpp()}, .{ 0, bf16Bits(scale_fp32) });
        k.op(.tile_regs_commit, .{});
        k.op(.tile_regs_wait, .{});
        k.op(.pack_tile, .{ 0, exp_max_diff, 0 });
        k.op(.pack_tile, .{ 1, exp_max_diff_2, 0 });
        k.op(.pack_tile, .{ 2, cur_max, 0 });
        k.op(.pack_tile, .{ 3, cur_sum, 0 });
        k.op(.tile_regs_release, .{});
        llk.push(k, cur_max, 1);
        llk.push(k, cur_sum, 1);
        llk.push(k, exp_max_diff, 1);
        llk.push(k, exp_max_diff_2, 1);
    }
    loop.yield(.{});
    llk.pop(k, prev_sum, n);
    llk.pop(k, worker_sum, n);
}

pub const MatmulBlocks = struct {
    /// Rows of tiles of `in0` and `out`.
    m: u32,
    /// Columns of tiles of `in1` and `out`.
    n: Value,
    /// Inner tiles.
    k: Value,
    in0_num_subblocks: u32,
    in1_num_subblocks: Value,
    in0_block_w: Value,
    subblock_h: u32,
    subblock_w: Value,
    transpose: bool,
};

/// `matmul_blocks`: `out[M, N] = in0[M, K] @ in1[K, N]` (`in1` tiles
/// transposed), subblock by subblock, the attention mask added to each
/// when `add_mask` (from `zero + mask`).
pub fn matmulBlocks(k: *Builder, in0: Cb, in1: Cb, out: Cb, b: MatmulBlocks, add_mask: ?Value, mask: Cb, zero: Cb) void {
    const tr: u32 = @intFromBool(b.transpose);
    k.call(matmul_h, "matmul_block_init", .{}, .{ in0, in1, tr, b.subblock_w, b.subblock_h, b.in0_block_w });
    const out_subblock_tiles = b.subblock_w.mul(b.subblock_h);
    const in0_subblock_tiles = b.in0_block_w.mul(b.subblock_h);
    llk.reconfigDataFormat(k, in1, in0);
    llk.wait(k, in1, b.k.mul(b.n));
    llk.reserve(k, out, b.n.mul(b.m));
    var sb0_loop, const sb0 = range(k, b.in0_num_subblocks);
    {
        llk.wait(k, in0, sb0.add(1).mul(in0_subblock_tiles));
        var sb1_loop, const sb1 = range(k, b.in1_num_subblocks);
        {
            k.op(.tile_regs_acquire, .{});
            var inner_loop, const inner = range(k, b.in0_block_w);
            k.call(matmul_h, "matmul_block", .{}, .{ in0, in1, sb0.mul(in0_subblock_tiles).add(inner), sb1.mul(b.subblock_w).add(inner.mul(b.n)), 0, tr, b.subblock_w, b.subblock_h, b.in0_block_w });
            inner_loop.yield(.{});
            if (add_mask) |cond| {
                var if_mask = k.openIf(cond);
                {
                    llk.wait(k, mask, out_subblock_tiles);
                    llk.wait(k, zero, 1);
                    llk.reconfigDataFormat(k, zero, mask);
                    k.call("api/compute/eltwise_binary.h", "add_init", .{}, .{ zero, mask, true });
                    var m_loop, const t = range(k, out_subblock_tiles);
                    k.op(.add_tiles, .{ zero, mask, 0, t, t });
                    m_loop.yield(.{});
                    llk.reconfigDataFormat(k, in1, in0);
                    k.call(matmul_h, "matmul_block_init", .{}, .{ in0, in1, tr, b.subblock_w, b.subblock_h, b.in0_block_w });
                }
                if_mask.yieldThen(.{});
            }
            k.op(.tile_regs_commit, .{});
            k.op(.tile_regs_wait, .{});
            const col = sb1.mul(b.subblock_w);
            var r_loop, const r = range(k, b.subblock_h);
            {
                var c_loop, const c = range(k, b.subblock_w);
                k.op(.pack_tile, .{ r.mul(b.subblock_w).add(c), out, r.mul(b.n).add(col).add(c), true });
                c_loop.yield(.{});
            }
            r_loop.yield(.{});
            k.op(.tile_regs_release, .{});
        }
        sb1_loop.yield(.{});
        llk.push(k, out, b.n.mul(b.subblock_h));
    }
    sb0_loop.yield(.{});
    llk.pop(k, in1, b.k.mul(b.n));
}

// ---------------------------------------------------------------- dataflow

/// Fills tile `tile` of the tiles at L1 `base` with the 32-bit word `word`.
pub fn fillTile(k: *Builder, base: Value, tile_bytes: u32, tile: anytype, word: u32) void {
    const ptr = k.l1Ptr(base.add(k.lift(tile).mul(tile_bytes)), 32);
    var loop, const i = range(k, tile_bytes / 4);
    k.storeL1(ptr, i, word, 32);
    loop.yield(.{});
}

/// `generate_bcast_col_scalar`: the bf16 of `packed` down the first column
/// of a tile of `cb` (the rest left as is), pushed.
pub fn bcastColScalar(k: *Builder, cb: Cb, packed_scalar: u32) void {
    k.op(.cb_reserve_back, .{ cb, 1 });
    const ptr = k.l1Ptr(k.op(.get_write_ptr, .{cb}), 16);
    var faces, const f = range(k, 2);
    {
        var rows, const r = range(k, 16);
        k.storeL1(ptr, f.mul(512).add(r.mul(16)), packed_scalar >> 16, 16);
        rows.yield(.{});
    }
    faces.yield(.{});
    k.op(.cb_push_back, .{ cb, 1 });
}

// ---------------------------------------------------------------- work split

/// `get_dynamic_Sk_chunk_t`: tiles per key chunk, the fixed `sk_chunk_t`,
/// or with none the power of two (up to `max`) covering the positions up
/// to `cur_pos`.
pub fn dynamicChunkTiles(k: *Builder, cur_pos: Value, sk_chunk_t: u32, max: u32) Value {
    if (sk_chunk_t != 0) return k.int(sk_chunk_t);
    // nearest_pow_of_2_up_to_8(cur_pos / 32 + 1): x - 1 with its two next
    // bits set, plus one.
    var x = cur_pos.div(32);
    x = x.bitOr(x.shr(1));
    x = x.bitOr(x.shr(2));
    const p = x.add(1);
    return k.select(p.gt(max), max, p);
}

pub const Workload = struct {
    num_chunks: Value,
    start: Value,
    end: Value,
};

/// `get_workload_for_core` without a sliding window: the key chunks of
/// positions `0..=cur_pos` the core `core_num` of `cores` reduces, the last
/// ones to the first cores.
pub fn workloadForCore(k: *Builder, cur_pos: Value, core_num: Value, cores: u32, chunk_size: Value) Workload {
    const valid = cur_pos.add(1).add(chunk_size).sub(1).div(chunk_size).mul(chunk_size);
    const num_chunks = valid.div(chunk_size);
    // More cores than chunks: one chunk each for the first ones.
    const one_each = num_chunks.lt(cores);
    const has_one = core_num.lt(num_chunks);
    const single_start = k.select(has_one, num_chunks.sub(core_num).sub(1), 0);
    const single_end = k.select(has_one, num_chunks.sub(core_num), 0);
    // Else even shares, the remainder to the last cores.
    const per_core = num_chunks.div(cores);
    const residuals = num_chunks.rem(cores);
    const reversed = k.int(cores - 1).sub(core_num);
    const share_start = reversed.mul(per_core).add(reversed.min(residuals));
    const share_end = share_start.add(per_core).add(k.select(reversed.lt(residuals), 1, 0));
    return .{
        .num_chunks = num_chunks,
        .start = k.select(one_each, single_start, share_start),
        .end = k.select(one_each, single_end, share_end),
    };
}
