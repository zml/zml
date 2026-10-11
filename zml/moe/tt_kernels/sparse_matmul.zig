//! tt-metal's sparse matmul, ported: the host program factory of
//! `ttnn::sparse_matmul` (`matmul/device/sparse/factory/
//! sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`, tt-metal
//! d04395ed) with its device kernels (`sparse_matmul_kernels.zig`).
//!
//! The matmul runs on a rectangle of cores, each computing a column block
//! of the output; the first core reads `a` and multicasts it to the others,
//! and tells them, a sparsity entry at a time, whether to compute it. The
//! output is expanded: the skipped matmuls leave zeros. Supported: the
//! sparse `b` and sparse `a` modes of the MoE layers, `b` optionally
//! transposed (the experts' `[out, in]` weights), batches counted at
//! runtime (no `nnz`).

const std = @import("std");

const Compiler = @import("../../Compiler.zig");
const zml = @import("../../zml.zig");
const dsl = zml.kernel.ttkernel.dsl;
const program = dsl.program;
const kernels = @import("sparse_matmul_kernels.zig");
const llk = @import("../../attention/tt_kernels/llk.zig");
const Tensor = zml.Tensor;

const cb = kernels.cb;
const C = program.Arg;

/// Bytes a DRAM page is aligned to (Blackhole).
const dram_alignment = 64;
/// `MCAST_INPUT_BUFFERING_DEPTH` of `matmul_utilities.hpp`.
const mcast_input_buffering_depth = 2;

fn val(x: u32) C {
    return .{ .value = x };
}

/// `MatmulMultiCoreReuseMultiCast1DProgramConfig` with `mcast_in0`.
pub const ProgramConfig = struct {
    /// `(cols, rows)` of the cores, from (0, 0).
    grid: [2]u32,
    in0_block_w: u32,
    out_subblock_h: u32,
    out_subblock_w: u32,
    out_block_h: u32,
    out_block_w: u32,
    per_core_m: u32,
    per_core_n: u32,

    /// gpt-oss's `_build_sparse_matmul_config(m, n)`: the most cores
    /// splitting the `n` columns evenly, within `max_grid`.
    pub fn forColumns(m: u32, n: u32, max_grid: [2]u32) ProgramConfig {
        const n_tiles = std.math.divCeil(u32, n, 32) catch unreachable;
        var best: u32 = 1;
        var best_xy: [2]u32 = .{ 1, 1 };
        var cores: u32 = 1;
        while (cores <= @min(max_grid[0] * max_grid[1], n_tiles)) : (cores += 1) {
            if (n_tiles % cores != 0) continue;
            var cy: u32 = 1;
            while (cy <= max_grid[1]) : (cy += 1) {
                if (cores % cy == 0) {
                    const cx = cores / cy;
                    if (cx <= max_grid[0] and cores > best) {
                        best = cores;
                        best_xy = .{ cx, cy };
                    }
                    break;
                }
            }
        }
        const per_core_n = n_tiles / best;
        return .{
            .grid = best_xy,
            .in0_block_w = 1,
            .out_subblock_h = 1,
            .out_subblock_w = 1,
            .out_block_h = 1,
            .out_block_w = per_core_n,
            .per_core_m = @max(32, m) / 32,
            .per_core_n = per_core_n,
        };
    }
};

pub const Mode = enum {
    /// `a` `[A, M, K]`, sparsity `[A, E]`: `out[r, e] = a[r] @ b[e]`.
    sparse_b,
    /// `a` `[A, E, M, K]`, sparsity `[A, E]`: `out[r, e] = a[r, e] @ b[e]`.
    sparse_a,
};

pub const Dims = struct {
    mode: Mode,
    /// Rows of the sparsity, `A`.
    rows: u32,
    experts: u32,
    m: u32,
    k: u32,
    n: u32,
    /// `b` is `[E, N, K]` rather than `[E, K, N]`.
    transpose_b: bool,
    a_dtype: program.DataType = .bf16,
    b_dtype: program.DataType = .bf16,
    out_dtype: program.DataType = .bf16,
};

/// The program (`SparseMatmulMultiCoreReuseMcast1DProgramFactory::create`).
/// io-tensors: `a`, `b`, the sparsity (bf16, row major, a row per `A`), the
/// zeroed output `[A, E, M, N]`, updated in place.
pub fn sparseMatmulProgram(allocator: std.mem.Allocator, d: Dims, cfg: ProgramConfig, compute_config: program.ComputeConfig) ![]u8 {
    const mt = d.m / 32;
    const kt = d.k / 32;
    const nt = d.n / 32;
    const batch_b = d.experts;
    const batch_a = d.rows;
    if (kt % cfg.in0_block_w != 0) return error.InvalidConfig;

    const num_blocks_y = (mt - 1) / cfg.per_core_m + 1;
    const num_blocks_x = (nt - 1) / cfg.per_core_n + 1;
    const num_cores = num_blocks_y * num_blocks_x;
    if (num_cores != cfg.grid[0] * cfg.grid[1]) return error.NonRectangularGrid;
    const num_blocks = kt / cfg.in0_block_w;

    const in0_tile = d.a_dtype.tileSize();
    const in1_tile = d.b_dtype.tileSize();
    const out_tile = d.out_dtype.tileSize();
    const in0_aligned = std.mem.alignForward(u32, in0_tile, dram_alignment);
    const in1_aligned = std.mem.alignForward(u32, in1_tile, dram_alignment);
    const sparsity_page = std.mem.alignForward(u32, batch_b * 2, dram_alignment);

    const in0_block_h = cfg.out_block_h;
    const in1_block_w = cfg.out_block_w;
    const in0_num_blocks_y = cfg.per_core_m / cfg.out_block_h;
    const in1_num_blocks_x = cfg.per_core_n / cfg.out_block_w;
    const buffering: u32 = if (batch_a * batch_b * num_blocks > 1) mcast_input_buffering_depth else 1;
    const in0_cb_tiles = in0_block_h * cfg.in0_block_w * buffering;
    const in1_cb_tiles = cfg.out_block_w * cfg.in0_block_w * buffering;
    const out_block_tiles = cfg.out_block_h * cfg.out_block_w;
    const in0_num_subblocks = cfg.out_block_h / cfg.out_subblock_h;
    const in0_block_num_tiles = cfg.out_subblock_h * cfg.in0_block_w * in0_num_subblocks;
    const in1_num_subblocks = cfg.out_block_w / cfg.out_subblock_w;

    const in1_stride_w: u32 = if (d.transpose_b) kt else 1;
    const in1_stride_h: u32 = if (d.transpose_b) 1 else nt;
    const kc: kernels.Config = .{
        .in0_stride_w = 1,
        .in0_stride_h = kt,
        .in0_next_inner_dim_block_stride = cfg.in0_block_w,
        .in0_next_h_dim_block_stride = in0_block_h * kt,
        .in0_block_w = cfg.in0_block_w,
        .in0_block_h = in0_block_h,
        .in0_block_num_tiles = in0_block_num_tiles,
        .in0_aligned_tile_bytes = in0_aligned,
        .num_blocks_inner_dim = num_blocks,
        .num_blocks_w_dim = in1_num_blocks_x,
        .num_blocks_h_dim = in0_num_blocks_y,
        .in0_mcast_num_dests = num_cores - 1,
        .in0_mcast_num_cores = num_cores - 1,
        .mt_kt = mt * kt,
        .batch_a = batch_a,
        .batch_b = batch_b,
        .sparsity_page_bytes = sparsity_page,
        .bcast_a = d.mode == .sparse_b,
        .in1_stride_w = in1_stride_w,
        .in1_stride_h = in1_stride_h,
        .in1_next_block_stride = cfg.in0_block_w * in1_stride_h,
        .in1_next_w_dim_block_stride = in1_block_w * in1_stride_w,
        .in1_block_w = in1_block_w,
        .in1_block_h = cfg.in0_block_w,
        .in1_block_num_tiles = in1_block_w * cfg.in0_block_w,
        .in1_aligned_tile_bytes = in1_aligned,
        .kt_nt = kt * nt,
        .out_stride_w = 1,
        .out_stride_h = nt,
        .out_next_subblock_stride_w = cfg.out_subblock_w,
        .out_next_subblock_stride_h = cfg.out_subblock_h * nt,
        .out_next_w_dim_block_stride = cfg.out_block_w,
        .out_next_h_dim_block_stride = cfg.out_block_h * nt,
        .out_subblock_w = cfg.out_subblock_w,
        .out_subblock_h = cfg.out_subblock_h,
        .out_subblock_tile_count = cfg.out_subblock_w * cfg.out_subblock_h,
        .mt_nt = mt * nt,
        .in0_num_subblocks = in0_num_subblocks,
        .in0_subblock_num_tiles = cfg.out_subblock_h * cfg.in0_block_w,
        .in1_num_subblocks = in1_num_subblocks,
        .in1_per_core_w = cfg.out_subblock_w * in1_num_subblocks,
        .batch_compute = batch_a * batch_b,
        .transpose_b = d.transpose_b,
        .fp32_dest_acc_en = compute_config.fp32_dest_acc_en,
    };

    const all: program.CoreRange = .grid(cfg.grid[0], cfg.grid[1]);
    const sender: program.CoreRange = .single(.{ .x = 0, .y = 0 });
    // The others, row-major from the core after the sender.
    var receiver_ranges: std.ArrayList(program.CoreRange) = .empty;
    if (cfg.grid[0] > 1) try receiver_ranges.append(allocator, .{ .start = .{ .x = 1, .y = 0 }, .end = .{ .x = cfg.grid[0] - 1, .y = 0 } });
    if (cfg.grid[1] > 1) try receiver_ranges.append(allocator, .{ .start = .{ .x = 0, .y = 1 }, .end = .{ .x = cfg.grid[0] - 1, .y = cfg.grid[1] - 1 } });

    // Partials share the output's buffer when their formats agree.
    const interm_dtype: program.DataType = if (compute_config.fp32_dest_acc_en) .f32 else d.out_dtype;
    var cbs: std.ArrayList(program.CircularBuffer) = .empty;
    try cbs.append(allocator, .{ .index = cb.in0, .dtype = d.a_dtype, .page_size = in0_aligned, .num_pages = in0_cb_tiles, .cores = all });
    try cbs.append(allocator, .{ .index = cb.in1, .dtype = d.b_dtype, .page_size = in1_aligned, .num_pages = in1_cb_tiles, .cores = all });
    try cbs.append(allocator, .{ .index = cb.sparsity0, .dtype = .bf16, .page_size = sparsity_page, .num_pages = 1, .cores = all });
    try cbs.append(allocator, .{ .index = cb.sparsity1, .dtype = .bf16, .page_size = sparsity_page, .num_pages = 1, .cores = all });
    if (interm_dtype == d.out_dtype) {
        try cbs.append(allocator, .{ .index = cb.out, .dtype = d.out_dtype, .page_size = out_tile, .num_pages = out_block_tiles, .cores = all, .aliases = try allocator.dupe(program.CircularBuffer.Alias, &.{.{ .index = cb.intermed0, .dtype = interm_dtype, .page_size = out_tile }}) });
    } else {
        try cbs.append(allocator, .{ .index = cb.out, .dtype = d.out_dtype, .page_size = out_tile, .num_pages = out_block_tiles, .cores = all });
        try cbs.append(allocator, .{ .index = cb.intermed0, .dtype = interm_dtype, .page_size = interm_dtype.tileSize(), .num_pages = out_block_tiles, .cores = all });
    }

    // Last column block of the last core, when the columns don't split
    // evenly.
    const last_per_core_n = if (nt % cfg.per_core_n == 0) cfg.per_core_n else nt % cfg.per_core_n;
    const last_out_block_w = if (last_per_core_n % cfg.out_block_w == 0) cfg.out_block_w else last_per_core_n % cfg.out_block_w;
    const last_out_num_blocks_w = (last_per_core_n - 1) / cfg.out_block_w + 1;
    const last_block_num_nonzero_subblocks_w = (last_out_block_w - 1) / cfg.out_subblock_w + 1;
    const last_subblock_of_last_block_w = if (last_out_block_w % cfg.out_subblock_w == 0) cfg.out_subblock_w else last_out_block_w % cfg.out_subblock_w;
    const last_block_padded_subblock_tiles_addr_skip = out_tile * (cfg.out_subblock_w - last_subblock_of_last_block_w);
    const last_block_padded_block_tiles_w_skip = (cfg.out_subblock_w * cfg.out_subblock_h) * (cfg.out_block_w / cfg.out_subblock_w - last_block_num_nonzero_subblocks_w);

    const top_left: program.CoreCoord = .{ .x = 0, .y = 0 };
    const bottom_right: program.CoreCoord = .{ .x = cfg.grid[0] - 1, .y = cfg.grid[1] - 1 };
    var sender_rt: std.ArrayList(program.CoreArgs) = .empty;
    var receiver_rt: std.ArrayList(program.CoreArgs) = .empty;
    var in1_rt: std.ArrayList(program.CoreArgs) = .empty;
    for (0..num_cores) |ii| {
        const i: u32 = @intCast(ii);
        const core: program.CoreCoord = .{ .x = i % cfg.grid[0], .y = i / cfg.grid[0] };
        const idx_x = i % num_blocks_x;
        const idx_y = i / num_blocks_x;
        if (i == 0) {
            try sender_rt.append(allocator, .{ .core = core, .args = try allocator.dupe(C, &.{
                .{ .tensor_address = 0 },                             val(kt * cfg.per_core_m * idx_y),
                // The writer's NoC, 1, walks the rectangle from the bottom right.
                .{ .core_noc = .{ .core = bottom_right, .dim = 0 } }, .{ .core_noc = .{ .core = bottom_right, .dim = 1 } },
                .{ .core_noc = .{ .core = top_left, .dim = 0 } },     .{ .core_noc = .{ .core = top_left, .dim = 1 } },
                val(cfg.out_block_h),                                 .{ .tensor_address = 2 },
            }) });
        } else {
            try receiver_rt.append(allocator, .{ .core = core, .args = try allocator.dupe(C, &.{
                .{ .core_noc = .{ .core = top_left, .dim = 0 } }, .{ .core_noc = .{ .core = top_left, .dim = 1 } },
            }) });
        }
        const last = idx_x == num_blocks_x - 1;
        try in1_rt.append(allocator, .{ .core = core, .args = try allocator.dupe(C, &.{
            .{ .tensor_address = 1 },
            // This core's first column of `b`, a column `in1_stride_w` tiles
            // on.
            val(cfg.per_core_n * idx_x * in1_stride_w),
            val(0),
            val(0),
            val(0),
            val(0),
            .{ .tensor_address = 2 },
            .{ .tensor_address = 3 },
            val(idx_x * cfg.per_core_n + idx_y * cfg.per_core_m * nt),
            val(if (last) last_out_block_w else cfg.out_block_w),
            val(cfg.out_block_h / cfg.out_subblock_h),
            val(cfg.out_subblock_h),
            val(0),
            val(cfg.out_block_w / cfg.out_subblock_w),
            val(if (last) last_block_num_nonzero_subblocks_w else cfg.out_block_w / cfg.out_subblock_w),
            val(if (last) last_subblock_of_last_block_w else cfg.out_subblock_w),
            val(if (last) last_block_padded_subblock_tiles_addr_skip else 0),
            val(if (last) last_block_padded_block_tiles_w_skip else 0),
            val(0),
            val(0),
            val(if (last) last_out_num_blocks_w else in1_num_blocks_x),
            val(0),
            val(0),
            val(0),
            val(0),
            val(0),
        }) });
    }

    const ctx = try dsl.newContext();
    defer ctx.deinit();
    var ks: std.ArrayList(program.Kernel) = .empty;
    // in0 sender and receivers on RISC-V 0: the compute threads read the
    // batch validity from its mailbox.
    try ks.append(allocator, .{
        .thread = .writer,
        .source = try llk.source(allocator, ctx, "sparse_matmul_in0_sender", .writer, kernels.in0Sender, kc),
        .cores = sender,
        .compile_time_args = try allocator.dupe(C, &.{ .{ .cb = cb.in0 }, .{ .cb = cb.sparsity0 }, .{ .semaphore = 0 }, .{ .semaphore = 1 }, .{ .tensor_accessor_args = 0 }, .{ .tensor_accessor_args = 2 } }),
        .runtime_args = sender_rt.items,
    });
    if (receiver_ranges.items.len > 0) try ks.append(allocator, .{
        .thread = .writer,
        .source = try llk.source(allocator, ctx, "sparse_matmul_in0_receiver", .writer, kernels.in0Receiver, kc),
        .cores = receiver_ranges.items[0],
        .more_cores = receiver_ranges.items[1..],
        .compile_time_args = try allocator.dupe(C, &.{ .{ .cb = cb.in0 }, .{ .semaphore = 0 }, .{ .semaphore = 1 } }),
        .runtime_args = receiver_rt.items,
    });
    try ks.append(allocator, .{
        .thread = .reader,
        .source = try llk.source(allocator, ctx, "sparse_matmul_in1_reader_writer", .reader, kernels.in1ReaderWriter, kc),
        .cores = all,
        .compile_time_args = try allocator.dupe(C, &.{ .{ .cb = cb.in1 }, .{ .cb = cb.out }, .{ .cb = cb.sparsity1 }, .{ .tensor_accessor_args = 1 }, .{ .tensor_accessor_args = 2 }, .{ .tensor_accessor_args = 3 } }),
        .runtime_args = in1_rt.items,
    });
    try ks.append(allocator, .{
        .thread = .{ .compute = compute_config },
        .source = try llk.source(allocator, ctx, "sparse_matmul_compute", .compute, kernels.compute, kc),
        .cores = all,
        .compile_time_args = try allocator.dupe(C, &.{ .{ .cb = cb.in0 }, .{ .cb = cb.in1 }, .{ .cb = cb.out }, .{ .cb = cb.intermed0 } }),
    });

    const p: program.Program = .{
        .kernels = ks.items,
        .circular_buffers = cbs.items,
        .semaphores = &.{ .{ .cores = all, .initial_value = llk.invalid }, .{ .cores = all, .initial_value = llk.invalid } },
        .operands = &.{.{ .index = 2, .layout = .row_major }},
    };
    return p.artifact(allocator);
}

pub const Options = struct {
    /// `(cols, rows)` of the cores available, from (0, 0).
    grid: [2]u32 = .{ 8, 8 },
    compute: program.ComputeConfig = .{ .math_fidelity = .hifi4, .fp32_dest_acc_en = false, .math_approx_mode = false },
};

fn run(a: Tensor, b: Tensor, sparsity: Tensor, out_shape: zml.Shape, d: Dims, opts: Options) Tensor {
    const allocator = Compiler.current().arena.allocator();
    const cfg = ProgramConfig.forColumns(d.m, d.n, opts.grid);
    const art = sparseMatmulProgram(allocator, d, cfg, opts.compute) catch |err|
        std.debug.panic("sparseMatmul: {}", .{err});
    const zeros = Tensor.constant(out_shape.dtype().zero()).broad(out_shape);
    return zml.ops.ttKernel(&.{ a, b, sparsity.convert(.bf16), zeros }, &.{}, .{
        .kernel_id = "zml.tt.sparse_matmul",
        .artifact = art,
        .inouts = &.{3},
    })[0];
}

fn dtypeOf(t: Tensor) program.DataType {
    return zml.kernel.ttkernel.from(t.dtype());
}

/// `out[r, e] = a[r] @ b[e]^T` where `sparsity[r, e]` is nonzero, zeros
/// elsewhere: `a` `[rows, m, k]`, `b` `[experts, n, k]`, `sparsity`
/// `[rows, experts]`, `out` `[rows, experts, m, n]`.
pub fn sparseB(a: Tensor, b: Tensor, sparsity: Tensor, opts: Options) Tensor {
    const rows = a.dim(0);
    const e = b.dim(0);
    const d: Dims = .{
        .mode = .sparse_b,
        .rows = @intCast(rows),
        .experts = @intCast(e),
        .m = @intCast(a.dim(1)),
        .k = @intCast(a.dim(2)),
        .n = @intCast(b.dim(1)),
        .transpose_b = true,
        .a_dtype = dtypeOf(a),
        .b_dtype = dtypeOf(b),
        .out_dtype = dtypeOf(a),
    };
    const out: zml.Shape = .init(.{ rows, e, a.dim(1), b.dim(1) }, a.dtype());
    return run(a.reshape(.{ rows, 1, a.dim(1), a.dim(2) }), b.reshape(.{ 1, e, b.dim(1), b.dim(2) }), sparsity.reshape(.{ rows, 1, 1, e }), out, d, opts);
}

/// `out[r, e] = a[r, e] @ b[e]^T` where `sparsity[r, e]` is nonzero, zeros
/// elsewhere: `a` `[rows, experts, m, k]`, `b` `[experts, n, k]`,
/// `sparsity` `[rows, experts]`, `out` `[rows, experts, m, n]`.
pub fn sparseA(a: Tensor, b: Tensor, sparsity: Tensor, opts: Options) Tensor {
    const rows = a.dim(0);
    const e = b.dim(0);
    const d: Dims = .{
        .mode = .sparse_a,
        .rows = @intCast(rows),
        .experts = @intCast(e),
        .m = @intCast(a.dim(2)),
        .k = @intCast(a.dim(3)),
        .n = @intCast(b.dim(1)),
        .transpose_b = true,
        .a_dtype = dtypeOf(a),
        .b_dtype = dtypeOf(b),
        .out_dtype = dtypeOf(a),
    };
    const out: zml.Shape = .init(.{ rows, e, a.dim(2), b.dim(1) }, a.dtype());
    return run(a, b.reshape(.{ 1, e, b.dim(1), b.dim(2) }), sparsity.reshape(.{ 1, 1, rows, e }), out, d, opts);
}
