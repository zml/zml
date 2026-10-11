//! tt-metal's flash-attention prefill, ported: the host program factory of
//! `ttnn::transformer::scaled_dot_product_attention` and its chunked (paged)
//! variant (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/
//! sdpa_program_factory.cpp`, tt-metal d04395ed), its device kernels in
//! the TTKernel DSL (`sdpa_kernels.zig`).
//!
//! Supported: causal attention, Q/K/V tiled in DRAM, either over K/V tensors
//! or over a paged KV cache from a chunk start read at runtime (tt-metal's
//! "flexible" chunked prefill); fp32 DST, tt-metal's standard compute path
//! (its bf16-DST streaming kernel rewinds circular buffer write pointers,
//! which the DSL does not express). Not ported: masks, sinks, MLA, sliding
//! and block-diagonal windows, ring and joint attention.

const std = @import("std");

const Compiler = @import("../../Compiler.zig");
const zml = @import("../../zml.zig");
const dsl = zml.kernel.ttkernel.dsl;
const program = dsl.program;
const kernels = @import("sdpa_kernels.zig");
const llk = @import("llk.zig");
const Tensor = zml.Tensor;

pub const Options = struct {
    scale: f32,
    /// Queries and keys per chunk, multiples of 32.
    q_chunk_size: u32 = 32,
    k_chunk_size: u32 = 32,
    /// `(cols, rows)` of the cores the program uses, from (0, 0).
    grid: [2]u32 = .{ 8, 8 },
};

/// Causal attention of `q` (`[B, heads, Sq, head_dim]`) over `k`, `v`
/// (`[B, kv_heads, Sk, head_dim]`).
pub fn sdpa(q: Tensor, k: Tensor, v: Tensor, opts: Options) Tensor {
    const allocator = Compiler.current().arena.allocator();
    const art = prefillProgram(allocator, q.shape(), k.shape(), v.shape(), null, opts) catch |err|
        std.debug.panic("sdpa: {}", .{err});
    return zml.ops.ttKernel(&.{ q, k, v }, &.{q.shape()}, .{ .kernel_id = "zml.tt.sdpa", .artifact = art })[0];
}

/// Causal attention of the chunk `q` (`[B, heads, Sq, head_dim]`), at
/// sequence position `chunk_start_idx[0]` (`[1]` i32, a multiple of the
/// chunk sizes), over the paged caches `k_cache`, `v_cache` (`[blocks,
/// kv_heads, block_size, head_dim]`) holding the sequence so far through
/// `page_table` (`[B, max_blocks]` i32, `max_blocks` a multiple of 8). The
/// chunk's own keys must already be in the cache.
pub fn chunkedSdpa(q: Tensor, k_cache: Tensor, v_cache: Tensor, page_table: Tensor, chunk_start_idx: Tensor, opts: Options) Tensor {
    const allocator = Compiler.current().arena.allocator();
    const art = prefillProgram(allocator, q.shape(), k_cache.shape(), v_cache.shape(), .{ .page_table = page_table.shape() }, opts) catch |err|
        std.debug.panic("chunkedSdpa: {}", .{err});
    return zml.ops.ttKernel(&.{ q, k_cache, v_cache, page_table, chunk_start_idx }, &.{q.shape()}, .{
        .kernel_id = "zml.tt.chunked_sdpa",
        .artifact = art,
    })[0];
}

/// `determine_largest_subblock_size` of `sdpa_subblock_utils.hpp`.
fn largestSubblock(block_h: u32, block_w: u32, dst_size: u32, max_h: u32) [2]u32 {
    const subblocks = [_][2]u32{
        .{ 2, 4 }, .{ 4, 2 }, .{ 1, 8 }, .{ 8, 1 }, .{ 1, 7 }, .{ 7, 1 }, .{ 2, 3 }, .{ 3, 2 }, .{ 1, 6 }, .{ 6, 1 },
        .{ 1, 5 }, .{ 5, 1 }, .{ 2, 2 }, .{ 1, 4 }, .{ 4, 1 }, .{ 1, 3 }, .{ 3, 1 }, .{ 1, 2 }, .{ 2, 1 }, .{ 1, 1 },
    };
    for (subblocks) |s| {
        if (s[0] * s[1] > dst_size or s[0] > max_h) continue;
        if (block_h % s[0] != 0 or block_w % s[1] != 0) continue;
        return s;
    }
    return .{ 1, 1 };
}

/// `find_valid_granularity` of `sdpa_subblock_utils.hpp`.
fn validGranularity(tile_count: u32, max_granularity: u32) u32 {
    var g = @min(tile_count, max_granularity);
    while (g > 1 and tile_count % g != 0) g -= 1;
    return g;
}

const Paged = struct { page_table: zml.Shape };

/// The program of `sdpa` (io-tensors `q`, `k`, `v`, output) or, when
/// `paged`, of `chunkedSdpa` (io-tensors `q`, `k_cache`, `v_cache`,
/// `page_table`, `chunk_start_idx`, output).
pub fn prefillProgram(allocator: std.mem.Allocator, q: zml.Shape, k: zml.Shape, v: zml.Shape, paged: ?Paged, opts: Options) ![]u8 {
    const io_q = 0;
    const io_k = 1;
    const io_v = 2;
    const io_page_table = 3;
    const io_chunk_start = 4;
    const io_out: u32 = if (paged != null) 5 else 3;

    if (q.dtype() != .bf16 or k.dtype() != .bf16 or v.dtype() != .bf16) return error.UnsupportedDataType;
    const is_causal = true;
    const is_chunked = paged != null;

    const b: u32 = @intCast(q.dim(0));
    const nqh: u32 = @intCast(q.dim(1));
    const sq: u32 = @intCast(q.dim(2));
    const dh: u32 = @intCast(q.dim(3));
    const nkh: u32 = @intCast(k.dim(1));
    const nvh: u32 = @intCast(v.dim(1));

    // Chunked prefill: the keys are the cached prefix plus the chunk.
    var block_size_t: u32 = 0;
    var page_table_stick_size: u32 = 0;
    var sk: u32 = @intCast(k.dim(2));
    if (paged) |p| {
        const block_size: u32 = @intCast(k.dim(2));
        const max_blocks: u32 = @intCast(p.page_table.dim(1));
        block_size_t = block_size / 32;
        page_table_stick_size = max_blocks * 4;
        if (page_table_stick_size % 32 != 0) return error.PageTableStickSize;
        sk = sq + max_blocks * block_size;
    }

    // tt-metal's causal SDPA aligns queries and keys.
    if (!is_chunked and sq != sk) return error.CausalLengthMismatch;

    const qc = opts.q_chunk_size;
    const kc = opts.k_chunk_size;
    const padded_sq = std.mem.alignForward(u32, sq, qc);
    const padded_sk = std.mem.alignForward(u32, sk, kc);
    const skt = padded_sk / 32;
    const dht = dh / 32;
    const vdht = dht;
    const valid_sqt = std.math.divCeil(u32, sq, 32) catch unreachable;
    const valid_skt = std.math.divCeil(u32, sk, 32) catch unreachable;
    const sq_chunk_t = qc / 32;
    const sk_chunk_t = kc / 32;
    const q_num_chunks = padded_sq / qc;

    // HiFi2, approximate math, fp32 DST: the standard compute kernel.
    const fp32_dest_acc_en = true;
    const exp_approx_mode = true;

    const grid_x = opts.grid[0];
    const grid_y = opts.grid[1];
    const num_cores = grid_x * grid_y;
    const core_grid: program.CoreRange = .grid(grid_x, grid_y);

    // Global Q scheduling: the flat (B, heads, q chunks) space over the
    // cores, in pairs when causal so zigzag remapping balances the work.
    const total_q_chunks = b * nqh * q_num_chunks;
    const pair_distribute = is_causal and q_num_chunks % 2 == 0;
    var base_chunks: u32 = undefined;
    var cores_doing_extra: u32 = undefined;
    var extra_chunks: u32 = undefined;
    if (pair_distribute) {
        const total_pairs = total_q_chunks / 2;
        base_chunks = (total_pairs / num_cores) * 2;
        cores_doing_extra = total_pairs % num_cores;
        extra_chunks = 2;
    } else {
        base_chunks = total_q_chunks / num_cores;
        cores_doing_extra = total_q_chunks % num_cores;
        extra_chunks = 1;
    }
    const max_chunks_per_core = base_chunks + (if (cores_doing_extra > 0) extra_chunks else 0);
    const q_buffer_factor: u32 = if (max_chunks_per_core > 1) 2 else 1;

    const dst_size: u32 = if (fp32_dest_acc_en) 4 else 8;
    const qk_sub = largestSubblock(sq_chunk_t, sk_chunk_t, dst_size, std.math.maxInt(u32));
    const qk_out_subblock_h = qk_sub[0];
    const qk_out_subblock_w = qk_sub[1];

    const q_tiles = sq_chunk_t * dht * q_buffer_factor;
    const k_tiles = sk_chunk_t * dht * 2;
    const v_tiles = sk_chunk_t * vdht * 2;
    const qk_tiles = sq_chunk_t * sk_chunk_t;
    const out_im_tiles = sq_chunk_t * vdht;
    const statistics_tiles = sq_chunk_t;
    // The lightweight causal mask: a -inf tile and the diagonal tile.
    const mask_tiles: u32 = 2;

    const out_sub = largestSubblock(sq_chunk_t, vdht, dst_size, std.math.maxInt(u32));
    const out_out_subblock_h = out_sub[0];
    const out_out_subblock_w = out_sub[1];
    const out0_t = sq_chunk_t * vdht;

    var dht_granularity = @min(dht, vdht, dst_size);
    while (dht_granularity > 1 and (dht % dht_granularity != 0 or vdht % dht_granularity != 0)) dht_granularity -= 1;

    // Circular buffers, numbered in allocation order; the QK scores and
    // the sums in fp32, as the DST.
    var cbs: std.ArrayList(program.CircularBuffer) = .empty;
    const alloc = struct {
        fn cb(list: *std.ArrayList(program.CircularBuffer), a: std.mem.Allocator, pages: u32, page_size: u32, dtype: program.DataType, cores: program.CoreRange) !u32 {
            const index: u32 = @intCast(list.items.len);
            try list.append(a, .{ .index = index, .dtype = dtype, .page_size = page_size, .num_pages = pages, .cores = cores });
            return index;
        }
    }.cb;
    const tile = program.DataType.bf16.tileSize();
    const tile_f32 = program.DataType.f32.tileSize();
    const cb_q_in = try alloc(&cbs, allocator, q_tiles, tile, .bf16, core_grid);
    const cb_k_in = try alloc(&cbs, allocator, k_tiles, tile, .bf16, core_grid);
    const cb_v_in = try alloc(&cbs, allocator, v_tiles, tile, .bf16, core_grid);
    const cb_mask_in = try alloc(&cbs, allocator, mask_tiles, tile, .bf16, core_grid);
    const cb_identity_scale_in = try alloc(&cbs, allocator, 1, tile, .bf16, core_grid);
    const cb_col_identity = try alloc(&cbs, allocator, 1, tile, .bf16, core_grid);
    var cb_page_table: u32 = undefined;
    var cb_chunk_start_compute: u32 = undefined;
    var cb_chunk_start_writer: u32 = undefined;
    if (is_chunked) {
        cb_page_table = try alloc(&cbs, allocator, 1, page_table_stick_size, .si32, core_grid);
        cb_chunk_start_compute = try alloc(&cbs, allocator, 1, 32, .si32, core_grid);
        cb_chunk_start_writer = try alloc(&cbs, allocator, 1, 32, .si32, core_grid);
    }
    const cb_qk_im = try alloc(&cbs, allocator, qk_tiles, tile_f32, .f32, core_grid);
    const cb_out_im_a = try alloc(&cbs, allocator, out_im_tiles, tile, .bf16, core_grid);
    const cb_out_im_b = try alloc(&cbs, allocator, out_im_tiles, tile, .bf16, core_grid);
    const cb_max_a = try alloc(&cbs, allocator, statistics_tiles, tile, .bf16, core_grid);
    const cb_max_b = try alloc(&cbs, allocator, statistics_tiles, tile, .bf16, core_grid);
    const cb_sum_a = try alloc(&cbs, allocator, statistics_tiles, tile_f32, .f32, core_grid);
    const cb_sum_b = try alloc(&cbs, allocator, statistics_tiles, tile_f32, .f32, core_grid);
    const cb_exp_max_diff = try alloc(&cbs, allocator, statistics_tiles, tile, .bf16, core_grid);
    const cb_out = try alloc(&cbs, allocator, out0_t, tile, .bf16, core_grid);

    // Compile-time args: the kernels' circular buffers, then the accessor
    // args of their tensors; the rest is baked in.
    const C = program.Arg;
    const val = struct {
        fn f(x: u32) C {
            return .{ .value = x };
        }
    }.f;
    var reader_ct: std.ArrayList(C) = .empty;
    for ([_]u32{ cb_q_in, cb_k_in, cb_v_in }) |i| try reader_ct.append(allocator, .{ .cb = i });
    if (is_chunked) for ([_]u32{ cb_page_table, cb_chunk_start_compute, cb_chunk_start_writer }) |i| try reader_ct.append(allocator, .{ .cb = i });
    for ([_]u32{ io_q, io_k, io_v }) |i| try reader_ct.append(allocator, .{ .tensor_accessor_args = i });
    if (is_chunked) for ([_]u32{ io_page_table, io_chunk_start }) |i| try reader_ct.append(allocator, .{ .tensor_accessor_args = i });

    var writer_ct: std.ArrayList(C) = .empty;
    for ([_]u32{ cb_mask_in, cb_identity_scale_in, cb_col_identity, cb_out }) |i| try writer_ct.append(allocator, .{ .cb = i });
    if (is_chunked) try writer_ct.append(allocator, .{ .cb = cb_chunk_start_writer });
    try writer_ct.append(allocator, .{ .tensor_accessor_args = io_out });

    var compute_ct: std.ArrayList(C) = .empty;
    for ([_]u32{
        cb_q_in,     cb_k_in,     cb_v_in,  cb_mask_in, cb_identity_scale_in, cb_col_identity, cb_out,   cb_qk_im,
        cb_out_im_a, cb_out_im_b, cb_max_a, cb_max_b,   cb_sum_a,             cb_sum_b,        cb_exp_max_diff,
    }) |i| try compute_ct.append(allocator, .{ .cb = i });
    if (is_chunked) try compute_ct.append(allocator, .{ .cb = cb_chunk_start_compute });

    const cfg: kernels.Config = .{
        .nqh = nqh,
        .nkh = nkh,
        .nvh = nvh,
        .skt = skt,
        .valid_sqt = valid_sqt,
        .valid_skt = valid_skt,
        .dht = dht,
        .vdht = vdht,
        .sq_chunk_t = sq_chunk_t,
        .q_num_chunks = q_num_chunks,
        .sk_chunk_t = sk_chunk_t,
        .chunked = is_chunked,
        .block_size_t = block_size_t,
        .page_table_page_bytes = page_table_stick_size,
        .tile_bytes = tile,
        .cb_identity_scale = cb_identity_scale_in,
        .zigzag = is_causal,
        .scale_fp32 = @bitCast(opts.scale),
        .identity_scalar_packed = 0x3F80_3F80,
        .qk_subblock_w = qk_out_subblock_w,
        .qk_subblock_h = qk_out_subblock_h,
        .qk_in0_num_subblocks = sq_chunk_t / qk_out_subblock_h,
        .qk_in1_num_subblocks = sk_chunk_t / qk_out_subblock_w,
        .out_subblock_w = out_out_subblock_w,
        .out_subblock_h = out_out_subblock_h,
        .out_in0_num_subblocks = sq_chunk_t / out_out_subblock_h,
        .out_in1_num_subblocks = vdht / out_out_subblock_w,
        .dst_size = dst_size,
        .stats_granularity = validGranularity(sq_chunk_t, dst_size),
        .sub_exp_granularity = validGranularity(sk_chunk_t, dst_size),
        .dht_granularity = dht_granularity,
        .exp_approx = exp_approx_mode,
    };
    // The defines `compute_common.hpp` reads.
    const defines = try allocator.dupe(program.Define, &.{
        .{ .name = "EXP_APPROX_MODE", .value = if (exp_approx_mode) "1" else "0" },
        .{ .name = "REDUCE_OP", .value = "(PoolType::MAX)" },
        .{ .name = "REDUCE_DIM", .value = "(ReduceDim::REDUCE_ROW)" },
    });

    const flexible: u32 = @intFromBool(is_chunked);
    const reader_rt = try allocator.alloc(program.CoreArgs, num_cores);
    const writer_rt = try allocator.alloc(program.CoreArgs, num_cores);
    const compute_rt = try allocator.alloc(program.CoreArgs, num_cores);
    for (0..num_cores) |ii| {
        const i: u32 = @intCast(ii);
        const core: program.CoreCoord = .{ .x = i % grid_x, .y = i / grid_x };
        var q_start = i * base_chunks + @min(i, cores_doing_extra) * extra_chunks;
        var q_count = base_chunks + (if (i < cores_doing_extra) extra_chunks else 0);
        if (q_start >= total_q_chunks) {
            q_start = total_q_chunks;
            q_count = 0;
        } else if (q_start + q_count > total_q_chunks) {
            q_count = total_q_chunks - q_start;
        }
        const page_table_arg: C = if (is_chunked) .{ .tensor_address = io_page_table } else val(0);
        const chunk_start_arg: C = if (is_chunked) .{ .tensor_address = io_chunk_start } else val(0);
        reader_rt[ii] = .{ .core = core, .args = try allocator.dupe(C, &.{
            .{ .tensor_address = io_q }, .{ .tensor_address = io_k }, .{ .tensor_address = io_v }, val(0), page_table_arg, val(0),
            chunk_start_arg,             val(i),                      val(1),                      val(0), val(0),         val(q_start),
            val(q_count),                val(0),                      val(0),                      val(0), val(0),
        }) };
        writer_rt[ii] = .{ .core = core, .args = try allocator.dupe(C, &.{
            .{ .tensor_address = io_out }, val(i),       val(1), val(flexible), val(0), val(0), val(0), val(0),
            val(q_start),                  val(q_count), val(0), val(0),        val(0), val(0),
        }) };
        compute_rt[ii] = .{ .core = core, .args = try allocator.dupe(C, &.{
            val(i), val(1), val(flexible), val(0), val(0), val(q_start), val(q_count),
        }) };
    }

    const ctx = try dsl.newContext();
    defer ctx.deinit();
    const p: program.Program = .{
        .kernels = &.{
            .{
                .thread = .reader,
                .source = try llk.source(allocator, ctx, "sdpa_reader", .reader, kernels.reader, cfg),
                .cores = core_grid,
                .compile_time_args = reader_ct.items,
                .runtime_args = reader_rt,
            },
            .{
                .thread = .writer,
                .source = try llk.source(allocator, ctx, "sdpa_writer", .writer, kernels.writer, cfg),
                .cores = core_grid,
                .compile_time_args = writer_ct.items,
                .runtime_args = writer_rt,
            },
            .{
                .thread = .{ .compute = .{ .math_fidelity = .hifi2, .fp32_dest_acc_en = fp32_dest_acc_en, .math_approx_mode = true } },
                .source = try llk.source(allocator, ctx, "sdpa", .compute, kernels.compute, cfg),
                .defines = defines,
                .cores = core_grid,
                .compile_time_args = compute_ct.items,
                .runtime_args = compute_rt,
            },
        },
        .circular_buffers = cbs.items,
        .operands = if (is_chunked) &.{
            .{ .index = io_page_table, .layout = .row_major },
            .{ .index = io_chunk_start, .layout = .row_major },
        } else &.{},
    };
    return p.artifact(allocator);
}
