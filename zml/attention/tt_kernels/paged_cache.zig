//! tt-metal's paged KV cache operations, ported: the host program factories
//! of `ttnn::experimental::paged_update_cache` and `paged_fill_cache`
//! (`ttnn/cpp/ttnn/operations/experimental/paged_cache/device/`, tt-metal
//! d04395ed) with their device kernels, in the TTKernel DSL
//! (`paged_cache_kernels.zig`).
//!
//! Caches use tt-metal's layout: `[blocks, heads, block_size, head_dim]`,
//! tiled; user `b`'s position `p` lives in block `page_table[b, p /
//! block_size]` at row `p % block_size`.

const std = @import("std");

const Compiler = @import("../../Compiler.zig");
const zml = @import("../../zml.zig");
const dsl = zml.kernel.ttkernel.dsl;
const program = dsl.program;
const kernels = @import("paged_cache_kernels.zig");
const llk = @import("llk.zig");
const Tensor = zml.Tensor;

/// Bytes a DRAM page is aligned to (Blackhole).
const dram_alignment = 64;

fn dataType(dt: zml.DataType) program.DataType {
    return zml.kernel.ttkernel.from(dt);
}

/// Writes `input` (`[1, B, heads, head_dim]`) into `cache` at position
/// `update_idxs[b]` (`[B]` i32, `-1` skips the user) of each user's sequence,
/// mapped through `page_table` (`[B, max_blocks]` i32). Returns the updated
/// cache; the kernel writes it in place.
pub fn pagedUpdateCache(cache: Tensor, input: Tensor, update_idxs: Tensor, page_table: Tensor) Tensor {
    const allocator = Compiler.current().arena.allocator();
    const art = updateCacheProgram(allocator, cache.shape(), input.shape(), update_idxs.shape(), page_table.shape()) catch |err|
        std.debug.panic("pagedUpdateCache: {}", .{err});
    return zml.ops.ttKernel(&.{ input, update_idxs, page_table, cache }, &.{}, .{
        .kernel_id = "zml.tt.paged_update_cache",
        .artifact = art,
        .inouts = &.{3},
    })[0];
}

/// The program of `pagedUpdateCache` (index-tensor mode, paged, no shared
/// cache, bf16 accumulation). Its io-tensors are `input`, `update_idxs`,
/// `page_table`, then `cache`.
pub fn updateCacheProgram(allocator: std.mem.Allocator, cache: zml.Shape, input: zml.Shape, update_idxs: zml.Shape, page_table: zml.Shape) ![]u8 {
    const io_input = 0;
    const io_index = 1;
    const io_page_table = 2;
    const io_cache = 3;

    const cache_dtype = dataType(cache.dtype());
    const input_dtype = dataType(input.dtype());
    const cache_tile_size = cache_dtype.tileSize();
    const input_tile_size = input_dtype.tileSize();
    const interm_dtype: program.DataType = .bf16;
    const interm_tile_size = interm_dtype.tileSize();

    const index_stick_size: u32 = std.mem.alignForward(u32, @intCast(update_idxs.byteSize()), dram_alignment);
    const block_size: u32 = @intCast(cache.dim(2));
    const block_size_t = block_size / 32;
    const max_blocks_per_seq: u32 = @intCast(page_table.dim(1));
    const page_table_stick_size: u32 = std.mem.alignForward(u32, @intCast(page_table.dim(1) * page_table.dtype().sizeOf()), dram_alignment);

    const head_dim: u32 = @intCast(input.dim(-1));
    const wt = head_dim / 32;
    const st = block_size_t;
    const wbytes = head_dim * 2;
    const cache_total_tiles: u32 = @intCast(@divExact(cache.dim(0) * cache.dim(1) * cache.dim(2) * cache.dim(3), 1024));
    const cache_batch_num_tiles: u32 = cache_total_tiles / @as(u32, @intCast(cache.dim(0)));
    const users: u32 = @intCast(input.dim(1));
    const num_heads: u32 = @intCast(cache.dim(1));
    if (input.dim(2) > 32) return error.TooManyHeads;

    // One core per user, holding its input shard.
    const cores: program.CoreRange = .{ .start = .{ .x = 0, .y = 0 }, .end = .{ .x = users - 1, .y = 0 } };

    const cache_cb = 0;
    const input_cb = 1;
    const index_cb = 2;
    const page_table_cb = 3;
    const output_cb = 16;
    const interm0_cb = 24;
    const interm1_cb = 25;
    const interm2_cb = 26;
    const semaphore = 0;

    const C = program.Arg;
    const v = struct {
        fn val(x: u32) C {
            return .{ .value = x };
        }
    }.val;

    const reader_ct = [_]C{ .{ .cb = cache_cb }, .{ .cb = input_cb }, .{ .cb = index_cb }, .{ .cb = page_table_cb }, .{ .tensor_accessor_args = io_cache }, .{ .tensor_accessor_args = io_index }, .{ .tensor_accessor_args = io_page_table } };
    const writer_ct = [_]C{ .{ .cb = output_cb }, .{ .cb = interm0_cb }, .{ .cb = interm1_cb }, .{ .cb = interm2_cb }, .{ .cb = index_cb }, .{ .cb = page_table_cb }, .{ .tensor_accessor_args = io_cache } };
    const compute_ct = [_]C{ .{ .cb = input_cb }, .{ .cb = interm2_cb } };
    const cfg: kernels.UpdateConfig = .{
        .wt = wt,
        .num_heads = num_heads,
        .block_size = block_size,
        .block_size_t = block_size_t,
        .wbytes = wbytes,
        .index_page_bytes = index_stick_size,
        .page_table_page_bytes = page_table_stick_size,
    };
    _ = .{ cache_batch_num_tiles, max_blocks_per_seq, st, semaphore };
    const ctx = try dsl.newContext();
    defer ctx.deinit();

    const reader_rt = try allocator.alloc(program.CoreArgs, users);
    const writer_rt = try allocator.alloc(program.CoreArgs, users);
    for (0..users) |i| {
        const core: program.CoreCoord = .{ .x = @intCast(i), .y = 0 };
        reader_rt[i] = .{ .core = core, .args = try allocator.dupe(C, &.{
            .{ .tensor_address = io_cache }, v(0), .{ .tensor_address = io_index }, v(@intCast(i)), .{ .tensor_address = io_page_table }, v(0),
        }) };
        writer_rt[i] = .{ .core = core, .args = try allocator.dupe(C, &.{
            .{ .tensor_address = io_cache }, v(0), v(0), v(@intCast(i)), v(0), v(0), v(0),
        }) };
    }

    const p: program.Program = .{
        .kernels = &.{
            .{
                .thread = .reader,
                .source = try llk.source(allocator, ctx, "paged_update_cache_reader", .reader, kernels.updateReader, cfg),
                .cores = cores,
                .compile_time_args = &reader_ct,
                .runtime_args = reader_rt,
            },
            .{
                .thread = .writer,
                .source = try llk.source(allocator, ctx, "paged_update_cache_writer", .writer, kernels.updateWriter, cfg),
                .cores = cores,
                .compile_time_args = &writer_ct,
                .runtime_args = writer_rt,
            },
            .{
                .thread = .{ .compute = .{ .fp32_dest_acc_en = false } },
                .source = try llk.source(allocator, ctx, "paged_update_cache", .compute, kernels.updateCompute, cfg),
                .cores = cores,
                .compile_time_args = &compute_ct,
            },
        },
        .circular_buffers = &.{
            .{ .index = cache_cb, .dtype = cache_dtype, .page_size = cache_tile_size, .num_pages = 2 * wt, .cores = cores },
            .{ .index = input_cb, .dtype = input_dtype, .page_size = input_tile_size, .num_pages = wt, .cores = cores, .tensor = io_input },
            .{ .index = interm0_cb, .dtype = interm_dtype, .page_size = interm_tile_size, .num_pages = 2 * wt, .cores = cores, .aliases = &.{
                .{ .index = interm1_cb, .dtype = interm_dtype, .page_size = interm_tile_size },
            } },
            .{ .index = interm2_cb, .dtype = interm_dtype, .page_size = interm_tile_size, .num_pages = 2 * wt, .cores = cores },
            .{ .index = output_cb, .dtype = cache_dtype, .page_size = cache_tile_size, .num_pages = users * wt, .cores = cores },
            .{ .index = index_cb, .dtype = .si32, .page_size = program.DataType.si32.tileSize(), .num_pages = 1, .cores = cores },
            .{ .index = page_table_cb, .dtype = .si32, .page_size = page_table_stick_size, .num_pages = 1, .cores = cores },
        },
        .operands = &.{
            .{ .index = io_input, .buffer = .l1, .memory_layout = .height_sharded, .grid = .{ users, 1 }, .cores = &.{cores} },
            .{ .index = io_index, .layout = .row_major },
            .{ .index = io_page_table, .layout = .row_major },
        },
    };
    return p.artifact(allocator);
}

/// tt-metal's `split_work_to_cores` (row-wise): `total` units over the cores of a
/// `cols x rows` grid in row-major order, the first ones taking one more.
pub const WorkSplit = struct {
    num_cores: u32,
    group_1_cores: u32,
    units_group_1: u32,
    units_group_2: u32,

    pub fn init(cols: u32, rows: u32, total: u32) WorkSplit {
        if (total == 0) return .{ .num_cores = 0, .group_1_cores = 0, .units_group_1 = 0, .units_group_2 = 0 };
        const num_cores = @min(total, cols * rows);
        const per_core = total / num_cores;
        const more = total % num_cores;
        return if (more == 0)
            .{ .num_cores = num_cores, .group_1_cores = num_cores, .units_group_1 = per_core, .units_group_2 = 0 }
        else
            .{ .num_cores = num_cores, .group_1_cores = more, .units_group_1 = per_core + 1, .units_group_2 = per_core };
    }

    pub fn units(self: WorkSplit, core: u32) u32 {
        if (core < self.group_1_cores) return self.units_group_1;
        if (core < self.num_cores) return self.units_group_2;
        return 0;
    }
};

pub const FillOptions = struct {
    /// `(cols, rows)` of the cores the program may use, from (0, 0).
    grid: [2]u32 = .{ 8, 8 },
};

/// Writes the prefill `input` (`[users, heads, seq, head_dim]`) into
/// `cache`: row `s` of user `u` goes to position `s` of the sequence of
/// `page_table` row `batch_idx[u]` (`[users]` i32). Returns the updated
/// cache; the kernel writes it in place.
pub fn pagedFillCache(cache: Tensor, input: Tensor, page_table: Tensor, batch_idx: Tensor, opts: FillOptions) Tensor {
    const allocator = Compiler.current().arena.allocator();
    const art = fillCacheProgram(allocator, cache.shape(), input.shape(), page_table.shape(), batch_idx.shape(), opts) catch |err|
        std.debug.panic("pagedFillCache: {}", .{err});
    return zml.ops.ttKernel(&.{ input, page_table, batch_idx, cache }, &.{}, .{
        .kernel_id = "zml.tt.paged_fill_cache",
        .artifact = art,
        .inouts = &.{3},
    })[0];
}

/// The program of `pagedFillCache` (`paged_fill_cache_program_factory.cpp`,
/// batch-index tensor, no valid-length tensor). Its io-tensors are `input`,
/// `page_table`, `batch_idx`, then `cache`.
pub fn fillCacheProgram(allocator: std.mem.Allocator, cache: zml.Shape, input: zml.Shape, page_table: zml.Shape, batch_idx: zml.Shape, opts: FillOptions) ![]u8 {
    const io_input = 0;
    const io_page_table = 1;
    const io_batch_idx = 2;
    const io_cache = 3;

    const dtype = dataType(input.dtype());
    const tile_size = dtype.tileSize();
    const input_batch: u32 = @intCast(input.dim(0));
    const num_heads: u32 = @intCast(input.dim(1));
    const input_seq_len: u32 = std.mem.alignForward(u32, @intCast(input.dim(2)), 32);
    const block_size: u32 = @intCast(cache.dim(2));
    const head_dim: u32 = @intCast(input.dim(3));
    const input_seq_len_t = input_seq_len / 32;
    const wt = head_dim / 32;
    const block_size_t = block_size / 32;
    const num_blocks_of_work_per_batch = num_heads * input_seq_len_t;
    const num_blocks_of_work = input_batch * num_blocks_of_work_per_batch;
    const num_blocks_of_work_per_head = input_seq_len_t;
    const page_table_stick_size: u32 = std.mem.alignForward(u32, @intCast(page_table.dim(-1) * 4), dram_alignment);
    const log2_page_table_stick_size: u32 = std.math.log2_int(u32, page_table_stick_size);
    const batch_idx_stick_size = 4;
    const batch_idx_num_elements: u32 = @intCast(batch_idx.count());
    if (batch_idx_num_elements != input_batch) return error.BatchIdxSize;

    const split: WorkSplit = .init(opts.grid[0], opts.grid[1], num_blocks_of_work);
    const cores: program.CoreRange = .grid(opts.grid[0], opts.grid[1]);

    const input_cb = 0;
    const page_table_cb = 1;
    const batch_idx_cb = 2;
    const valid_seq_len_cb = 3;

    const C = program.Arg;
    const v = struct {
        fn val(x: u32) C {
            return .{ .value = x };
        }
    }.val;

    const reader_ct = [_]C{ .{ .cb = input_cb }, .{ .tensor_accessor_args = io_input } };
    const writer_ct = [_]C{ .{ .cb = input_cb }, .{ .cb = page_table_cb }, .{ .cb = batch_idx_cb }, .{ .tensor_accessor_args = io_cache }, .{ .tensor_accessor_args = io_page_table }, .{ .tensor_accessor_args = io_batch_idx } };
    const batch_idx_page_size = std.mem.alignForward(u32, batch_idx_stick_size * batch_idx_num_elements, dram_alignment);
    const cfg: kernels.FillConfig = .{
        .wt = wt,
        .num_heads = num_heads,
        .block_size_t = block_size_t,
        .num_blocks_of_work_per_head = num_blocks_of_work_per_head,
        .num_blocks_per_batch = num_blocks_of_work_per_batch,
        .batch_idx_num_elements = batch_idx_num_elements,
        .batch_idx_page_bytes = batch_idx_page_size,
        .page_table_page_bytes = page_table_stick_size,
    };
    _ = .{ log2_page_table_stick_size, valid_seq_len_cb };
    const ctx = try dsl.newContext();
    defer ctx.deinit();

    const num_cores = opts.grid[0] * opts.grid[1];
    const reader_rt = try allocator.alloc(program.CoreArgs, num_cores);
    const writer_rt = try allocator.alloc(program.CoreArgs, num_cores);
    var blocks_written: u32 = 0;
    for (0..num_cores) |ii| {
        const i: u32 = @intCast(ii);
        const core: program.CoreCoord = .{ .x = i % opts.grid[0], .y = i / opts.grid[0] };
        const rows = split.units(i);
        reader_rt[i] = .{ .core = core, .args = try allocator.dupe(C, &.{
            .{ .tensor_address = io_input }, v(blocks_written * wt), v(rows), v(0),
        }) };
        writer_rt[i] = .{ .core = core, .args = try allocator.dupe(C, &.{
            .{ .tensor_address = io_cache },     .{ .tensor_address = io_page_table }, v(blocks_written), v(rows),
            .{ .tensor_address = io_batch_idx }, v(0),                                 v(0),
        }) };
        blocks_written += rows;
    }

    const p: program.Program = .{
        .kernels = &.{
            .{
                .thread = .reader,
                .source = try llk.source(allocator, ctx, "paged_fill_cache_reader", .reader, kernels.fillReader, cfg),
                .cores = cores,
                .compile_time_args = &reader_ct,
                .runtime_args = reader_rt,
            },
            .{
                .thread = .writer,
                .source = try llk.source(allocator, ctx, "paged_fill_cache_writer", .writer, kernels.fillWriter, cfg),
                .cores = cores,
                .compile_time_args = &writer_ct,
                .runtime_args = writer_rt,
            },
        },
        .circular_buffers = &.{
            .{ .index = input_cb, .dtype = dtype, .page_size = tile_size, .num_pages = 2 * wt, .cores = cores },
            .{ .index = page_table_cb, .dtype = .si32, .page_size = page_table_stick_size, .num_pages = 1, .cores = cores },
            .{ .index = batch_idx_cb, .dtype = .si32, .page_size = batch_idx_page_size, .num_pages = 1, .cores = cores },
        },
        .operands = &.{
            .{ .index = io_page_table, .layout = .row_major },
            .{ .index = io_batch_idx, .layout = .row_major },
        },
    };
    return p.artifact(allocator);
}
