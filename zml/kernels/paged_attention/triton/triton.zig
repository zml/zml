const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../../../zml.zig");
const triton = zml.kernel.triton;
const AttentionOptions = @import("../paged_attention.zig").AttentionOptions;
const kernels = @import("unified_attention.zig");
const kernels_sparse = @import("sparse_unified_attention.zig");
const kernels_oneapi = @import("unified_attention_oneapi.zig");

const log = std.log.scoped(.@"zml/attention/triton");

fn isOneapiTarget() bool {
    return zml.Compiler.current().platform.target == .oneapi;
}

fn use2dKernel(all_decode: bool, batch_size: usize, num_kv_heads: usize) bool {
    // Intel decode spills the 2D whole-sequence kernel; force the 3D split-K path.
    if (all_decode and isOneapiTarget()) return false;
    // prefill uses 2D; decode uses 3D until the batch is large enough to
    // provide at least 128 2D launch programs across KV heads.
    if (all_decode) {
        const seq_threshold_3d = @divFloor(128, num_kv_heads);
        return batch_size > seq_threshold_3d;
    }

    return true;
}

pub const Config2D = struct {
    block_m: usize,
    block_q: usize,
    tile_size: usize,
    num_warps: usize,
    num_stages: usize,
    total_q_blocks: usize,
};

fn isCudaComputeCapability(expected: zml.platform.cuda.ComputeCapability) bool {
    const platform = zml.module.CompilationContext.current().platform;
    return if (zml.platform.cuda.computeCapability(platform)) |cc| cc.eql(expected) else false;
}

/// Largest `BLOCK_M * HEAD_SIZE_PADDED` the 2D kernel survives. That product is
/// the shape of both the query tile and the f32 accumulator carried through the
/// tile loop; above it the kernel faults with CUDA_ERROR_MISALIGNED_ADDRESS.
///
/// Measured on a GB300 with Gemma-4-31B, whose global-attention layers are
/// head_dim 512 (its sliding ones are 256):
///
///   64x256, 32x512, 16x512  ok        64x512, 128x512  fault
///
/// The bound is on this product alone. Two things that do *not* move it, so do
/// not reach for them if a new head size faults: TILE_SIZE, which at 64 pushes
/// total shared memory to ~160 KB and still passes at BLOCK_M 32 while 64x512
/// faults at 98 KB; and num_warps, where 8 halves the per-thread register share
/// and still faults -- which is also why the obvious "the accumulator outgrows
/// the register file" story cannot be the whole explanation. The underlying
/// Triton codegen bug is not diagnosed; this only keeps us out of it.
const max_query_tile_elements: usize = 16 * 1024;

fn select2dConfig(options: paged.PagedAttentionOptions) Config2D {
    const max_num_stages_2d: usize = if (options.head_dim <= 128) 4 else 2;

    // Until we test on other platforms, gate the fix to GB300
    const is_gb300 = isCudaComputeCapability(.{ .major = 10, .minor = 3 });

    var num_stages_2d: usize, var num_warps: usize, var tile_size: usize = if (!options.all_decode) .{ 1, 2, 64 } else .{ 3, 2, options.block_size };

    var block_m = options.block_m;
    var block_q = options.block_q;
    if (options.max_seqlen_q >= 256) {
        if (options.head_dim >= 256) {
            block_m = 64;
            tile_size = 16;
        } else {
            block_m = 128;
        }

        if (is_gb300) {
            // Halving BLOCK_M for every doubling of the head keeps that product
            // constant: 128 at head_dim 128, 64 at 256, 32 at 512. The first two
            // are the values this branch has always picked.
            const head_dim_padded = std.math.ceilPowerOfTwoAssert(usize, options.head_dim);
            block_m = @min(128, max_query_tile_elements / head_dim_padded);
        }
        num_stages_2d = 1;
        num_warps = 4;
    }
    block_q = block_m / options.numQueriesPerKv();
    const total_q_blocks = options.num_tokens / block_q + options.batch_size;

    return .{
        .block_m = block_m,
        .block_q = block_q,
        .tile_size = tile_size,
        .num_warps = num_warps,
        .num_stages = @min(max_num_stages_2d, num_stages_2d),
        .total_q_blocks = total_q_blocks,
    };
}

pub const Config3D = struct {
    const AttentionConfig = struct {
        tile_size: usize,
        num_segments_per_seq: usize,
        num_warps: usize,
        num_stages: usize,
        block_q: usize,
        block_m: usize,
        total_q_blocks: usize,
    };
    const ReduceConfig = struct {
        tile_size: usize,
        num_segments_per_seq: usize,
        num_warps: usize,
        num_stages: usize,
        block_q: usize,
        block_m: usize,
    };
    attention: AttentionConfig,
    reduce: ReduceConfig,
};

fn select3dConfig(options: paged.PagedAttentionOptions) Config3D {
    var reduce_num_warps: usize = 2;
    // Intel decode needs more warps to spread the work and avoid register spill.
    const attn_warps: usize = if (options.all_decode and isOneapiTarget()) 8 else 2;
    const tile_size = options.block_size;

    //const MAX_SEGMENTS: usize = @min(128, std.math.divCeil(usize, max_seqlen_k, tile_size));
    var num_segments = std.math.divCeil(usize, options.target_num_prgms, options.num_2d_prgms) catch unreachable;
    num_segments = std.math.ceilPowerOfTwoAssert(usize, num_segments);
    num_segments = @min(num_segments, 128);
    if (options.all_decode and !isOneapiTarget()) {
        // Keep the number of segments small to then limit reduce cost
        // Didn't change the computation of Intel decode at the momment
        // Need to be tested
        num_segments = @min(num_segments, 16);
    }
    const min_segments: usize = if (tile_size <= 16) 16 else 8;
    num_segments = @max(num_segments, min_segments);
    if (num_segments == min_segments) {
        reduce_num_warps = 1;
    }

    return .{
        .attention = .{
            .tile_size = tile_size,
            .num_segments_per_seq = num_segments,
            .num_warps = attn_warps,
            .num_stages = 1,
            .block_m = options.block_m,
            .block_q = options.block_q,
            .total_q_blocks = options.total_q_blocks,
        },
        .reduce = .{
            .tile_size = tile_size,
            .num_segments_per_seq = num_segments,
            .num_warps = reduce_num_warps,
            .num_stages = 1,
            .block_m = options.block_m,
            .block_q = options.block_q,
        },
    };
}

pub fn getCuCount() usize {
    const platform = zml.Compiler.current().platform;
    if (platform.devices.len == 0) return 1;
    const attribute = platform.devices[0].pjrt_desc.attribute(platform.pjrt_api, "core_count") orelse return 1;
    if (attribute.int64 <= 0) return 1;
    return @intCast(attribute.int64);
}

const SparseConfig = struct {
    tile_size: usize,
    num_segments: usize,
};

fn selectSparseConfig(count: usize, batch_size: usize, num_kv_heads: usize, cu_count: usize, all_decode: bool) SparseConfig {
    std.debug.assert(count > 0 and batch_size > 0 and num_kv_heads > 0);
    if (!all_decode) return .{ .tile_size = 32, .num_segments = 1 };

    const target_num_prgms = cu_count * 4;
    const occupancy_splits = std.math.ceilPowerOfTwoAssert(usize, std.math.divCeil(usize, target_num_prgms * 4, batch_size * num_kv_heads) catch unreachable);
    const tile: usize = if (occupancy_splits > 128) 16 else 32;
    const tile_splits = std.math.ceilPowerOfTwoAssert(usize, std.math.divCeil(usize, count, tile) catch unreachable);

    return .{
        .tile_size = tile,
        .num_segments = @min(tile_splits, @min(if (tile == 16) @as(usize, 256) else 128, @max(8, occupancy_splits))),
    };
}

pub const paged = struct {
    pub const Options = struct {
        batch_size: usize,
        max_num_pages: usize,
        max_seqlen_q: usize,
        is_prefill: bool,

        pub fn isPrefill(self: Options) bool {
            return self.is_prefill;
        }

        pub fn maxNumPages(self: Options) usize {
            return self.max_num_pages;
        }
    };

    pub const Parameters = struct {
        block_table: zml.Tensor,
        seq_lens: zml.Tensor,
        query_start_len: zml.Tensor,
        options_: Options,

        pub fn init(options_: Options) Parameters {
            return .{
                .block_table = .init(.{ .b = options_.batch_size, .p = options_.max_num_pages }, .i32),
                .seq_lens = .init(.{ .b = options_.batch_size }, .i32),
                .query_start_len = .init(.{ .b = options_.batch_size + 1 }, .i32),
                .options_ = options_,
            };
        }

        pub fn allocationSize(self: Parameters) usize {
            var allocation_size: usize = 0;

            allocation_size += self.block_table.byteSize();
            allocation_size += self.seq_lens.byteSize();
            allocation_size += self.query_start_len.byteSize();

            return allocation_size;
        }

        pub fn options(self: Parameters) Options {
            return self.options_;
        }
    };

    pub const PagedAttentionOptions = struct {
        cu_count: usize,
        all_decode: bool,
        num_tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        batch_size: usize,
        block_size: usize,
        num_blocks: usize,
        max_num_block_per_seq: usize,
        sliding_window: usize,
        block_m: usize,
        block_q: usize,
        total_q_blocks: usize,
        target_num_prgms: usize,
        num_2d_prgms: usize,
        max_seqlen_q: usize,
        scale: ?f32,

        pub fn numQueriesPerKv(self: PagedAttentionOptions) usize {
            return self.num_heads / self.num_kv_heads;
        }

        pub fn maxSeqLenK(self: PagedAttentionOptions) usize {
            return self.max_num_block_per_seq * self.block_size;
        }
    };

    pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        if (opts.mask == .indices) {
            return pagedSparseAttention(parameters, q, k_cache, v_cache, opts);
        }

        const output = zml.ops.manualComputation(
            (struct {
                q: zml.Tensor,
                k_cache: zml.Tensor,
                v_cache: zml.Tensor,
                block_table: zml.Tensor,
                seq_lens: zml.Tensor,
                query_start_len: zml.Tensor,
                opts: AttentionOptions,
                options: Options,

                fn body(self: @This(), _: zml.Shape) zml.Tensor {
                    const parameters_: Parameters = .{
                        .block_table = self.block_table,
                        .seq_lens = self.seq_lens,
                        .query_start_len = self.query_start_len,
                        .options_ = self.options,
                    };

                    const cu_count = getCuCount();
                    const num_heads: usize = @intCast(self.q.dim(.hkv) * self.q.dim(.hg));
                    const num_kv_heads: usize = @intCast(self.k_cache.dim(.hkv));
                    const num_queries_per_kv: usize = num_heads / num_kv_heads;
                    // Intel decode: pack exactly one GQA group per tile (block_q == 1) so the
                    // single decode query token doesn't carry masked-out fp32 acc lanes.
                    // oneAPI decode keeps one GQA group per tile, padded to a power of two so tt.make_range emits legal Triton IR.
                    const block_m: usize = if (!self.options.is_prefill and isOneapiTarget())
                        std.math.ceilPowerOfTwoAssert(usize, num_queries_per_kv)
                    else if (num_queries_per_kv <= 16) 16 else std.math.ceilPowerOfTwoAssert(usize, num_queries_per_kv);
                    const block_q: usize = block_m / num_queries_per_kv;
                    const num_tokens: usize = @intCast(self.q.dim(.b));
                    const num_seqs: usize = @intCast(parameters_.block_table.dim(.b));
                    const total_q_blocks: usize = num_tokens / block_q + num_seqs;
                    const target_num_prgms: usize = cu_count * 4;
                    const num_2d_prgms: usize = total_q_blocks * num_kv_heads;

                    const sliding_window: usize = self.opts.mask.slidingWindow() orelse 0;

                    const paged_attention_opts: PagedAttentionOptions = .{
                        .cu_count = getCuCount(),
                        .all_decode = !self.options.is_prefill,
                        .num_tokens = num_tokens,
                        .num_heads = num_heads,
                        .num_kv_heads = num_kv_heads,
                        .head_dim = @intCast(self.q.dim(.hd)),
                        .batch_size = @intCast(parameters_.block_table.dim(.b)),
                        .block_size = @intCast(self.k_cache.dim(.k_chunk)),
                        .num_blocks = @intCast(self.k_cache.dim(.page)),
                        .max_num_block_per_seq = @intCast(parameters_.block_table.dim(.p)),
                        .sliding_window = sliding_window,
                        .block_m = block_m,
                        .block_q = block_q,
                        .total_q_blocks = total_q_blocks,
                        .target_num_prgms = target_num_prgms,
                        .num_2d_prgms = num_2d_prgms,
                        .max_seqlen_q = self.options.max_seqlen_q,
                        .scale = self.opts.scale,
                    };

                    const use_2d_kernel = use2dKernel(
                        paged_attention_opts.all_decode,
                        paged_attention_opts.batch_size,
                        paged_attention_opts.num_kv_heads,
                    );
                    const output = if (use_2d_kernel)
                        pagedAttention2d(parameters_, self.q, self.k_cache, self.v_cache, self.opts, paged_attention_opts)
                    else if (isOneapiTarget())
                        pagedAttention3dOneapi(parameters_, self.q, self.k_cache, self.v_cache, self.opts, paged_attention_opts)
                    else
                        pagedAttention3d(parameters_, self.q, self.k_cache, self.v_cache, self.opts, paged_attention_opts);

                    return output;
                }
            }).body,
            .{
                .q = q,
                .k_cache = k_cache,
                .v_cache = v_cache,
                .block_table = parameters.block_table,
                .seq_lens = parameters.seq_lens,
                .query_start_len = parameters.query_start_len,
                .opts = opts,
                .options = parameters.options_,
            },
            q.shape(),
        );

        return output;
    }

    pub fn pagedSparseAttention(parameters: Parameters, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        stdx.debug.assert(opts.mask == .indices, "Expected indices mask but got {}", .{opts.mask});
        const indices = opts.mask.indices;
        stdx.debug.assert(indices.rank() == 2 and indices.dtype() == .i32, "Sparse indices must be an i32 [query, topk] tensor, got {f}", .{indices.shape()});
        stdx.debug.assert(indices.dim(.b) == q.dim(.b) and indices.dim(.topk) > 0, "Sparse indices {f} must match queries {f} with a nonempty selection capacity", .{ indices.shape(), q.shape() });
        stdx.debug.assert(indices.shape().hasTags(.{ .b, .topk }), "Sparse indices expect to have .b, .topk tags", .{});
        stdx.debug.assert(q.dtype() == k_cache.dtype() and q.dtype() == v_cache.dtype(), "Sparse attention requires matching query and KV dtypes", .{});

        return zml.ops.manualComputation(
            (struct {
                q: zml.Tensor,
                k_cache: zml.Tensor,
                v_cache: zml.Tensor,
                block_table: zml.Tensor,
                seq_lens: zml.Tensor,
                query_start_len: zml.Tensor,
                opts: AttentionOptions,
                options: Options,

                fn body(self: @This(), _: zml.Shape) zml.Tensor {
                    const parameters_: Parameters = .{
                        .block_table = self.block_table,
                        .seq_lens = self.seq_lens,
                        .query_start_len = self.query_start_len,
                        .options_ = self.options,
                    };

                    const config = selectSparseConfig(
                        @intCast(self.opts.mask.indices.dim(.topk)),
                        @intCast(self.block_table.dim(.b)),
                        @intCast(self.q.dim(.hkv)),
                        getCuCount(),
                        !self.options.is_prefill,
                    );

                    return if (config.num_segments == 1)
                        pagedSparseAttention2d(parameters_, self.q, self.k_cache, self.v_cache, self.opts, config)
                    else
                        pagedSparseAttention3d(parameters_, self.q, self.k_cache, self.v_cache, self.opts, config);
                }
            }).body,
            .{
                .q = q,
                .k_cache = k_cache,
                .v_cache = v_cache,
                .block_table = parameters.block_table,
                .seq_lens = parameters.seq_lens,
                .query_start_len = parameters.query_start_len,
                .opts = opts,
                .options = parameters.options_,
            },
            q.shape(),
        );
    }

    pub fn pagedSparseAttention2d(parameters: Parameters, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, opts: AttentionOptions, tuning: SparseConfig) zml.Tensor {
        std.debug.assert(tuning.num_segments == 1);

        const indices = opts.mask.indices;

        const group = q.dim(.hg);
        const num_kv_heads = q.dim(.hkv);
        std.debug.assert(num_kv_heads == k_cache.dim(.hkv) and num_kv_heads == v_cache.dim(.hkv));
        std.debug.assert(q.dim(.hd) == k_cache.dim(.hd) and q.dim(.hd) == v_cache.dim(.hd));

        const all_decode = !parameters.options_.is_prefill;

        const q_shape = q.shape().mergeAxes(.{ .h = .{ .hkv, .hg } });

        const qs = q_shape.computeElementStrides().constSlice();
        const ks = k_cache.shape().computeElementStrides().constSlice();
        const vs = v_cache.shape().computeElementStrides().constSlice();
        const is = indices.shape().computeElementStrides().constSlice();
        const ts = parameters.block_table.shape().computeElementStrides().constSlice();

        const head_size_padded: i64 = @intCast(std.math.ceilPowerOfTwoAssert(u64, @intCast(@max(16, q.dim(.hd)))));

        const config: kernels_sparse.SparseAttentionConfig = .{
            .dtype = triton.from(q.dtype()),
            .num_seqs = @intCast(parameters.block_table.dim(.b)),
            .num_query_heads = num_kv_heads * group,
            .num_queries_per_kv = group,
            .head_size = q.dim(.hd),
            .head_size_padded = head_size_padded,
            .block_m = @intCast(@max(16, std.math.ceilPowerOfTwoAssert(u64, @intCast(group)))),
            .block_size = k_cache.dim(.k_chunk),
            .tile_size = @intCast(tuning.tile_size),
            .indices_count = indices.dim(.topk),
            .num_segments = @intCast(tuning.num_segments),
            .decode_sequence_grid = all_decode and parameters.options_.max_seqlen_q == 1,
            .use_sinks = opts.sink != null,
            .scale = opts.scale orelse @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(q.dim(.hd))))),
            .block_table_stride = ts[parameters.block_table.axis(.b)],
            .query_strides = .{ qs[q_shape.axis(.b)], qs[q_shape.axis(.h)], qs[q_shape.axis(.hd)] },
            .key_strides = .{ ks[k_cache.axis(.page)], ks[k_cache.axis(.k_chunk)], ks[k_cache.axis(.hkv)], ks[k_cache.axis(.hd)] },
            .value_strides = .{ vs[v_cache.axis(.page)], vs[v_cache.axis(.k_chunk)], vs[v_cache.axis(.hkv)], vs[v_cache.axis(.hd)] },
            .indices_strides = .{ is[indices.axis(.b)], is[indices.axis(.topk)] },
        };

        const inputs: kernels_sparse.KernelSparseAttention2dPtr.Kernel.Inputs = .{
            .query_ptr = q,
            .key_ptr = k_cache,
            .value_ptr = v_cache,
            .indices_ptr = indices,
            .block_tables_ptr = parameters.block_table,
            .seq_lens_ptr = parameters.seq_lens,
            .query_start_len_ptr = parameters.query_start_len,
            .sink_ptr = if (opts.sink) |sink| sink.convert(.f32) else .scalar(0, .f32),
        };

        const grid: [3]i32 = .{
            if (config.decode_sequence_grid) config.num_seqs else @intCast(q.dim(.b)),
            @intCast(num_kv_heads),
            1,
        };

        const warps: i32 = if (all_decode and isOneapiTarget()) 8 else 4;

        const output = kernels_sparse.KernelSparseAttention2dPtr.Kernel.call(inputs, .{
            .output = q.shape(),
        }, .{
            .cfg = config,
            .grid = grid,
            .num_stages = 1,
            .num_warps = warps,
        });

        return output.output;
    }

    pub fn pagedSparseAttention3d(parameters: Parameters, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, opts: AttentionOptions, tuning: SparseConfig) zml.Tensor {
        std.debug.assert(tuning.num_segments > 1);

        const indices = opts.mask.indices;
        const group = q.dim(.hg);
        const num_kv_heads = q.dim(.hkv);
        std.debug.assert(num_kv_heads == k_cache.dim(.hkv) and num_kv_heads == v_cache.dim(.hkv));
        std.debug.assert(q.dim(.hd) == k_cache.dim(.hd) and q.dim(.hd) == v_cache.dim(.hd));

        const all_decode = !parameters.options_.is_prefill;

        const q_shape = q.shape().mergeAxes(.{ .h = .{ .hkv, .hg } });

        const qs = q_shape.computeElementStrides().constSlice();
        const ks = k_cache.shape().computeElementStrides().constSlice();
        const vs = v_cache.shape().computeElementStrides().constSlice();
        const is = indices.shape().computeElementStrides().constSlice();
        const ts = parameters.block_table.shape().computeElementStrides().constSlice();

        const head_size_padded: i64 = @intCast(std.math.ceilPowerOfTwoAssert(u64, @intCast(@max(16, q.dim(.hd)))));

        const config: kernels_sparse.SparseAttentionConfig = .{
            .dtype = triton.from(q.dtype()),
            .num_seqs = @intCast(parameters.block_table.dim(.b)),
            .num_query_heads = num_kv_heads * group,
            .num_queries_per_kv = group,
            .head_size = q.dim(.hd),
            .head_size_padded = head_size_padded,
            .block_m = @intCast(@max(16, std.math.ceilPowerOfTwoAssert(u64, @intCast(group)))),
            .block_size = k_cache.dim(.k_chunk),
            .tile_size = @intCast(tuning.tile_size),
            .indices_count = indices.dim(.topk),
            .num_segments = @intCast(tuning.num_segments),
            .decode_sequence_grid = all_decode and parameters.options_.max_seqlen_q == 1,
            .use_sinks = opts.sink != null,
            .scale = opts.scale orelse @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(q.dim(.hd))))),
            .block_table_stride = ts[parameters.block_table.axis(.b)],
            .query_strides = .{ qs[q_shape.axis(.b)], qs[q_shape.axis(.h)], qs[q_shape.axis(.hd)] },
            .key_strides = .{ ks[k_cache.axis(.page)], ks[k_cache.axis(.k_chunk)], ks[k_cache.axis(.hkv)], ks[k_cache.axis(.hd)] },
            .value_strides = .{ vs[v_cache.axis(.page)], vs[v_cache.axis(.k_chunk)], vs[v_cache.axis(.hkv)], vs[v_cache.axis(.hd)] },
            .indices_strides = .{ is[indices.axis(.b)], is[indices.axis(.topk)] },
        };

        const inputs: kernels_sparse.KernelSparseAttention3dPtr.Kernel.Inputs = .{
            .query_ptr = q,
            .key_ptr = k_cache,
            .value_ptr = v_cache,
            .indices_ptr = indices,
            .block_tables_ptr = parameters.block_table,
            .seq_lens_ptr = parameters.seq_lens,
            .query_start_len_ptr = parameters.query_start_len,
            .sink_ptr = if (opts.sink) |sink| sink.convert(.f32) else .scalar(0, .f32),
        };

        const partition = kernels_sparse.SparseSegments.init(config.indices_count, config.num_segments, config.tile_size);
        const grid: [3]i32 = .{
            if (config.decode_sequence_grid) config.num_seqs else @intCast(q.dim(.b)),
            @intCast(num_kv_heads),
            @intCast(partition.launched),
        };

        const warps: i32 = if (all_decode) 8 else 4;

        const stats_shape = zml.Shape.init(.{ q.dim(.b), config.num_segments, config.num_query_heads }, .f32);

        const partials = kernels_sparse.KernelSparseAttention3dPtr.Kernel.call(inputs, .{
            .output = .init(.{ q.dim(.b), config.num_query_heads, config.num_segments, head_size_padded }, .f32),
            .segm_max = stats_shape,
            .segm_expsum = stats_shape,
        }, .{
            .cfg = config,
            .grid = grid,
            .num_stages = 1,
            .num_warps = warps,
        });

        const head_tile = @min(64, head_size_padded);

        const output = kernels_sparse.SparseReducePtr.Kernel.call(.{
            .segm_output_ptr = partials.output,
            .segm_max_ptr = partials.segm_max,
            .segm_expsum_ptr = partials.segm_expsum,
            .query_start_len_ptr = parameters.query_start_len,
        }, .{ .output = q.shape() }, .{
            .cfg = .{
                .dtype = config.dtype,
                .num_seqs = config.num_seqs,
                .num_query_heads = config.num_query_heads,
                .head_size = config.head_size,
                .head_size_padded = head_size_padded,
                .head_tile = head_tile,
                .num_segments = config.num_segments,
                .launched_segments = partition.launched,
                .output_strides = config.query_strides,
            },
            .grid = .{ @intCast(q.dim(.b)), @intCast(config.num_query_heads), @intCast(@divExact(head_size_padded, head_tile)) },
            .num_stages = 1,
            .num_warps = 4,
        }).output;

        return output;
    }

    pub fn pagedAttention2d(parameters: Parameters, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, opts: AttentionOptions, paged_attention_opts: PagedAttentionOptions) zml.Tensor {
        const config = select2dConfig(paged_attention_opts);

        const kernel_config: kernels.KernelUnifiedAttention2dPtr.Config = .{
            .q_dtype = triton.from(q.dtype()),
            .kv_dtype = triton.from(k_cache.dtype()),
            .o_dtype = triton.from(q.dtype()),
            .num_query_heads = @intCast(paged_attention_opts.num_heads),
            .num_queries_per_kv = @intCast(paged_attention_opts.numQueriesPerKv()),
            .block_size = @intCast(paged_attention_opts.block_size),
            .tile_size = @intCast(config.tile_size),
            .head_size = @intCast(paged_attention_opts.head_dim),
            .head_size_padded = @intCast(std.math.ceilPowerOfTwoAssert(usize, paged_attention_opts.head_dim)),
            .use_alibi_slopes = false,
            .use_qq_bias = false,
            .use_softcap = false,
            .use_sinks = (opts.sink != null),
            .sliding_window = @intCast(paged_attention_opts.sliding_window),
            .block_q = @intCast(config.block_q),
            .block_m = @intCast(config.block_m),
            .use_fp8 = false,
            .all_decode = paged_attention_opts.all_decode,
            .is_causal = opts.mask.isCausal(),
        };
        log.debug("pagedAttention2d config: {any}", .{kernel_config});

        const dummy: zml.Tensor = .scalar(0, .i8);
        const block_table_strides = parameters.block_table.shape().computeElementStrides().constSlice();

        const q_shape = q.shape().mergeAxes(.{ .h = .{ .hkv, .hg } });
        const q_strides = q_shape.computeElementStrides().constSlice();
        const k_strides = k_cache.shape().computeElementStrides().constSlice();
        const v_strides = v_cache.shape().computeElementStrides().constSlice();

        const scale: f32 = paged_attention_opts.scale orelse @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(q.dim(.hd)))));
        const num_seqs = parameters.block_table.dim(0);

        const sink = if (opts.sink) |sink| sink.convert(.f32) else dummy;

        const output = kernels.KernelUnifiedAttention2dPtr.Kernel.call(
            .{
                .query_ptr = q,
                .key_cache_ptr = k_cache,
                .value_cache_ptr = v_cache,
                .sink_ptr = sink,
                .block_tables_ptr = parameters.block_table,
                .seq_lens_ptr = parameters.seq_lens,
                .alibi_slopes_ptr = dummy,
                .qq_bias_ptr = dummy,
                .scale_ptr = .scalar(scale, .f32),
                .k_scale_ptr = dummy,
                .v_scale_ptr = dummy,
                .out_scale_ptr = dummy,
                .softcap_ptr = dummy,
                .block_table_stride_ptr = .scalar(block_table_strides[0], .i64),
                .query_stride_0_ptr = .scalar(q_strides[0], .i64),
                .query_stride_1_ptr = .scalar(q_strides[1], .i64),
                .output_stride_0_ptr = .scalar(q_strides[0], .i64),
                .output_stride_1_ptr = .scalar(q_strides[1], .i64),
                .qq_bias_stride_0_ptr = dummy,
                .stride_k_cache_0_ptr = .scalar(k_strides[0], .i64),
                .stride_k_cache_1_ptr = .scalar(k_strides[1], .i64),
                .stride_k_cache_2_ptr = .scalar(k_strides[2], .i64),
                .stride_v_cache_0_ptr = .scalar(v_strides[0], .i64),
                .stride_v_cache_1_ptr = .scalar(v_strides[1], .i64),
                .stride_v_cache_2_ptr = .scalar(v_strides[2], .i64),
                .query_start_len_ptr = parameters.query_start_len,
                .num_seqs_ptr = .scalar(num_seqs, .i32),
            },
            .{ .output = q.shape() },
            .{
                .cfg = kernel_config,
                .grid = .{ @intCast(paged_attention_opts.num_kv_heads), @intCast(config.total_q_blocks), 1 },
                .num_stages = @intCast(config.num_stages),
                .num_warps = @intCast(config.num_warps),
            },
        );
        return output.output;
    }

    pub fn pagedAttention3d(parameters: Parameters, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, opts: AttentionOptions, paged_attention_opts: PagedAttentionOptions) zml.Tensor {
        const config = select3dConfig(paged_attention_opts);

        const head_size_padded: i64 = @intCast(std.math.ceilPowerOfTwoAssert(usize, paged_attention_opts.head_dim));
        const attn_kernel_config: kernels.KernelUnifiedAttention3dPtr.Config = .{
            .q_dtype = triton.from(q.dtype()),
            .kv_dtype = triton.from(k_cache.dtype()),
            .num_query_heads = @intCast(paged_attention_opts.num_heads),
            .num_queries_per_kv = @intCast(paged_attention_opts.numQueriesPerKv()),
            .block_size = @intCast(paged_attention_opts.block_size),
            .tile_size = @intCast(config.attention.tile_size),
            .head_size = @intCast(paged_attention_opts.head_dim),
            .head_size_padded = head_size_padded,
            .use_alibi_slopes = false,
            .use_qq_bias = false,
            .use_softcap = false,
            .use_sinks = (opts.sink != null),
            .sliding_window = @intCast(paged_attention_opts.sliding_window),
            .block_q = @intCast(config.attention.block_q),
            .block_m = @intCast(config.attention.block_m),
            .num_segments_per_seq = @intCast(config.attention.num_segments_per_seq),
            .all_decode = paged_attention_opts.all_decode,
            .is_causal = opts.mask.isCausal(),
        };
        log.debug("pagedAttention3d attention config: {any}", .{attn_kernel_config});

        const reduce_kernel_config: kernels.ReduceSegmentsPtr.Config = .{
            .o_dtype = triton.from(q.dtype()),
            .num_query_heads = @intCast(paged_attention_opts.num_heads),
            .tile_size = @intCast(config.reduce.tile_size),
            .head_size = @intCast(paged_attention_opts.head_dim),
            .head_size_padded = head_size_padded,
            .block_q = @intCast(config.reduce.block_q),
            .num_segments_per_seq = @intCast(config.reduce.num_segments_per_seq),
            .use_fp8 = false,
        };
        log.debug("pagedAttention3d reduce config: {any}", .{reduce_kernel_config});

        const dummy: zml.Tensor = .scalar(0, .i8);
        const block_table_strides = parameters.block_table.shape().computeElementStrides().constSlice();

        const q_shape = q.shape().mergeAxes(.{ .h = .{ .hkv, .hg } });
        const q_strides = q_shape.computeElementStrides().constSlice();
        const k_strides = k_cache.shape().computeElementStrides().constSlice();
        const v_strides = v_cache.shape().computeElementStrides().constSlice();

        const scale: f32 = paged_attention_opts.scale orelse @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(q.dim(.hd)))));
        const num_seqs = parameters.block_table.dim(0);

        const attn_grid: [3]i32 = .{
            @intCast(config.attention.total_q_blocks),
            @intCast(paged_attention_opts.num_kv_heads),
            @intCast(config.attention.num_segments_per_seq),
        };

        const sink = if (opts.sink) |sink| sink.convert(.f32) else dummy;

        const attn_output = kernels.KernelUnifiedAttention3dPtr.Kernel.call(
            .{
                .query_ptr = q,
                .key_cache_ptr = k_cache,
                .value_cache_ptr = v_cache,
                .sink_ptr = sink,
                .block_tables_ptr = parameters.block_table,
                .seq_lens_ptr = parameters.seq_lens,
                .alibi_slopes_ptr = dummy,
                .qq_bias_ptr = dummy,
                .scale_ptr = .scalar(scale, .f32),
                .k_scale_ptr = dummy,
                .v_scale_ptr = dummy,
                .softcap_ptr = dummy,
                .block_table_stride_ptr = .scalar(block_table_strides[0], .i64),
                .query_stride_0_ptr = .scalar(q_strides[0], .i64),
                .query_stride_1_ptr = .scalar(q_strides[1], .i64),
                .qq_bias_stride_0_ptr = dummy,
                .stride_k_cache_0_ptr = .scalar(k_strides[0], .i64),
                .stride_k_cache_1_ptr = .scalar(k_strides[1], .i64),
                .stride_k_cache_2_ptr = .scalar(k_strides[2], .i64),
                .stride_v_cache_0_ptr = .scalar(v_strides[0], .i64),
                .stride_v_cache_1_ptr = .scalar(v_strides[1], .i64),
                .stride_v_cache_2_ptr = .scalar(v_strides[2], .i64),
                .query_start_len_ptr = parameters.query_start_len,
                .num_seqs_ptr = .scalar(num_seqs, .i32),
            },
            .{
                .segm_output = zml.Shape.init(.{ paged_attention_opts.num_tokens, paged_attention_opts.num_heads, config.attention.num_segments_per_seq, std.math.ceilPowerOfTwoAssert(usize, paged_attention_opts.head_dim) }, .f32),
                .segm_max = zml.Shape.init(.{ paged_attention_opts.num_tokens, paged_attention_opts.num_heads, config.attention.num_segments_per_seq }, .f32),
                .segm_expsum = zml.Shape.init(.{ paged_attention_opts.num_tokens, paged_attention_opts.num_heads, config.attention.num_segments_per_seq }, .f32),
            },
            .{
                .cfg = attn_kernel_config,
                .grid = attn_grid,
                .num_stages = @intCast(config.attention.num_stages),
                .num_warps = @intCast(config.attention.num_warps),
            },
        );

        const output = kernels.ReduceSegmentsPtr.Kernel.call(
            .{
                .segm_output_ptr = attn_output.segm_output,
                .segm_max_ptr = attn_output.segm_max,
                .segm_expsum_ptr = attn_output.segm_expsum,
                .seq_lens_ptr = parameters.seq_lens,
                .num_seqs_ptr = .scalar(num_seqs, .i32),
                .out_scale_inv_ptr = dummy,
                .output_stride_0_ptr = .scalar(q_strides[0], .i64),
                .output_stride_1_ptr = .scalar(q_strides[1], .i64),
                .block_table_stride_ptr = .scalar(block_table_strides[0], .i64),
                .query_start_len_ptr = parameters.query_start_len,
            },
            .{ .output = q.shape() },
            .{
                .cfg = reduce_kernel_config,
                .grid = .{
                    @intCast(paged_attention_opts.num_tokens),
                    @intCast(paged_attention_opts.num_heads),
                    1,
                },
                .num_stages = @intCast(config.reduce.num_stages),
                .num_warps = @intCast(config.reduce.num_warps),
            },
        );

        return output.output;
    }

    /// oneAPI/Intel specialization of pagedAttention3d: routes the attention pass to
    /// the SIMD16-tuned kernel (compile-time KV strides, cached loads, same-page fast
    /// path) while reusing the shared segment-reduce kernel. See unified_attention_oneapi.zig.
    pub fn pagedAttention3dOneapi(parameters: Parameters, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, opts: AttentionOptions, paged_attention_opts: PagedAttentionOptions) zml.Tensor {
        const config = select3dConfig(paged_attention_opts);

        const head_size_padded: i64 = @intCast(std.math.ceilPowerOfTwoAssert(usize, paged_attention_opts.head_dim));

        const k_strides = k_cache.shape().computeElementStrides().constSlice();
        const v_strides = v_cache.shape().computeElementStrides().constSlice();

        const attn_kernel_config: kernels_oneapi.KernelUnifiedAttention3dPtr.Config = .{
            .q_dtype = triton.from(q.dtype()),
            .kv_dtype = triton.from(k_cache.dtype()),
            .num_query_heads = @intCast(paged_attention_opts.num_heads),
            .num_queries_per_kv = @intCast(paged_attention_opts.numQueriesPerKv()),
            .block_size = @intCast(paged_attention_opts.block_size),
            .tile_size = @intCast(config.attention.tile_size),
            .head_size = @intCast(paged_attention_opts.head_dim),
            .head_size_padded = head_size_padded,
            .use_alibi_slopes = false,
            .use_qq_bias = false,
            .use_softcap = false,
            .use_sinks = (opts.sink != null),
            .sliding_window = @intCast(paged_attention_opts.sliding_window),
            .is_causal = opts.mask.isCausal(),
            .block_q = @intCast(config.attention.block_q),
            .block_m = @intCast(config.attention.block_m),
            .num_segments_per_seq = @intCast(config.attention.num_segments_per_seq),
            .all_decode = paged_attention_opts.all_decode,
            .stride_k_cache_0 = k_strides[k_cache.axis(.page)],
            .stride_k_cache_1 = k_strides[k_cache.axis(.k_chunk)],
            .stride_k_cache_2 = k_strides[k_cache.axis(.hkv)],
            .stride_v_cache_0 = v_strides[v_cache.axis(.page)],
            .stride_v_cache_1 = v_strides[v_cache.axis(.k_chunk)],
            .stride_v_cache_2 = v_strides[v_cache.axis(.hkv)],
            // Intel: keep KV loads cached (.none); the streaming .cg hint lowers to
            // fully-uncached here and defeats the L2 prefetcher.
            .kv_cache_modifier = .none,
        };
        log.debug("pagedAttention3dOneapi attention config: {any}", .{attn_kernel_config});

        const reduce_kernel_config: kernels.ReduceSegmentsPtr.Config = .{
            .o_dtype = triton.from(q.dtype()),
            .num_query_heads = @intCast(paged_attention_opts.num_heads),
            .tile_size = @intCast(config.reduce.tile_size),
            .head_size = @intCast(paged_attention_opts.head_dim),
            .head_size_padded = head_size_padded,
            .block_q = @intCast(config.reduce.block_q),
            .num_segments_per_seq = @intCast(config.reduce.num_segments_per_seq),
            .use_fp8 = false,
        };
        log.debug("pagedAttention3d reduce config: {any}", .{reduce_kernel_config});

        const dummy: zml.Tensor = .scalar(0, .i8);
        const block_table_strides = parameters.block_table.shape().computeElementStrides().constSlice();

        const q_shape = q.shape().mergeAxes(.{ .h = .{ .hkv, .hg } });
        const q_strides = q_shape.computeElementStrides().constSlice();

        const scale: f32 = paged_attention_opts.scale orelse @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(q.dim(.hd)))));
        const num_seqs = parameters.block_table.dim(0);

        const attn_grid: [3]i32 = .{ @intCast(config.attention.total_q_blocks), @intCast(paged_attention_opts.num_kv_heads), @intCast(config.attention.num_segments_per_seq) };

        const sink = if (opts.sink) |sink| sink.convert(.f32) else dummy;

        const attn_output = kernels_oneapi.KernelUnifiedAttention3dPtr.Kernel.call(
            .{
                .query_ptr = q,
                .key_cache_ptr = k_cache,
                .value_cache_ptr = v_cache,
                .sink_ptr = sink,
                .block_tables_ptr = parameters.block_table,
                .seq_lens_ptr = parameters.seq_lens,
                .alibi_slopes_ptr = dummy,
                .qq_bias_ptr = dummy,
                .scale_ptr = .scalar(scale, .f32),
                .k_scale_ptr = dummy,
                .v_scale_ptr = dummy,
                .softcap_ptr = dummy,
                .block_table_stride_ptr = .scalar(block_table_strides[0], .i64),
                .query_stride_0_ptr = .scalar(q_strides[0], .i64),
                .query_stride_1_ptr = .scalar(q_strides[1], .i64),
                .qq_bias_stride_0_ptr = dummy,
                .stride_k_cache_0_ptr = .scalar(k_strides[k_cache.axis(.page)], .i64),
                .stride_k_cache_1_ptr = .scalar(k_strides[k_cache.axis(.k_chunk)], .i64),
                .stride_k_cache_2_ptr = .scalar(k_strides[k_cache.axis(.hkv)], .i64),
                .stride_v_cache_0_ptr = .scalar(v_strides[v_cache.axis(.page)], .i64),
                .stride_v_cache_1_ptr = .scalar(v_strides[v_cache.axis(.k_chunk)], .i64),
                .stride_v_cache_2_ptr = .scalar(v_strides[v_cache.axis(.hkv)], .i64),
                .query_start_len_ptr = parameters.query_start_len,
                .num_seqs_ptr = .scalar(num_seqs, .i32),
            },
            .{
                .segm_output = zml.Shape.init(.{ paged_attention_opts.num_tokens, paged_attention_opts.num_heads, config.attention.num_segments_per_seq, std.math.ceilPowerOfTwoAssert(usize, paged_attention_opts.head_dim) }, .f32),
                .segm_max = zml.Shape.init(.{ paged_attention_opts.num_tokens, paged_attention_opts.num_heads, config.attention.num_segments_per_seq }, .f32),
                .segm_expsum = zml.Shape.init(.{ paged_attention_opts.num_tokens, paged_attention_opts.num_heads, config.attention.num_segments_per_seq }, .f32),
            },
            .{
                .cfg = attn_kernel_config,
                .grid = attn_grid,
                .num_stages = @intCast(config.attention.num_stages),
                .num_warps = @intCast(config.attention.num_warps),
            },
        );

        const output = kernels.ReduceSegmentsPtr.Kernel.call(
            .{
                .segm_output_ptr = attn_output.segm_output,
                .segm_max_ptr = attn_output.segm_max,
                .segm_expsum_ptr = attn_output.segm_expsum,
                .seq_lens_ptr = parameters.seq_lens,
                .num_seqs_ptr = .scalar(num_seqs, .i32),
                .out_scale_inv_ptr = dummy,
                .output_stride_0_ptr = .scalar(q_strides[0], .i64),
                .output_stride_1_ptr = .scalar(q_strides[1], .i64),
                .block_table_stride_ptr = .scalar(block_table_strides[0], .i64),
                .query_start_len_ptr = parameters.query_start_len,
            },
            .{ .output = q.shape() },
            .{
                .cfg = reduce_kernel_config,
                .grid = .{ @intCast(paged_attention_opts.num_tokens), @intCast(paged_attention_opts.num_heads), 1 },
                .num_stages = @intCast(config.reduce.num_stages),
                .num_warps = @intCast(config.reduce.num_warps),
            },
        );

        return output.output;
    }
};
