const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const triton = zml.kernel.triton;
const AttentionOptions = @import("paged_attention.zig").AttentionOptions;
const MlaOptions = @import("paged_attention.zig").Mla.Options;
const mha_kernels = @import("triton_kernels/mha.zig");
const unified_kernels = @import("triton_kernels/unified_attention.zig");
const kernels_oneapi = @import("triton_kernels/unified_attention_oneapi.zig");
const mla_kernels = @import("triton_kernels/unified_sparse_mla.zig");

const log = std.log.scoped(.@"zml/attention/triton");

fn isOneapiTarget() bool {
    return zml.module.CompilationContext.current().platform.target == .oneapi;
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

fn select2dConfig(options: paged.PagedAttentionOptions) Config2D {
    const max_num_stages_2d: usize = if (options.head_dim <= 128) 4 else 2;

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

fn getCuCount() usize {
    const platform = zml.module.CompilationContext.current().platform;

    return @intCast(platform.devices[0].pjrt_desc.attribute(platform.pjrt_api, "core_count").?.int64);
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
        const output = zml.ops.manualComputation(
            .{
                q,
                k_cache,
                v_cache,
                parameters.block_table,
                parameters.seq_lens,
                parameters.query_start_len,
            },
            q.shape(),
            .{
                .opts = opts,
                .options = parameters.options_,
            },
            (struct {
                fn body(ctx_: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, _: zml.Shape) zml.Tensor {
                    const q_ = sharded_inputs[0];
                    const k_cache_ = sharded_inputs[1];
                    const v_cache_ = sharded_inputs[2];
                    const parameters_: Parameters = .{ .block_table = sharded_inputs[3], .seq_lens = sharded_inputs[4], .query_start_len = sharded_inputs[5], .options_ = ctx_.options };

                    const cu_count = getCuCount();
                    const num_heads: usize = @intCast(q_.dim(.hkv) * q_.dim(.hg));
                    const num_kv_heads: usize = @intCast(k_cache_.dim(.hkv));
                    const num_queries_per_kv: usize = num_heads / num_kv_heads;
                    // Intel decode: pack exactly one GQA group per tile (block_q == 1) so the
                    // single decode query token doesn't carry masked-out fp32 acc lanes.
                    // oneAPI decode keeps one GQA group per tile, padded to a power of two so tt.make_range emits legal Triton IR.
                    const block_m: usize = if (!ctx_.options.is_prefill and isOneapiTarget())
                        std.math.ceilPowerOfTwoAssert(usize, num_queries_per_kv)
                    else if (num_queries_per_kv <= 16) 16 else std.math.ceilPowerOfTwoAssert(usize, num_queries_per_kv);
                    const block_q: usize = block_m / num_queries_per_kv;
                    const num_tokens: usize = @intCast(q_.dim(.b));
                    const num_seqs: usize = @intCast(parameters_.block_table.dim(.b));
                    const total_q_blocks: usize = num_tokens / block_q + num_seqs;
                    const target_num_prgms: usize = cu_count * 4;
                    const num_2d_prgms: usize = total_q_blocks * num_kv_heads;

                    const paged_attention_opts: PagedAttentionOptions = .{
                        .cu_count = getCuCount(),
                        .all_decode = !ctx_.options.is_prefill,
                        .num_tokens = num_tokens,
                        .num_heads = num_heads,
                        .num_kv_heads = num_kv_heads,
                        .head_dim = @intCast(q_.dim(.hd)),
                        .batch_size = @intCast(parameters_.block_table.dim(.b)),
                        .block_size = @intCast(k_cache_.dim(.k_chunk)),
                        .num_blocks = @intCast(k_cache_.dim(.page)),
                        .max_num_block_per_seq = @intCast(parameters_.block_table.dim(.p)),
                        .sliding_window = if (ctx_.opts.sliding_window < 0) 0 else @intCast(ctx_.opts.sliding_window),
                        .block_m = block_m,
                        .block_q = block_q,
                        .total_q_blocks = total_q_blocks,
                        .target_num_prgms = target_num_prgms,
                        .num_2d_prgms = num_2d_prgms,
                        .max_seqlen_q = ctx_.options.max_seqlen_q,
                        .scale = ctx_.opts.scale,
                    };

                    const use_2d_kernel = use2dKernel(
                        paged_attention_opts.all_decode,
                        paged_attention_opts.batch_size,
                        paged_attention_opts.num_kv_heads,
                    );
                    const output = if (use_2d_kernel)
                        pagedAttention2d(parameters_, q_, k_cache_, v_cache_, ctx_.opts, paged_attention_opts)
                    else if (isOneapiTarget())
                        pagedAttention3dOneapi(parameters_, q_, k_cache_, v_cache_, ctx_.opts, paged_attention_opts)
                    else
                        pagedAttention3d(parameters_, q_, k_cache_, v_cache_, ctx_.opts, paged_attention_opts);

                    return output;
                }
            }).body,
        );

        return output;
    }

    pub fn pagedAttention2d(parameters: Parameters, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, opts: AttentionOptions, paged_attention_opts: PagedAttentionOptions) zml.Tensor {
        const config = select2dConfig(paged_attention_opts);

        const kernel_config: unified_kernels.KernelUnifiedAttention2dPtr.Config = .{
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
            .use_sinks = false,
            .sliding_window = @intCast(paged_attention_opts.sliding_window),
            .block_q = @intCast(config.block_q),
            .block_m = @intCast(config.block_m),
            .use_fp8 = false,
            .all_decode = paged_attention_opts.all_decode,
            .is_causal = opts.is_causal,
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

        const output = unified_kernels.KernelUnifiedAttention2dPtr.Kernel.call(
            .{
                .query_ptr = q,
                .key_cache_ptr = k_cache,
                .value_cache_ptr = v_cache,
                .sink_ptr = dummy,
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
        const attn_kernel_config: unified_kernels.KernelUnifiedAttention3dPtr.Config = .{
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
            .use_sinks = false,
            .sliding_window = @intCast(paged_attention_opts.sliding_window),
            .block_q = @intCast(config.attention.block_q),
            .block_m = @intCast(config.attention.block_m),
            .num_segments_per_seq = @intCast(config.attention.num_segments_per_seq),
            .all_decode = paged_attention_opts.all_decode,
            .is_causal = opts.is_causal,
        };
        log.debug("pagedAttention3d attention config: {any}", .{attn_kernel_config});

        const reduce_kernel_config: unified_kernels.ReduceSegmentsPtr.Config = .{
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
        const attn_output = unified_kernels.KernelUnifiedAttention3dPtr.Kernel.call(
            .{
                .query_ptr = q,
                .key_cache_ptr = k_cache,
                .value_cache_ptr = v_cache,
                .sink_ptr = dummy,
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

        const output = unified_kernels.ReduceSegmentsPtr.Kernel.call(
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
        _ = opts;

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
            .use_sinks = false,
            .sliding_window = @intCast(paged_attention_opts.sliding_window),
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

        const reduce_kernel_config: unified_kernels.ReduceSegmentsPtr.Config = .{
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
        const attn_output = kernels_oneapi.KernelUnifiedAttention3dPtr.Kernel.call(
            .{
                .query_ptr = q,
                .key_cache_ptr = k_cache,
                .value_cache_ptr = v_cache,
                .sink_ptr = dummy,
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

        const output = unified_kernels.ReduceSegmentsPtr.Kernel.call(
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

    fn pagedSparseMla2d(q: zml.Tensor, kv_cache: zml.Tensor, sink: ?zml.Tensor, topk: zml.Tensor, parameters: Parameters, paged_opts: PagedSparseMlaOptions) zml.Tensor {
        const rope_rank: i64 = @intCast(paged_opts.rope_rank);
        const nope_rank: i64 = @intCast(paged_opts.head_dim - paged_opts.rope_rank);
        const kv_lora_rank: i64 = @intCast(std.math.ceilPowerOfTwoAssert(usize, @intCast(nope_rank)));

        const out_shape = q.shape();

        const q_strides = q.shape().computeElementStrides().constSlice();
        const out_strides = out_shape.computeElementStrides().constSlice();
        const kv_strides = kv_cache.shape().computeElementStrides().constSlice();

        const sink_: zml.Tensor = sink orelse .scalar(0, .i8);
        const sm_scale: f32 = paged_opts.scale orelse @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(paged_opts.head_dim))));

        const config: Config2D = .{
            .block_m = paged_opts.block_m,
            .block_q = 1,
            .total_q_blocks = paged_opts.total_q_blocks,
            .num_warps = 4,
            .num_stages = 2,
            .tile_size = @min(@as(usize, @intCast(topk.dim(.topk))), 16),
        };

        const attn_grid: [3]i32 = .{ @intCast(config.total_q_blocks * @divExact(paged_opts.num_heads, paged_opts.block_m)), 1, 1 };

        const kernel_config: mla_kernels.Kernel.Config = .{
            .q_dtype = triton.from(q.dtype()),
            .kv_dtype = triton.from(kv_cache.dtype()),
            .sink_dtype = triton.from(sink_.dtype()),
            .o_dtype = triton.from(q.dtype()),
            .num_query_heads = @intCast(paged_opts.num_heads),
            .num_queries_per_kv = @intCast(paged_opts.num_heads),
            .block_size = @intCast(paged_opts.block_size),
            .block_m = @intCast(paged_opts.block_m),
            .topk_count = topk.dim(.topk),
            .rope_rank = rope_rank,
            .qk_lora_rank = nope_rank,
            .kv_lora_rank = kv_lora_rank,
            .rope_offset = nope_rank,
            .value_rank = @intCast(paged_opts.head_dim),
            .tile_size = @intCast(config.tile_size),
            .use_attn_sink = if (sink) |_| true else false,
            .all_decode = !parameters.options_.is_prefill,
        };
        log.debug("pagedSparseMla2d config: {any}", .{kernel_config});

        const out = mla_kernels.Kernel.call(.{
            .query_ptr = q,
            .key_cache_ptr = kv_cache,
            .value_cache_ptr = kv_cache,
            .attn_sink_ptr = sink_,
            .block_tables_ptr = parameters.block_table,
            .topk_indices_ptr = topk,
            .seq_lens_ptr = parameters.seq_lens,
            .scale_ptr = zml.Tensor.constant(zml.DataType.f32.constant(sm_scale)),
            .block_table_stride_ptr = zml.Tensor.constant(zml.DataType.i64.constant(parameters.block_table.dim(.p))),
            .query_stride_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(q_strides[0])),
            .query_stride_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(q_strides[1])),
            .output_stride_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(out_strides[0])),
            .output_stride_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(out_strides[1])),
            .stride_k_cache_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(kv_strides[kv_cache.shape().axis(.page)])),
            .stride_k_cache_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(kv_strides[kv_cache.shape().axis(.k_chunk)])),
            .stride_k_cache_2_ptr = zml.Tensor.constant(zml.DataType.i64.constant(kv_strides[kv_cache.shape().axis(.hkv)])),
            .stride_v_cache_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(kv_strides[kv_cache.shape().axis(.page)])),
            .stride_v_cache_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(kv_strides[kv_cache.shape().axis(.k_chunk)])),
            .stride_v_cache_2_ptr = zml.Tensor.constant(zml.DataType.i64.constant(kv_strides[kv_cache.shape().axis(.hkv)])),
            .query_start_len_ptr = parameters.query_start_len,
            .num_seqs_ptr = zml.Tensor.constant(.{ .i32 = @as(i32, @intCast(parameters.block_table.dim(.b))) }).reshape(.{1}),
        }, .{
            .output = out_shape,
        }, .{
            .cfg = kernel_config,
            .grid = attn_grid,
            .num_warps = @intCast(config.num_warps),
            .num_stages = @intCast(config.num_stages),
        });

        return out.output.reshape(q.shape());
    }

    fn tokenToSequence(query_start_len: zml.Tensor, query_count: i64) zml.Tensor {
        const sequence_count = query_start_len.dim(.b) - 1;
        const starts = query_start_len.slice1d(.b, .{ .end = sequence_count }).rename(.{ .b = .seq });
        const ends = query_start_len.slice1d(.b, .{ .start = 1 }).rename(.{ .b = .seq });
        const query_x_sequence_shape = zml.Shape.init(.{ .q = query_count, .seq = sequence_count }, .i32);
        const query = zml.Tensor.iota(query_x_sequence_shape, .q).convert(.i32);
        const in_range = query.cmp(.GE, starts.broad(query_x_sequence_shape)).convert(.i32)
            .mul(query.cmp(.LT, ends.broad(query_x_sequence_shape)).convert(.i32))
            .cmp(.NE, zml.Tensor.zeroes(query_x_sequence_shape));
        const sequence = zml.Tensor.iota(query_x_sequence_shape, .seq).convert(.i32);
        return zml.Tensor.select(in_range, sequence, zml.Tensor.zeroes(query_x_sequence_shape)).sum(.seq).squeeze(.seq);
    }

    pub fn topkToPhysical(parameters: anytype, topk: zml.Tensor, tokens_pos: zml.Tensor, block_size: i64) zml.Tensor {
        const topk_i32 = topk.convert(.i32);
        const topk_shape = topk_i32.shape();

        stdx.debug.assert(topk_shape.hasTags(.{ .q, .topk }), "paged MLA topk must have .q and .topk axes, got {f}", .{topk_shape});
        stdx.debug.assert(tokens_pos.shape().hasTags(.{.q}), "paged MLA token positions must have a .q axis, got {f}", .{tokens_pos.shape()});

        const query_to_sequence = tokenToSequence(parameters.query_start_len, topk_i32.dim(.q));
        const sequence_ends = parameters.query_start_len.slice1d(.b, .{ .start = 1 }).rename(.{ .b = .seq });
        const last_query = sequence_ends.gather(.{ .seq = query_to_sequence }, .{}).sub(.scalar(1, .i32));
        const last_token_pos = tokens_pos
            .gather(.{ .q = last_query.rename(.{ .q = .lookup }) }, .{})
            .rename(.{ .lookup = .q })
            .convert(.i32);
        const seq_lens = parameters.seq_lens.rename(.{ .b = .seq }).gather(.{ .seq = query_to_sequence }, .{});
        const first_visible_token = last_token_pos.addConstant(1).sub(seq_lens);
        const relative_topk = topk_i32.sub(first_visible_token.broad(topk_shape));

        const block_size_scalar = zml.Tensor.scalar(@as(i32, @intCast(block_size)), .i32).broad(topk_shape);
        const zero = zml.Tensor.zeroes(topk_shape);
        const valid_nonnegative = relative_topk.cmp(.GE, zero);
        const valid_in_sequence = relative_topk.cmp(.LT, seq_lens.broad(topk_shape));
        const valid_topk = valid_nonnegative.logical(.AND, valid_in_sequence);

        const safe_topk = zml.Tensor.select(valid_topk, relative_topk, zero);
        const logical_block = safe_topk.div(block_size_scalar);
        const slot = safe_topk.remainder(block_size_scalar);

        const sequence = query_to_sequence.broad(topk_shape);
        const block_table = parameters.block_table.rename(.{ .b = .seq });
        const physical_block = block_table.gather(.{ .seq = sequence, .p = logical_block }, .{});

        const physical_topk = physical_block.mul(block_size_scalar).add(slot);

        return zml.Tensor.select(valid_topk, physical_topk, zml.Tensor.scalar(-1, .i32).broad(topk_shape));
    }

    pub const PagedSparseMlaOptions = struct {
        head_dim: usize,
        num_heads: usize,
        block_size: usize,
        block_m: usize,
        rope_rank: usize,
        scale: ?f32,
        total_q_blocks: usize,
    };

    pub fn pagedSparseMla(parameters: Parameters, q: zml.Tensor, kv_cache: zml.Tensor, sink: ?zml.Tensor, topk: zml.Tensor, tokens_pos: zml.Tensor, opts: MlaOptions) zml.Tensor {
        return zml.ops.manualComputation(
            .{
                q,
                kv_cache,
                sink,
                topk,
                tokens_pos,
                parameters.block_table,
                parameters.seq_lens,
                parameters.query_start_len,
            },
            q.shape(),
            .{
                .opts = opts,
                .options = parameters.options_,
            },
            (struct {
                fn body(ctx_: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, _: zml.Shape) zml.Tensor {
                    const q_ = sharded_inputs[0];
                    const kv_cache_ = sharded_inputs[1];

                    const block_size = kv_cache_.dim(.k_chunk);

                    const parameters_: Parameters = .{
                        .block_table = sharded_inputs[5],
                        .seq_lens = sharded_inputs[6],
                        .query_start_len = sharded_inputs[7],
                        .options_ = ctx_.options,
                    };

                    const topk_final = topkToPhysical(parameters_, sharded_inputs[3], sharded_inputs[4], block_size);
                    stdx.debug.assert(topk_final.dim(.q) == q_.dim(.q), "expected topk q dim ({}) to match q dim ({})", .{ topk_final.dim(.q), q_.dim(.q) });

                    const num_heads: usize = @intCast(q_.dim(.h));
                    const block_m: usize = if (num_heads <= 16) 16 else std.math.ceilPowerOfTwoAssert(usize, num_heads);
                    stdx.debug.assert(@mod(num_heads, block_m) == 0, "expected q heads ({}) to be divisible by block_m ({})", .{ num_heads, block_m });

                    const paged_opts: PagedSparseMlaOptions = .{
                        .head_dim = @intCast(q_.dim(.hd)),
                        .num_heads = num_heads,
                        .block_size = @intCast(kv_cache_.dim(.k_chunk)),
                        .block_m = block_m,
                        .rope_rank = @intCast(ctx_.opts.rope_rank),
                        .scale = ctx_.opts.scale,
                        .total_q_blocks = @intCast(q_.dim(.q)),
                    };

                    return pagedSparseMla2d(
                        q_,
                        kv_cache_,
                        sharded_inputs[2],
                        topk_final,
                        parameters_,
                        paged_opts,
                    );
                }
            }).body,
        );
    }
};

pub const flashattn = struct {
    fn scalarI64(value: i64) zml.Tensor {
        return zml.Tensor.constant(zml.DataType.i64.constant(value));
    }

    fn strideFor(t: zml.Tensor, comptime tag: @EnumLiteral()) i64 {
        const strides = t.shape().computeElementStrides().constSlice();
        if (t.shape().hasTag(tag)) |axis| return strides[axis];
        return 0;
    }

    fn blockSizeM(seqlen_q: i64) i64 {
        return if (seqlen_q == 1) 16 else 64;
    }

    pub const Parameters = struct {
        pub const InitOptions = struct {};

        pub fn init(opts: InitOptions) Parameters {
            _ = opts;
            return .{};
        }
    };

    pub const Metadata = struct {
        pub const InitOptions = struct {};

        pub fn init(opts: InitOptions) Metadata {
            _ = opts;
            return .{};
        }
    };

    pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, metadata: Metadata, parameters: Parameters) zml.Tensor {
        _ = metadata;
        _ = parameters;

        stdx.debug.assert(q.shape().hasTags(.{ .q, .h, .hd }), "triton.flashattn expects q to have tags .q, .h, .hd, got {f}", .{q.shape()});
        stdx.debug.assert(k.shape().hasTags(.{ .k, .h, .hd }), "triton.flashattn expects k to have tags .k, .h, .hd, got {f}", .{k.shape()});
        stdx.debug.assert(v.shape().hasTags(.{ .k, .h, .hd }), "triton.flashattn expects v to have tags .k, .h, .hd, got {f}", .{v.shape()});

        const q_sharded = q.withPartitioning(.{ .h = .model });
        const k_sharded = k.withPartitioning(.{ .h = .model });
        const v_sharded = v.withPartitioning(.{ .h = .model });

        return zml.ops.manualComputation(
            .{ q_sharded, k_sharded, v_sharded, token_index },
            q_sharded.shape(),
            {},
            (struct {
                fn body(_: void, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, _: zml.Shape) zml.Tensor {
                    stdx.debug.assert(sharded_inputs.len == 4, "triton.flashattn manualComputation expects 4 inputs, got {}", .{sharded_inputs.len});

                    const q_ = sharded_inputs[0];
                    const k_ = sharded_inputs[1];
                    const v_ = sharded_inputs[2];

                    const bs: i64 = if (q_.shape().hasTag(.b)) |_| q_.dim(.b) else 1;
                    const seqlen_q = q_.dim(.q);
                    const seqlen_k = k_.dim(.k);
                    const num_q_heads = q_.dim(.h);
                    const num_kv_heads = k_.dim(.h);
                    const head_dim = q_.dim(.hd);
                    const head_dim_pow2: i64 = @intCast(std.math.ceilPowerOfTwoAssert(usize, @intCast(head_dim)));
                    const block_m = blockSizeM(seqlen_q);
                    const block_n: i64 = 64;
                    const num_m_blocks = std.math.divCeil(i64, seqlen_q, block_m) catch unreachable;

                    // We still have a rectangle layout, q and k haven't been compacted.
                    // So each sequence have the same number of queries, keys
                    const cu_seqlens_q: zml.Tensor = .arange(.{ .end = seqlen_q * (bs + 1), .step = seqlen_q }, .i32);
                    const cu_seqlens_k: zml.Tensor = .arange(.{ .end = seqlen_k * (bs + 1), .step = seqlen_k }, .i32);

                    const softmax_lse = zml.Tensor.uninitialized(zml.Shape.init(.{
                        .h = num_q_heads,
                        .q = seqlen_q,
                    }, .f32));
                    const alibi_slopes = zml.Tensor.zeroes(zml.Shape.init(.{ .h = num_q_heads }, .f32));
                    const dummy = zml.Tensor.constant(zml.DataType.f32.zero());

                    const sm_scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(head_dim))));
                    const sm_scale_ptr = zml.Tensor.constant(zml.DataType.f32.constant(sm_scale));
                    const kernel_config: mha_kernels.MhaFwd.Config = .{
                        .q_dtype = triton.from(q_.dtype()),
                        .kv_dtype = triton.from(k_.dtype()),
                        .out_dtype = triton.from(q_.dtype()),
                        .SEQLEN_Q = seqlen_q,
                        .SEQLEN_K = seqlen_k,
                        .IS_CAUSAL = true,
                        .NUM_Q_HEADS = num_q_heads,
                        .NUM_K_HEADS = num_kv_heads,
                        .PRELOAD_V = false,
                        .BLOCK_M = block_m,
                        .BLOCK_N = block_n,
                        .BLOCK_DMODEL = head_dim,
                        .BLOCK_DMODEL_POW2 = head_dim_pow2,
                        .BLOCK_DMODEL_PE = 0,
                        .IS_FP8 = false,
                        .VARLEN = true,
                        .BATCH = 1,
                        .NUM_XCD = 8,
                        .USE_INT64_STRIDES = true,
                        .ENABLE_SINK = false,
                        .SLIDING_WINDOW = 0,
                        .HEAD_STRIDE_ALIGNED_8 = @mod(strideFor(q_, .h), 8) == 0,
                    };
                    log.debug("flashattn config: {any}", .{kernel_config});

                    const output = mha_kernels.MhaFwd.Kernel.call(
                        .{
                            .q_ptr = q_,
                            .k_ptr = k_,
                            .v_ptr = v_,
                            .descale_q_ptr = dummy,
                            .descale_k_ptr = dummy,
                            .descale_v_ptr = dummy,
                            .alibi_slopes_ptr = alibi_slopes,
                            .softmax_lse_ptr = softmax_lse,
                            .sink_ptr = dummy,
                            .stride_qz_in_ptr = scalarI64(strideFor(q_, .b)),
                            .stride_qh_in_ptr = scalarI64(strideFor(q_, .h)),
                            .stride_qm_in_ptr = scalarI64(strideFor(q_, .q)),
                            .stride_qk_in_ptr = scalarI64(strideFor(q_, .hd)),
                            .stride_kz_in_ptr = scalarI64(strideFor(k_, .b)),
                            .stride_kh_in_ptr = scalarI64(strideFor(k_, .h)),
                            .stride_kn_in_ptr = scalarI64(strideFor(k_, .k)),
                            .stride_kk_in_ptr = scalarI64(strideFor(k_, .hd)),
                            .stride_vz_in_ptr = scalarI64(strideFor(v_, .b)),
                            .stride_vh_in_ptr = scalarI64(strideFor(v_, .h)),
                            .stride_vn_in_ptr = scalarI64(strideFor(v_, .k)),
                            .stride_vk_in_ptr = scalarI64(strideFor(v_, .hd)),
                            .stride_descale_q_z_in_ptr = scalarI64(0),
                            .stride_descale_k_z_in_ptr = scalarI64(0),
                            .stride_descale_v_z_in_ptr = scalarI64(0),
                            .stride_oz_in_ptr = scalarI64(strideFor(q_, .b)),
                            .stride_oh_in_ptr = scalarI64(strideFor(q_, .h)),
                            .stride_om_in_ptr = scalarI64(strideFor(q_, .q)),
                            .stride_on_in_ptr = scalarI64(strideFor(q_, .hd)),
                            .stride_alibi_z_in_ptr = scalarI64(0),
                            .stride_alibi_h_in_ptr = scalarI64(strideFor(alibi_slopes, .h)),
                            .stride_lse_z_in_ptr = scalarI64(0),
                            .stride_lse_h_in_ptr = scalarI64(strideFor(softmax_lse, .h)),
                            .stride_lse_m_in_ptr = scalarI64(strideFor(softmax_lse, .q)),
                            .sm_scale_ptr = sm_scale_ptr,
                            .cu_seqlens_q = cu_seqlens_q,
                            .cu_seqlens_k = cu_seqlens_k,
                        },
                        .{ .out = q_.shape() },
                        .{
                            .cfg = kernel_config,
                            .grid = .{ @intCast(num_m_blocks * num_q_heads), 1, 1 },
                            .num_stages = 1,
                            .num_warps = 4,
                        },
                    );

                    return output.out;
                }
            }).body,
        );
    }
};
