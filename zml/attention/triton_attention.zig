const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const AttentionOptions = @import("paged_attention.zig").AttentionOptions;

const kernels = @import("triton_kernels.zig");
const tri = zml.kernel.triton;

const log = std.log.scoped(.@"zml/attention/triton");

const toDType = tri.from;

fn use2dKernel(head_size: usize, sliding_window: usize, all_decode: bool, max_seqlen_q: usize, max_seqlen_k: usize, target_num_prgms: usize, num_2d_prgms: usize) bool {
    _ = head_size;
    _ = all_decode;
    _ = max_seqlen_q;
    return sliding_window > 0 or max_seqlen_k <= 512 or num_2d_prgms > target_num_prgms;
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
    const attn_warps: usize = 2;
    const tile_size = options.block_size;

    //const MAX_SEGMENTS: usize = @min(128, std.math.divCeil(usize, max_seqlen_k, tile_size));
    var num_segments = std.math.divCeil(usize, options.target_num_prgms, options.num_2d_prgms) catch unreachable;
    num_segments = std.math.ceilPowerOfTwoAssert(usize, num_segments);
    num_segments = @min(num_segments, 128);
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

        pub fn onMemory(self: Parameters, memory: zml.platform.Memory.Kind) Parameters {
            return .{
                .options_ = self.options_,
                .block_table = self.block_table.onMemory(memory),
                .seq_lens = self.seq_lens.onMemory(memory),
                .query_start_len = self.query_start_len.onMemory(memory),
            };
        }

        pub fn toMemory(self: Parameters, memory: zml.platform.Memory.Kind) Parameters {
            return .{
                .options_ = self.options_,
                .block_table = self.block_table.toMemory(memory),
                .seq_lens = self.seq_lens.toMemory(memory),
                .query_start_len = self.query_start_len.toMemory(memory),
            };
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
                    const block_m: usize = if (num_queries_per_kv <= 16) 16 else std.math.ceilPowerOfTwoAssert(usize, num_queries_per_kv);
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
                        paged_attention_opts.head_dim,
                        paged_attention_opts.sliding_window,
                        paged_attention_opts.all_decode,
                        paged_attention_opts.max_seqlen_q,
                        paged_attention_opts.maxSeqLenK(),
                        paged_attention_opts.target_num_prgms,
                        paged_attention_opts.num_2d_prgms,
                    );
                    const output = if (use_2d_kernel)
                        pagedAttention2d(parameters_, q_, k_cache_, v_cache_, ctx_.opts, paged_attention_opts)
                    else
                        pagedAttention3d(parameters_, q_, k_cache_, v_cache_, ctx_.opts, paged_attention_opts);

                    return output;
                }
            }).body,
        );

        return output;
    }

    pub fn pagedAttention2d(parameters: Parameters, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, opts: AttentionOptions, paged_attention_opts: PagedAttentionOptions) zml.Tensor {
        _ = opts;
        const config = select2dConfig(paged_attention_opts);

        const kernel_config: kernels.KernelUnifiedAttention2dPtr.Config = .{
            .q_dtype = toDType(q.dtype()),
            .kv_dtype = toDType(k_cache.dtype()),
            .o_dtype = toDType(q.dtype()),
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
        };
        log.debug("pagedAttention2d config: {any}", .{kernel_config});

        const dummy = zml.Tensor.constant(zml.DataType.i8.zero());
        const block_table_strides = parameters.block_table.shape().computeElementStrides().constSlice();
        const block_table_strides_ptr = zml.Tensor.constant(zml.DataType.i64.constant(block_table_strides[0]));
        const q_shape = q.shape().mergeAxes(.{ .h = .{ .hkv, .hg } });
        const q_strides = q_shape.computeElementStrides().constSlice();
        const q_strides_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(q_strides[0]));
        const q_strides_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(q_strides[1]));
        const k_strides = k_cache.shape().computeElementStrides().constSlice();
        const k_strides_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[0]));
        const k_strides_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[1]));
        const k_strides_2_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[2]));
        const v_strides = v_cache.shape().computeElementStrides().constSlice();
        const v_strides_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[0]));
        const v_strides_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[1]));
        const v_strides_2_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[2]));
        const num_seqs_ptr = zml.Tensor.constant(zml.DataType.i32.constant(parameters.block_table.dim(0)));
        const scale: f32 = paged_attention_opts.scale orelse @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(q.dim(.hd)))));
        const scale_ptr = zml.Tensor.constant(zml.DataType.f32.constant(scale));

        const output = kernels.KernelUnifiedAttention2dPtr.Kernel.call(
            .{
                .query_ptr = q,
                .key_cache_ptr = k_cache,
                .value_cache_ptr = v_cache,
                .sink_ptr = dummy,
                .block_tables_ptr = parameters.block_table,
                .seq_lens_ptr = parameters.seq_lens,
                .alibi_slopes_ptr = dummy,
                .qq_bias_ptr = dummy,
                .scale_ptr = scale_ptr,
                .k_scale_ptr = dummy,
                .v_scale_ptr = dummy,
                .out_scale_ptr = dummy,
                .softcap_ptr = dummy,
                .block_table_stride_ptr = block_table_strides_ptr,
                .query_stride_0_ptr = q_strides_0_ptr,
                .query_stride_1_ptr = q_strides_1_ptr,
                .output_stride_0_ptr = q_strides_0_ptr,
                .output_stride_1_ptr = q_strides_1_ptr,
                .qq_bias_stride_0_ptr = dummy,
                .stride_k_cache_0_ptr = k_strides_0_ptr,
                .stride_k_cache_1_ptr = k_strides_1_ptr,
                .stride_k_cache_2_ptr = k_strides_2_ptr,
                .stride_v_cache_0_ptr = v_strides_0_ptr,
                .stride_v_cache_1_ptr = v_strides_1_ptr,
                .stride_v_cache_2_ptr = v_strides_2_ptr,
                .query_start_len_ptr = parameters.query_start_len,
                .num_seqs_ptr = num_seqs_ptr,
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
        _ = opts;

        const config = select3dConfig(paged_attention_opts);

        const head_size_padded: i64 = @intCast(std.math.ceilPowerOfTwoAssert(usize, paged_attention_opts.head_dim));
        const attn_kernel_config: kernels.KernelUnifiedAttention3dPtr.Config = .{
            .q_dtype = toDType(q.dtype()),
            .kv_dtype = toDType(k_cache.dtype()),
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
        };
        log.debug("pagedAttention3d attention config: {any}", .{attn_kernel_config});

        const reduce_kernel_config: kernels.ReduceSegmentsPtr.Config = .{
            .o_dtype = toDType(q.dtype()),
            .num_query_heads = @intCast(paged_attention_opts.num_heads),
            .tile_size = @intCast(config.reduce.tile_size),
            .head_size = @intCast(paged_attention_opts.head_dim),
            .head_size_padded = head_size_padded,
            .block_q = @intCast(config.reduce.block_q),
            .num_segments_per_seq = @intCast(config.reduce.num_segments_per_seq),
            .use_fp8 = false,
        };
        log.debug("pagedAttention3d reduce config: {any}", .{reduce_kernel_config});

        const dummy = zml.Tensor.constant(zml.DataType.i8.zero());
        const block_table_strides = parameters.block_table.shape().computeElementStrides().constSlice();
        const block_table_strides_ptr = zml.Tensor.constant(zml.DataType.i64.constant(block_table_strides[0]));
        const q_shape = q.shape().mergeAxes(.{ .h = .{ .hkv, .hg } });
        const q_strides = q_shape.computeElementStrides().constSlice();
        const q_strides_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(q_strides[0]));
        const q_strides_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(q_strides[1]));
        const k_strides = k_cache.shape().computeElementStrides().constSlice();
        const k_strides_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[0]));
        const k_strides_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[1]));
        const k_strides_2_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[2]));
        const v_strides = v_cache.shape().computeElementStrides().constSlice();
        const v_strides_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[0]));
        const v_strides_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[1]));
        const v_strides_2_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[2]));
        const num_seqs_ptr = zml.Tensor.constant(zml.DataType.i32.constant(parameters.block_table.dim(0)));
        const scale: f32 = paged_attention_opts.scale orelse @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(q.dim(.hd)))));
        const scale_ptr = zml.Tensor.constant(zml.DataType.f32.constant(scale));

        const attn_grid: [3]i32 = .{ @intCast(config.attention.total_q_blocks), @intCast(paged_attention_opts.num_kv_heads), @intCast(config.attention.num_segments_per_seq) };
        const attn_output = kernels.KernelUnifiedAttention3dPtr.Kernel.call(
            .{
                .query_ptr = q,
                .key_cache_ptr = k_cache,
                .value_cache_ptr = v_cache,
                .sink_ptr = dummy,
                .block_tables_ptr = parameters.block_table,
                .seq_lens_ptr = parameters.seq_lens,
                .alibi_slopes_ptr = dummy,
                .qq_bias_ptr = dummy,
                .scale_ptr = scale_ptr,
                .k_scale_ptr = dummy,
                .v_scale_ptr = dummy,
                .softcap_ptr = dummy,
                .block_table_stride_ptr = block_table_strides_ptr,
                .query_stride_0_ptr = q_strides_0_ptr,
                .query_stride_1_ptr = q_strides_1_ptr,
                .qq_bias_stride_0_ptr = dummy,
                .stride_k_cache_0_ptr = k_strides_0_ptr,
                .stride_k_cache_1_ptr = k_strides_1_ptr,
                .stride_k_cache_2_ptr = k_strides_2_ptr,
                .stride_v_cache_0_ptr = v_strides_0_ptr,
                .stride_v_cache_1_ptr = v_strides_1_ptr,
                .stride_v_cache_2_ptr = v_strides_2_ptr,
                .query_start_len_ptr = parameters.query_start_len,
                .num_seqs_ptr = num_seqs_ptr,
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
                .num_seqs_ptr = num_seqs_ptr,
                .out_scale_inv_ptr = dummy,
                .output_stride_0_ptr = q_strides_0_ptr,
                .output_stride_1_ptr = q_strides_1_ptr,
                .block_table_stride_ptr = block_table_strides_ptr,
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
        stdx.debug.assert(q.shape().hasTag(.b) == null or q.dim(.b) == 1, "triton.flashattn only supports batch size 1, got {f}", .{q.shape()});
        stdx.debug.assert(k.shape().hasTag(.b) == null or k.dim(.b) == 1, "triton.flashattn only supports batch size 1, got {f}", .{k.shape()});
        stdx.debug.assert(v.shape().hasTag(.b) == null or v.dim(.b) == 1, "triton.flashattn only supports batch size 1, got {f}", .{v.shape()});

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
                    const token_index_ = sharded_inputs[3];

                    const seqlen_q = q_.dim(.q);
                    const seqlen_k = k_.dim(.k);
                    const num_q_heads = q_.dim(.h);
                    const num_kv_heads = k_.dim(.h);
                    const head_dim = q_.dim(.hd);
                    const head_dim_pow2: i64 = @intCast(std.math.ceilPowerOfTwoAssert(usize, @intCast(head_dim)));
                    const block_m = blockSizeM(seqlen_q);
                    const block_n: i64 = 64;
                    const num_m_blocks = std.math.divCeil(i64, seqlen_q, block_m) catch unreachable;

                    const seqused_k = token_index_.addConstant(seqlen_q).reshape(.{1});
                    const zero = zml.Tensor.constant(token_index_.dtype().zero()).reshape(.{1});
                    const cu_seqlens_k = zml.Tensor.concatenate(&.{ zero, seqused_k }, 0)
                        .convert(.i32);

                    const cu_seqlens_q = zml.Tensor.constantTensor(zml.Shape.init(.{2}, .i32), std.mem.sliceAsBytes(&[2]i32{ 0, @intCast(seqlen_q) }));

                    const softmax_lse = zml.Tensor.uninitialized(zml.Shape.init(.{
                        .h = num_q_heads,
                        .q = seqlen_q,
                    }, .f32));
                    const alibi_slopes = zml.Tensor.zeroes(zml.Shape.init(.{ .h = num_q_heads }, .f32));
                    const dummy = zml.Tensor.constant(zml.DataType.f32.zero());

                    const sm_scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(head_dim))));
                    const sm_scale_ptr = zml.Tensor.constant(zml.DataType.f32.constant(sm_scale));
                    const kernel_config: kernels.MhaFwd.Config = .{
                        .q_dtype = toDType(q_.dtype()),
                        .kv_dtype = toDType(k_.dtype()),
                        .out_dtype = toDType(q_.dtype()),
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

                    const output = kernels.MhaFwd.Kernel.call(
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
