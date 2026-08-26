const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const triton = zml.kernel.triton;
const AttentionOptions = @import("paged_attention.zig").AttentionOptions;
const MlaOptions = @import("paged_attention.zig").Mla.Options;
const kernels = @import("triton_kernels/unified_attention.zig");
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

const SparseMlaLaunchConfig = struct {
    block_m: usize,
    tile_size: usize,
    num_tiles: usize,
    num_splits: usize,
    direct_programs: usize,
};

const SparseMlaTuning = struct {
    max_block_m: usize = 16,
    tile_size: usize = 16,
    max_splits: usize = 128,
    main_num_warps: usize = 4,
    main_num_stages: usize = 2,
    grouped_reduce_threshold: usize = 32,
    splits_per_group: usize = 16,
    grouped_reduce_num_warps: usize = 1,
    parallel_reduce_min_splits: usize = std.math.maxInt(usize),
    parallel_reduce_num_warps: usize = 1,
};

fn isCudaComputeCapability(expected: []const u8) bool {
    const platform = zml.module.CompilationContext.current().platform;
    if (platform.target != .cuda) return false;
    const devices = platform.pjrt_client.devices(platform.pjrt_api);
    if (devices.len == 0) return false;
    const actual = zml.platform.cuda.tryGetComputeCapabilities(platform, devices[0]) orelse return false;
    return std.mem.eql(u8, actual, expected);
}

fn sparseMlaTuning(paged_opts: paged.PagedSparseMlaOptions) SparseMlaTuning {
    var tuning: SparseMlaTuning = .{};
    const is_sm103 = isCudaComputeCapability("10.3");

    // GB300 (sm_103): a single wide query benefits from more head blocks and
    // wider sparse tiles. The split selector below still derives each layer's
    // split count from query/head/top-k shapes (16 for DSV4 CSA/HCA and 4 for
    // its 128-entry full-attention layers).
    if (is_sm103 and
        paged_opts.all_decode and
        paged_opts.total_q_blocks == 1 and
        paged_opts.value_rank >= 512)
    {
        tuning.max_block_m = 8;
        tuning.tile_size = 32;
        tuning.parallel_reduce_min_splits = 16;
        tuning.parallel_reduce_num_warps = 2;
    }
    // Full 256-query prefill chunks already expose enough query parallelism.
    // Retain BLOCK_M=16 and the generic split model, but process wider sparse
    // tiles. The predicate uses the post-sharding flattened query shape.
    if (is_sm103 and
        !paged_opts.all_decode and
        paged_opts.total_q_blocks >= 256 and
        paged_opts.value_rank >= 512)
    {
        tuning.tile_size = 32;
    }

    stdx.debug.assert(std.math.isPowerOfTwo(tuning.max_block_m), "MLA block_m ({}) must be a power of two", .{tuning.max_block_m});
    stdx.debug.assert(std.math.isPowerOfTwo(tuning.tile_size), "MLA tile size ({}) must be a power of two", .{tuning.tile_size});
    stdx.debug.assert(tuning.tile_size <= 32, "MLA tile size ({}) must not exceed 32", .{tuning.tile_size});
    stdx.debug.assert(std.math.isPowerOfTwo(tuning.main_num_warps), "MLA main-kernel warps ({}) must be a power of two", .{tuning.main_num_warps});
    stdx.debug.assert(std.math.isPowerOfTwo(tuning.grouped_reduce_num_warps), "MLA grouped-reducer warps ({}) must be a power of two", .{tuning.grouped_reduce_num_warps});
    return tuning;
}

fn selectSparseMlaLaunchConfig(
    query_count: usize,
    num_heads: usize,
    topk_count: usize,
    cu_count_: usize,
    requested_splits: ?u8,
    tuning: SparseMlaTuning,
) SparseMlaLaunchConfig {
    stdx.debug.assert(query_count > 0, "sparse MLA requires at least one query", .{});
    stdx.debug.assert(num_heads > 0, "sparse MLA requires at least one query head", .{});
    stdx.debug.assert(topk_count > 0, "sparse MLA requires at least one top-k entry", .{});

    const block_m = @min(std.math.ceilPowerOfTwoAssert(usize, num_heads), tuning.max_block_m);

    const topk_padded = std.math.ceilPowerOfTwoAssert(usize, topk_count);
    const tile_size = @min(topk_padded, tuning.tile_size);
    const num_tiles = std.math.divCeil(usize, topk_count, tile_size) catch unreachable;
    const head_blocks = std.math.divCeil(usize, num_heads, block_m) catch unreachable;
    const direct_programs = query_count * head_blocks;
    const cu_count = @max(cu_count_, 1);

    var max_splits: usize = 1;
    while (max_splits * 2 <= @min(num_tiles, tuning.max_splits)) max_splits *= 2;

    if (requested_splits) |requested| {
        const num_splits: usize = requested;
        stdx.debug.assert(std.math.isPowerOfTwo(num_splits), "MLA num_kv_splits ({}) must be a power of two", .{num_splits});
        stdx.debug.assert(num_splits <= 128, "MLA num_kv_splits ({}) must not exceed 128", .{num_splits});
        stdx.debug.assert(num_splits <= num_tiles, "MLA num_kv_splits ({}) must not exceed sparse tile count ({})", .{ num_splits, num_tiles });
        return .{
            .block_m = block_m,
            .tile_size = tile_size,
            .num_tiles = num_tiles,
            .num_splits = num_splits,
            .direct_programs = direct_programs,
        };
    }

    var best_splits: usize = 1;
    var best_cost: usize = std.math.maxInt(usize);
    const candidates = [_]usize{ 1, 2, 4, 8, 16, 32, 64, 128 };
    for (candidates) |num_splits| {
        if (num_splits > max_splits) break;
        const programs = direct_programs * num_splits;
        const rounds = std.math.divCeil(usize, programs, cu_count) catch unreachable;
        const tiles_per_program = std.math.divCeil(usize, num_tiles, num_splits) catch unreachable;
        const cost = rounds * tiles_per_program;
        // Candidates are ordered by split count, so ties retain 2D or the lower-overhead 3D launch.
        if (cost < best_cost) {
            best_cost = cost;
            best_splits = num_splits;
        }
    }

    return .{
        .block_m = block_m,
        .tile_size = tile_size,
        .num_tiles = num_tiles,
        .num_splits = best_splits,
        .direct_programs = direct_programs,
    };
}

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
    if (platform.devices.len == 0) return 1;
    const attribute = platform.devices[0].pjrt_desc.attribute(platform.pjrt_api, "core_count") orelse return 1;
    if (attribute.int64 <= 0) return 1;
    return @intCast(attribute.int64);
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
            .use_sinks = false,
            .sliding_window = @intCast(paged_attention_opts.sliding_window),
            .block_q = @intCast(config.attention.block_q),
            .block_m = @intCast(config.attention.block_m),
            .num_segments_per_seq = @intCast(config.attention.num_segments_per_seq),
            .all_decode = paged_attention_opts.all_decode,
            .is_causal = opts.is_causal,
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

    fn pagedSparseMlaKernel(q: zml.Tensor, kv_cache: zml.Tensor, sink: ?zml.Tensor, topk: zml.Tensor, active_query_count: zml.Tensor, paged_opts: PagedSparseMlaOptions) zml.Tensor {
        const tuning = sparseMlaTuning(paged_opts);
        const value_rank: i64 = @intCast(paged_opts.value_rank);
        const rope_rank: i64 = if (paged_opts.value_rank == paged_opts.qk_rank) 0 else @intCast(paged_opts.rope_rank);
        stdx.debug.assert(std.math.isPowerOfTwo(paged_opts.value_rank), "sparse MLA value rank ({}) must be a power of two", .{paged_opts.value_rank});
        stdx.debug.assert(rope_rank == 0 or std.math.isPowerOfTwo(@as(usize, @intCast(rope_rank))), "sparse MLA separate RoPE rank ({}) must be a power of two", .{rope_rank});
        stdx.debug.assert(value_rank + rope_rank == paged_opts.qk_rank, "sparse MLA expects either a full-width value or adjacent value/RoPE regions, got qk={} value={} rope={}", .{ paged_opts.qk_rank, value_rank, rope_rank });

        const out_shape = q.shape().set(.hd, value_rank);

        const q_strides = q.shape().computeElementStrides().constSlice();
        const out_strides = out_shape.computeElementStrides().constSlice();
        const kv_strides = kv_cache.shape().computeElementStrides().constSlice();

        const sink_: zml.Tensor = sink orelse .scalar(0, .i8);
        const use_sink = sink != null;
        const sm_scale: f32 = paged_opts.scale orelse @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(paged_opts.qk_rank))));

        const launch = selectSparseMlaLaunchConfig(
            paged_opts.total_q_blocks,
            paged_opts.num_heads,
            @intCast(topk.dim(.topk)),
            getCuCount(),
            paged_opts.num_kv_splits,
            tuning,
        );

        const kernel_config: mla_kernels.Config = .{
            .q_dtype = triton.from(q.dtype()),
            .kv_dtype = triton.from(kv_cache.dtype()),
            .sink_dtype = triton.from(sink_.dtype()),
            .o_dtype = triton.from(q.dtype()),
            .num_query_heads = @intCast(paged_opts.num_heads),
            .block_size = @intCast(paged_opts.block_size),
            .block_m = @intCast(launch.block_m),
            .topk_count = topk.dim(.topk),
            .rope_rank = rope_rank,
            .value_rank = value_rank,
            .rope_offset = value_rank,
            .tile_size = @intCast(launch.tile_size),
            .num_splits = @intCast(launch.num_splits),
            .use_attn_sink = use_sink,
            .all_decode = paged_opts.all_decode,
        };
        log.debug("pagedSparseMla launch: {any}, kernel: {any}", .{ launch, kernel_config });

        const kernel_inputs: mla_kernels.Kernel2D.Inputs = .{
            .query_ptr = q,
            .kv_cache_ptr = kv_cache,
            .attn_sink_ptr = sink_,
            .topk_indices_ptr = topk,
            .active_query_count_ptr = active_query_count,
            .scale_ptr = zml.Tensor.constant(zml.DataType.f32.constant(sm_scale)),
            .query_stride_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(q_strides[0])),
            .query_stride_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(q_strides[1])),
            .output_stride_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(out_strides[0])),
            .output_stride_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(out_strides[1])),
            .stride_cache_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(kv_strides[kv_cache.shape().axis(.page)])),
            .stride_cache_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(kv_strides[kv_cache.shape().axis(.k_chunk)])),
        };

        if (launch.num_splits == 1) {
            const out = mla_kernels.Kernel2D.call(
                kernel_inputs,
                .{ .output = out_shape },
                .{
                    .cfg = kernel_config,
                    .grid = .{ @intCast(launch.direct_programs), 1, 1 },
                    .num_warps = @intCast(tuning.main_num_warps),
                    .num_stages = @intCast(tuning.main_num_stages),
                },
            );
            return out.output.reshape(out_shape);
        }

        const num_splits: i64 = @intCast(launch.num_splits);
        const kernel_3d_inputs: mla_kernels.Kernel3D.Inputs = .{
            .query_ptr = kernel_inputs.query_ptr,
            .kv_cache_ptr = kernel_inputs.kv_cache_ptr,
            .attn_sink_ptr = kernel_inputs.attn_sink_ptr,
            .topk_indices_ptr = kernel_inputs.topk_indices_ptr,
            .active_query_count_ptr = kernel_inputs.active_query_count_ptr,
            .scale_ptr = kernel_inputs.scale_ptr,
            .query_stride_0_ptr = kernel_inputs.query_stride_0_ptr,
            .query_stride_1_ptr = kernel_inputs.query_stride_1_ptr,
            .output_stride_0_ptr = kernel_inputs.output_stride_0_ptr,
            .output_stride_1_ptr = kernel_inputs.output_stride_1_ptr,
            .stride_cache_0_ptr = kernel_inputs.stride_cache_0_ptr,
            .stride_cache_1_ptr = kernel_inputs.stride_cache_1_ptr,
        };
        const partials = mla_kernels.Kernel3D.call(
            kernel_3d_inputs,
            .{
                .partial_output = zml.Shape.init(.{ q.dim(.q), @as(i64, @intCast(paged_opts.num_heads)), num_splits, value_rank }, .f32),
                .partial_lse = zml.Shape.init(.{ q.dim(.q), @as(i64, @intCast(paged_opts.num_heads)), num_splits }, .f32),
            },
            .{
                .cfg = kernel_config,
                .grid = .{ @intCast(launch.direct_programs), @intCast(launch.num_splits), 1 },
                .num_warps = @intCast(tuning.main_num_warps),
                .num_stages = @intCast(tuning.main_num_stages),
            },
        );
        var reduce_partial_output = partials.partial_output;
        var reduce_partial_lse = partials.partial_lse;
        var reduce_num_splits = num_splits;
        if (num_splits > tuning.grouped_reduce_threshold) {
            const grouped_splits: i64 = @divExact(num_splits, @as(i64, @intCast(tuning.splits_per_group)));
            const grouped = mla_kernels.Reduce3DPartials.call(
                .{
                    .input = partials.partial_output,
                    .input_lse = partials.partial_lse,
                    .active_query_count_ptr = active_query_count,
                },
                .{
                    .partial_output = zml.Shape.init(.{ q.dim(.q), @as(i64, @intCast(paged_opts.num_heads)), grouped_splits, value_rank }, .f32),
                    .partial_lse = zml.Shape.init(.{ q.dim(.q), @as(i64, @intCast(paged_opts.num_heads)), grouped_splits }, .f32),
                },
                .{
                    .cfg = .{
                        .num_query_heads = @intCast(paged_opts.num_heads),
                        .value_rank = value_rank,
                        .num_input_splits = num_splits,
                        .num_output_splits = grouped_splits,
                        .all_decode = paged_opts.all_decode,
                    },
                    .grid = .{ @intCast(q.dim(.q)), @intCast(paged_opts.num_heads * @as(usize, @intCast(grouped_splits))), 1 },
                    .num_warps = @intCast(tuning.grouped_reduce_num_warps),
                    .num_stages = 1,
                },
            );
            reduce_partial_output = grouped.partial_output;
            reduce_partial_lse = grouped.partial_lse;
            reduce_num_splits = grouped_splits;
        }

        const out = mla_kernels.Reduce3D.call(
            .{
                .partial_output_ptr = reduce_partial_output,
                .partial_lse_ptr = reduce_partial_lse,
                .attn_sink_ptr = sink_,
                .active_query_count_ptr = active_query_count,
                .output_stride_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(out_strides[0])),
                .output_stride_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(out_strides[1])),
            },
            .{ .output = out_shape },
            .{
                .cfg = .{
                    .sink_dtype = triton.from(sink_.dtype()),
                    .o_dtype = triton.from(q.dtype()),
                    .num_query_heads = @intCast(paged_opts.num_heads),
                    .value_rank = value_rank,
                    .num_splits = reduce_num_splits,
                    .use_attn_sink = use_sink,
                    .all_decode = paged_opts.all_decode,
                },
                .grid = .{ @intCast(q.dim(.q)), @intCast(paged_opts.num_heads), 1 },
                .num_warps = @intCast(if (reduce_num_splits >= tuning.parallel_reduce_min_splits)
                    tuning.parallel_reduce_num_warps
                else
                    1),
                .num_stages = 1,
            },
        );

        return out.output.reshape(out_shape);
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
        qk_rank: usize,
        value_rank: usize,
        num_heads: usize,
        block_size: usize,
        rope_rank: usize,
        scale: ?f32,
        total_q_blocks: usize,
        num_kv_splits: ?u8,
        all_decode: bool,
    };

    pub fn pagedSparseMla(parameters: Parameters, q: zml.Tensor, kv_cache: zml.Tensor, sink: ?zml.Tensor, topk: zml.Tensor, tokens_pos: zml.Tensor, opts: MlaOptions) zml.Tensor {
        const output_shape = q.shape().set(.hd, opts.valueRank(q.dim(.hd)));
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
            output_shape,
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
                    const active_query_count = sharded_inputs[7]
                        .slice1d(.b, .{ .start = sharded_inputs[7].dim(.b) - 1 })
                        .squeeze(.b);
                    stdx.debug.assert(topk_final.dim(.q) == q_.dim(.q), "expected topk q dim ({}) to match q dim ({})", .{ topk_final.dim(.q), q_.dim(.q) });

                    const num_heads: usize = @intCast(q_.dim(.h));
                    const paged_opts: PagedSparseMlaOptions = .{
                        .qk_rank = @intCast(q_.dim(.hd)),
                        .value_rank = @intCast(ctx_.opts.valueRank(q_.dim(.hd))),
                        .num_heads = num_heads,
                        .block_size = @intCast(kv_cache_.dim(.k_chunk)),
                        .rope_rank = @intCast(ctx_.opts.rope_rank),
                        .scale = ctx_.opts.scale,
                        .total_q_blocks = @intCast(q_.dim(.q)),
                        .num_kv_splits = ctx_.opts.num_kv_splits,
                        .all_decode = !ctx_.options.is_prefill,
                    };

                    return pagedSparseMlaKernel(
                        q_,
                        kv_cache_,
                        sharded_inputs[2],
                        topk_final,
                        active_query_count,
                        paged_opts,
                    );
                }
            }).body,
        );
    }
};

test "sparse MLA launch selection balances occupancy and split overhead" {
    const saturated = selectSparseMlaLaunchConfig(120, 16, 512, 120, null, .{});
    try std.testing.expectEqual(@as(usize, 1), saturated.num_splits);

    const under_occupied = selectSparseMlaLaunchConfig(1, 16, 512, 120, null, .{});
    try std.testing.expectEqual(@as(usize, 32), under_occupied.num_splits);

    const tile_limited = selectSparseMlaLaunchConfig(1, 16, 32, 120, null, .{});
    try std.testing.expectEqual(@as(usize, 2), tile_limited.num_splits);

    const missing_cu_count = selectSparseMlaLaunchConfig(1, 16, 512, 0, null, .{});
    try std.testing.expectEqual(@as(usize, 1), missing_cu_count.num_splits);
}

test "sparse MLA launch selection honors valid explicit splits" {
    const forced = selectSparseMlaLaunchConfig(1, 16, 64, 120, 4, .{});
    try std.testing.expectEqual(@as(usize, 4), forced.num_splits);

    const one_tile = selectSparseMlaLaunchConfig(1, 16, 5, 120, null, .{});
    try std.testing.expectEqual(@as(usize, 8), one_tile.tile_size);
    try std.testing.expectEqual(@as(usize, 1), one_tile.num_splits);
}

test "sparse MLA launch selection supports arbitrary local head counts" {
    const cases = [_]struct {
        heads: usize,
        block_m: usize,
        programs: usize,
    }{
        .{ .heads = 1, .block_m = 1, .programs = 3 },
        .{ .heads = 6, .block_m = 8, .programs = 3 },
        .{ .heads = 8, .block_m = 8, .programs = 3 },
        .{ .heads = 16, .block_m = 16, .programs = 3 },
        .{ .heads = 24, .block_m = 16, .programs = 6 },
    };
    for (cases) |case| {
        const launch = selectSparseMlaLaunchConfig(3, case.heads, 64, 120, 1, .{});
        try std.testing.expectEqual(case.block_m, launch.block_m);
        try std.testing.expectEqual(case.programs, launch.direct_programs);
    }
}

test "sparse MLA launch selection derives GB300 decode splits from shapes" {
    const tuning: SparseMlaTuning = .{
        .max_block_m = 8,
        .tile_size = 32,
        .parallel_reduce_min_splits = 16,
        .parallel_reduce_num_warps = 2,
    };

    const csa = selectSparseMlaLaunchConfig(1, 64, 640, 152, null, tuning);
    try std.testing.expectEqual(@as(usize, 8), csa.block_m);
    try std.testing.expectEqual(@as(usize, 32), csa.tile_size);
    try std.testing.expectEqual(@as(usize, 16), csa.num_splits);

    const full = selectSparseMlaLaunchConfig(1, 64, 128, 152, null, tuning);
    try std.testing.expectEqual(@as(usize, 4), full.num_splits);
}

test "sparse MLA emits 2D and 3D Triton kernels" {
    const platform = zml.testing.env();
    var compilation = zml.module.CompilationContext.init(std.testing.allocator, std.testing.io, platform, .{});
    defer compilation.deinit();
    compilation.activate();
    defer compilation.deactivate();

    const block = @import("mlir").Block.init(&.{}, &.{});
    compilation.pushBlock(block);
    defer compilation.popBlock();

    const q = zml.Tensor.zeroes(zml.Shape.init(.{ .q = 1, .h = 6, .hd = 576 }, .bf16));
    const kv = zml.Tensor.zeroes(zml.Shape.init(.{ .page = 32, .k_chunk = 1, .hkv = 1, .hd = 576 }, .bf16));
    const sink = zml.Tensor.zeroes(zml.Shape.init(.{ .h = 6 }, .bf16));
    const topk = zml.Tensor.zeroes(zml.Shape.init(.{ .q = 1, .topk = 32 }, .i32));
    const two_d_opts: paged.PagedSparseMlaOptions = .{
        .qk_rank = 576,
        .value_rank = 512,
        .num_heads = 6,
        .block_size = 1,
        .rope_rank = 64,
        .scale = null,
        .total_q_blocks = 1,
        .num_kv_splits = 1,
        .all_decode = true,
    };
    const output_shape = q.shape().set(.hd, 512);

    const two_d = paged.pagedSparseMlaKernel(q, kv, sink, topk, two_d_opts);
    try std.testing.expect(two_d.shape().eql(output_shape));
    try std.testing.expect(two_d.value().owner().verify());

    var three_d_opts = two_d_opts;
    three_d_opts.num_kv_splits = 2;
    const three_d = paged.pagedSparseMlaKernel(q, kv, sink, topk, three_d_opts);
    try std.testing.expect(three_d.shape().eql(output_shape));
    try std.testing.expect(three_d.value().owner().verify());

    const dsv4_q = zml.Tensor.zeroes(zml.Shape.init(.{ .q = 1, .h = 6, .hd = 512 }, .bf16));
    const dsv4_kv = zml.Tensor.zeroes(zml.Shape.init(.{ .page = 32, .k_chunk = 1, .hkv = 1, .hd = 512 }, .bf16));
    var dsv4_opts = two_d_opts;
    dsv4_opts.qk_rank = 512;
    dsv4_opts.value_rank = 512;
    const dsv4_output_shape = dsv4_q.shape();

    const dsv4_two_d = paged.pagedSparseMlaKernel(dsv4_q, dsv4_kv, sink, topk, dsv4_opts);
    try std.testing.expect(dsv4_two_d.shape().eql(dsv4_output_shape));
    try std.testing.expect(dsv4_two_d.value().owner().verify());

    dsv4_opts.num_kv_splits = 2;
    const dsv4_three_d = paged.pagedSparseMlaKernel(dsv4_q, dsv4_kv, sink, topk, dsv4_opts);
    try std.testing.expect(dsv4_three_d.shape().eql(dsv4_output_shape));
    try std.testing.expect(dsv4_three_d.value().owner().verify());
}
