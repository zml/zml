const std = @import("std");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const stdx = @import("stdx");

const zml = @import("../zml.zig");
const AttentionOptions = @import("paged_attention.zig").AttentionOptions;

const log = std.log.scoped(.@"zml/attention/triton");

fn use2dKernel(head_size: usize, sliding_window: usize, all_decode: bool, max_seqlen_q: usize, max_seqlen_k: usize, target_num_prgms: usize, num_2d_prgms: usize) bool {
    _ = head_size; // autofix
    _ = all_decode; // autofix
    _ = max_seqlen_q; // autofix
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

pub const KernelKind = enum {
    kernel_unified_attention_2d_ptr,
    kernel_unified_attention_3d_ptr,
    reduce_segments_ptr,
};

pub const GenerationConfig2D = struct {
    pub const Dimensions = struct {
        num_tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_blocks: usize,
        block_size: usize,
        batch_size: usize,
        num_blocks_per_seq: usize,
        num_qq_tokens: ?usize = null,
        max_seqlen_q: usize,
    };

    pub const FeatureFlags = struct {
        use_alibi_slopes: bool,
        use_softcap: bool,
        use_sinks: bool,
        sliding_window: usize,
        use_fp8: bool,
        all_decode: bool,
    };

    dimensions: Dimensions,
    config: Config2D,
    feature_flags: FeatureFlags,
};

pub const GenerationConfig3D = struct {
    pub const Dimensions = struct {
        num_tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        num_blocks: usize,
        block_size: usize,
        batch_size: usize,
        num_blocks_per_seq: usize,
        num_qq_tokens: ?usize = null,
        cu_count: usize,
    };

    pub const FeatureFlags = struct {
        use_alibi_slopes: bool,
        use_softcap: bool,
        use_sinks: bool,
        sliding_window: usize,
        all_decode: bool,
    };

    dimensions: Dimensions,
    config: Config3D.AttentionConfig,
    feature_flags: FeatureFlags,
};

pub const GenerationConfigReduce = struct {
    pub const Dimensions = struct {
        num_tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        batch_size: usize,
        num_blocks_per_seq: usize,
        cu_count: usize,
    };

    pub const FeatureFlags = struct {
        use_fp8: bool,
    };

    dimensions: Dimensions,
    config: Config3D.ReduceConfig,
    feature_flags: FeatureFlags,
};

pub const GenerationConfig = union(KernelKind) {
    kernel_unified_attention_2d_ptr: GenerationConfig2D,
    kernel_unified_attention_3d_ptr: GenerationConfig3D,
    reduce_segments_ptr: GenerationConfigReduce,
};

fn getGenerateBinPath(allocator: std.mem.Allocator) ![]const u8 {
    const runfiles = bazel.runfiles(bazel_builtin.current_repository) catch unreachable;

    var sandbox_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = runfiles.rlocation("zml/zml/attention/triton/sandbox", &sandbox_path_buf) catch unreachable;

    return std.fs.path.join(allocator, &.{ sandbox_path.?, "bin", "generate" });
}

fn generateTtir(allocator: std.mem.Allocator, io: std.Io, config: GenerationConfig) ![:0]const u8 {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const writer_buffer = try arena.allocator().alloc(u8, 4096);
    const reader_buffer = try arena.allocator().alloc(u8, 4096);

    const compilation_context = zml.module.CompilationContext.current();
    const platform = compilation_context.platform;
    const triton_runtime = platform.triton_runtime.?;

    var writer = triton_runtime.process.stdin.?.writer(io, writer_buffer);
    try writer.interface.print("{f}\n", .{std.json.fmt(config, .{ .emit_null_optional_fields = false })});
    try writer.interface.flush();

    var reader = triton_runtime.process.stdout.?.reader(io, reader_buffer);
    var allocating: std.Io.Writer.Allocating = .init(arena.allocator());
    _ = try reader.interface.streamDelimiter(&allocating.writer, '\n');
    _ = try reader.interface.discardShort(1);

    const response: std.json.Value = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), allocating.written(), .{});
    return if (response.object.get("ok").?.bool)
        try allocator.dupeZ(u8, response.object.get("result").?.string)
    else
        error.GenerationFailed;
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
                .block_table = .init(.{ .b = options_.batch_size, .num_pages_per_seq = options_.max_num_pages }, .i32),
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

    pub const Context = struct {
        is_prefill: bool,
        max_seqlen_q: usize,

        pub fn init(parameters: Parameters, num_heads: i64, num_kv_heads: i64, head_dim: i64, page_size: i64) Context {
            _ = num_heads; // autofix
            _ = num_kv_heads; // autofix
            _ = head_dim; // autofix
            _ = page_size; // autofix
            return .{ .is_prefill = parameters.options_.is_prefill, .max_seqlen_q = parameters.options_.max_seqlen_q };
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

        pub fn numQueriesPerKv(self: PagedAttentionOptions) usize {
            return self.num_heads / self.num_kv_heads;
        }

        pub fn maxSeqLenK(self: PagedAttentionOptions) usize {
            return self.max_num_block_per_seq * self.block_size;
        }
    };

    pub fn pagedAttention(parameters: Parameters, context: Context, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, layer_index: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        const output = zml.ops.manualComputation(
            .{
                q,
                k_cache,
                v_cache,
                parameters.block_table,
                parameters.seq_lens,
                parameters.query_start_len,
                layer_index,
            },
            q.shape(),
            .{
                .opts = opts,
                .options = parameters.options_,
                .context = context,
            },
            (struct {
                fn body(ctx_: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, _: zml.Shape) zml.Tensor {
                    const q_ = sharded_inputs[0];
                    const k_cache_ = sharded_inputs[1];
                    const v_cache_ = sharded_inputs[2];
                    const layer_index_ = sharded_inputs[6];
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
                        .max_num_block_per_seq = @intCast(parameters_.block_table.dim(.num_pages_per_seq)),
                        .sliding_window = if (ctx_.opts.sliding_window < 0) 0 else @intCast(ctx_.opts.sliding_window),
                        .block_m = block_m,
                        .block_q = block_q,
                        .total_q_blocks = total_q_blocks,
                        .target_num_prgms = target_num_prgms,
                        .num_2d_prgms = num_2d_prgms,
                        .max_seqlen_q = ctx_.options.max_seqlen_q,
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
                        pagedAttention2d(parameters_, ctx_.context, q_, k_cache_, v_cache_, layer_index_, ctx_.opts, paged_attention_opts)
                    else
                        pagedAttention3d(parameters_, ctx_.context, q_, k_cache_, v_cache_, layer_index_, ctx_.opts, paged_attention_opts);

                    return output;
                }
            }).body,
        );

        return output;
    }

    pub fn pagedAttention2d(parameters: Parameters, context: Context, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, layer_index: zml.Tensor, opts: AttentionOptions, paged_attention_opts: PagedAttentionOptions) zml.Tensor {
        _ = context; // autofix
        _ = layer_index; // autofix
        _ = opts; // autofix
        const config = select2dConfig(paged_attention_opts);
        const generation_config: GenerationConfig = .{
            .kernel_unified_attention_2d_ptr = .{
                .dimensions = .{
                    .batch_size = paged_attention_opts.batch_size,
                    .block_size = paged_attention_opts.block_size,
                    .head_dim = paged_attention_opts.head_dim,
                    .num_blocks = paged_attention_opts.num_blocks,
                    .num_blocks_per_seq = paged_attention_opts.max_num_block_per_seq,
                    .num_heads = paged_attention_opts.num_heads,
                    .num_kv_heads = paged_attention_opts.num_kv_heads,
                    .num_tokens = paged_attention_opts.num_tokens,
                    .max_seqlen_q = parameters.options_.max_seqlen_q,
                },
                .config = config,
                .feature_flags = .{
                    .all_decode = paged_attention_opts.all_decode,
                    .sliding_window = paged_attention_opts.sliding_window,
                    .use_alibi_slopes = false,
                    .use_fp8 = false,
                    .use_sinks = false,
                    .use_softcap = false,
                },
            },
        };

        log.debug("pagedAttention2d config: {any}", .{generation_config});

        var threaded_io: std.Io.Threaded = .init_single_threaded;
        threaded_io.allocator = std.heap.c_allocator;
        defer threaded_io.deinit();

        const io = threaded_io.io();

        const ttir = generateTtir(std.heap.c_allocator, io, generation_config) catch unreachable;
        defer std.heap.c_allocator.free(ttir);

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
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(q.dim(.hd)))));
        const scale_ptr = zml.Tensor.constant(zml.DataType.f32.constant(scale));

        const output = zml.ops.triton(.{
            q,
            k_cache,
            v_cache,
            dummy, // sink_ptr
            parameters.block_table,
            parameters.seq_lens,
            dummy, // alibi_slopes_ptr
            dummy, // qq_bias_ptr
            scale_ptr,
            dummy, // k_scale_ptr
            dummy, // v_scale_ptr
            dummy, // out_scale_ptr
            dummy, // softcap_ptr
            block_table_strides_ptr,
            q_strides_0_ptr,
            q_strides_1_ptr,
            q_strides_0_ptr,
            q_strides_1_ptr,
            dummy, // qq_bias_stride_0_ptr
            k_strides_0_ptr,
            k_strides_1_ptr,
            k_strides_2_ptr,
            v_strides_0_ptr,
            v_strides_1_ptr,
            v_strides_2_ptr,
            parameters.query_start_len,
            num_seqs_ptr,
        }, .{q.shape()}, .{
            .debug = false,
            .name = "kernel_unified_attention_2d_ptr",
            .ir = ttir,
            .grid = .{ @intCast(paged_attention_opts.num_kv_heads), @intCast(config.total_q_blocks), 1 },
            .num_stages = @intCast(config.num_stages),
            .num_warps = @intCast(config.num_warps),
        });
        return output[0];
    }

    pub fn pagedAttention3d(parameters: Parameters, context: Context, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, layer_index: zml.Tensor, opts: AttentionOptions, paged_attention_opts: PagedAttentionOptions) zml.Tensor {
        _ = context; // autofix
        _ = layer_index; // autofix
        _ = opts; // autofix

        const config = select3dConfig(paged_attention_opts);

        const attn_generation_config: GenerationConfig = .{
            .kernel_unified_attention_3d_ptr = .{
                .dimensions = .{
                    .batch_size = paged_attention_opts.batch_size,
                    .block_size = paged_attention_opts.block_size,
                    .head_dim = paged_attention_opts.head_dim,
                    .num_blocks = paged_attention_opts.num_blocks,
                    .num_blocks_per_seq = paged_attention_opts.max_num_block_per_seq,
                    .num_heads = paged_attention_opts.num_heads,
                    .num_kv_heads = paged_attention_opts.num_kv_heads,
                    .num_tokens = paged_attention_opts.num_tokens,
                    .cu_count = paged_attention_opts.cu_count,
                },
                .config = config.attention,
                .feature_flags = .{
                    .all_decode = paged_attention_opts.all_decode,
                    .sliding_window = paged_attention_opts.sliding_window,
                    .use_alibi_slopes = false,
                    .use_sinks = false,
                    .use_softcap = false,
                },
            },
        };

        log.debug("pagedAttention3d attention config: {any}", .{attn_generation_config});

        var threaded_io: std.Io.Threaded = .init_single_threaded;
        threaded_io.allocator = std.heap.c_allocator;
        defer threaded_io.deinit();

        const io = threaded_io.io();
        const attn_ttir = generateTtir(std.heap.c_allocator, io, attn_generation_config) catch unreachable;
        defer std.heap.c_allocator.free(attn_ttir);

        const reduce_generation_config: GenerationConfig = .{
            .reduce_segments_ptr = .{
                .dimensions = .{
                    .num_tokens = paged_attention_opts.num_tokens,
                    .num_heads = paged_attention_opts.num_heads,
                    .num_kv_heads = paged_attention_opts.num_kv_heads,
                    .head_dim = paged_attention_opts.head_dim,
                    .block_size = paged_attention_opts.block_size,
                    .batch_size = paged_attention_opts.batch_size,
                    .num_blocks_per_seq = paged_attention_opts.max_num_block_per_seq,
                    .cu_count = paged_attention_opts.cu_count,
                },
                .config = config.reduce,
                .feature_flags = .{
                    .use_fp8 = false,
                },
            },
        };
        log.debug("pagedAttention3d reduce config: {any}", .{reduce_generation_config});
        const reduce_ttir = generateTtir(std.heap.c_allocator, io, reduce_generation_config) catch unreachable;
        defer std.heap.c_allocator.free(reduce_ttir);

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
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(q.dim(.hd)))));
        const scale_ptr = zml.Tensor.constant(zml.DataType.f32.constant(scale));

        const attn_grid: [3]i32 = .{ @intCast(config.attention.total_q_blocks), @intCast(paged_attention_opts.num_kv_heads), @intCast(config.attention.num_segments_per_seq) };
        const attn_output = zml.ops.triton(.{
            q,
            k_cache,
            v_cache,
            dummy, // sink_ptr
            parameters.block_table,
            parameters.seq_lens,
            dummy, // alibi_slopes_ptr
            dummy, // qq_bias_ptr
            scale_ptr,
            dummy, // k_scale_ptr
            dummy, // v_scale_ptr
            dummy, // softcap_ptr
            block_table_strides_ptr,
            q_strides_0_ptr,
            q_strides_1_ptr,
            dummy, // qq_bias_stride_0_ptr
            k_strides_0_ptr,
            k_strides_1_ptr,
            k_strides_2_ptr,
            v_strides_0_ptr,
            v_strides_1_ptr,
            v_strides_2_ptr,
            parameters.query_start_len,
            num_seqs_ptr,
        }, .{
            zml.Shape.init(.{ paged_attention_opts.num_tokens, paged_attention_opts.num_heads, config.attention.num_segments_per_seq, std.math.ceilPowerOfTwoAssert(usize, paged_attention_opts.head_dim) }, .f32),
            zml.Shape.init(.{ paged_attention_opts.num_tokens, paged_attention_opts.num_heads, config.attention.num_segments_per_seq }, .f32),
            zml.Shape.init(.{ paged_attention_opts.num_tokens, paged_attention_opts.num_heads, config.attention.num_segments_per_seq }, .f32),
        }, .{
            .debug = false,
            .name = "kernel_unified_attention_3d_ptr",
            .ir = attn_ttir,
            .grid = attn_grid,
            .num_stages = @intCast(config.attention.num_stages),
            .num_warps = @intCast(config.attention.num_warps),
        });

        const output = zml.ops.triton(.{
            attn_output[0],
            attn_output[1],
            attn_output[2],
            parameters.seq_lens,
            num_seqs_ptr,
            dummy, // out_scale_inv_ptr
            q_strides_0_ptr,
            q_strides_1_ptr,
            block_table_strides_ptr,
            parameters.query_start_len,
        }, .{
            q.shape(),
        }, .{
            .debug = false,
            .name = "reduce_segments_ptr",
            .ir = reduce_ttir,
            .grid = .{ @intCast(paged_attention_opts.num_tokens), @intCast(paged_attention_opts.num_heads), 1 },
            .num_stages = @intCast(config.reduce.num_stages),
            .num_warps = @intCast(config.reduce.num_warps),
        });

        return output[0];
    }
};

pub const Runtime = struct {
    process: std.process.Child,

    pub fn init(allocator: std.mem.Allocator, io: std.Io) !Runtime {
        const path = try getGenerateBinPath(allocator);
        defer allocator.free(path);

        const process = try std.process.spawn(io, .{ .argv = &.{path}, .stdin = .pipe, .stdout = .pipe });
        errdefer process.kill(io);

        return .{ .process = process };
    }

    pub fn deinit(self: *Runtime, io: std.Io) void {
        // NOTE(Corendos): This is not portable, but I couldn't find a better way with the current std.Io.
        _ = std.c.kill(self.process.id.?, .INT);
        _ = self.process.wait(io) catch unreachable;
    }
};
