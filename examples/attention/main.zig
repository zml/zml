const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

//const ttir = @embedFile("kernel_unified_attention_2d_ptr.ttir");

//def select_2d_config(
//    block_size: int,
//    head_size: int,
//    all_decode: bool,
//    max_seqlen_q: int,
//    num_queries_per_kv: int,
//) -> dict:
//    block_m = 16 if num_queries_per_kv <= 16 else next_power_of_2(num_queries_per_kv)
//    tile_size = 64
//    max_num_stages_2d = 4 if head_size <= 128 else 2
//
//    if not all_decode:
//        num_stages_2d = 1
//        num_warps = 2
//    else:
//        num_stages_2d = 3
//        num_warps = 2
//        tile_size = block_size
//
//    if max_seqlen_q >= 256:
//        block_m = 128
//        num_stages_2d = 1
//        num_warps = 4
//
//    block_q = max(1, block_m // num_queries_per_kv)
//    return {
//        "BLOCK_M": block_m,
//        "BLOCK_Q": block_q,
//        "TILE_SIZE": tile_size,
//        "num_warps": num_warps,
//        "num_stages": min(max_num_stages_2d, num_stages_2d),
//    }

pub const Config2D = struct {
    block_m: usize,
    block_q: usize,
    tile_size: usize,
    num_warps: usize,
    num_stages: usize,
};

fn select2dConfig(block_size: usize, head_size: usize, all_decode: bool, max_seqlen_q: usize, num_queries_per_kv: usize) Config2D {
    var block_m: usize = if (num_queries_per_kv <= 16)
        16
    else
        std.math.ceilPowerOfTwoAssert(usize, num_queries_per_kv);

    const max_num_stages_2d: usize = if (head_size <= 128) 4 else 2;

    var num_stages_2d: usize, var num_warps: usize, const tile_size: usize = if (!all_decode) .{ 1, 2, 64 } else .{ 3, 2, block_size };

    if (max_seqlen_q >= 256) {
        block_m = 128;
        num_stages_2d = 1;
        num_warps = 4;
    }

    const block_q = @max(1, @divFloor(block_m, num_queries_per_kv));
    return .{
        .block_m = block_m,
        .block_q = block_q,
        .tile_size = tile_size,
        .num_warps = num_warps,
        .num_stages = @min(max_num_stages_2d, num_stages_2d),
    };
}

pub const Config3D = struct {
    const AttentionConfig = struct {
        tile_size: usize,
        num_segments_per_seq: usize,
        num_warps: usize,
        num_stages: usize,
    };
    const ReduceConfig = struct {
        tile_size: usize,
        num_segments_per_seq: usize,
        num_warps: usize,
        num_stages: usize,
    };
    attention: AttentionConfig,
    reduce: ReduceConfig,
};

fn select3dConfig(head_size: usize, block_size: usize, element_size: usize, max_seqlen_k: usize, target_num_prgms: usize, num_2d_prgms: usize) Config3D {
    _ = head_size; // autofix
    _ = element_size; // autofix
    _ = max_seqlen_k; // autofix
    var reduce_num_warps: usize = 2;
    const attn_warps: usize = 2;
    const tile_size = block_size;

    //const MAX_SEGMENTS: usize = @min(128, std.math.divCeil(usize, max_seqlen_k, tile_size));
    var num_segments = std.math.divCeil(usize, target_num_prgms, num_2d_prgms) catch unreachable;
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
        },
        .reduce = .{
            .tile_size = tile_size,
            .num_segments_per_seq = num_segments,
            .num_warps = reduce_num_warps,
            .num_stages = 1,
        },
    };
}

pub const KernelKind = enum {
    @"2d",
    @"3d",
    reduce,
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
    };

    pub const FeatureFlags = struct {
        use_alibi_slopes: bool,
        use_softcap: bool,
        use_sinks: bool,
        sliding_window: usize,
        all_decode: bool,
    };

    dimensions: Dimensions,
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
    };

    pub const FeatureFlags = struct {
        use_fp8: bool,
    };

    dimensions: Dimensions,
    feature_flags: FeatureFlags,
};

pub const GenerationConfig = union(KernelKind) {
    @"2d": GenerationConfig2D,
    @"3d": GenerationConfig3D,
    reduce: GenerationConfigReduce,
};

fn generateTtir(allocator: std.mem.Allocator, io: std.Io, config: GenerationConfig) ![:0]const u8 {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    var list: std.ArrayList([]const u8) = .empty;
    try list.append(arena.allocator(), "/home/corendos/triton-playground/.venv/bin/python");
    try list.append(arena.allocator(), "-m");
    try list.append(arena.allocator(), "tools.generate_unified_attention_ttir");
    try list.append(arena.allocator(), "--kernel");
    const kernel_name = switch (config) {
        .@"2d" => "kernel_unified_attention_2d_ptr",
        .@"3d" => "kernel_unified_attention_3d_ptr",
        .reduce => "reduce_segments_ptr",
    };
    try list.append(arena.allocator(), kernel_name);
    try list.append(arena.allocator(), "--config");
    switch (config) {
        .@"2d" => try list.append(arena.allocator(), try std.fmt.allocPrint(arena.allocator(), "{f}", .{std.json.fmt(config.@"2d", .{ .emit_null_optional_fields = false })})),
        .@"3d" => try list.append(arena.allocator(), try std.fmt.allocPrint(arena.allocator(), "{f}", .{std.json.fmt(config.@"3d", .{ .emit_null_optional_fields = false })})),
        .reduce => try list.append(arena.allocator(), try std.fmt.allocPrint(arena.allocator(), "{f}", .{std.json.fmt(config.reduce, .{ .emit_null_optional_fields = false })})),
    }
    const result = try std.process.run(arena.allocator(), io, .{ .argv = list.items, .cwd = .{ .path = "/home/corendos/triton-playground" } });
    return try allocator.dupeZ(u8, result.stdout);
}

pub const Attention2d = struct {
    pub fn forward(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, block_table: zml.Tensor, seq_lens: zml.Tensor, query_start_len: zml.Tensor, io: std.Io) zml.Tensor {
        const generation_config: GenerationConfig = .{
            .@"2d" = .{
                .dimensions = .{
                    .batch_size = @intCast(block_table.dim(.b)),
                    .block_size = @intCast(k.dim(.blocks)),
                    .head_dim = @intCast(k.dim(.hd)),
                    .num_blocks = @intCast(k.dim(.pages)),
                    .num_blocks_per_seq = @intCast(block_table.dim(.num_pages)),
                    .num_heads = @intCast(q.dim(.h)),
                    .num_kv_heads = @intCast(k.dim(.h)),
                    .num_tokens = @intCast(q.dim(.s)),
                },
                .feature_flags = .{
                    .all_decode = false,
                    .sliding_window = 0,
                    .use_alibi_slopes = false,
                    .use_fp8 = false,
                    .use_sinks = false,
                    .use_softcap = false,
                },
            },
        };
        const ttir = generateTtir(std.heap.c_allocator, io, generation_config) catch unreachable;
        defer std.heap.c_allocator.free(ttir);

        const dummy = zml.Tensor.constant(zml.DataType.i8.zero());
        const block_table_strides = block_table.shape().computeElementStrides().constSlice();
        const block_table_strides_ptr = zml.Tensor.constant(zml.DataType.i64.constant(block_table_strides[0]));
        const q_strides = q.shape().computeElementStrides().constSlice();
        const q_strides_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(q_strides[0]));
        const q_strides_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(q_strides[1]));
        const k_strides = k.shape().computeElementStrides().constSlice();
        const k_strides_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[0]));
        const k_strides_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[1]));
        const k_strides_2_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[2]));
        const v_strides = k.shape().computeElementStrides().constSlice();
        const v_strides_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[0]));
        const v_strides_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[1]));
        const v_strides_2_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[2]));
        const num_seqs_ptr = zml.Tensor.constant(zml.DataType.i32.constant(block_table.dim(0)));
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(q.dim(.hd)))));
        const scale_ptr = zml.Tensor.constant(zml.DataType.f32.constant(scale));

        const num_queries_per_kv: usize = @intCast(@divExact(q.dim(.h), k.dim(.h)));
        const config = select2dConfig(@intCast(k.dim(.blocks)), @intCast(q.dim(.hd)), false, 128, num_queries_per_kv);

        const num_kv_heads: i32 = @intCast(k.dim(.h));
        const num_tokens: usize = @intCast(q.dim(0));
        const num_seqs: usize = @intCast(block_table.dim(0));
        const total_num_q_blocks: i32 = @intCast(num_tokens / config.block_q + num_seqs);

        const output = zml.ops.triton(.{
            q,
            k,
            v,
            dummy, // sink_ptr
            block_table,
            seq_lens,
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
            query_start_len,
            num_seqs_ptr,
        }, .{q.shape()}, .{
            .debug = false,
            .name = "kernel_unified_attention_2d_ptr",
            .ir = ttir,
            .grid = .{ num_kv_heads, total_num_q_blocks, 1 },
            .num_stages = @intCast(config.num_stages),
            .num_warps = @intCast(config.num_warps),
        });
        return output[0];
    }
};

pub const Attention3d = struct {
    pub fn forward(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, block_table: zml.Tensor, seq_lens: zml.Tensor, query_start_len: zml.Tensor, io: std.Io) zml.Tensor {
        const attn_generation_config: GenerationConfig = .{
            .@"3d" = .{
                .dimensions = .{
                    .batch_size = @intCast(block_table.dim(.b)),
                    .block_size = @intCast(k.dim(.blocks)),
                    .head_dim = @intCast(k.dim(.hd)),
                    .num_blocks = @intCast(k.dim(.pages)),
                    .num_blocks_per_seq = @intCast(block_table.dim(.num_pages)),
                    .num_heads = @intCast(q.dim(.h)),
                    .num_kv_heads = @intCast(k.dim(.h)),
                    .num_tokens = @intCast(q.dim(.s)),
                },
                .feature_flags = .{
                    .all_decode = false,
                    .sliding_window = 0,
                    .use_alibi_slopes = false,
                    .use_sinks = false,
                    .use_softcap = false,
                },
            },
        };
        const attn_ttir = generateTtir(std.heap.c_allocator, io, attn_generation_config) catch unreachable;
        defer std.heap.c_allocator.free(attn_ttir);

        const reduce_generation_config: GenerationConfig = .{
            .reduce = .{
                .dimensions = .{
                    .num_tokens = @intCast(q.dim(.s)),
                    .num_heads = @intCast(q.dim(.h)),
                    .num_kv_heads = @intCast(k.dim(.h)),
                    .head_dim = @intCast(k.dim(.hd)),
                    .block_size = @intCast(k.dim(.blocks)),
                    .batch_size = @intCast(block_table.dim(.b)),
                    .num_blocks_per_seq = @intCast(block_table.dim(.num_pages)),
                },
                .feature_flags = .{
                    .use_fp8 = false,
                },
            },
        };
        const reduce_ttir = generateTtir(std.heap.c_allocator, io, reduce_generation_config) catch unreachable;
        defer std.heap.c_allocator.free(reduce_ttir);

        const dummy = zml.Tensor.constant(zml.DataType.i8.zero());
        const block_table_strides = block_table.shape().computeElementStrides().constSlice();
        const block_table_strides_ptr = zml.Tensor.constant(zml.DataType.i64.constant(block_table_strides[0]));
        const q_strides = q.shape().computeElementStrides().constSlice();
        const q_strides_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(q_strides[0]));
        const q_strides_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(q_strides[1]));
        const k_strides = k.shape().computeElementStrides().constSlice();
        const k_strides_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[0]));
        const k_strides_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[1]));
        const k_strides_2_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[2]));
        const v_strides = k.shape().computeElementStrides().constSlice();
        const v_strides_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[0]));
        const v_strides_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[1]));
        const v_strides_2_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[2]));
        const num_seqs_ptr = zml.Tensor.constant(zml.DataType.i32.constant(block_table.dim(0)));
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(q.dim(.hd)))));
        const scale_ptr = zml.Tensor.constant(zml.DataType.f32.constant(scale));

        const num_kv_heads: usize = @intCast(k.dim(.h));
        const num_heads: usize = @intCast(k.dim(.h));
        const num_tokens: usize = @intCast(q.dim(0));
        const num_seqs: usize = @intCast(block_table.dim(0));
        const head_dim: usize = @intCast(q.dim(.hd));
        const num_queries_per_kv: usize = @intCast(@divExact(q.dim(.h), k.dim(.h)));
        const block_m = if (num_queries_per_kv <= 16) 16 else std.math.ceilPowerOfTwoAssert(usize, num_queries_per_kv);
        const block_q = block_m / num_queries_per_kv;
        const total_num_q_blocks: usize = num_tokens / block_q + num_seqs;
        const target_num_prgms: usize = 128 * 4;
        const num_2d_prgms: usize = total_num_q_blocks * @as(usize, @intCast(num_kv_heads));

        const config = select3dConfig(@intCast(q.dim(.hd)), @intCast(k.dim(.blocks)), 2, @intCast(k.dim(.blocks) * block_table.dim(.num_pages)), target_num_prgms, num_2d_prgms);

        const attn_output = zml.ops.triton(.{
            q,
            k,
            v,
            dummy, // sink_ptr
            block_table,
            seq_lens,
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
            query_start_len,
            num_seqs_ptr,
        }, .{
            zml.Shape.init(.{ num_tokens, num_heads, config.attention.num_segments_per_seq, std.math.ceilPowerOfTwoAssert(usize, head_dim) }, .f32),
            zml.Shape.init(.{ num_tokens, num_heads, config.attention.num_segments_per_seq }, .f32),
            zml.Shape.init(.{ num_tokens, num_heads, config.attention.num_segments_per_seq }, .f32),
        }, .{
            .debug = false,
            .name = "kernel_unified_attention_3d_ptr",
            .ir = attn_ttir,
            .grid = .{ @intCast(total_num_q_blocks), @intCast(num_kv_heads), @intCast(config.attention.num_segments_per_seq) },
            .num_stages = @intCast(config.attention.num_stages),
            .num_warps = @intCast(config.attention.num_warps),
        });

        const output = zml.ops.triton(.{
            attn_output[0],
            attn_output[1],
            attn_output[2],
            seq_lens,
            num_seqs_ptr,
            dummy, // out_scale_inv_ptr
            q_strides_0_ptr,
            q_strides_1_ptr,
            block_table_strides_ptr,
            query_start_len,
        }, .{
            q.shape(),
        }, .{
            .debug = false,
            .name = "reduce_segments_ptr",
            .ir = reduce_ttir,
            .grid = .{ @intCast(num_tokens), @intCast(num_heads), 1 },
            .num_stages = @intCast(config.reduce.num_stages),
            .num_warps = @intCast(config.reduce.num_warps),
        });

        return output[0];
    }
};

pub fn main(init: std.process.Init) !void {
    const CliArgs = struct {
        pub const help =
            \\ attention --activations=activations.safetensors
        ;
        activations: []const u8,
    };

    const allocator = init.gpa;
    const io = init.io;

    const args = stdx.flags.parse(init.minimal.args, CliArgs);

    // Auto-select platform
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const q: zml.Tensor = .init(.{ .s = 128, .h = 32, .hd = 128 }, .bf16);
    const k_cache: zml.Tensor = .init(.{ .pages = 64, .blocks = 16, .h = 8, .hd = 128 }, .bf16);
    const v_cache: zml.Tensor = .init(.{ .pages = 64, .blocks = 16, .h = 8, .hd = 128 }, .bf16);
    const block_table: zml.Tensor = .init(.{ .b = 16, .num_pages = 64 }, .i32);
    const seq_lens: zml.Tensor = .init(.{ .b = 16 }, .i32);
    const query_start_len: zml.Tensor = .init(.{ .b = 17 }, .i32);

    var exe = try platform.compileFn(allocator, io, Attention3d.forward, .{ q, k_cache, v_cache, block_table, seq_lens, query_start_len, io });
    defer exe.deinit();

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();
    _ = random; // autofix

    const batch_size: usize = @intCast(seq_lens.dim(.b));
    const query_len = @divExact(128, batch_size);
    const context_len = 128;
    const seqlen = context_len + query_len;
    _ = seqlen; // autofix

    var activations_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.activations);
    defer activations_registry.deinit();

    var activations_store: zml.io.TensorStore = .fromRegistry(allocator, &activations_registry);
    defer activations_store.deinit();

    var q_buffer = try loadBuffer(allocator, io, platform, "q", &activations_store);
    defer q_buffer.deinit();
    var k_cache_buffer = try loadBuffer(allocator, io, platform, "k", &activations_store);
    defer k_cache_buffer.deinit();
    var v_cache_buffer = try loadBuffer(allocator, io, platform, "v", &activations_store);
    defer v_cache_buffer.deinit();
    var block_table_buffer = try loadBuffer(allocator, io, platform, "block_table", &activations_store);
    defer block_table_buffer.deinit();
    var seq_lens_buffer = try loadBuffer(allocator, io, platform, "seq_lens", &activations_store);
    defer seq_lens_buffer.deinit();
    var query_start_len_buffer = try loadBuffer(allocator, io, platform, "query_start_len", &activations_store);
    defer query_start_len_buffer.deinit();

    //var q_buffer = try createRandomBuffer(allocator, io, platform, q.shape(), random);
    //defer q_buffer.deinit();
    //var k_cache_buffer = try createRandomBuffer(allocator, io, platform, k_cache.shape(), random);
    //defer k_cache_buffer.deinit();
    //var v_cache_buffer = try createRandomBuffer(allocator, io, platform, v_cache.shape(), random);
    //defer v_cache_buffer.deinit();
    //var seq_lens_buffer = b: {
    //    const slice: zml.Slice = try .alloc(allocator, seq_lens.shape());
    //    defer slice.free(allocator);
    //    @memset(slice.items(i32), @intCast(seqlen));

    //    break :b try zml.Buffer.fromSlice(io, platform, slice);
    //};
    //defer seq_lens_buffer.deinit();

    //var query_start_len_buffer = b: {
    //    const slice: zml.Slice = try .alloc(allocator, query_start_len.shape());
    //    defer slice.free(allocator);
    //    const raw = slice.items(i32);
    //    raw[0] = 0;
    //    for (1..batch_size + 1) |i| {
    //        raw[i] = raw[i - 1] + @as(i32, @intCast(query_len));
    //    }
    //    break :b try zml.Buffer.fromSlice(io, platform, slice);
    //};
    //defer query_start_len_buffer.deinit();

    //var block_table_buffer = b: {
    //    const slice: zml.Slice = try .alloc(allocator, block_table.shape());
    //    defer slice.free(allocator);

    //    for (0..batch_size) |i| {
    //        const max_num_pages: usize = @intCast(block_table.dim(.num_pages));
    //        const current_block_table = slice.items(i32)[i * max_num_pages ..][0..max_num_pages];
    //        @memset(current_block_table, std.math.maxInt(i32));
    //        const num_pages = (seqlen + 16 - 1) / 16;
    //        for (0..num_pages) |j| {
    //            current_block_table[j] = @intCast(j);
    //        }
    //    }

    //    break :b try zml.Buffer.fromSlice(io, platform, slice);
    //};
    //defer block_table_buffer.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);

    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{ q_buffer, k_cache_buffer, v_cache_buffer, block_table_buffer, seq_lens_buffer, query_start_len_buffer });
    exe.call(exe_args, &exe_results);

    var output = exe_results.get(zml.Buffer);
    defer output.deinit();

    const slice = try output.toSliceAlloc(allocator, io);
    defer slice.free(allocator);

    std.log.info("slice: {d}", .{slice});
}

fn loadBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, name: []const u8, store: *zml.io.TensorStore) !zml.Buffer {
    const buffer = try allocator.alloc(u8, 4096);
    defer allocator.free(buffer);

    var reader = try store.getReader(name, io, buffer);
    defer reader.deinit();

    var slice: zml.Slice = try .alloc(allocator, reader.tensor.shape);
    defer slice.free(allocator);

    var writer: std.Io.Writer = .fixed(slice.data());

    _ = try reader.interface.streamRemaining(&writer);
    try writer.flush();

    return try .fromSlice(io, platform, slice);
}

fn createRandomBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape, random: std.Random) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    switch (shape.dtype()) {
        inline else => |v| {
            const ZigType = v.toZigType();
            switch (comptime v.class()) {
                .bool => unreachable,
                .integer => {
                    for (slice.items(ZigType)) |*e| e.* = random.int(ZigType);
                },
                .float => {
                    for (slice.items(ZigType)) |*e| {
                        const value = random.float(f32);
                        e.* = switch (ZigType) {
                            f64, f32 => value,
                            f16 => @floatCast(value),
                            inline else => |T| if (@hasDecl(T, "fromF32")) T.fromF32(value) else unreachable,
                        };
                    }
                },
                .complex => unreachable,
            }
        },
    }

    return .fromSlice(io, platform, slice);
}
