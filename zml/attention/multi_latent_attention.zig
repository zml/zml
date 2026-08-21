const std = @import("std");

const zml = @import("../zml.zig");
const paged_attn = @import("paged_attention.zig");
const triton_attn = @import("triton_attention.zig");
const kernel = @import("triton_kernels/unified_sparse_mla.zig");

const stdx = zml.stdx;

const log = std.log.scoped(.mla);

const LaunchConfig = struct {
    block_m: i64,
    tile_size: i64,
    num_tiles: usize,
    num_splits: usize,
    direct_programs: usize,
};

fn getCoreCount() usize {
    const platform = zml.module.CompilationContext.current().platform;
    if (platform.devices.len == 0) return 1;
    const attribute = platform.devices[0].pjrt_desc.attribute(platform.pjrt_api, "core_count") orelse return 1;
    if (attribute.int64 <= 0) return 1;
    return @intCast(attribute.int64);
}

fn selectLaunchConfig(
    query_count: i64,
    num_heads: i64,
    topk_count: i64,
    core_count_: usize,
    requested_splits: ?u8,
) LaunchConfig {
    stdx.debug.assert(query_count > 0, "sparse MLA requires at least one query", .{});
    stdx.debug.assert(num_heads > 0, "sparse MLA requires at least one query head", .{});
    stdx.debug.assert(topk_count > 0, "sparse MLA requires at least one top-k entry", .{});

    const block_m = @min(num_heads, 16);
    stdx.debug.assert(@mod(num_heads, block_m) == 0, "expected q heads ({}) to be divisible by block_m ({})", .{ num_heads, block_m });

    const topk_padded = std.math.ceilPowerOfTwoAssert(usize, @intCast(topk_count));
    const tile_size: usize = @min(topk_padded, 16);
    const num_tiles = std.math.divCeil(usize, @intCast(topk_count), tile_size) catch unreachable;
    const direct_programs: usize = @intCast(query_count * @divExact(num_heads, block_m));
    const core_count = @max(core_count_, 1);

    var max_splits: usize = 1;
    while (max_splits * 2 <= @min(num_tiles, 16)) max_splits *= 2;

    if (requested_splits) |requested| {
        const splits: usize = requested;
        stdx.debug.assert(std.math.isPowerOfTwo(splits), "MLA num_kv_splits ({}) must be a power of two", .{splits});
        stdx.debug.assert(splits <= 16, "MLA num_kv_splits ({}) must not exceed 16", .{splits});
        stdx.debug.assert(splits <= num_tiles, "MLA num_kv_splits ({}) must not exceed sparse tile count ({})", .{ splits, num_tiles });
        return .{
            .block_m = block_m,
            .tile_size = @intCast(tile_size),
            .num_tiles = num_tiles,
            .num_splits = splits,
            .direct_programs = direct_programs,
        };
    }

    var best_splits: usize = 1;
    var best_score: f64 = -1.0;
    const topk_work: f64 = @floatFromInt(topk_count);
    const candidates = [_]usize{ 1, 2, 4, 8, 16 };
    for (candidates) |splits| {
        if (splits > max_splits) break;
        const programs = direct_programs * splits;
        const rounds = std.math.divCeil(usize, programs, core_count) catch unreachable;
        const utilization = @as(f64, @floatFromInt(programs)) /
            @as(f64, @floatFromInt(rounds * core_count));
        // AIter models each extra split as roughly 84 tokens of launch/reduce overhead.
        const work_efficiency = topk_work / (topk_work + 84.1 * @as(f64, @floatFromInt(splits)));
        const score = utilization * work_efficiency;
        if (score > best_score) {
            best_score = score;
            best_splits = splits;
        }
    }

    return .{
        .block_m = block_m,
        .tile_size = @intCast(tile_size),
        .num_tiles = num_tiles,
        .num_splits = best_splits,
        .direct_programs = direct_programs,
    };
}

const Triton = struct {
    fn sparseAttentionShard(
        q: zml.Tensor,
        key_cache: zml.Tensor,
        value_cache: zml.Tensor,
        sink: zml.Tensor,
        use_sink: bool,
        block_tables: zml.Tensor,
        block_table_stride: i64,
        topk_indices: zml.Tensor,
        seq_lens: zml.Tensor,
        query_start_len: zml.Tensor,
        num_seqs: i64,
        opts: AttentionOptions,
        all_decode: bool,
    ) zml.Tensor {
        const rope_rank = opts.rope_rank;
        const q_dim = q.dim(.hd);
        const q_heads = q.dim(.h);
        const nope_rank = q_dim - rope_rank;
        const value_rank = opts.value_rank orelse q_dim;
        const kernel_lora_rank: i64 = @intCast(std.math.ceilPowerOfTwoAssert(usize, @intCast(nope_rank)));

        stdx.debug.assert(q_dim > rope_rank, "expected q head dim ({}) to include a rope tail of {}", .{ q_dim, rope_rank });
        stdx.debug.assert(key_cache.dim(.hd) == q_dim, "expected q and key cache head dims to match, got q={} kv={}", .{ q_dim, key_cache.dim(.hd) });
        stdx.debug.assert(value_rank <= value_cache.dim(.hd), "expected value rank ({}) to fit in value cache head dim ({})", .{ value_rank, value_cache.dim(.hd) });
        stdx.debug.assert(std.math.isPowerOfTwo(@as(usize, @intCast(kernel_lora_rank))), "expected kernel lora rank ({}) to be a power of two", .{kernel_lora_rank});
        stdx.debug.assert(std.math.isPowerOfTwo(@as(usize, @intCast(value_rank))), "expected value rank ({}) to be a power of two", .{value_rank});
        stdx.debug.assert(topk_indices.dim(.q) == q.dim(.q), "expected topk q dim ({}) to match q dim ({})", .{ topk_indices.dim(.q), q.dim(.q) });

        const launch = selectLaunchConfig(
            q.dim(.q),
            q_heads,
            topk_indices.dim(.topk),
            getCoreCount(),
            opts.num_kv_splits,
        );
        log.debug("sparse MLA launch: {any}", .{launch});

        const q_strides = q.shape().computeElementStrides().constSlice();
        const out_shape = q.shape().setDim(.hd, value_rank);
        const out_strides = out_shape.computeElementStrides().constSlice();
        const k_strides = key_cache.shape().computeElementStrides().constSlice();
        const v_strides = value_cache.shape().computeElementStrides().constSlice();
        const sm_scale = opts.scale orelse 1.0 / std.math.sqrt(@as(f32, @floatFromInt(q_dim)));

        const kernel_inputs: kernel.Kernel.Inputs = .{
            .query_ptr = q,
            .key_cache_ptr = key_cache,
            .value_cache_ptr = value_cache,
            .attn_sink_ptr = sink,
            .block_tables_ptr = block_tables,
            .topk_indices_ptr = topk_indices,
            .seq_lens_ptr = seq_lens,
            .scale_ptr = zml.Tensor.scalar(sm_scale, .f32),
            .block_table_stride_ptr = zml.Tensor.scalar(block_table_stride, .i64),
            .query_stride_0_ptr = zml.Tensor.scalar(q_strides[0], .i64),
            .query_stride_1_ptr = zml.Tensor.scalar(q_strides[1], .i64),
            .output_stride_0_ptr = zml.Tensor.scalar(out_strides[0], .i64),
            .output_stride_1_ptr = zml.Tensor.scalar(out_strides[1], .i64),
            .stride_k_cache_0_ptr = zml.Tensor.scalar(k_strides[key_cache.shape().axis(.page)], .i64),
            .stride_k_cache_1_ptr = zml.Tensor.scalar(k_strides[key_cache.shape().axis(.k_chunk)], .i64),
            .stride_k_cache_2_ptr = zml.Tensor.scalar(k_strides[key_cache.shape().axis(.hkv)], .i64),
            .stride_v_cache_0_ptr = zml.Tensor.scalar(v_strides[value_cache.shape().axis(.page)], .i64),
            .stride_v_cache_1_ptr = zml.Tensor.scalar(v_strides[value_cache.shape().axis(.k_chunk)], .i64),
            .stride_v_cache_2_ptr = zml.Tensor.scalar(v_strides[value_cache.shape().axis(.hkv)], .i64),
            .query_start_len_ptr = query_start_len,
            .num_seqs_ptr = zml.Tensor.scalar(num_seqs, .i32),
        };
        const kernel_cfg: kernel.Kernel.Config = .{
            .q_dtype = zml.kernel.triton.from(q.dtype()),
            .kv_dtype = zml.kernel.triton.from(key_cache.dtype()),
            .sink_dtype = zml.kernel.triton.from(sink.dtype()),
            .o_dtype = zml.kernel.triton.from(q.dtype()),
            .num_query_heads = q_heads,
            .num_queries_per_kv = q_heads,
            .block_size = key_cache.dim(.k_chunk),
            .topk_count = topk_indices.dim(.topk),
            .block_m = launch.block_m,
            .rope_rank = rope_rank,
            .qk_lora_rank = nope_rank,
            .kv_lora_rank = kernel_lora_rank,
            .rope_offset = nope_rank,
            .value_rank = value_rank,
            .tile_size = launch.tile_size,
            .num_splits = @intCast(launch.num_splits),
            .use_attn_sink = use_sink,
            .all_decode = all_decode,
        };

        if (launch.num_splits == 1) {
            const output = kernel.Kernel.call(
                kernel_inputs,
                .{ .output = out_shape },
                .{
                    .cfg = kernel_cfg,
                    .grid = .{ @intCast(launch.direct_programs), 1, 1 },
                    .num_warps = 4,
                    .num_stages = 2,
                },
            );
            return output.output;
        }

        const num_splits: i64 = @intCast(launch.num_splits);
        const split_kernel_inputs: kernel.SplitKernel.Inputs = .{
            .query_ptr = kernel_inputs.query_ptr,
            .key_cache_ptr = kernel_inputs.key_cache_ptr,
            .value_cache_ptr = kernel_inputs.value_cache_ptr,
            .attn_sink_ptr = kernel_inputs.attn_sink_ptr,
            .block_tables_ptr = kernel_inputs.block_tables_ptr,
            .topk_indices_ptr = kernel_inputs.topk_indices_ptr,
            .seq_lens_ptr = kernel_inputs.seq_lens_ptr,
            .scale_ptr = kernel_inputs.scale_ptr,
            .block_table_stride_ptr = kernel_inputs.block_table_stride_ptr,
            .query_stride_0_ptr = kernel_inputs.query_stride_0_ptr,
            .query_stride_1_ptr = kernel_inputs.query_stride_1_ptr,
            .output_stride_0_ptr = kernel_inputs.output_stride_0_ptr,
            .output_stride_1_ptr = kernel_inputs.output_stride_1_ptr,
            .stride_k_cache_0_ptr = kernel_inputs.stride_k_cache_0_ptr,
            .stride_k_cache_1_ptr = kernel_inputs.stride_k_cache_1_ptr,
            .stride_k_cache_2_ptr = kernel_inputs.stride_k_cache_2_ptr,
            .stride_v_cache_0_ptr = kernel_inputs.stride_v_cache_0_ptr,
            .stride_v_cache_1_ptr = kernel_inputs.stride_v_cache_1_ptr,
            .stride_v_cache_2_ptr = kernel_inputs.stride_v_cache_2_ptr,
            .query_start_len_ptr = kernel_inputs.query_start_len_ptr,
            .num_seqs_ptr = kernel_inputs.num_seqs_ptr,
        };
        const partials = kernel.SplitKernel.call(
            split_kernel_inputs,
            .{
                .partial_output = zml.Shape.init(.{ q.dim(.q), q_heads, num_splits, value_rank }, .f32),
                .partial_lse = zml.Shape.init(.{ q.dim(.q), q_heads, num_splits }, .f32),
            },
            .{
                .cfg = kernel_cfg,
                .grid = .{ @intCast(launch.direct_programs), @intCast(launch.num_splits), 1 },
                .num_warps = 4,
                .num_stages = 2,
            },
        );
        const output = kernel.ReduceKernel.call(
            .{
                .partial_output_ptr = partials.partial_output,
                .partial_lse_ptr = partials.partial_lse,
                .attn_sink_ptr = sink,
                .output_stride_0_ptr = zml.Tensor.scalar(out_strides[0], .i64),
                .output_stride_1_ptr = zml.Tensor.scalar(out_strides[1], .i64),
            },
            .{ .output = out_shape },
            .{
                .cfg = .{
                    .sink_dtype = zml.kernel.triton.from(sink.dtype()),
                    .o_dtype = zml.kernel.triton.from(q.dtype()),
                    .num_query_heads = q_heads,
                    .value_rank = value_rank,
                    .num_splits = num_splits,
                    .use_attn_sink = use_sink,
                },
                .grid = .{ @intCast(q.dim(.q)), @intCast(q_heads), 1 },
                .num_warps = 1,
                .num_stages = 1,
            },
        );
        return output.output;
    }

    pub fn sparseAttention(q: zml.Tensor, kv: zml.Tensor, topk: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        // q: [batch, q, h, hd], kv: [batch, kv, hd], topk: [batch, seq, topk]
        const batch = q.dim(.batch);
        const q_len = q.dim(.q);
        const q_final = q.merge(.{ .q = .{ .batch, .q } });
        const q_dim = q_final.dim(.hd);

        stdx.debug.assert(kv.dim(.hd) == q_dim, "expected q and kv head dims to match, got q={} kv={}", .{ q_dim, kv.dim(.hd) });
        stdx.debug.assert(topk.dim(.seq) == q_len, "expected topk seq dim ({}) to match q dim ({})", .{ topk.dim(.seq), q_len });

        const kv_final = kv.merge(.{ .kv = .{ .batch, .kv } });
        const key_cache = kv_final.reshape(.{
            .page = kv_final.dim(.kv),
            .k_chunk = 1,
            .hkv = 1,
            .hd = q_dim,
        });
        const value_cache = kv_final.reshape(.{
            .page = kv_final.dim(.kv),
            .k_chunk = 1,
            .hkv = 1,
            .hd = q_dim,
        });

        const topk_i64 = topk.convert(.i64);
        const batch_offsets = zml.Tensor.iota(topk_i64.shape(), .batch)
            .convert(.i64)
            .mul(zml.Tensor.scalar(kv.dim(.kv), .i64).broad(topk_i64.shape()));
        const valid_topk = topk_i64.cmp(.GE, zml.Tensor.scalar(0, .i64).broad(topk_i64.shape()));
        const topk_final = zml.Tensor.select(
            valid_topk,
            topk_i64.add(batch_offsets),
            zml.Tensor.scalar(-1, .i64).broad(topk_i64.shape()),
        ).merge(.{ .q = .{ .batch, .seq } }).convert(.i32);
        const dummy = zml.Tensor.scalar(0, q.dtype());
        const query_start_len = zml.Tensor.arange(.{ .end = batch + 1 }, .i32)
            .mul(zml.Tensor.scalar(q_len, .i32));
        const seq_lens = zml.Tensor.scalar(kv.dim(.kv), .i32)
            .broad(.init(.{ .batch = batch }, .i32));
        const output = sparseAttentionShard(
            q_final,
            key_cache,
            value_cache,
            dummy,
            false,
            zml.Tensor.scalar(0, .i32),
            1,
            topk_final,
            seq_lens,
            query_start_len,
            batch,
            opts,
            q_len == 1,
        );
        return output.reshape(q.shape().setDim(.hd, opts.value_rank orelse q_dim));
    }

    pub const paged = struct {
        const Self = @This();

        pub const Options = struct {
            batch_size: usize,
            max_num_pages: usize,
            max_seqlen_q: usize,
            is_prefill: bool,

            pub fn isPrefill(self: Self.Options) bool {
                return self.is_prefill;
            }

            pub fn maxNumPages(self: Self.Options) usize {
                return self.max_num_pages;
            }
        };

        pub const Parameters = struct {
            block_table: zml.Tensor,
            seq_lens: zml.Tensor,
            query_start_len: zml.Tensor,
            num_warps: i32 = 4,
            num_stages: i32 = 2,
            options_: Self.Options,

            pub fn init(options_: Self.Options) Self.Parameters {
                return .{
                    .block_table = .init(.{ .b = options_.batch_size, .p = options_.max_num_pages }, .i32),
                    .seq_lens = .init(.{ .b = options_.batch_size }, .i32),
                    .query_start_len = .init(.{ .b = options_.batch_size + 1 }, .i32),
                    .options_ = options_,
                };
            }

            pub fn allocationSize(self: Self.Parameters) usize {
                var allocation_size: usize = 0;

                allocation_size += self.block_table.byteSize();
                allocation_size += self.seq_lens.byteSize();
                allocation_size += self.query_start_len.byteSize();

                return allocation_size;
            }

            pub fn options(self: Self.Parameters) Self.Options {
                return self.options_;
            }

            pub fn onMemory(self: Self.Parameters, memory: zml.platform.Memory.Kind) Self.Parameters {
                return .{
                    .options_ = self.options_,
                    .block_table = self.block_table.onMemory(memory),
                    .seq_lens = self.seq_lens.onMemory(memory),
                    .query_start_len = self.query_start_len.onMemory(memory),
                };
            }

            pub fn toMemory(self: Self.Parameters, memory: zml.platform.Memory.Kind) Self.Parameters {
                return .{
                    .options_ = self.options_,
                    .block_table = self.block_table.toMemory(memory),
                    .seq_lens = self.seq_lens.toMemory(memory),
                    .query_start_len = self.query_start_len.toMemory(memory),
                };
            }
        };

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

        fn topkToPhysical(parameters: anytype, topk: zml.Tensor, tokens_pos: zml.Tensor, block_size: i64) zml.Tensor {
            const topk_i32 = topk.convert(.i32);
            const topk_shape = topk_i32.shape();

            stdx.debug.assert(topk_shape.hasTags(.{ .q, .topk }), "paged MLA topk must have .q and .topk axes, got {f}", .{topk_shape});
            stdx.debug.assert(tokens_pos.shape().hasTags(.{.q}), "paged MLA token positions must have a .q axis, got {f}", .{tokens_pos.shape()});

            const query_to_sequence = tokenToSequence(parameters.query_start_len, topk_i32.dim(.q));
            const sequence_ends = parameters.query_start_len.slice1d(.b, .{ .start = 1 }).rename(.{ .b = .seq });
            const last_query = sequence_ends.gather(.{ .seq = query_to_sequence }, .{}).subConstant(1);
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

        fn sparseAttentionShard(
            q: zml.Tensor,
            kv_cache: zml.Tensor,
            sink: zml.Tensor,
            use_sink: bool,
            topk: zml.Tensor,
            tokens_pos: zml.Tensor,
            parameters: triton_attn.paged.Parameters,
            opts: AttentionOptions,
        ) zml.Tensor {
            // q: [q, h, hd]
            // kv_cache: [page, k_chunk, hkv=1, hd]
            // topk: [q, topk] absolute logical token ids, with -1 for padding.
            const block_size = kv_cache.dim(.k_chunk);
            const topk_final = topkToPhysical(parameters, topk, tokens_pos, block_size);
            return Triton.sparseAttentionShard(
                q,
                kv_cache,
                kv_cache,
                sink,
                use_sink,
                parameters.block_table,
                parameters.block_table.dim(.p),
                topk_final,
                parameters.seq_lens,
                parameters.query_start_len,
                parameters.block_table.dim(.b),
                opts,
                !parameters.options_.is_prefill,
            );
        }

        pub fn sparseAttention(
            q: zml.Tensor,
            kv: zml.Tensor,
            sink: ?zml.Tensor,
            topk: zml.Tensor,
            tokens_pos: zml.Tensor,
            parameters: triton_attn.paged.Parameters,
            opts: AttentionOptions,
        ) zml.Tensor {
            stdx.debug.assert(q.shape().hasTags(.{ .q, .h, .hd }), "expected q to have tags .q, .h, .hd after flattening, got {f}", .{q.shape()});
            stdx.debug.assert(kv.shape().hasTags(.{ .page, .k_chunk, .hkv, .hd }), "expected paged MLA KV cache to have tags .page, .k_chunk, .hkv, .hd, got {f}", .{kv.shape()});
            stdx.debug.assert(q.dim(.hd) > opts.rope_rank, "expected q head dim ({}) to include a rope tail of {}", .{ q.dim(.hd), opts.rope_rank });
            stdx.debug.assert(kv.dim(.hd) == q.dim(.hd), "expected q and kv cache head dims to match, got q={} kv={}", .{ q.dim(.hd), kv.dim(.hd) });

            const out_shape = q.shape().setDim(.hd, opts.value_rank orelse q.dim(.hd));
            const use_sink = sink != null;
            const sink_tensor = if (sink) |sink_| sink_ else zml.Tensor
                .zeroes(zml.Shape.init(.{ .h = q.dim(.h) }, q.dtype()))
                .withPartitioning(.{ .h = .model });

            return zml.ops.manualComputation(
                .{
                    q,
                    kv,
                    sink_tensor,
                    topk,
                    tokens_pos,
                    parameters.block_table,
                    parameters.seq_lens,
                    parameters.query_start_len,
                },
                out_shape,
                .{
                    .opts = opts,
                    .options = parameters.options_,
                    .use_sink = use_sink,
                },
                (struct {
                    fn body(ctx_: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, _: zml.Shape) zml.Tensor {
                        const parameters_: triton_attn.paged.Parameters = .{
                            .block_table = sharded_inputs[5],
                            .seq_lens = sharded_inputs[6],
                            .query_start_len = sharded_inputs[7],
                            .options_ = ctx_.options,
                        };

                        return Self.sparseAttentionShard(
                            sharded_inputs[0],
                            sharded_inputs[1],
                            sharded_inputs[2],
                            ctx_.use_sink,
                            sharded_inputs[3],
                            sharded_inputs[4],
                            parameters_,
                            ctx_.opts,
                        );
                    }
                }).body,
            );
        }
    };
};

pub const AttentionOptions = struct {
    rope_rank: i64,
    value_rank: ?i64 = null,
    scale: ?f32 = null,
    /// null selects automatically; 1 forces direct mode; other values must be powers of two up to 16.
    num_kv_splits: ?u8 = null,
};

test "sparse MLA launch selection balances occupancy and split overhead" {
    const saturated = selectLaunchConfig(120, 16, 512, 120, null);
    try std.testing.expectEqual(@as(usize, 1), saturated.num_splits);

    const under_occupied = selectLaunchConfig(1, 16, 512, 120, null);
    try std.testing.expectEqual(@as(usize, 16), under_occupied.num_splits);

    const tile_limited = selectLaunchConfig(1, 16, 32, 120, null);
    try std.testing.expectEqual(@as(usize, 2), tile_limited.num_splits);

    const missing_core_count = selectLaunchConfig(1, 16, 512, 0, null);
    try std.testing.expectEqual(@as(usize, 1), missing_core_count.num_splits);
}

test "sparse MLA launch selection honors valid explicit splits" {
    const forced = selectLaunchConfig(1, 16, 64, 120, 4);
    try std.testing.expectEqual(@as(usize, 4), forced.num_splits);

    const one_tile = selectLaunchConfig(1, 16, 5, 120, null);
    try std.testing.expectEqual(@as(i64, 8), one_tile.tile_size);
    try std.testing.expectEqual(@as(usize, 1), one_tile.num_splits);
}

test "sparse MLA emits direct and split Triton kernels" {
    const platform = zml.testing.env();
    var compilation = zml.module.CompilationContext.init(std.testing.allocator, std.testing.io, platform, .{});
    defer compilation.deinit();
    compilation.activate();
    defer compilation.deactivate();

    const block = @import("mlir").Block.init(&.{}, &.{});
    compilation.pushBlock(block);
    defer compilation.popBlock();

    const q = zml.Tensor.zeroes(zml.Shape.init(.{ .batch = 1, .q = 1, .h = 16, .hd = 128 }, .bf16));
    const kv = zml.Tensor.zeroes(zml.Shape.init(.{ .batch = 1, .kv = 32, .hd = 128 }, .bf16));
    const topk = zml.Tensor.zeroes(zml.Shape.init(.{ .batch = 1, .seq = 1, .topk = 32 }, .i32));

    const direct = Triton.sparseAttention(q, kv, topk, .{ .rope_rank = 64, .num_kv_splits = 1 });
    try std.testing.expect(direct.shape().eql(q.shape()));
    try std.testing.expect(direct.value().owner().verify());

    const split = Triton.sparseAttention(q, kv, topk, .{ .rope_rank = 64, .num_kv_splits = 2 });
    try std.testing.expect(split.shape().eql(q.shape()));
    try std.testing.expect(split.value().owner().verify());

    const q_flat = zml.Tensor.zeroes(zml.Shape.init(.{ .q = 1, .h = 16, .hd = 128 }, .bf16));
    const cache = zml.Tensor.zeroes(zml.Shape.init(.{ .page = 32, .k_chunk = 1, .hkv = 1, .hd = 128 }, .bf16));
    const sink = zml.Tensor.zeroes(zml.Shape.init(.{ .h = 16 }, .bf16));
    const topk_flat = zml.Tensor.zeroes(zml.Shape.init(.{ .q = 1, .topk = 32 }, .i32));
    const seq_lens = zml.Tensor.scalar(32, .i32).reshape(.{1});
    const query_start_len = zml.Tensor.arange(.{ .end = 2 }, .i32);

    const direct_with_sink = Triton.sparseAttentionShard(
        q_flat,
        cache,
        cache,
        sink,
        true,
        zml.Tensor.scalar(0, .i32),
        1,
        topk_flat,
        seq_lens,
        query_start_len,
        1,
        .{ .rope_rank = 64, .num_kv_splits = 1 },
        true,
    );
    try std.testing.expect(direct_with_sink.value().owner().verify());

    const split_with_sink = Triton.sparseAttentionShard(
        q_flat,
        cache,
        cache,
        sink,
        true,
        zml.Tensor.scalar(0, .i32),
        1,
        topk_flat,
        seq_lens,
        query_start_len,
        1,
        .{ .rope_rank = 64, .num_kv_splits = 2 },
        true,
    );
    try std.testing.expect(split_with_sink.value().owner().verify());
}

pub const Backend = enum {
    triton,

    pub fn auto(platform: *const zml.Platform) Backend {
        return switch (platform.target) {
            .cuda => .triton,
            .rocm => .triton,
            .oneapi => .triton,
            else => stdx.debug.panic("Paged attention is not supported on {s} yet", .{@tagName(platform.target)}),
        };
    }
};

pub const Parameters = union(Backend) {
    triton: void,
};

pub fn sparseAttention(q: zml.Tensor, kv: zml.Tensor, topk: zml.Tensor, parameters: Parameters, opts: AttentionOptions) zml.Tensor {
    return switch (parameters) {
        .triton => Triton.sparseAttention(q, kv, topk, opts),
    };
}

pub const paged = struct {
    pub const Parameters = union(Backend) {
        triton: Triton.paged.Parameters,
    };
};

fn vanillaSparseAttention(q: zml.Tensor, kv: zml.Tensor, sink: ?zml.Tensor, topk: zml.Tensor, opts: AttentionOptions) zml.Tensor {
    const mask = topk.cmp(.GE, zml.Tensor.zeroes(topk.shape())).insertAxes(.topk, .{.h});
    const selected_kv = kv.gather(.{ .kv = topk }, .{}).rename(.{ .b = .q, .topk = .kv }).convert(.f32);

    const dims = zml.nn.collectDims(.{ .h, .q, .kv, .hd }, &.{ q, kv }, .strict) catch {
        stdx.debug.panic("Inputs have incompatible shapes (q: {f}, kv: {f}).", .{ q, kv });
    };

    const sqrt_head_dim = opts.scale orelse 1.0 / std.math.sqrt(@as(f32, @floatFromInt(dims.hd)));
    const q_32 = q.convert(.f32);
    var scores = q_32.dot(selected_kv, .hd).scale(sqrt_head_dim);
    scores = zml.Tensor.select(mask.broad(scores.shape()), scores, zml.Tensor.constant(scores.dtype().minValue()));

    const sink_shape = q.shape().set(.hd, 1);
    const attn_sink = sink orelse stdx.debug.panic("ragged MLA attention requires an attention sink", .{});
    const sink_ = attn_sink.insertAxes(0, .{.q}).insertAxes(.last, .{.hd}).broad(sink_shape);
    const scores_sink = zml.Tensor.concatenate(&.{ scores, sink_.convert(scores.dtype()) }, .kv);

    const attn_weights = scores_sink.softmax(.kv);
    const attn_weights_non_sink = attn_weights.slice(&.{
        .{},
        .{},
        .{ .end = topk.dim(.topk) },
    });
    return attn_weights_non_sink.dot(selected_kv, .kv).convert(q.dtype());
}

pub fn pagedSparseAttention(
    q: zml.Tensor,
    kv: zml.Tensor,
    sink: ?zml.Tensor,
    topk: zml.Tensor,
    tokens_pos: zml.Tensor,
    parameters: paged_attn.Parameters,
    opts: AttentionOptions,
) zml.Tensor {
    return switch (parameters) {
        .triton => |params| blk: {
            break :blk Triton.paged.sparseAttention(
                q.rename(.{ .b = .q }),
                kv,
                sink,
                topk.rename(.{ .b = .q }),
                tokens_pos.rename(.{ .b = .q }),
                params,
                opts,
            );
        },
        else => {
            const kv_final = kv.merge(.{ .kv = .{ .page, .k_chunk, .hkv } });
            return vanillaSparseAttention(q, kv_final, sink, topk, opts);
        },
    };
}
