const std = @import("std");

const zml = @import("../zml.zig");
const paged_attn = @import("paged_attention.zig");
const triton_attn = @import("triton_attention.zig");
const kernel = @import("triton_kernels/unified_sparse_mla.zig");

const stdx = zml.stdx;

const log = std.log.scoped(.mla);

const Triton = struct {
    pub fn sparseAttention(q: zml.Tensor, kv: zml.Tensor, topk: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        // q: [batch, q, h, hd], kv: [batch, kv, hd], topk: [batch, seq, topk]
        const rope_rank: i64 = 64;
        const batch = q.dim(.batch);
        const q_len = q.dim(.q);
        const q_final = q.merge(.{ .q = .{ .batch, .q } });
        const q_dim = q_final.dim(.hd);
        const q_heads = q_final.dim(.h);
        const nope_rank = q_dim - rope_rank;
        const kernel_lora_rank: i64 = @intCast(std.math.ceilPowerOfTwoAssert(usize, @intCast(nope_rank)));

        stdx.debug.assert(q_dim > rope_rank, "expected q head dim ({}) to include a rope tail of {}", .{ q_dim, rope_rank });
        stdx.debug.assert(std.math.isPowerOfTwo(@as(usize, @intCast(kernel_lora_rank))), "expected kernel lora rank ({}) to be a power of two", .{kernel_lora_rank});
        stdx.debug.assert(std.math.isPowerOfTwo(@as(usize, @intCast(q_dim))), "expected value rank ({}) to be a power of two", .{q_dim});
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

        const block_m: i64 = @min(q_heads, 16);
        stdx.debug.assert(@mod(q_heads, block_m) == 0, "expected q heads ({}) to be divisible by block_m ({})", .{ q_heads, block_m });

        const q_strides = q_final.shape().computeElementStrides().constSlice();
        const out_shape: zml.Shape = .init(.{ .q = q_final.dim(.q), .h = q_heads, .hd = q_dim }, q.dtype());
        const out_strides = out_shape.computeElementStrides().constSlice();
        const k_strides = key_cache.shape().computeElementStrides().constSlice();
        const v_strides = value_cache.shape().computeElementStrides().constSlice();

        const sm_scale = opts.scale orelse 1.0 / std.math.sqrt(@as(f32, @floatFromInt(q_dim)));

        const out = kernel.Kernel.call(.{
            .query_ptr = q_final,
            .key_cache_ptr = key_cache,
            .value_cache_ptr = value_cache,
            .attn_sink_ptr = opts.attn_sink,
            .block_tables_ptr = zml.Tensor.constant(zml.DataType.i32.zero()).reshape(.{1}),
            .topk_indices_ptr = topk_final,
            .seq_lens_ptr = zml.Tensor.constant(.{ .i32 = @as(i32, @intCast(kv.dim(.kv))) }).broad(.init(.{ .batch = batch }, .i32)),
            .scale_ptr = zml.Tensor.constant(zml.DataType.f32.constant(sm_scale)),
            .block_table_stride_ptr = zml.Tensor.constant(zml.DataType.i64.constant(1)),
            .query_stride_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(q_strides[0])),
            .query_stride_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(q_strides[1])),
            .output_stride_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(out_strides[0])),
            .output_stride_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(out_strides[1])),
            .stride_k_cache_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[0])),
            .stride_k_cache_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[1])),
            .stride_k_cache_2_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[2])),
            .stride_v_cache_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[0])),
            .stride_v_cache_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[1])),
            .stride_v_cache_2_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[2])),
            .query_start_len_ptr = zml.Tensor.arange(.{ .end = batch + 1 }, .i32).mul(zml.Tensor.scalar(q_len, .i32)),
            .num_seqs_ptr = zml.Tensor.constant(.{ .i32 = @as(i32, @intCast(batch)) }).reshape(.{1}),
        }, .{
            .output = out_shape,
        }, .{
            .cfg = .{
                .q_dtype = zml.kernel.triton.from(q_final.dtype()),
                .kv_dtype = zml.kernel.triton.from(kv_final.dtype()),
                .sink_dtype = zml.kernel.triton.from(opts.attn_sink.dtype()),
                .o_dtype = zml.kernel.triton.from(q_final.dtype()),
                .num_query_heads = q_heads,
                .num_queries_per_kv = q_heads,
                .block_size = 1,
                .topk_count = topk_final.dim(.topk),
                .block_m = block_m,
                .rope_rank = rope_rank,
                .qk_lora_rank = nope_rank,
                .kv_lora_rank = kernel_lora_rank,
                .rope_offset = nope_rank,
                .value_rank = q_dim,
                .tile_size = @min(topk_final.dim(.topk), 16),
                .use_attn_sink = true,
                .all_decode = q_len == 1,
            },
            .grid = .{ @intCast(q_final.dim(.q) * @divExact(q_heads, block_m)), 1, 1 },
            .num_warps = @intCast(4),
            .num_stages = @intCast(2),
        });

        return out.output.reshape(q.shape());
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
            sink_: ?zml.Tensor,
            topk: zml.Tensor,
            tokens_pos: zml.Tensor,
            parameters: triton_attn.paged.Parameters,
            opts: AttentionOptions,
        ) zml.Tensor {
            // q: [q, h, hd]
            // kv_cache: [page, k_chunk, hkv=1, hd]
            // topk: [q, topk] absolute logical token ids, with -1 for padding.
            const rope_rank = opts.rope_rank;
            const q_dim = q.dim(.hd);
            const q_heads = q.dim(.h);
            const nope_rank = q_dim - rope_rank;
            const kernel_lora_rank: i64 = @intCast(std.math.ceilPowerOfTwoAssert(usize, @intCast(nope_rank)));
            stdx.debug.assert(std.math.isPowerOfTwo(@as(usize, @intCast(kernel_lora_rank))), "expected kernel lora rank ({}) to be a power of two", .{kernel_lora_rank});

            const block_size = kv_cache.dim(.k_chunk);

            const topk_final = topkToPhysical(parameters, topk, tokens_pos, block_size);
            stdx.debug.assert(topk_final.dim(.q) == q.dim(.q), "expected topk q dim ({}) to match q dim ({})", .{ topk_final.dim(.q), q.dim(.q) });

            const block_m: i64 = @min(q_heads, 16);
            stdx.debug.assert(@mod(q_heads, block_m) == 0, "expected q heads ({}) to be divisible by block_m ({})", .{ q_heads, block_m });

            const q_strides = q.shape().computeElementStrides().constSlice();
            const out_shape = q.shape();
            const out_strides = out_shape.computeElementStrides().constSlice();
            const k_strides = kv_cache.shape().computeElementStrides().constSlice();
            const v_strides = kv_cache.shape().computeElementStrides().constSlice();
            const sm_scale = opts.scale orelse 1.0 / std.math.sqrt(@as(f32, @floatFromInt(q_dim)));

            const sink = sink_ orelse zml.Tensor.zeroes(zml.Shape.init(.{ .h = q.dim(.h) }, q.dtype()));

            const out = kernel.Kernel.call(.{
                .query_ptr = q,
                .key_cache_ptr = kv_cache,
                .value_cache_ptr = kv_cache,
                .attn_sink_ptr = sink,
                .block_tables_ptr = parameters.block_table,
                .topk_indices_ptr = topk_final,
                .seq_lens_ptr = parameters.seq_lens,
                .scale_ptr = zml.Tensor.constant(zml.DataType.f32.constant(sm_scale)),
                .block_table_stride_ptr = zml.Tensor.constant(zml.DataType.i64.constant(parameters.block_table.dim(.p))),
                .query_stride_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(q_strides[0])),
                .query_stride_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(q_strides[1])),
                .output_stride_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(out_strides[0])),
                .output_stride_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(out_strides[1])),
                .stride_k_cache_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[kv_cache.shape().axis(.page)])),
                .stride_k_cache_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[kv_cache.shape().axis(.k_chunk)])),
                .stride_k_cache_2_ptr = zml.Tensor.constant(zml.DataType.i64.constant(k_strides[kv_cache.shape().axis(.hkv)])),
                .stride_v_cache_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[kv_cache.shape().axis(.page)])),
                .stride_v_cache_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[kv_cache.shape().axis(.k_chunk)])),
                .stride_v_cache_2_ptr = zml.Tensor.constant(zml.DataType.i64.constant(v_strides[kv_cache.shape().axis(.hkv)])),
                .query_start_len_ptr = parameters.query_start_len,
                .num_seqs_ptr = zml.Tensor.constant(.{ .i32 = @as(i32, @intCast(parameters.block_table.dim(.b))) }).reshape(.{1}),
            }, .{
                .output = out_shape,
            }, .{
                .cfg = .{
                    .q_dtype = zml.kernel.triton.from(q.dtype()),
                    .kv_dtype = zml.kernel.triton.from(kv_cache.dtype()),
                    .sink_dtype = zml.kernel.triton.from(sink.dtype()),
                    .o_dtype = zml.kernel.triton.from(q.dtype()),
                    .num_query_heads = q_heads,
                    .num_queries_per_kv = q_heads,
                    .block_size = block_size,
                    .topk_count = topk_final.dim(.topk),
                    .block_m = block_m,
                    .rope_rank = rope_rank,
                    .qk_lora_rank = nope_rank,
                    .kv_lora_rank = kernel_lora_rank,
                    .rope_offset = nope_rank,
                    .value_rank = q_dim,
                    .tile_size = @min(topk_final.dim(.topk), 16),
                    .use_attn_sink = if (sink_) |_| true else false,
                    .all_decode = !parameters.options_.is_prefill,
                },
                .grid = .{ @intCast(q.dim(.q) * @divExact(q_heads, block_m)), 1, 1 },
                .num_warps = 4,
                .num_stages = 2,
            });

            return out.output.reshape(q.shape());
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
            stdx.debug.assert(std.math.isPowerOfTwo(@as(usize, @intCast(q.dim(.hd)))), "expected value rank ({}) to be a power of two", .{q.dim(.hd)});
            stdx.debug.assert(kv.dim(.hd) == q.dim(.hd), "expected q and kv cache head dims to match, got q={} kv={}", .{ q.dim(.hd), kv.dim(.hd) });

            return zml.ops.manualComputation(
                .{
                    q,
                    kv,
                    sink,
                    topk,
                    tokens_pos,
                    parameters.block_table,
                    parameters.seq_lens,
                    parameters.query_start_len,
                    sink,
                },
                q.shape(),
                .{
                    .opts = opts,
                    .options = parameters.options_,
                },
                (struct {
                    fn body(ctx_: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, _: zml.Shape) zml.Tensor {
                        const parameters_: triton_attn.paged.Parameters = .{
                            .block_table = sharded_inputs[5],
                            .seq_lens = sharded_inputs[6],
                            .query_start_len = sharded_inputs[7],
                            .options_ = ctx_.options,
                        };

                        return sparseAttentionShard(
                            sharded_inputs[0],
                            sharded_inputs[1],
                            sharded_inputs[2],
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
    scale: ?f32 = null,
};

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
