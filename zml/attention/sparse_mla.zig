const std = @import("std");
const stdx = @import("stdx");
const zml = @import("../zml.zig");
const triton = @import("triton_attention.zig");
const fly = @import("fly_kernels/sparse_mla.zig");
const MlaOptions = @import("paged_attention.zig").Mla.Options;

pub const Backend = enum {
    triton,
    fly,

    pub fn auto(platform: *const zml.Platform, dtype: zml.DataType) Backend {
        return switch (platform.target) {
            .rocm => switch (zml.platform.rocm.computeCapability(platform) orelse return .triton) {
                .gfx942 => switch (dtype) {
                    .bf16 => .fly,
                    else => .triton,
                },
                else => .triton,
            },
            else => .triton,
        };
    }

    fn call(self: Backend, q: zml.Tensor, kv_cache: zml.Tensor, sink: ?zml.Tensor, topk: zml.Tensor, active_query_count: zml.Tensor, opts: Options) zml.Tensor {
        const backend = switch (self) {
            .fly => auto(zml.Compiler.current().platform, q.dtype()),
            .triton => .triton,
        };
        switch (backend) {
            .fly => if (fly.pagedAttention(q, kv_cache, sink, topk, active_query_count, opts, triton.getCuCount())) |output| return output,
            .triton => {},
        }
        return triton.paged.pagedSparseMlaKernel(q, kv_cache, sink, topk, active_query_count, opts);
    }
};

pub const Options = struct {
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

pub const Config = struct {
    block_m: usize,
    tile_size: usize,
    num_splits: usize,
    direct_programs: usize,
    main_num_warps: usize = 4,
    main_num_stages: usize = 2,
    grouped_reduce_threshold: usize = 32,
    splits_per_group: usize = 16,
    grouped_reduce_num_warps: usize = 1,
    parallel_reduce_min_splits: usize = std.math.maxInt(usize),
    parallel_reduce_num_warps: usize = 1,
};

pub fn launchConfig(paged_opts: Options, topk_count: usize, cu_count_: usize) Config {
    stdx.debug.assert(paged_opts.total_q_blocks > 0, "sparse MLA requires at least one query", .{});
    stdx.debug.assert(paged_opts.num_heads > 0, "sparse MLA requires at least one query head", .{});
    stdx.debug.assert(topk_count > 0, "sparse MLA requires at least one top-k entry", .{});

    const padded_heads = std.math.ceilPowerOfTwoAssert(usize, paged_opts.num_heads);
    const padded_topk = std.math.ceilPowerOfTwoAssert(usize, topk_count);
    var config: Config = .{
        .block_m = @min(padded_heads, 16),
        .tile_size = @min(padded_topk, 16),
        .num_splits = 1,
        .direct_programs = undefined,
    };
    const cc = zml.platform.cuda.computeCapability(zml.Compiler.current().platform);
    const is_sm103 = if (cc) |value| value.eql(.{ .major = 10, .minor = 3 }) else false;

    // GB300 (sm_103): a single wide query benefits from more head blocks and
    // wider sparse tiles. The split selection below still derives each layer's
    // split count from query/head/top-k shapes (16 for DSV4 CSA/HCA and 4 for
    // its 128-entry full-attention layers).
    if (is_sm103 and
        paged_opts.all_decode and
        paged_opts.total_q_blocks == 1 and
        paged_opts.value_rank >= 512)
    {
        config.block_m = @min(padded_heads, 8);
        config.tile_size = @min(padded_topk, 32);
        config.parallel_reduce_min_splits = 16;
        config.parallel_reduce_num_warps = 2;
    }
    // Full 256-query prefill chunks already expose enough query parallelism.
    // Retain BLOCK_M=16 and the generic split model, but process wider sparse
    // tiles. The predicate uses the post-sharding flattened query shape.
    if (is_sm103 and
        !paged_opts.all_decode and
        paged_opts.total_q_blocks >= 256 and
        paged_opts.value_rank >= 512)
    {
        config.tile_size = @min(padded_topk, 32);
    }

    const num_tiles = std.math.divCeil(usize, topk_count, config.tile_size) catch unreachable;
    const head_blocks = std.math.divCeil(usize, paged_opts.num_heads, config.block_m) catch unreachable;
    config.direct_programs = paged_opts.total_q_blocks * head_blocks;
    const cu_count = @max(cu_count_, 1);

    if (paged_opts.num_kv_splits) |requested| {
        const num_splits: usize = requested;
        stdx.debug.assert(std.math.isPowerOfTwo(num_splits), "MLA num_kv_splits ({}) must be a power of two", .{num_splits});
        stdx.debug.assert(num_splits <= 128, "MLA num_kv_splits ({}) must not exceed 128", .{num_splits});
        stdx.debug.assert(num_splits <= num_tiles, "MLA num_kv_splits ({}) must not exceed sparse tile count ({})", .{ num_splits, num_tiles });
        config.num_splits = num_splits;
    } else {
        var best_cost: usize = std.math.maxInt(usize);
        const candidates = [_]usize{ 1, 2, 4, 8, 16, 32, 64, 128 };
        for (candidates) |num_splits| {
            if (num_splits > num_tiles) break;
            const programs = config.direct_programs * num_splits;
            const rounds = std.math.divCeil(usize, programs, cu_count) catch unreachable;
            const tiles_per_program = std.math.divCeil(usize, num_tiles, num_splits) catch unreachable;
            const cost = rounds * tiles_per_program;
            // Candidates are ordered by split count, so ties retain 2D or the lower-overhead 3D launch.
            if (cost < best_cost) {
                best_cost = cost;
                config.num_splits = num_splits;
            }
        }
    }

    return config;
}

pub fn pagedAttention(parameters: triton.paged.Parameters, q: zml.Tensor, kv_cache: zml.Tensor, sink: ?zml.Tensor, topk: zml.Tensor, tokens_pos: zml.Tensor, opts: MlaOptions) zml.Tensor {
    const output_shape = q.shape().set(.hd, opts.value_rank);
    return zml.ops.manualComputation(
        (struct {
            q: zml.Tensor,
            kv_cache: zml.Tensor,
            sink: ?zml.Tensor,
            topk: zml.Tensor,
            tokens_pos: zml.Tensor,
            block_table: zml.Tensor,
            seq_lens: zml.Tensor,
            query_start_len: zml.Tensor,
            opts: MlaOptions,
            options: triton.paged.Options,

            fn body(self: @This(), _: zml.Shape) zml.Tensor {
                const block_size = self.kv_cache.dim(.k_chunk);

                const parameters_: triton.paged.Parameters = .{
                    .block_table = self.block_table,
                    .seq_lens = self.seq_lens,
                    .query_start_len = self.query_start_len,
                    .options_ = self.options,
                };

                const topk_final = triton.paged.topkToPhysical(parameters_, self.topk, self.tokens_pos, block_size);
                const active_query_count = self.query_start_len
                    .slice(.b, .{ .start = self.query_start_len.dim(.b) - 1 })
                    .squeeze(.b);
                stdx.debug.assert(topk_final.dim(.q) == self.q.dim(.q), "expected topk q dim ({}) to match q dim ({})", .{ topk_final.dim(.q), self.q.dim(.q) });

                const num_heads: usize = @intCast(self.q.dim(.h));
                const paged_opts: Options = .{
                    .qk_rank = @intCast(self.q.dim(.hd)),
                    .value_rank = @intCast(self.opts.value_rank),
                    .num_heads = num_heads,
                    .block_size = @intCast(self.kv_cache.dim(.k_chunk)),
                    .rope_rank = @intCast(self.opts.rope_rank),
                    .scale = self.opts.scale,
                    .total_q_blocks = @intCast(self.q.dim(.q)),
                    .num_kv_splits = self.opts.num_kv_splits,
                    .all_decode = !self.options.is_prefill,
                };

                return self.opts.backend.call(
                    self.q,
                    self.kv_cache,
                    self.sink,
                    topk_final,
                    active_query_count,
                    paged_opts,
                );
            }
        }).body,
        .{
            .q = q,
            .kv_cache = kv_cache,
            .sink = sink,
            .topk = topk,
            .tokens_pos = tokens_pos,
            .block_table = parameters.block_table,
            .seq_lens = parameters.seq_lens,
            .query_start_len = parameters.query_start_len,
            .opts = opts,
            .options = parameters.options_,
        },
        output_shape,
        .{ .manual_axes = .{ .data, .model } },
    );
}
