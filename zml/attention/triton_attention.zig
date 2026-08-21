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

const max_num_warps: usize = 8;
const max_num_stages: usize = 4;
const max_block_m: usize = 1024;
const max_tile_size: usize = 1024;

pub const TuningValidationError = error{
    InvalidQueriesPerKv,
    InvalidBlockM,
    InvalidTileSize,
    BlockMNotDivisibleByQueriesPerKv,
    InvalidNumSegmentsPerSeq,
    InvalidNumWarps,
    InvalidNumStages,
};

/// Compile-time launch knobs for the whole-sequence unified-attention kernel.
/// Shape-dependent values such as `block_q` and the launch grid are derived
/// after sharding and deliberately are not part of this configuration.
pub const Tuning2D = struct {
    block_m: usize,
    tile_size: usize,
    num_warps: usize,
    num_stages: usize,
};

/// Compile-time launch knobs shared by the split-K attention and reduction
/// kernels. Both kernels use the same tile and segment count.
pub const Tuning3D = struct {
    block_m: usize,
    tile_size: usize,
    num_segments_per_seq: usize,
    attention_num_warps: usize,
    attention_num_stages: usize,
    reduce_num_warps: usize,
    reduce_num_stages: usize,
};

/// Selects the existing heuristic or forces one unified-attention kernel
/// family. Keep `.automatic` first in a tuning search as its safe fallback.
pub const TuningConfig = union(enum) {
    automatic,
    two_d: Tuning2D,
    three_d: Tuning3D,

    pub fn validate(self: TuningConfig, num_queries_per_kv: usize) TuningValidationError!void {
        if (num_queries_per_kv == 0) return error.InvalidQueriesPerKv;
        switch (self) {
            .automatic => return,
            .two_d => |config| {
                try validateCommon(config.block_m, config.tile_size, num_queries_per_kv);
                try validateLaunch(config.num_warps, config.num_stages);
            },
            .three_d => |config| {
                try validateCommon(config.block_m, config.tile_size, num_queries_per_kv);
                if (config.num_segments_per_seq == 0 or
                    !std.math.isPowerOfTwo(config.num_segments_per_seq) or
                    config.num_segments_per_seq > 128)
                    return error.InvalidNumSegmentsPerSeq;
                try validateLaunch(config.attention_num_warps, config.attention_num_stages);
                try validateLaunch(config.reduce_num_warps, config.reduce_num_stages);
            },
        }
    }

    fn validateCommon(block_m: usize, tile_size: usize, num_queries_per_kv: usize) TuningValidationError!void {
        if (block_m == 0 or block_m > max_block_m or !std.math.isPowerOfTwo(block_m))
            return error.InvalidBlockM;
        if (tile_size == 0 or tile_size > max_tile_size or !std.math.isPowerOfTwo(tile_size))
            return error.InvalidTileSize;
        if (block_m % num_queries_per_kv != 0) return error.BlockMNotDivisibleByQueriesPerKv;
    }

    fn validateLaunch(num_warps: usize, num_stages: usize) TuningValidationError!void {
        if (num_warps == 0 or !std.math.isPowerOfTwo(num_warps) or num_warps > max_num_warps)
            return error.InvalidNumWarps;
        if (num_stages == 0 or num_stages > max_num_stages)
            return error.InvalidNumStages;
    }

    fn assertValid(self: TuningConfig, num_queries_per_kv: usize) void {
        self.validate(num_queries_per_kv) catch |err|
            stdx.debug.panic("invalid Triton unified-attention tuning config: {t}", .{err});
    }
};

fn validatedQueriesPerKv(num_heads: usize, num_kv_heads: usize) TuningValidationError!usize {
    if (num_kv_heads == 0 or num_heads < num_kv_heads or num_heads % num_kv_heads != 0)
        return error.InvalidQueriesPerKv;
    return @divExact(num_heads, num_kv_heads);
}

fn isOneapiTarget() bool {
    return zml.module.CompilationContext.current().platform.target == .oneapi;
}

fn use2dKernel(target: zml.Target, all_decode: bool, batch_size: usize, num_kv_heads: usize) bool {
    // Intel decode spills the 2D whole-sequence kernel; force the 3D split-K path.
    if (all_decode and target == .oneapi) return false;
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

fn selectSparseMlaLaunchConfig(
    query_count: usize,
    num_heads: usize,
    topk_count: usize,
    cu_count_: usize,
    requested_splits: ?u8,
) SparseMlaLaunchConfig {
    stdx.debug.assert(query_count > 0, "sparse MLA requires at least one query", .{});
    stdx.debug.assert(num_heads > 0, "sparse MLA requires at least one query head", .{});
    stdx.debug.assert(topk_count > 0, "sparse MLA requires at least one top-k entry", .{});

    const block_m = @min(num_heads, 16);
    stdx.debug.assert(@mod(num_heads, block_m) == 0, "expected q heads ({}) to be divisible by block_m ({})", .{ num_heads, block_m });

    const topk_padded = std.math.ceilPowerOfTwoAssert(usize, topk_count);
    const tile_size = @min(topk_padded, 16);
    const num_tiles = std.math.divCeil(usize, topk_count, tile_size) catch unreachable;
    const direct_programs = query_count * @divExact(num_heads, block_m);
    const cu_count = @max(cu_count_, 1);

    var max_splits: usize = 1;
    while (max_splits * 2 <= @min(num_tiles, 16)) max_splits *= 2;

    if (requested_splits) |requested| {
        const num_splits: usize = requested;
        stdx.debug.assert(std.math.isPowerOfTwo(num_splits), "MLA num_kv_splits ({}) must be a power of two", .{num_splits});
        stdx.debug.assert(num_splits <= 16, "MLA num_kv_splits ({}) must not exceed 16", .{num_splits});
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
    const candidates = [_]usize{ 1, 2, 4, 8, 16 };
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

fn select2dConfig(options: paged.PagedAttentionOptions, tuning: TuningConfig) Config2D {
    switch (tuning) {
        .two_d => |explicit| {
            tuning.assertValid(options.numQueriesPerKv());
            const block_q = explicit.block_m / options.numQueriesPerKv();
            return .{
                .block_m = explicit.block_m,
                .block_q = block_q,
                .tile_size = explicit.tile_size,
                .num_warps = explicit.num_warps,
                .num_stages = explicit.num_stages,
                .total_q_blocks = options.num_tokens / block_q + options.batch_size,
            };
        },
        .automatic => {},
        .three_d => stdx.debug.panic("3D tuning config cannot be used with the 2D unified-attention kernel", .{}),
    }

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

fn select3dConfig(options: paged.PagedAttentionOptions, tuning: TuningConfig, target: zml.Target) Config3D {
    switch (tuning) {
        .three_d => |explicit| {
            tuning.assertValid(options.numQueriesPerKv());
            const block_q = explicit.block_m / options.numQueriesPerKv();
            const total_q_blocks = options.num_tokens / block_q + options.batch_size;
            return .{
                .attention = .{
                    .tile_size = explicit.tile_size,
                    .num_segments_per_seq = explicit.num_segments_per_seq,
                    .num_warps = explicit.attention_num_warps,
                    .num_stages = explicit.attention_num_stages,
                    .block_m = explicit.block_m,
                    .block_q = block_q,
                    .total_q_blocks = total_q_blocks,
                },
                .reduce = .{
                    .tile_size = explicit.tile_size,
                    .num_segments_per_seq = explicit.num_segments_per_seq,
                    .num_warps = explicit.reduce_num_warps,
                    .num_stages = explicit.reduce_num_stages,
                    .block_m = explicit.block_m,
                    .block_q = block_q,
                },
            };
        },
        .automatic => {},
        .two_d => stdx.debug.panic("2D tuning config cannot be used with the 3D unified-attention kernel", .{}),
    }

    var reduce_num_warps: usize = 2;
    // Intel decode needs more warps to spread the work and avoid register spill.
    const attn_warps: usize = if (options.all_decode and target == .oneapi) 8 else 2;
    const tile_size = options.block_size;

    //const MAX_SEGMENTS: usize = @min(128, std.math.divCeil(usize, max_seqlen_k, tile_size));
    var num_segments = std.math.divCeil(usize, options.target_num_prgms, options.num_2d_prgms) catch unreachable;
    num_segments = std.math.ceilPowerOfTwoAssert(usize, num_segments);
    num_segments = @min(num_segments, 128);
    if (options.all_decode and target != .oneapi) {
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

fn platformCuCount(platform: *const zml.Platform) !usize {
    if (platform.devices.len == 0) return error.MissingDevices;
    const attribute = platform.devices[0].pjrt_desc.attribute(platform.pjrt_api, "core_count") orelse
        return error.MissingCoreCount;
    if (attribute.int64 <= 0) return error.InvalidCoreCount;
    return @intCast(attribute.int64);
}

pub const paged = struct {
    pub const Options = struct {
        batch_size: usize,
        max_num_pages: usize,
        max_seqlen_q: usize,
        /// Representative KV length for automatic tuning. Zero means unknown
        /// and leaves the built-in heuristic in place.
        max_seqlen_k: usize = 0,
        is_prefill: bool,
        /// Representative active prefill rows used only by automatic tuning.
        /// When both batch counts are null, `is_prefill` selects a pure
        /// prefill or pure decode workload over the full batch.
        batch_size_prefill: ?usize = null,
        /// Representative active one-token decode rows used only by automatic
        /// tuning. Explicit prefill and decode counts may sum to less than
        /// `batch_size`; the remaining rows are benchmarked as inactive.
        batch_size_decode: ?usize = null,
        tuning: TuningConfig = .automatic,

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

    const max_autotune_candidates = 12;
    // Keep the synthetic K and V cache allocations together below 256 MiB.
    // Production cache pools are commonly multi-gigabyte and their `.page`
    // extent does not affect Triton launch configuration or cache strides.
    const synthetic_kv_cache_budget_bytes = 256 * 1024 * 1024;
    const AutotuneCandidates = stdx.BoundedArray(TuningConfig, max_autotune_candidates);

    const prefill_tuning_candidates = [_]TuningConfig{
        .{ .two_d = .{ .block_m = 64, .tile_size = 16, .num_warps = 4, .num_stages = 1 } },
        .{ .two_d = .{ .block_m = 64, .tile_size = 32, .num_warps = 4, .num_stages = 1 } },
        .{ .two_d = .{ .block_m = 64, .tile_size = 64, .num_warps = 4, .num_stages = 1 } },
        .{ .two_d = .{ .block_m = 128, .tile_size = 32, .num_warps = 4, .num_stages = 1 } },
        .{ .two_d = .{ .block_m = 128, .tile_size = 64, .num_warps = 2, .num_stages = 1 } },
        .{ .two_d = .{ .block_m = 128, .tile_size = 64, .num_warps = 4, .num_stages = 2 } },
    };

    const decode_tuning_candidates = [_]TuningConfig{
        .{ .three_d = .{
            .block_m = 16,
            .tile_size = 16,
            .num_segments_per_seq = 8,
            .attention_num_warps = 2,
            .attention_num_stages = 1,
            .reduce_num_warps = 1,
            .reduce_num_stages = 1,
        } },
        .{ .three_d = .{
            .block_m = 16,
            .tile_size = 16,
            .num_segments_per_seq = 32,
            .attention_num_warps = 2,
            .attention_num_stages = 1,
            .reduce_num_warps = 1,
            .reduce_num_stages = 1,
        } },
        .{ .three_d = .{
            .block_m = 16,
            .tile_size = 16,
            .num_segments_per_seq = 16,
            .attention_num_warps = 4,
            .attention_num_stages = 1,
            .reduce_num_warps = 1,
            .reduce_num_stages = 1,
        } },
        .{ .three_d = .{
            .block_m = 16,
            .tile_size = 16,
            .num_segments_per_seq = 16,
            .attention_num_warps = 2,
            .attention_num_stages = 1,
            .reduce_num_warps = 2,
            .reduce_num_stages = 1,
        } },
        .{ .two_d = .{ .block_m = 16, .tile_size = 16, .num_warps = 2, .num_stages = 2 } },
        .{ .two_d = .{ .block_m = 32, .tile_size = 16, .num_warps = 4, .num_stages = 2 } },
    };

    fn derivePagedAttentionOptions(
        platform: *const zml.Platform,
        options: Options,
        q_shape: zml.Shape,
        k_cache_shape: zml.Shape,
        block_table_shape: zml.Shape,
        opts: AttentionOptions,
    ) !PagedAttentionOptions {
        const num_heads: usize = @intCast(q_shape.dim(.hkv) * q_shape.dim(.hg));
        const num_kv_heads: usize = @intCast(k_cache_shape.dim(.hkv));
        const num_queries_per_kv = try validatedQueriesPerKv(num_heads, num_kv_heads);
        try options.tuning.validate(num_queries_per_kv);

        const automatic_block_m: usize = if (!options.is_prefill and platform.target == .oneapi)
            std.math.ceilPowerOfTwoAssert(usize, num_queries_per_kv)
        else if (num_queries_per_kv <= 16)
            16
        else
            std.math.ceilPowerOfTwoAssert(usize, num_queries_per_kv);
        const block_m = switch (options.tuning) {
            .automatic => automatic_block_m,
            inline .two_d, .three_d => |config| config.block_m,
        };
        const block_q = @divExact(block_m, num_queries_per_kv);
        const num_tokens: usize = @intCast(q_shape.dim(.b));
        const num_seqs: usize = @intCast(block_table_shape.dim(.b));
        const total_q_blocks = num_tokens / block_q + num_seqs;
        const cu_count = try platformCuCount(platform);
        const num_2d_prgms = total_q_blocks * num_kv_heads;
        const block_size: usize = @intCast(k_cache_shape.dim(.k_chunk));
        if (num_tokens == 0 or num_seqs == 0 or num_2d_prgms == 0 or
            block_size == 0 or q_shape.dim(.hd) <= 0)
            return error.InvalidAutotuneWorkload;

        return .{
            .cu_count = cu_count,
            .all_decode = !options.is_prefill,
            .num_tokens = num_tokens,
            .num_heads = num_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = @intCast(q_shape.dim(.hd)),
            .batch_size = num_seqs,
            .block_size = block_size,
            .num_blocks = @intCast(k_cache_shape.dim(.page)),
            .max_num_block_per_seq = @intCast(block_table_shape.dim(.p)),
            .sliding_window = if (opts.sliding_window < 0) 0 else @intCast(opts.sliding_window),
            .block_m = block_m,
            .block_q = block_q,
            .total_q_blocks = total_q_blocks,
            .target_num_prgms = cu_count * 4,
            .num_2d_prgms = num_2d_prgms,
            .max_seqlen_q = options.max_seqlen_q,
            .scale = opts.scale,
        };
    }

    fn materializeAutomaticConfig(options: PagedAttentionOptions, target: zml.Target) TuningConfig {
        if (use2dKernel(target, options.all_decode, options.batch_size, options.num_kv_heads)) {
            const config = select2dConfig(options, .automatic);
            return .{ .two_d = .{
                .block_m = config.block_m,
                .tile_size = config.tile_size,
                .num_warps = config.num_warps,
                .num_stages = config.num_stages,
            } };
        }

        const config = select3dConfig(options, .automatic, target);
        return .{ .three_d = .{
            .block_m = config.attention.block_m,
            .tile_size = config.attention.tile_size,
            .num_segments_per_seq = config.attention.num_segments_per_seq,
            .attention_num_warps = config.attention.num_warps,
            .attention_num_stages = config.attention.num_stages,
            .reduce_num_warps = config.reduce.num_warps,
            .reduce_num_stages = config.reduce.num_stages,
        } };
    }

    fn candidateTileSize(config: TuningConfig) usize {
        return switch (config) {
            .automatic => unreachable,
            inline .two_d, .three_d => |explicit| explicit.tile_size,
        };
    }

    fn paddedSequenceLength(sequence_length: usize, tile_size: usize) ?usize {
        const rounded = std.math.add(usize, sequence_length, tile_size - 1) catch return null;
        return rounded & ~(tile_size - 1);
    }

    fn pageTableCapacity(options: Options, block_size: usize) ?usize {
        return std.math.mul(usize, options.max_num_pages, block_size) catch null;
    }

    fn syntheticCachePageCount(k_cache_shape: zml.Shape, active_sequences: usize, max_seqlen_k: usize) !usize {
        if (k_cache_shape.dim(.page) <= 0 or k_cache_shape.dim(.k_chunk) <= 0 or
            active_sequences == 0 or max_seqlen_k == 0)
            return error.InvalidAutotuneWorkload;

        const production_pages: usize = @intCast(k_cache_shape.dim(.page));
        const block_size: usize = @intCast(k_cache_shape.dim(.k_chunk));
        const pages_per_sequence = std.math.divCeil(usize, max_seqlen_k, block_size) catch
            return error.InvalidAutotuneWorkload;
        const workload_pages = std.math.mul(usize, active_sequences, pages_per_sequence) catch
            std.math.maxInt(usize);

        var elements_per_page: usize = 1;
        for (k_cache_shape.setDim(.page, 1).dims()) |dim| {
            if (dim <= 0) return error.InvalidAutotuneWorkload;
            elements_per_page = std.math.mul(usize, elements_per_page, @intCast(dim)) catch
                return error.InvalidAutotuneWorkload;
        }
        const bytes_per_page = std.math.mul(usize, elements_per_page, k_cache_shape.dtype().sizeOf()) catch
            return error.InvalidAutotuneWorkload;
        const combined_kv_bytes_per_page = std.math.mul(usize, bytes_per_page, 2) catch
            return error.InvalidAutotuneWorkload;
        const budget_pages = @max(@as(usize, 1), synthetic_kv_cache_budget_bytes / combined_kv_bytes_per_page);
        return @max(@as(usize, 1), @min(production_pages, workload_pages, budget_pages));
    }

    fn candidateFitsPageTable(config: TuningConfig, options: Options, block_size: usize) bool {
        const capacity = pageTableCapacity(options, block_size) orelse return false;
        const padded_k = paddedSequenceLength(options.max_seqlen_k, candidateTileSize(config)) orelse return false;
        return padded_k <= capacity;
    }

    fn pageTableSafeBaseline(config_: TuningConfig, options: Options, block_size: usize) ?TuningConfig {
        var config = config_;
        // Automatic decode normally uses the KV page size as its tile size.
        // Explicit Triton tiles must be powers of two, though page sizes need
        // not be, so first materialize the closest valid tile no larger than
        // the heuristic's choice.
        const initial_tile_size = candidateTileSize(config);
        if (initial_tile_size == 0) return null;
        switch (config) {
            .automatic => unreachable,
            inline .two_d, .three_d => |*explicit| {
                explicit.tile_size = std.math.floorPowerOfTwo(usize, initial_tile_size);
            },
        }
        while (!candidateFitsPageTable(config, options, block_size)) {
            const tile_size = candidateTileSize(config);
            if (tile_size <= 1) return null;
            switch (config) {
                .automatic => unreachable,
                inline .two_d, .three_d => |*explicit| explicit.tile_size = tile_size / 2,
            }
        }
        return config;
    }

    fn appendCandidate(
        candidates: *AutotuneCandidates,
        config: TuningConfig,
        options: Options,
        kernel_options: PagedAttentionOptions,
    ) void {
        config.validate(kernel_options.numQueriesPerKv()) catch return;
        if (!candidateFitsPageTable(config, options, kernel_options.block_size)) return;
        for (candidates.constSlice()) |existing| {
            if (std.meta.eql(existing, config)) return;
        }
        candidates.appendAssumeCapacity(config);
    }

    fn tuningCandidates(options: Options, kernel_options: PagedAttentionOptions, target: zml.Target) !AutotuneCandidates {
        var candidates: AutotuneCandidates = .empty;
        const baseline = pageTableSafeBaseline(
            materializeAutomaticConfig(kernel_options, target),
            options,
            kernel_options.block_size,
        ) orelse return error.InsufficientPageTableCapacity;
        try baseline.validate(kernel_options.numQueriesPerKv());
        if (!candidateFitsPageTable(baseline, options, kernel_options.block_size))
            return error.InvalidAutotuneBaseline;
        // Candidate zero is the current heuristic and remains the deterministic
        // fallback if timing cannot distinguish close candidates.
        candidates.appendAssumeCapacity(baseline);
        const curated = if (options.is_prefill)
            prefill_tuning_candidates[0..]
        else
            decode_tuning_candidates[0..];
        for (curated) |candidate| appendCandidate(&candidates, candidate, options, kernel_options);
        return candidates;
    }

    fn scatteredPageId(index: usize, num_blocks: usize) usize {
        if (num_blocks <= 1) return 0;

        // An affine permutation with a stride coprime to the page count visits
        // every physical page exactly once per cycle. This avoids benchmarking
        // only adjacent cache pages while remaining deterministic and cheap to
        // regenerate for every workload.
        var stride = num_blocks / 2 + 1;
        while (std.math.gcd(stride, num_blocks) != 1) {
            stride += 1;
            if (stride == num_blocks) stride = 1;
        }
        const offset = num_blocks / 3;
        return (offset + (index % num_blocks) * stride) % num_blocks;
    }

    fn representativePageId(
        sequence_index: usize,
        page_index: usize,
        pages_per_sequence: usize,
        active_sequences: usize,
        fallback_index: usize,
        num_blocks: usize,
    ) usize {
        const logical_page = if (sequence_index < active_sequences and page_index < pages_per_sequence)
            sequence_index * pages_per_sequence + page_index
        else
            fallback_index;
        return scatteredPageId(logical_page, num_blocks);
    }

    fn supportsAutotuneMemoryKind(kind: ?zml.Memory.Kind) bool {
        return switch (kind orelse .default) {
            .default, .device => true,
            .host_unpinned, .host_pinned => false,
        };
    }

    fn supportsAutotuneDtype(dtype: zml.DataType) bool {
        return switch (dtype) {
            .bf16, .f16, .f32 => true,
            else => false,
        };
    }

    const RepresentativeBatch = struct {
        prefill: usize,
        decode: usize,

        fn active(self: RepresentativeBatch) usize {
            return self.prefill + self.decode;
        }
    };

    fn representativeBatch(options: Options) !RepresentativeBatch {
        const counts: RepresentativeBatch = if (options.batch_size_prefill == null and options.batch_size_decode == null)
            if (options.is_prefill)
                .{ .prefill = options.batch_size, .decode = 0 }
            else
                .{ .prefill = 0, .decode = options.batch_size }
        else
            .{
                .prefill = options.batch_size_prefill orelse 0,
                .decode = options.batch_size_decode orelse 0,
            };

        if (counts.active() > options.batch_size or options.is_prefill != (counts.prefill != 0))
            return error.InvalidRepresentativeBatch;
        return counts;
    }

    fn validateRepresentativeBatch(options: Options, total_query_tokens: usize) !RepresentativeBatch {
        if (options.max_seqlen_k > std.math.maxInt(i32) or total_query_tokens > std.math.maxInt(i32))
            return error.InvalidAutotuneWorkload;

        const counts = try representativeBatch(options);
        if (total_query_tokens < counts.decode) return error.InvalidRepresentativeBatch;
        const prefill_tokens = total_query_tokens - counts.decode;
        if ((counts.prefill == 0 and prefill_tokens != 0) or
            (counts.prefill != 0 and prefill_tokens < counts.prefill))
            return error.InvalidRepresentativeBatch;

        const prefill_tokens_per_sequence = if (counts.prefill == 0) 0 else prefill_tokens / counts.prefill;
        const prefill_sequences_with_extra_token = if (counts.prefill == 0) 0 else prefill_tokens % counts.prefill;
        const maximum_prefill_tokens = prefill_tokens_per_sequence + @intFromBool(prefill_sequences_with_extra_token != 0);
        if (maximum_prefill_tokens > options.max_seqlen_q or maximum_prefill_tokens > options.max_seqlen_k)
            return error.InvalidRepresentativeBatch;
        return counts;
    }

    fn fillRepresentativeMetadata(
        options: Options,
        total_query_tokens: usize,
        seq_lens: []i32,
        query_start_len: []i32,
    ) !RepresentativeBatch {
        if (seq_lens.len != options.batch_size or query_start_len.len != options.batch_size + 1)
            return error.InvalidRepresentativeMetadata;
        const counts = try validateRepresentativeBatch(options, total_query_tokens);
        const prefill_tokens = total_query_tokens - counts.decode;

        const prefill_tokens_per_sequence = if (counts.prefill == 0) 0 else prefill_tokens / counts.prefill;
        const prefill_sequences_with_extra_token = if (counts.prefill == 0) 0 else prefill_tokens % counts.prefill;

        @memset(seq_lens, 0);
        @memset(query_start_len, 0);
        @memset(seq_lens[0..counts.active()], @intCast(options.max_seqlen_k));

        var query_offset: usize = 0;
        for (0..counts.prefill) |index| {
            query_offset += prefill_tokens_per_sequence + @intFromBool(index < prefill_sequences_with_extra_token);
            query_start_len[index + 1] = @intCast(query_offset);
        }
        for (0..counts.decode) |index| {
            query_offset += 1;
            query_start_len[counts.prefill + index + 1] = @intCast(query_offset);
        }
        for (counts.active()..options.batch_size) |index| {
            query_start_len[index + 1] = @intCast(query_offset);
        }
        if (query_offset != total_query_tokens) return error.InvalidRepresentativeMetadata;
        return counts;
    }

    const AutotuneProgram = struct {
        exe: zml.Exe,
        arguments: zml.Exe.Arguments,
        results: zml.Exe.Results,
        timer: ?zml.ExecutionTimer = null,
        validated: bool = false,

        fn deinit(self: *AutotuneProgram, allocator: std.mem.Allocator) void {
            self.results.deinit(allocator);
            self.arguments.deinit(allocator);
            self.exe.deinit();
        }
    };

    const ReferenceState = union(enum) {
        uninitialized,
        ready: zml.Slice,
        failed: anyerror,
    };

    const AutotuneContext = struct {
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        partitioner: zml.Sharding.Partitioner,
        shardings: []const zml.Sharding,
        parameters: Parameters,
        q_shape: zml.Shape,
        k_cache_shape: zml.Shape,
        v_cache_shape: zml.Shape,
        q_sharding: zml.Sharding,
        k_cache_sharding: zml.Sharding,
        v_cache_sharding: zml.Sharding,
        block_table_sharding: zml.Sharding,
        seq_lens_sharding: zml.Sharding,
        query_start_len_sharding: zml.Sharding,
        attention_options: AttentionOptions,
        baseline: TuningConfig,
        queries_per_kv: usize,

        parameter_buffers: ?zml.Bufferized(Parameters) = null,
        q_buffer: ?zml.Buffer = null,
        k_cache_buffer: ?zml.Buffer = null,
        v_cache_buffer: ?zml.Buffer = null,
        reference: ReferenceState = .uninitialized,
        compilation_index: usize = 0,

        fn deinit(self: *AutotuneContext) void {
            switch (self.reference) {
                .ready => |reference| reference.free(self.allocator),
                .uninitialized, .failed => {},
            }
            if (self.parameter_buffers) |*buffers| zml.Buffer.deinitAll(Parameters, buffers);
            if (self.q_buffer) |*buffer| buffer.deinit();
            if (self.k_cache_buffer) |*buffer| buffer.deinit();
            if (self.v_cache_buffer) |*buffer| buffer.deinit();
        }

        fn deterministicBuffer(
            self: *AutotuneContext,
            shape: zml.Shape,
            sharding: zml.Sharding,
            seed: u64,
        ) !zml.Buffer {
            const host = try zml.Slice.alloc(self.allocator, shape);
            defer host.free(self.allocator);
            var prng = std.Random.DefaultPrng.init(seed);
            const random = prng.random();
            switch (shape.dtype()) {
                .bf16 => for (host.items(zml.floats.BFloat16)) |*value| {
                    value.* = .fromF32(random.float(f32) * 2 - 1);
                },
                .f16 => for (host.items(f16)) |*value| {
                    value.* = @floatCast(random.float(f32) * 2 - 1);
                },
                .f32 => for (host.items(f32)) |*value| {
                    value.* = random.float(f32) * 2 - 1;
                },
                else => return error.UnsupportedAutotuneDtype,
            }
            return .fromSlice(self.io, self.platform, host, sharding);
        }

        fn ensureInputs(self: *AutotuneContext) !void {
            if (self.q_buffer != null) return;

            const options = self.parameters.options_;
            const batch_size = options.batch_size;
            const max_num_pages = options.max_num_pages;
            const block_size: usize = @intCast(self.k_cache_shape.dim(.k_chunk));
            const num_blocks: usize = @intCast(self.k_cache_shape.dim(.page));
            const total_query_tokens: usize = @intCast(self.q_shape.dim(.b));
            if (batch_size == 0 or max_num_pages == 0 or block_size == 0 or num_blocks == 0 or total_query_tokens == 0)
                return error.InvalidAutotuneWorkload;
            const capacity = pageTableCapacity(options, block_size) orelse return error.InvalidAutotuneWorkload;
            if (options.max_seqlen_k == 0 or options.max_seqlen_k > capacity)
                return error.InvalidAutotuneWorkload;
            if (options.max_seqlen_k > std.math.maxInt(i32) or
                total_query_tokens > std.math.maxInt(i32) or
                num_blocks > std.math.maxInt(i32))
                return error.InvalidAutotuneWorkload;
            const representative_batch = try validateRepresentativeBatch(options, total_query_tokens);
            const pages_per_sequence = std.math.divCeil(usize, options.max_seqlen_k, block_size) catch
                return error.InvalidAutotuneWorkload;

            var q_buffer = try self.deterministicBuffer(self.q_shape, self.q_sharding, 0);
            errdefer q_buffer.deinit();
            var k_cache_buffer = try self.deterministicBuffer(self.k_cache_shape, self.k_cache_sharding, 1);
            errdefer k_cache_buffer.deinit();
            var v_cache_buffer = try self.deterministicBuffer(self.v_cache_shape, self.v_cache_sharding, 2);
            errdefer v_cache_buffer.deinit();

            const block_table_len = std.math.mul(usize, batch_size, max_num_pages) catch
                return error.InvalidAutotuneWorkload;
            const block_table = try self.allocator.alloc(i32, block_table_len);
            defer self.allocator.free(block_table);
            for (block_table, 0..) |*page, index| {
                const sequence_index = index / max_num_pages;
                const page_index = index % max_num_pages;
                page.* = @intCast(representativePageId(
                    sequence_index,
                    page_index,
                    pages_per_sequence,
                    representative_batch.active(),
                    index,
                    num_blocks,
                ));
            }

            const seq_lens = try self.allocator.alloc(i32, batch_size);
            defer self.allocator.free(seq_lens);

            const query_start_len = try self.allocator.alloc(i32, batch_size + 1);
            defer self.allocator.free(query_start_len);
            _ = try fillRepresentativeMetadata(options, total_query_tokens, seq_lens, query_start_len);

            var block_table_buffer: zml.Buffer = try .fromBytes(
                self.io,
                self.platform,
                self.parameters.block_table.shape(),
                self.block_table_sharding,
                std.mem.sliceAsBytes(block_table),
            );
            errdefer block_table_buffer.deinit();
            var seq_lens_buffer: zml.Buffer = try .fromBytes(
                self.io,
                self.platform,
                self.parameters.seq_lens.shape(),
                self.seq_lens_sharding,
                std.mem.sliceAsBytes(seq_lens),
            );
            errdefer seq_lens_buffer.deinit();
            var query_start_len_buffer: zml.Buffer = try .fromBytes(
                self.io,
                self.platform,
                self.parameters.query_start_len.shape(),
                self.query_start_len_sharding,
                std.mem.sliceAsBytes(query_start_len),
            );
            errdefer query_start_len_buffer.deinit();

            self.parameter_buffers = .{
                .block_table = block_table_buffer,
                .seq_lens = seq_lens_buffer,
                .query_start_len = query_start_len_buffer,
            };
            self.q_buffer = q_buffer;
            self.k_cache_buffer = k_cache_buffer;
            self.v_cache_buffer = v_cache_buffer;
        }

        fn compileCandidate(self: *AutotuneContext, config: TuningConfig) !AutotuneProgram {
            return self.compileProgram(config, .device);
        }

        fn compileProgram(
            self: *AutotuneContext,
            config: TuningConfig,
            execution_timing: zml.CompilationOptions.ExecutionTiming,
        ) !AutotuneProgram {
            try config.validate(self.queries_per_kv);
            try self.ensureInputs();

            var parameters = self.parameters;
            parameters.options_.tuning = config;
            const program_name = try std.fmt.allocPrint(
                self.allocator,
                "triton_unified_attention_autotune_{d}",
                .{self.compilation_index},
            );
            defer self.allocator.free(program_name);
            self.compilation_index += 1;

            var exe = try self.platform.compileFn(
                self.allocator,
                self.io,
                pagedAttention,
                .{
                    parameters,
                    zml.Tensor.fromShape(self.q_shape),
                    zml.Tensor.fromShape(self.k_cache_shape),
                    zml.Tensor.fromShape(self.v_cache_shape),
                    self.attention_options,
                },
                .{
                    .program_name = program_name,
                    .shardings = self.shardings,
                    .partitioner = self.partitioner,
                    .execution_timing = execution_timing,
                },
            );
            errdefer exe.deinit();

            var arguments = try exe.args(self.allocator);
            errdefer arguments.deinit(self.allocator);
            arguments.set(.{
                self.parameter_buffers.?,
                self.q_buffer.?,
                self.k_cache_buffer.?,
                self.v_cache_buffer.?,
            });
            const results = try exe.results(self.allocator);
            return .{ .exe = exe, .arguments = arguments, .results = results };
        }

        fn createReference(self: *AutotuneContext) !zml.Slice {
            var program = try self.compileProgram(self.baseline, .none);
            defer program.deinit(self.allocator);
            try program.exe.tryCallOpts(
                self.io,
                program.arguments,
                &program.results,
                .{ .wait = true, .allow_input_donation = false },
            );
            var output = program.results.get(zml.Buffer);
            defer output.deinit();
            return output.toSliceAlloc(self.allocator, self.io);
        }

        fn ensureReference(self: *AutotuneContext) !void {
            switch (self.reference) {
                .ready => return,
                .failed => |err| return err,
                .uninitialized => {},
            }
            const reference = self.createReference() catch |err| {
                self.reference = .{ .failed = err };
                return err;
            };
            self.reference = .{ .ready = reference };
        }

        fn validateOutput(self: *AutotuneContext, output: zml.Buffer) !void {
            const reference = switch (self.reference) {
                .ready => |reference| reference,
                .uninitialized, .failed => return error.MissingReferenceOutput,
            };
            try zml.testing.expectClose(self.io, reference, output, .{
                .absolute_tolerance = 1e-2,
                .relative_tolerance = 1e-2,
                .epsilon_relative = 1e-6,
            });
        }

        fn measure(self: *AutotuneContext, program: *AutotuneProgram, repetitions: usize) !std.Io.Duration {
            try self.ensureReference();
            const timer = if (program.timer) |*timer| timer else timer: {
                program.timer = try zml.ExecutionTimer.attach(&program.exe);
                break :timer &program.timer.?;
            };

            var total_ns: i96 = 0;
            var remaining = repetitions;
            if (!program.validated) {
                try timer.reset();
                program.exe.tryCallOpts(
                    self.io,
                    program.arguments,
                    &program.results,
                    .{ .wait = true, .allow_input_donation = false },
                ) catch |err| {
                    program.results.releaseBuffers();
                    return err;
                };
                const duration = timer.read() catch |err| {
                    program.results.releaseBuffers();
                    return err;
                };
                var output = program.results.get(zml.Buffer);
                defer output.deinit();
                try self.validateOutput(output);
                program.validated = true;
                total_ns = duration.nanoseconds;
                remaining -= 1;
            }

            if (remaining != 0) {
                const duration = try timer.measureCall(self.io, program.arguments, &program.results, remaining);
                total_ns = std.math.add(i96, total_ns, duration.nanoseconds) catch
                    return error.DurationOverflow;
            }
            return .fromNanoseconds(total_ns);
        }

        fn deinitProgram(self: *AutotuneContext, program: *AutotuneProgram) void {
            program.deinit(self.allocator);
        }
    };

    fn writeTuningConfig(writer: *std.Io.Writer, config: TuningConfig) std.Io.Writer.Error!void {
        switch (config) {
            .automatic => try writer.writeAll("automatic"),
            .two_d => |explicit| try writer.print(
                "2d:{d},{d},{d},{d}",
                .{ explicit.block_m, explicit.tile_size, explicit.num_warps, explicit.num_stages },
            ),
            .three_d => |explicit| try writer.print(
                "3d:{d},{d},{d},{d},{d},{d},{d}",
                .{
                    explicit.block_m,
                    explicit.tile_size,
                    explicit.num_segments_per_seq,
                    explicit.attention_num_warps,
                    explicit.attention_num_stages,
                    explicit.reduce_num_warps,
                    explicit.reduce_num_stages,
                },
            ),
        }
    }

    fn writeShardingAssignment(
        writer: *std.Io.Writer,
        allocator: std.mem.Allocator,
        label: []const u8,
        sharding: zml.Sharding,
    ) !void {
        const assignment = try sharding.data.deviceAssignment(allocator);
        defer allocator.free(assignment);
        try writer.print("|{s}_assignment=", .{label});
        for (assignment, 0..) |device, index| {
            if (index != 0) try writer.writeByte(',');
            try writer.print("{d}", .{device});
        }
    }

    fn writeInputMemoryKind(writer: *std.Io.Writer, label: []const u8, kind: ?zml.Memory.Kind) !void {
        try writer.print("|{s}_memory=", .{label});
        if (kind) |value| {
            try writer.print("{t}", .{value});
        } else {
            try writer.writeAll("unspecified");
        }
    }

    fn tuningCacheKey(
        allocator: std.mem.Allocator,
        platform: *const zml.Platform,
        parameters: Parameters,
        q_shape: zml.Shape,
        k_cache_shape: zml.Shape,
        v_cache_shape: zml.Shape,
        synthetic_k_cache_shape: zml.Shape,
        synthetic_v_cache_shape: zml.Shape,
        local_q_shape: zml.Shape,
        local_k_cache_shape: zml.Shape,
        local_v_cache_shape: zml.Shape,
        partitioning: zml.Sharding.Partitioning,
        q_sharding: zml.Sharding,
        k_cache_sharding: zml.Sharding,
        v_cache_sharding: zml.Sharding,
        block_table_sharding: zml.Sharding,
        seq_lens_sharding: zml.Sharding,
        query_start_len_sharding: zml.Sharding,
        input_memory_kinds: [6]?zml.Memory.Kind,
        representative_batch: RepresentativeBatch,
        attention_options: AttentionOptions,
        candidates: []const TuningConfig,
    ) ![]u8 {
        var key: std.Io.Writer.Allocating = .init(allocator);
        errdefer key.deinit();
        const device_assignment = try partitioning.deviceAssignment(allocator);
        defer allocator.free(device_assignment);
        try key.writer.print(
            "triton-unified-attention/v5|target={t}|devices={d}|device={s}|cores={d}|partitioner={t}|partitions={d}|replicas={d}|q={f}|k={f}|v={f}|synthetic_k={f}|synthetic_v={f}|synthetic_kv_budget={d}",
            .{
                platform.target,
                platform.devices.len,
                platform.devices[0].kind(),
                try platformCuCount(platform),
                partitioning.partitioner,
                partitioning.numPartitions(),
                partitioning.numReplicas(),
                q_shape,
                k_cache_shape,
                v_cache_shape,
                synthetic_k_cache_shape,
                synthetic_v_cache_shape,
                synthetic_kv_cache_budget_bytes,
            },
        );
        try key.writer.print(
            "|local_q={f}|local_k={f}|local_v={f}|block_table={f}|seq_lens={f}|query_start_len={f}|q_sharding={f}|k_sharding={f}|v_sharding={f}|block_table_sharding={f}|seq_lens_sharding={f}|query_start_len_sharding={f}|batch={d}|prefill_batch={d}|decode_batch={d}|pages={d}|max_q={d}|max_k={d}|prefill={}|causal={}|window={d}|primary_assignment=",
            .{
                local_q_shape,
                local_k_cache_shape,
                local_v_cache_shape,
                parameters.block_table.shape(),
                parameters.seq_lens.shape(),
                parameters.query_start_len.shape(),
                q_sharding,
                k_cache_sharding,
                v_cache_sharding,
                block_table_sharding,
                seq_lens_sharding,
                query_start_len_sharding,
                parameters.options_.batch_size,
                representative_batch.prefill,
                representative_batch.decode,
                parameters.options_.max_num_pages,
                parameters.options_.max_seqlen_q,
                parameters.options_.max_seqlen_k,
                parameters.options_.is_prefill,
                attention_options.is_causal,
                attention_options.sliding_window,
            },
        );
        for (device_assignment, 0..) |device, index| {
            if (index != 0) try key.writer.writeByte(',');
            try key.writer.print("{d}", .{device});
        }
        try writeShardingAssignment(&key.writer, allocator, "q", q_sharding);
        try writeShardingAssignment(&key.writer, allocator, "k", k_cache_sharding);
        try writeShardingAssignment(&key.writer, allocator, "v", v_cache_sharding);
        try writeShardingAssignment(&key.writer, allocator, "block_table", block_table_sharding);
        try writeShardingAssignment(&key.writer, allocator, "seq_lens", seq_lens_sharding);
        try writeShardingAssignment(&key.writer, allocator, "query_start_len", query_start_len_sharding);
        try writeInputMemoryKind(&key.writer, "block_table", input_memory_kinds[0]);
        try writeInputMemoryKind(&key.writer, "seq_lens", input_memory_kinds[1]);
        try writeInputMemoryKind(&key.writer, "query_start_len", input_memory_kinds[2]);
        try writeInputMemoryKind(&key.writer, "q", input_memory_kinds[3]);
        try writeInputMemoryKind(&key.writer, "k", input_memory_kinds[4]);
        try writeInputMemoryKind(&key.writer, "v", input_memory_kinds[5]);
        try key.writer.writeByte('|');
        if (attention_options.scale) |scale| {
            try key.writer.print("scale={x}|", .{@as(u32, @bitCast(scale))});
        } else {
            try key.writer.writeAll("scale=default|");
        }
        for (candidates, 0..) |candidate, index| {
            try key.writer.print("c{d}=", .{index});
            try writeTuningConfig(&key.writer, candidate);
            try key.writer.writeByte('|');
        }
        return key.toOwnedSlice();
    }

    fn resolveAutomaticTuning(
        parameters: Parameters,
        q: zml.Tensor,
        k_cache: zml.Tensor,
        v_cache: zml.Tensor,
        attention_options: AttentionOptions,
    ) !TuningConfig {
        const compilation = zml.module.CompilationContext.current();
        const platform = compilation.platform;
        if (platform.target != .cuda and platform.target != .rocm) return .automatic;
        if (!platform.autotuneEnabled()) return .automatic;
        if (!platform.executionTimerAvailable()) return .automatic;
        if (parameters.options_.max_seqlen_k == 0) return .automatic;
        if (!supportsAutotuneDtype(q.dtype()) or
            !supportsAutotuneDtype(k_cache.dtype()) or
            !supportsAutotuneDtype(v_cache.dtype()))
            return .automatic;
        if (q.dim(.b) <= 0) return error.InvalidAutotuneWorkload;
        const representative_batch = try validateRepresentativeBatch(parameters.options_, @intCast(q.dim(.b)));

        const input_memory_kinds = [6]?zml.Memory.Kind{
            parameters.block_table.inputMemoryKind(),
            parameters.seq_lens.inputMemoryKind(),
            parameters.query_start_len.inputMemoryKind(),
            q.inputMemoryKind(),
            k_cache.inputMemoryKind(),
            v_cache.inputMemoryKind(),
        };
        // Synthetic benchmark buffers are currently allocated in accelerator
        // memory. Host-resident inputs can have materially different transfer
        // and execution behavior, so preserve the production compilation and
        // skip tuning instead of caching a non-representative measurement.
        for (input_memory_kinds) |memory_kind| {
            if (!supportsAutotuneMemoryKind(memory_kind)) return .automatic;
        }

        const local_q_shape = try compilation.partitioning.localShapeForShape(q.shape());
        const local_k_cache_shape = try compilation.partitioning.localShapeForShape(k_cache.shape());
        const local_v_cache_shape = try compilation.partitioning.localShapeForShape(v_cache.shape());
        const local_block_table_shape = try compilation.partitioning.localShapeForShape(parameters.block_table.shape());
        const local_seq_lens_shape = try compilation.partitioning.localShapeForShape(parameters.seq_lens.shape());
        const local_query_start_len_shape = try compilation.partitioning.localShapeForShape(parameters.query_start_len.shape());

        if (parameters.options_.batch_size != @as(usize, @intCast(parameters.block_table.dim(.b))) or
            parameters.options_.max_num_pages != @as(usize, @intCast(parameters.block_table.dim(.p))) or
            parameters.seq_lens.dim(.b) != parameters.block_table.dim(.b) or
            parameters.query_start_len.dim(.b) != parameters.block_table.dim(.b) + 1 or
            parameters.options_.max_seqlen_q == 0)
            return error.InvalidAutotuneWorkload;

        // Unified attention currently supports model/head sharding. Sharding
        // tokens, pages, or metadata would require different synthetic inputs.
        if (local_q_shape.dim(.b) != q.dim(.b) or
            local_k_cache_shape.dim(.page) != k_cache.dim(.page) or
            local_k_cache_shape.dim(.k_chunk) != k_cache.dim(.k_chunk) or
            local_block_table_shape.dim(.b) != parameters.block_table.dim(.b) or
            local_block_table_shape.dim(.p) != parameters.block_table.dim(.p) or
            !local_seq_lens_shape.eqlWithTags(parameters.seq_lens.shape()) or
            !local_query_start_len_shape.eqlWithTags(parameters.query_start_len.shape()) or
            !k_cache.shape().eqlWithTags(v_cache.shape()) or
            !local_k_cache_shape.eqlWithTags(local_v_cache_shape))
            return error.UnsupportedAutotuneSharding;

        const capacity = std.math.mul(
            usize,
            @intCast(parameters.block_table.dim(.p)),
            @intCast(local_k_cache_shape.dim(.k_chunk)),
        ) catch return error.InvalidAutotuneWorkload;
        if (parameters.options_.max_seqlen_k == 0 or parameters.options_.max_seqlen_k > capacity)
            return error.InvalidAutotuneWorkload;

        const synthetic_cache_pages = try syntheticCachePageCount(
            k_cache.shape(),
            representative_batch.active(),
            parameters.options_.max_seqlen_k,
        );
        const synthetic_k_cache_shape = k_cache.shape().setDim(.page, @intCast(synthetic_cache_pages));
        const synthetic_v_cache_shape = v_cache.shape().setDim(.page, @intCast(synthetic_cache_pages));

        const kernel_options = try derivePagedAttentionOptions(
            platform,
            parameters.options_,
            local_q_shape,
            local_k_cache_shape,
            local_block_table_shape,
            attention_options,
        );
        var candidates = try tuningCandidates(parameters.options_, kernel_options, platform.target);
        if (candidates.len == 1) return candidates.get(0);

        const q_sharding = compilation.partitioning.selectSharding(q.shape()) catch return candidates.get(0);
        const k_cache_sharding = compilation.partitioning.selectSharding(k_cache.shape()) catch return candidates.get(0);
        const v_cache_sharding = compilation.partitioning.selectSharding(v_cache.shape()) catch return candidates.get(0);
        const block_table_sharding = compilation.partitioning.selectSharding(parameters.block_table.shape()) catch return candidates.get(0);
        const seq_lens_sharding = compilation.partitioning.selectSharding(parameters.seq_lens.shape()) catch return candidates.get(0);
        const query_start_len_sharding = compilation.partitioning.selectSharding(parameters.query_start_len.shape()) catch return candidates.get(0);
        const cache_key = tuningCacheKey(
            compilation.arena.allocator(),
            platform,
            parameters,
            q.shape(),
            k_cache.shape(),
            v_cache.shape(),
            synthetic_k_cache_shape,
            synthetic_v_cache_shape,
            local_q_shape,
            local_k_cache_shape,
            local_v_cache_shape,
            compilation.partitioning,
            q_sharding,
            k_cache_sharding,
            v_cache_sharding,
            block_table_sharding,
            seq_lens_sharding,
            query_start_len_sharding,
            input_memory_kinds,
            representative_batch,
            attention_options,
            candidates.constSlice(),
        ) catch |err| {
            log.warn("unable to build unified-attention autotune cache key ({t}); using candidate zero", .{err});
            return candidates.get(0);
        };
        defer compilation.arena.allocator().free(cache_key);

        var context: AutotuneContext = .{
            .allocator = compilation.allocator,
            .io = compilation.io,
            .platform = platform,
            .partitioner = compilation.partitioning.partitioner,
            .shardings = compilation.partitioning.shardings,
            .parameters = parameters,
            .q_shape = q.shape(),
            .k_cache_shape = synthetic_k_cache_shape,
            .v_cache_shape = synthetic_v_cache_shape,
            .q_sharding = q_sharding,
            .k_cache_sharding = k_cache_sharding,
            .v_cache_sharding = v_cache_sharding,
            .block_table_sharding = block_table_sharding,
            .seq_lens_sharding = seq_lens_sharding,
            .query_start_len_sharding = query_start_len_sharding,
            .attention_options = attention_options,
            .baseline = candidates.get(0),
            .queries_per_kv = kernel_options.numQueriesPerKv(),
        };

        // Candidate compilation creates its own CompilationContext. Suspend
        // the outer graph context so nested activation cannot alias it.
        compilation.deactivate();
        defer compilation.activate();
        defer context.deinit();
        const result = platform.autotune(
            context.allocator,
            context.io,
            cache_key,
            &context,
            candidates.constSlice(),
            AutotuneContext.compileCandidate,
            AutotuneContext.measure,
            AutotuneContext.deinitProgram,
            .{},
        ) catch |err| {
            log.warn("unified-attention benchmarking failed ({t}); using candidate zero", .{err});
            return context.baseline;
        };
        log.info(
            "unified attention selected candidate {d}: {any} ({t}; median {f}, MAD {f})",
            .{ result.candidate_index, result.config, result.source, result.median, result.mad },
        );
        return result.config;
    }

    pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        var resolved_parameters = parameters;
        if (parameters.options_.tuning == .automatic) {
            resolved_parameters.options_.tuning = resolveAutomaticTuning(
                parameters,
                q,
                k_cache,
                v_cache,
                opts,
            ) catch |err| fallback: {
                log.warn("unified-attention autotuning failed ({t}); using the built-in heuristic", .{err});
                break :fallback .automatic;
            };
        }

        const output = zml.ops.manualComputation(
            .{
                q,
                k_cache,
                v_cache,
                resolved_parameters.block_table,
                resolved_parameters.seq_lens,
                resolved_parameters.query_start_len,
            },
            q.shape(),
            .{
                .opts = opts,
                .options = resolved_parameters.options_,
            },
            (struct {
                fn body(ctx_: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, _: zml.Shape) zml.Tensor {
                    const q_ = sharded_inputs[0];
                    const k_cache_ = sharded_inputs[1];
                    const v_cache_ = sharded_inputs[2];
                    const parameters_: Parameters = .{ .block_table = sharded_inputs[3], .seq_lens = sharded_inputs[4], .query_start_len = sharded_inputs[5], .options_ = ctx_.options };
                    const platform = zml.module.CompilationContext.current().platform;
                    const paged_attention_opts = derivePagedAttentionOptions(
                        platform,
                        ctx_.options,
                        q_.shape(),
                        k_cache_.shape(),
                        parameters_.block_table.shape(),
                        ctx_.opts,
                    ) catch |err| stdx.debug.panic("invalid unified-attention workload: {t}", .{err});

                    const use_2d_kernel = switch (ctx_.options.tuning) {
                        .automatic => use2dKernel(
                            platform.target,
                            paged_attention_opts.all_decode,
                            paged_attention_opts.batch_size,
                            paged_attention_opts.num_kv_heads,
                        ),
                        .two_d => true,
                        .three_d => false,
                    };
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
        const config = select2dConfig(paged_attention_opts, parameters.options_.tuning);

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
        const config = select3dConfig(
            paged_attention_opts,
            parameters.options_.tuning,
            zml.module.CompilationContext.current().platform.target,
        );

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

        const config = select3dConfig(
            paged_attention_opts,
            parameters.options_.tuning,
            zml.module.CompilationContext.current().platform.target,
        );

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

    fn pagedSparseMlaKernel(q: zml.Tensor, kv_cache: zml.Tensor, sink: ?zml.Tensor, topk: zml.Tensor, parameters: Parameters, paged_opts: PagedSparseMlaOptions) zml.Tensor {
        const rope_rank: i64 = @intCast(paged_opts.rope_rank);
        const nope_rank: i64 = @intCast(paged_opts.head_dim - paged_opts.rope_rank);
        const kv_lora_rank: i64 = @intCast(std.math.ceilPowerOfTwoAssert(usize, @intCast(nope_rank)));

        const out_shape = q.shape();

        const q_strides = q.shape().computeElementStrides().constSlice();
        const out_strides = out_shape.computeElementStrides().constSlice();
        const kv_strides = kv_cache.shape().computeElementStrides().constSlice();

        const sink_: zml.Tensor = sink orelse .scalar(0, .i8);
        const use_sink = sink != null;
        const sm_scale: f32 = paged_opts.scale orelse @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(paged_opts.head_dim))));

        const launch = selectSparseMlaLaunchConfig(
            paged_opts.total_q_blocks,
            paged_opts.num_heads,
            @intCast(topk.dim(.topk)),
            getCuCount(),
            paged_opts.num_kv_splits,
        );

        const kernel_config: mla_kernels.Config = .{
            .q_dtype = triton.from(q.dtype()),
            .kv_dtype = triton.from(kv_cache.dtype()),
            .sink_dtype = triton.from(sink_.dtype()),
            .o_dtype = triton.from(q.dtype()),
            .num_query_heads = @intCast(paged_opts.num_heads),
            .num_queries_per_kv = @intCast(paged_opts.num_heads),
            .block_size = @intCast(paged_opts.block_size),
            .block_m = @intCast(launch.block_m),
            .topk_count = topk.dim(.topk),
            .rope_rank = rope_rank,
            .qk_lora_rank = nope_rank,
            .kv_lora_rank = kv_lora_rank,
            .rope_offset = nope_rank,
            .value_rank = @intCast(paged_opts.head_dim),
            .tile_size = @intCast(launch.tile_size),
            .num_splits = @intCast(launch.num_splits),
            .use_attn_sink = use_sink,
            .all_decode = !parameters.options_.is_prefill,
        };
        log.debug("pagedSparseMla launch: {any}, kernel: {any}", .{ launch, kernel_config });

        const kernel_inputs: mla_kernels.Kernel2D.Inputs = .{
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
        };

        if (launch.num_splits == 1) {
            const out = mla_kernels.Kernel2D.call(
                kernel_inputs,
                .{ .output = out_shape },
                .{
                    .cfg = kernel_config,
                    .grid = .{ @intCast(launch.direct_programs), 1, 1 },
                    .num_warps = 4,
                    .num_stages = 2,
                },
            );
            return out.output.reshape(q.shape());
        }

        const num_splits: i64 = @intCast(launch.num_splits);
        const kernel_3d_inputs: mla_kernels.Kernel3D.Inputs = .{
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
        const partials = mla_kernels.Kernel3D.call(
            kernel_3d_inputs,
            .{
                .partial_output = zml.Shape.init(.{ q.dim(.q), @as(i64, @intCast(paged_opts.num_heads)), num_splits, @as(i64, @intCast(paged_opts.head_dim)) }, .f32),
                .partial_lse = zml.Shape.init(.{ q.dim(.q), @as(i64, @intCast(paged_opts.num_heads)), num_splits }, .f32),
            },
            .{
                .cfg = kernel_config,
                .grid = .{ @intCast(launch.direct_programs), @intCast(launch.num_splits), 1 },
                .num_warps = 4,
                .num_stages = 2,
            },
        );
        const out = mla_kernels.Reduce3D.call(
            .{
                .partial_output_ptr = partials.partial_output,
                .partial_lse_ptr = partials.partial_lse,
                .attn_sink_ptr = sink_,
                .output_stride_0_ptr = zml.Tensor.constant(zml.DataType.i64.constant(out_strides[0])),
                .output_stride_1_ptr = zml.Tensor.constant(zml.DataType.i64.constant(out_strides[1])),
            },
            .{ .output = out_shape },
            .{
                .cfg = .{
                    .sink_dtype = triton.from(sink_.dtype()),
                    .o_dtype = triton.from(q.dtype()),
                    .num_query_heads = @intCast(paged_opts.num_heads),
                    .value_rank = @intCast(paged_opts.head_dim),
                    .num_splits = num_splits,
                    .use_attn_sink = use_sink,
                },
                .grid = .{ @intCast(q.dim(.q)), @intCast(paged_opts.num_heads), 1 },
                .num_warps = 1,
                .num_stages = 1,
            },
        );

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
        rope_rank: usize,
        scale: ?f32,
        total_q_blocks: usize,
        num_kv_splits: ?u8,
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
                    const paged_opts: PagedSparseMlaOptions = .{
                        .head_dim = @intCast(q_.dim(.hd)),
                        .num_heads = num_heads,
                        .block_size = @intCast(kv_cache_.dim(.k_chunk)),
                        .rope_rank = @intCast(ctx_.opts.rope_rank),
                        .scale = ctx_.opts.scale,
                        .total_q_blocks = @intCast(q_.dim(.q)),
                        .num_kv_splits = ctx_.opts.num_kv_splits,
                    };

                    return pagedSparseMlaKernel(
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

test "sparse MLA launch selection balances occupancy and split overhead" {
    const saturated = selectSparseMlaLaunchConfig(120, 16, 512, 120, null);
    try std.testing.expectEqual(@as(usize, 1), saturated.num_splits);

    const under_occupied = selectSparseMlaLaunchConfig(1, 16, 512, 120, null);
    try std.testing.expectEqual(@as(usize, 16), under_occupied.num_splits);

    const tile_limited = selectSparseMlaLaunchConfig(1, 16, 32, 120, null);
    try std.testing.expectEqual(@as(usize, 2), tile_limited.num_splits);

    const missing_cu_count = selectSparseMlaLaunchConfig(1, 16, 512, 0, null);
    try std.testing.expectEqual(@as(usize, 1), missing_cu_count.num_splits);
}

test "sparse MLA launch selection honors valid explicit splits" {
    const forced = selectSparseMlaLaunchConfig(1, 16, 64, 120, 4);
    try std.testing.expectEqual(@as(usize, 4), forced.num_splits);

    const one_tile = selectSparseMlaLaunchConfig(1, 16, 5, 120, null);
    try std.testing.expectEqual(@as(usize, 8), one_tile.tile_size);
    try std.testing.expectEqual(@as(usize, 1), one_tile.num_splits);
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

    const q = zml.Tensor.zeroes(zml.Shape.init(.{ .q = 1, .h = 16, .hd = 128 }, .bf16));
    const kv = zml.Tensor.zeroes(zml.Shape.init(.{ .page = 32, .k_chunk = 1, .hkv = 1, .hd = 128 }, .bf16));
    const sink = zml.Tensor.zeroes(zml.Shape.init(.{ .h = 16 }, .bf16));
    const topk = zml.Tensor.zeroes(zml.Shape.init(.{ .q = 1, .topk = 32 }, .i32));
    const parameters: paged.Parameters = .{
        .block_table = zml.Tensor.zeroes(zml.Shape.init(.{ .b = 1, .p = 1 }, .i32)),
        .seq_lens = zml.Tensor.zeroes(zml.Shape.init(.{ .b = 1 }, .i32)),
        .query_start_len = zml.Tensor.zeroes(zml.Shape.init(.{ .b = 2 }, .i32)),
        .options_ = .{
            .batch_size = 1,
            .max_num_pages = 1,
            .max_seqlen_q = 1,
            .is_prefill = false,
        },
    };
    const two_d_opts: paged.PagedSparseMlaOptions = .{
        .head_dim = 128,
        .num_heads = 16,
        .block_size = 1,
        .rope_rank = 64,
        .scale = null,
        .total_q_blocks = 1,
        .num_kv_splits = 1,
    };

    const two_d = paged.pagedSparseMlaKernel(q, kv, sink, topk, parameters, two_d_opts);
    try std.testing.expect(two_d.shape().eql(q.shape()));
    try std.testing.expect(two_d.value().owner().verify());

    var three_d_opts = two_d_opts;
    three_d_opts.num_kv_splits = 2;
    const three_d = paged.pagedSparseMlaKernel(q, kv, sink, topk, parameters, three_d_opts);
    try std.testing.expect(three_d.shape().eql(q.shape()));
    try std.testing.expect(three_d.value().owner().verify());
}

test "unified attention tuning validation" {
    const valid_2d: TuningConfig = .{ .two_d = .{
        .block_m = 128,
        .tile_size = 64,
        .num_warps = 4,
        .num_stages = 2,
    } };
    try valid_2d.validate(4);

    const valid_3d: TuningConfig = .{ .three_d = .{
        .block_m = 16,
        .tile_size = 32,
        .num_segments_per_seq = 16,
        .attention_num_warps = 2,
        .attention_num_stages = 1,
        .reduce_num_warps = 1,
        .reduce_num_stages = 1,
    } };
    try valid_3d.validate(4);
    const automatic: TuningConfig = .automatic;
    try std.testing.expectError(error.InvalidQueriesPerKv, automatic.validate(0));
    try automatic.validate(4);

    var invalid = valid_2d;
    invalid.two_d.block_m = 24;
    try std.testing.expectError(error.InvalidBlockM, invalid.validate(4));
    invalid = valid_2d;
    invalid.two_d.block_m = 0;
    try std.testing.expectError(error.InvalidBlockM, invalid.validate(4));
    invalid = valid_2d;
    invalid.two_d.block_m = max_block_m * 2;
    try std.testing.expectError(error.InvalidBlockM, invalid.validate(4));
    invalid = valid_2d;
    invalid.two_d.tile_size = 48;
    try std.testing.expectError(error.InvalidTileSize, invalid.validate(4));
    invalid = valid_2d;
    invalid.two_d.tile_size = 0;
    try std.testing.expectError(error.InvalidTileSize, invalid.validate(4));
    invalid = valid_2d;
    invalid.two_d.tile_size = max_tile_size * 2;
    try std.testing.expectError(error.InvalidTileSize, invalid.validate(4));
    try std.testing.expectError(error.InvalidQueriesPerKv, valid_2d.validate(0));
    try std.testing.expectError(error.BlockMNotDivisibleByQueriesPerKv, valid_2d.validate(3));
    invalid = valid_2d;
    invalid.two_d.num_warps = 0;
    try std.testing.expectError(error.InvalidNumWarps, invalid.validate(4));
    invalid = valid_2d;
    invalid.two_d.num_warps = 16;
    try std.testing.expectError(error.InvalidNumWarps, invalid.validate(4));
    invalid = valid_2d;
    invalid.two_d.num_stages = 5;
    try std.testing.expectError(error.InvalidNumStages, invalid.validate(4));

    invalid = valid_3d;
    invalid.three_d.num_segments_per_seq = 0;
    try std.testing.expectError(error.InvalidNumSegmentsPerSeq, invalid.validate(4));
    invalid = valid_3d;
    invalid.three_d.num_segments_per_seq = 256;
    try std.testing.expectError(error.InvalidNumSegmentsPerSeq, invalid.validate(4));
    invalid = valid_3d;
    invalid.three_d.reduce_num_warps = 3;
    try std.testing.expectError(error.InvalidNumWarps, invalid.validate(4));
    invalid = valid_3d;
    invalid.three_d.attention_num_stages = 0;
    try std.testing.expectError(error.InvalidNumStages, invalid.validate(4));

    try std.testing.expectEqual(@as(usize, 4), try validatedQueriesPerKv(32, 8));
    try std.testing.expectError(error.InvalidQueriesPerKv, validatedQueriesPerKv(32, 0));
    try std.testing.expectError(error.InvalidQueriesPerKv, validatedQueriesPerKv(4, 8));
    try std.testing.expectError(error.InvalidQueriesPerKv, validatedQueriesPerKv(31, 8));
}

test "unified attention tuning resolves local derived values" {
    const options: paged.PagedAttentionOptions = .{
        .cu_count = 120,
        .all_decode = false,
        .num_tokens = 65,
        .num_heads = 32,
        .num_kv_heads = 8,
        .head_dim = 128,
        .batch_size = 3,
        .block_size = 16,
        .num_blocks = 512,
        .max_num_block_per_seq = 256,
        .sliding_window = 0,
        .block_m = 16,
        .block_q = 4,
        .total_q_blocks = 19,
        .target_num_prgms = 480,
        .num_2d_prgms = 152,
        .max_seqlen_q = 128,
        .scale = null,
    };

    const automatic = select2dConfig(options, .automatic);
    try std.testing.expectEqual(@as(usize, 16), automatic.block_m);
    try std.testing.expectEqual(@as(usize, 4), automatic.block_q);
    try std.testing.expectEqual(@as(usize, 19), automatic.total_q_blocks);
    try std.testing.expectEqual(@as(usize, 64), automatic.tile_size);
    try std.testing.expectEqual(@as(usize, 2), automatic.num_warps);
    try std.testing.expectEqual(@as(usize, 1), automatic.num_stages);

    const two_d = select2dConfig(options, .{ .two_d = .{
        .block_m = 32,
        .tile_size = 32,
        .num_warps = 4,
        .num_stages = 2,
    } });
    try std.testing.expectEqual(@as(usize, 8), two_d.block_q);
    try std.testing.expectEqual(@as(usize, 11), two_d.total_q_blocks);
    try std.testing.expectEqual(@as(usize, 32), two_d.tile_size);

    const three_d = select3dConfig(
        options,
        .{ .three_d = .{
            .block_m = 16,
            .tile_size = 32,
            .num_segments_per_seq = 8,
            .attention_num_warps = 4,
            .attention_num_stages = 2,
            .reduce_num_warps = 2,
            .reduce_num_stages = 1,
        } },
        .cuda,
    );
    try std.testing.expectEqual(@as(usize, 4), three_d.attention.block_q);
    try std.testing.expectEqual(@as(usize, 19), three_d.attention.total_q_blocks);
    try std.testing.expectEqual(@as(usize, 32), three_d.attention.tile_size);
    try std.testing.expectEqual(@as(usize, 8), three_d.attention.num_segments_per_seq);
    try std.testing.expectEqual(three_d.attention.tile_size, three_d.reduce.tile_size);
    try std.testing.expectEqual(three_d.attention.num_segments_per_seq, three_d.reduce.num_segments_per_seq);
    try std.testing.expectEqual(three_d.attention.block_q, three_d.reduce.block_q);
}

test "unified attention autotune candidates use an explicit page-table-safe baseline" {
    const options: paged.Options = .{
        .batch_size = 1,
        .max_num_pages = 5,
        .max_seqlen_q = 80,
        .max_seqlen_k = 80,
        .is_prefill = true,
    };
    const kernel_options: paged.PagedAttentionOptions = .{
        .cu_count = 120,
        .all_decode = false,
        .num_tokens = 80,
        .num_heads = 32,
        .num_kv_heads = 8,
        .head_dim = 128,
        .batch_size = 1,
        .block_size = 16,
        .num_blocks = 5,
        .max_num_block_per_seq = 5,
        .sliding_window = 0,
        .block_m = 16,
        .block_q = 4,
        .total_q_blocks = 21,
        .target_num_prgms = 480,
        .num_2d_prgms = 168,
        .max_seqlen_q = 80,
        .scale = null,
    };

    const candidates = try paged.tuningCandidates(options, kernel_options, .cuda);
    try std.testing.expect(candidates.len > 1);
    const baseline = candidates.get(0);
    try std.testing.expectEqual(@as(usize, 16), paged.candidateTileSize(baseline));
    for (candidates.constSlice()) |candidate| {
        try std.testing.expect(paged.candidateFitsPageTable(candidate, options, kernel_options.block_size));
        switch (candidate) {
            .automatic => return error.TestUnexpectedResult,
            .two_d, .three_d => {},
        }
    }
}

test "unified attention autotune normalizes a non-power-of-two page size" {
    const options: paged.Options = .{
        .batch_size = 1,
        .max_num_pages = 5,
        .max_seqlen_q = 1,
        .max_seqlen_k = 120,
        .is_prefill = false,
    };
    const kernel_options: paged.PagedAttentionOptions = .{
        .cu_count = 120,
        .all_decode = true,
        .num_tokens = 1,
        .num_heads = 32,
        .num_kv_heads = 8,
        .head_dim = 128,
        .batch_size = 1,
        .block_size = 24,
        .num_blocks = 5,
        .max_num_block_per_seq = 5,
        .sliding_window = 0,
        .block_m = 16,
        .block_q = 4,
        .total_q_blocks = 1,
        .target_num_prgms = 480,
        .num_2d_prgms = 8,
        .max_seqlen_q = 1,
        .scale = null,
    };

    const candidates = try paged.tuningCandidates(options, kernel_options, .cuda);
    const baseline = candidates.get(0);
    try baseline.validate(kernel_options.numQueriesPerKv());
    try std.testing.expectEqual(@as(usize, 8), paged.candidateTileSize(baseline));
    try std.testing.expect(paged.candidateFitsPageTable(baseline, options, kernel_options.block_size));
}

test "unified attention autotune scatters every cache page in a full cycle" {
    const expected = [_]usize{ 2, 6, 3, 0, 4, 1, 5 };
    var actual: [expected.len]usize = undefined;
    var seen: [expected.len]bool = @splat(false);
    for (&actual, 0..) |*page, index| {
        page.* = paged.scatteredPageId(index, expected.len);
        try std.testing.expect(!seen[page.*]);
        seen[page.*] = true;
        try std.testing.expectEqual(page.*, paged.scatteredPageId(index + expected.len, expected.len));
    }
    try std.testing.expectEqualSlices(usize, &expected, &actual);
}

test "unified attention autotune assigns disjoint active page ranges when uncapped" {
    const active_sequences = 3;
    const pages_per_sequence = 4;
    const num_blocks = active_sequences * pages_per_sequence;
    var seen: [num_blocks]bool = @splat(false);
    for (0..active_sequences) |sequence_index| {
        for (0..pages_per_sequence) |page_index| {
            const page = paged.representativePageId(
                sequence_index,
                page_index,
                pages_per_sequence,
                active_sequences,
                0,
                num_blocks,
            );
            try std.testing.expect(!seen[page]);
            seen[page] = true;
        }
    }
    for (seen) |was_used| try std.testing.expect(was_used);
}

test "unified attention autotune only synthesizes accelerator-memory inputs" {
    try std.testing.expect(paged.supportsAutotuneMemoryKind(null));
    try std.testing.expect(paged.supportsAutotuneMemoryKind(.default));
    try std.testing.expect(paged.supportsAutotuneMemoryKind(.device));
    try std.testing.expect(!paged.supportsAutotuneMemoryKind(.host_unpinned));
    try std.testing.expect(!paged.supportsAutotuneMemoryKind(.host_pinned));
}

test "unified attention autotune only synthesizes supported floating-point inputs" {
    try std.testing.expect(paged.supportsAutotuneDtype(.bf16));
    try std.testing.expect(paged.supportsAutotuneDtype(.f16));
    try std.testing.expect(paged.supportsAutotuneDtype(.f32));
    try std.testing.expect(!paged.supportsAutotuneDtype(.f8e4m3b11fnuz));
    try std.testing.expect(!paged.supportsAutotuneDtype(.i8));
}

test "unified attention autotune preserves a mixed prefill and decode layout" {
    const options: paged.Options = .{
        .batch_size = 8,
        .batch_size_prefill = 1,
        .batch_size_decode = 6,
        .max_num_pages = 16,
        .max_seqlen_q = 32,
        .max_seqlen_k = 256,
        .is_prefill = true,
    };
    var seq_lens: [8]i32 = undefined;
    var query_start_len: [9]i32 = undefined;
    const counts = try paged.fillRepresentativeMetadata(options, 38, &seq_lens, &query_start_len);

    try std.testing.expectEqual(@as(usize, 1), counts.prefill);
    try std.testing.expectEqual(@as(usize, 6), counts.decode);
    try std.testing.expectEqualSlices(i32, &.{ 256, 256, 256, 256, 256, 256, 256, 0 }, &seq_lens);
    try std.testing.expectEqualSlices(i32, &.{ 0, 32, 33, 34, 35, 36, 37, 38, 38 }, &query_start_len);
}

test "unified attention autotune defaults direct options to a pure workload" {
    const prefill = try paged.representativeBatch(.{
        .batch_size = 3,
        .max_num_pages = 1,
        .max_seqlen_q = 4,
        .is_prefill = true,
    });
    try std.testing.expectEqual(@as(usize, 3), prefill.prefill);
    try std.testing.expectEqual(@as(usize, 0), prefill.decode);

    const decode = try paged.representativeBatch(.{
        .batch_size = 3,
        .max_num_pages = 1,
        .max_seqlen_q = 1,
        .is_prefill = false,
    });
    try std.testing.expectEqual(@as(usize, 0), decode.prefill);
    try std.testing.expectEqual(@as(usize, 3), decode.decode);
}

test "unified attention autotune bounds the synthetic KV cache page pool" {
    const cache_shape = zml.Shape.init(.{
        .page = 100_000,
        .k_chunk = 16,
        .hkv = 8,
        .hd = 128,
    }, .bf16);

    // Each combined K+V page is 64 KiB, so the 256 MiB budget admits 4096.
    try std.testing.expectEqual(
        @as(usize, 896),
        try paged.syntheticCachePageCount(cache_shape, 7, 2048),
    );
    try std.testing.expectEqual(
        @as(usize, 4096),
        try paged.syntheticCachePageCount(cache_shape, 64, 2048),
    );
    try std.testing.expectEqual(
        @as(usize, 100),
        try paged.syntheticCachePageCount(cache_shape.setDim(.page, 100), 64, 2048),
    );
}
