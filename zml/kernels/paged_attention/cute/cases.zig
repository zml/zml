const std = @import("std");

pub const page_size = 16;
pub const topk = 8;
pub const Mask = enum { causal, non_causal, window, non_causal_window, indices };

pub const Case = struct {
    name: []const u8 = "standard",
    head_dim: usize = 128,
    batch: usize = 2,
    kv_heads: usize = 2,
    group: usize = 4,
    prefill: usize = 0,
    chunk: usize = 17,
    kv_len: usize = 65,
    kv_capacity: ?usize = null,
    inactive: usize = 0,
    token_capacity: ?usize = null,
    shuffle_pages: bool = false,
    skew: bool = false,
    mask: Mask = .causal,
    topk: usize = 8,
    empty_indices: bool = false,
    window: u32 = 17,
    sink: bool = false,
    sink_value: ?f32 = null,
    scale: ?f32 = null,

    pub fn tokens(c: Case) usize {
        return c.token_capacity orelse (c.prefill * c.chunk + c.batch - c.prefill - c.inactive);
    }
    pub fn pages(c: Case) usize {
        return std.math.divCeil(usize, c.kv_capacity orelse c.kv_len, page_size) catch unreachable;
    }
    pub fn heads(c: Case) usize {
        return c.kv_heads * c.group;
    }
    pub fn queryLen(c: Case, seq: usize) usize {
        return if (seq >= c.batch - c.inactive) 0 else if (seq < c.prefill) c.chunk else 1;
    }
    pub fn kvLen(c: Case, seq: usize) usize {
        const qlen = c.queryLen(seq);
        if (qlen == 0) return 0;
        const reduction = if (c.skew) (seq * 17) % c.kv_len else 0;
        return @max(qlen, c.kv_len - @min(reduction, c.kv_len - 1));
    }
};

// Eight geometry cases plus orthogonal semantics/boundary cases, not a Cartesian product.
pub const correctness = blk: {
    var result: [34]Case = undefined;
    var i: usize = 0;
    for ([_]usize{ 64, 128, 256, 512 }) |dim| {
        for ([_]bool{ false, true }) |mixed| {
            result[i] = .{ .head_dim = dim, .prefill = if (mixed) 1 else 0 };
            i += 1;
        }
    }
    result[8] = .{ .name = "mha_page_tail", .group = 1, .kv_len = 15 };
    result[9] = .{ .name = "mqa_page_exact", .kv_heads = 1, .group = 8, .kv_len = 16 };
    result[10] = .{ .name = "non_causal", .head_dim = 256, .prefill = 1, .mask = .non_causal };
    result[11] = .{ .name = "window_sink", .head_dim = 256, .prefill = 1, .mask = .window, .sink = true, .scale = 0.125 };
    result[12] = .{ .name = "indices", .head_dim = 256, .prefill = 1, .mask = .indices, .sink = true };
    result[13] = .{ .name = "long_decode", .head_dim = 512, .kv_heads = 1, .group = 2, .batch = 1, .kv_len = 2049 };
    result[14] = .{ .name = "skew", .head_dim = 256, .batch = 4, .prefill = 2, .skew = true, .kv_len = 129 };
    result[15] = .{ .name = "mixed_short_chunks", .batch = 4, .prefill = 2, .chunk = 16, .skew = true, .kv_len = 64 };
    result[16] = .{ .name = "empty_indices", .prefill = 1, .mask = .indices, .empty_indices = true };
    result[17] = .{ .name = "d32", .head_dim = 32, .group = 1, .kv_len = 33 };
    result[18] = .{ .name = "odd_group", .group = 5, .prefill = 1, .chunk = 9, .kv_len = 79 };
    result[19] = .{ .name = "group_tile_tail", .head_dim = 256, .batch = 1, .group = 19, .kv_len = 257 };
    result[20] = .{ .name = "shuffled_mixed", .head_dim = 512, .batch = 4, .prefill = 2, .group = 3, .chunk = 11, .kv_len = 129, .shuffle_pages = true };
    result[21] = .{ .name = "split_sink", .head_dim = 512, .batch = 1, .kv_len = 4097, .sink = true, .shuffle_pages = true };
    result[22] = .{ .name = "split_window_sink", .head_dim = 256, .batch = 1, .kv_len = 4097, .mask = .window, .window = 257, .sink = true };
    result[23] = .{ .name = "inactive_sequence", .batch = 4, .prefill = 1, .inactive = 1, .kv_len = 129 };
    result[24] = .{ .name = "non_causal_window", .head_dim = 64, .prefill = 1, .kv_len = 65, .mask = .non_causal_window, .window = 7 };
    result[25] = .{ .name = "empty_splits", .batch = 1, .kv_len = 1, .kv_capacity = 4096, .sink = true };
    result[26] = .{ .name = "d32_mixed", .head_dim = 32, .group = 3, .prefill = 1, .chunk = 9, .kv_len = 257, .shuffle_pages = true };
    result[27] = .{ .name = "d64_group_tail", .head_dim = 64, .group = 19, .kv_len = 257 };
    result[28] = .{ .name = "split_mixed_sink", .head_dim = 128, .batch = 1, .prefill = 1, .chunk = 3, .kv_len = 4097, .sink = true, .shuffle_pages = true };
    result[29] = .{ .name = "d512_prefill", .head_dim = 512, .batch = 2, .prefill = 2, .chunk = 65, .kv_len = 257, .sink = true, .shuffle_pages = true };
    result[30] = .{ .name = "negative_scale", .head_dim = 128, .prefill = 1, .kv_len = 257, .mask = .non_causal, .scale = -0.125 };
    result[31] = .{ .name = "zero_scale", .head_dim = 256, .prefill = 1, .kv_len = 257, .sink = true, .scale = 0 };
    result[32] = .{ .name = "d512_window_prefill", .head_dim = 512, .batch = 1, .prefill = 1, .chunk = 33, .kv_len = 513, .mask = .window, .window = 129, .sink = true };
    result[33] = .{ .name = "prefill_inactive", .head_dim = 64, .batch = 4, .prefill = 2, .inactive = 2, .kv_len = 257, .group = 5 };
    break :blk result;
};

pub const benchmarks = blk: {
    var result: [24]Case = undefined;
    var i: usize = 0;
    for ([_]usize{ 64, 128, 256, 512 }) |dim| {
        for ([_]usize{ 2, 8, 32 }) |batch| {
            for ([_]bool{ false, true }) |mixed| {
                result[i] = .{
                    .head_dim = dim,
                    .batch = batch,
                    .prefill = if (mixed) @max(1, batch / 4) else 0,
                    .chunk = 64,
                    .kv_len = 2048,
                };
                i += 1;
            }
        }
    }
    break :blk result;
};

// Selection-slot tails, GQA head tails, and logical-to-physical page gathers
// vary independently. Every nonempty fixture includes duplicate/invalid slots.
pub const sparseCorrectness = [_]Case{
    .{ .name = "sparse_decode_inactive", .head_dim = 64, .group = 5, .batch = 5, .inactive = 2, .token_capacity = 9, .mask = .indices, .topk = 65, .shuffle_pages = true },
    .{ .name = "sparse_tiny_decode_inactive", .head_dim = 128, .group = 1, .batch = 5, .inactive = 2, .token_capacity = 9, .mask = .indices, .topk = 8, .sink = true, .shuffle_pages = true },
    .{ .name = "sparse_tiny_d32_sink", .head_dim = 32, .group = 1, .mask = .indices, .topk = 1, .sink = true, .shuffle_pages = true },
    .{ .name = "sparse_tiny_d64_disabled_sink", .head_dim = 64, .group = 1, .mask = .indices, .topk = 7, .empty_indices = true, .sink = true, .sink_value = -std.math.inf(f32) },
    .{ .name = "sparse_tiny_d128_empty", .head_dim = 128, .group = 1, .mask = .indices, .topk = 8, .empty_indices = true },
    .{ .name = "sparse_tiny_d256_capacity", .head_dim = 256, .group = 1, .batch = 4, .prefill = 1, .inactive = 1, .chunk = 3, .token_capacity = 9, .mask = .indices, .topk = 5, .sink = true, .shuffle_pages = true },
    .{ .name = "sparse_tiny_d512_scale", .head_dim = 512, .group = 1, .mask = .indices, .topk = 8, .scale = -0.125, .shuffle_pages = true },
    .{ .name = "sparse_tiny_mha", .head_dim = 128, .group = 1, .mask = .indices, .topk = 8, .shuffle_pages = true },
    .{ .name = "sparse_d32_mha", .head_dim = 32, .group = 1, .mask = .indices, .topk = 32 },
    .{ .name = "sparse_d64_mqa", .head_dim = 64, .kv_heads = 1, .group = 8, .mask = .indices, .topk = 65 },
    .{ .name = "sparse_d128_mixed", .head_dim = 128, .prefill = 1, .chunk = 9, .mask = .indices, .topk = 65, .shuffle_pages = true },
    .{ .name = "sparse_d256_mixed_sink", .head_dim = 256, .prefill = 1, .chunk = 3, .mask = .indices, .topk = 32, .sink = true },
    .{ .name = "sparse_d512", .head_dim = 512, .mask = .indices, .topk = 65, .shuffle_pages = true },
    .{ .name = "sparse_group5", .head_dim = 64, .group = 5, .prefill = 1, .chunk = 5, .mask = .indices, .topk = 65 },
    .{ .name = "sparse_group19", .head_dim = 128, .group = 19, .mask = .indices, .topk = 65, .sink = true },
    .{ .name = "sparse_d512_group19", .head_dim = 512, .batch = 1, .kv_heads = 1, .group = 19, .mask = .indices, .topk = 32 },
    .{ .name = "sparse_long_selection", .head_dim = 128, .batch = 1, .kv_len = 2049, .mask = .indices, .topk = 512, .sink = true, .shuffle_pages = true },
    .{ .name = "sparse_long_tail", .head_dim = 256, .batch = 1, .kv_len = 4097, .mask = .indices, .topk = 513, .shuffle_pages = true },
    .{ .name = "sparse_split_tail", .head_dim = 64, .batch = 1, .kv_len = 4097, .mask = .indices, .topk = 1025, .sink = true },
    .{ .name = "sparse_all_invalid", .head_dim = 128, .prefill = 1, .chunk = 3, .mask = .indices, .topk = 65, .empty_indices = true },
    .{ .name = "sparse_all_invalid_sink", .head_dim = 256, .mask = .indices, .topk = 512, .empty_indices = true, .sink = true },
    .{ .name = "sparse_skew_inactive", .head_dim = 128, .batch = 4, .prefill = 1, .inactive = 1, .chunk = 5, .kv_len = 129, .mask = .indices, .topk = 65, .skew = true, .shuffle_pages = true },
    .{ .name = "sparse_spare_capacity", .head_dim = 128, .batch = 4, .prefill = 1, .inactive = 1, .chunk = 3, .token_capacity = 16, .kv_len = 129, .mask = .indices, .topk = 65, .shuffle_pages = true },
    .{ .name = "sparse_negative_scale", .head_dim = 64, .mask = .indices, .topk = 32, .scale = -0.125 },
    .{ .name = "sparse_all_invalid_disabled_sink", .head_dim = 128, .mask = .indices, .topk = 8, .empty_indices = true, .sink = true, .sink_value = -std.math.inf(f32) },
    .{ .name = "sparse_zero_scale", .head_dim = 128, .mask = .indices, .topk = 65, .scale = 0, .sink = true },
};

pub const sparseBenchmarks = blk: {
    var result: [25]Case = undefined;
    var i: usize = 0;
    for ([_]usize{ 64, 128, 256, 512 }) |dim| {
        for ([_]usize{ 1, 8 }) |batch| {
            for ([_]bool{ false, true }) |mixed| {
                result[i] = .{
                    .name = "sparse",
                    .head_dim = dim,
                    .batch = batch,
                    .prefill = if (mixed) @max(1, batch / 4) else 0,
                    .chunk = 16,
                    .kv_len = 8192,
                    .mask = .indices,
                    .topk = 512,
                    .shuffle_pages = true,
                };
                i += 1;
            }
        }
    }
    result[16] = .{ .name = "sparse_short", .head_dim = 128, .batch = 8, .kv_len = 8192, .mask = .indices, .topk = 32, .shuffle_pages = true };
    result[17] = .{ .name = "sparse_odd_tail", .head_dim = 128, .batch = 8, .group = 8, .kv_len = 8192, .mask = .indices, .topk = 65, .shuffle_pages = true };
    result[18] = .{ .name = "sparse_large_gqa", .head_dim = 128, .batch = 1, .group = 19, .kv_len = 8192, .mask = .indices, .topk = 2048, .shuffle_pages = true };
    result[19] = .{ .name = "sparse_mha", .head_dim = 128, .batch = 8, .group = 1, .kv_len = 8192, .mask = .indices, .topk = 512, .shuffle_pages = true };
    result[20] = .{ .name = "sparse_batch64", .head_dim = 128, .batch = 64, .kv_len = 8192, .mask = .indices, .topk = 512, .shuffle_pages = true };
    result[21] = .{ .name = "sparse_batch128", .head_dim = 128, .batch = 128, .kv_len = 8192, .mask = .indices, .topk = 512, .shuffle_pages = true };
    result[22] = .{ .name = "sparse_tiny_mha", .head_dim = 128, .batch = 8, .group = 1, .kv_len = 8192, .mask = .indices, .topk = 8, .shuffle_pages = true };
    result[23] = .{ .name = "sparse_gqa32", .head_dim = 128, .batch = 8, .group = 32, .kv_len = 8192, .mask = .indices, .topk = 512, .shuffle_pages = true };
    result[24] = .{ .name = "sparse_mqa16_d512", .head_dim = 512, .batch = 1, .kv_heads = 1, .group = 16, .kv_len = 8192, .mask = .indices, .topk = 2048, .shuffle_pages = true };
    break :blk result;
};

// These geometries expose backend resource limits; keep them separately
// selectable so one unsupported backend cannot interrupt the shared matrix.
pub const sparseStressBenchmarks = [_]Case{
    .{ .name = "sparse_mqa64_d512", .head_dim = 512, .batch = 1, .kv_heads = 1, .group = 64, .kv_len = 8192, .mask = .indices, .topk = 2048, .shuffle_pages = true },
    .{ .name = "sparse_mqa128_d512", .head_dim = 512, .batch = 1, .kv_heads = 1, .group = 128, .kv_len = 8192, .mask = .indices, .topk = 2048, .shuffle_pages = true },
};
