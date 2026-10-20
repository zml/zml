const std = @import("std");

pub const page_size = 16;
pub const topk = 8;
pub const Mask = enum { causal, non_causal, window, indices };

pub const Case = struct {
    name: []const u8 = "standard",
    head_dim: usize = 128,
    batch: usize = 2,
    kv_heads: usize = 2,
    group: usize = 4,
    prefill: usize = 0,
    chunk: usize = 17,
    kv_len: usize = 65,
    skew: bool = false,
    mask: Mask = .causal,
    empty_indices: bool = false,
    window: u32 = 17,
    sink: bool = false,
    scale: ?f32 = null,

    pub fn tokens(c: Case) usize {
        return c.prefill * c.chunk + c.batch - c.prefill;
    }
    pub fn pages(c: Case) usize {
        return std.math.divCeil(usize, c.kv_len, page_size) catch unreachable;
    }
    pub fn heads(c: Case) usize {
        return c.kv_heads * c.group;
    }
    pub fn queryLen(c: Case, seq: usize) usize {
        return if (seq < c.prefill) c.chunk else 1;
    }
    pub fn kvLen(c: Case, seq: usize) usize {
        const qlen = c.queryLen(seq);
        const reduction = if (c.skew) (seq * 17) % c.kv_len else 0;
        return @max(qlen, c.kv_len - @min(reduction, c.kv_len - 1));
    }
};

// Eight geometry cases plus orthogonal semantics/boundary cases, not a Cartesian product.
pub const correctness = blk: {
    var result: [17]Case = undefined;
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
