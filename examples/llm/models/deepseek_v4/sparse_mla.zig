const std = @import("std");

const zml = @import("zml");
const kernel = @import("unified_sparse_mla.zig");

const stdx = zml.stdx;

const log = std.log.scoped(.deepseek_v4);

// From: <https://github.com/tile-ai/tilelang/blob/4e873220313c7e4dbfa0538582bc32ac5c81b1eb/examples/deepseek_v4/sparse_attn_fwd_sm90.py#L162>
fn vanillaSparseAttention(
    q: zml.Tensor,
    kv: zml.Tensor,
    attn_sink: zml.Tensor,
    topk: zml.Tensor,
    scale: ?f32,
) zml.Tensor {
    // q = [batch, q, h, hd], kv = [batch, k, hd], topk = [batch, seq, topk]
    const mask = topk.cmp(.GE, zml.Tensor.zeroes(topk.shape())).insertAxes(.topk, .{.h});

    const selected_kv = kv.gather(.{ .kv = topk }, .{}).rename(.{ .seq = .q, .topk = .kv }).convert(.f32);

    const dims = zml.nn.collectDims(.{ .h, .q, .kv, .hd }, &.{ q, kv }, .strict) catch {
        stdx.debug.panic("Inputs have incompatible shapes (q: {f}, kv: {f}, attn_mask: ).", .{ q, kv });
    };

    const sqrt_head_dim = if (scale) |m| m else 1.0 / std.math.sqrt(@as(f32, @floatFromInt(dims.hd)));
    const head_scaling = zml.Tensor.scalar(sqrt_head_dim, selected_kv.dtype());

    const q_32 = q.convert(.f32);
    var scores = q_32.dot(selected_kv, .hd).mul(head_scaling);
    scores = zml.Tensor.select(mask.broad(scores.shape()), scores, zml.Tensor.constant(scores.dtype().minValue()));

    const sink_shape = q.shape().set(.hd, 1);
    const sink = attn_sink.insertAxes(0, .{ .batch, .q }).insertAxes(.last, .{.hd}).broad(sink_shape);
    const scores_sink = zml.Tensor.concatenate(&.{ scores, sink.convert(scores.dtype()) }, -1);

    const attn_weights = scores_sink.convert(.f32).softmax(.kv).convert(q_32.dtype());
    const attn_weights_non_sink = attn_weights.slice(&.{
        .{},
        .{},
        .{},
        .{ .end = topk.dim(.topk) },
    });
    const attn = attn_weights_non_sink.dot(selected_kv, .kv);
    return attn.convert(.bf16);
}

pub fn tritonSparseAttention(
    q: zml.Tensor,
    kv: zml.Tensor,
    attn_sink: zml.Tensor,
    topk: zml.Tensor,
    scale: ?f32,
) zml.Tensor {
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

    const sm_scale = scale orelse 1.0 / std.math.sqrt(@as(f32, @floatFromInt(q_dim)));

    const opts: Options = .{
        .num_warps = 4,
        .num_stages = 2,
    };

    const out = kernel.Kernel.call(.{
        .query_ptr = q_final,
        .key_cache_ptr = key_cache,
        .value_cache_ptr = value_cache,
        .attn_sink_ptr = attn_sink,
        .block_tables_ptr = zml.Tensor.constant(zml.DataType.i32.zero()).reshape(.{1}),
        .topk_indices_ptr = topk_final,
        .seq_lens_ptr = zml.Tensor.constant(.{ .i32 = @as(i32, @intCast(kv.dim(.kv))) }).broad(.init(.{ .batch = batch }, .i32)),
        .scale_ptr = zml.Tensor.constant(.{ .f32 = sm_scale }).reshape(.{1}),
        .block_table_stride_ptr = stride(1),
        .query_stride_0_ptr = stride(q_strides[0]),
        .query_stride_1_ptr = stride(q_strides[1]),
        .output_stride_0_ptr = stride(out_strides[0]),
        .output_stride_1_ptr = stride(out_strides[1]),
        .stride_k_cache_0_ptr = stride(k_strides[0]),
        .stride_k_cache_1_ptr = stride(k_strides[1]),
        .stride_k_cache_2_ptr = stride(k_strides[2]),
        .stride_v_cache_0_ptr = stride(v_strides[0]),
        .stride_v_cache_1_ptr = stride(v_strides[1]),
        .stride_v_cache_2_ptr = stride(v_strides[2]),
        .query_start_len_ptr = zml.Tensor.arange(.{ .end = batch + 1 }, .i32).mul(zml.Tensor.scalar(q_len, .i32)),
        .num_seqs_ptr = zml.Tensor.constant(.{ .i32 = @as(i32, @intCast(batch)) }).reshape(.{1}),
    }, .{
        .output = out_shape,
    }, .{
        .cfg = .{
            .q_dtype = zml.kernel.triton.from(q_final.dtype()),
            .kv_dtype = zml.kernel.triton.from(kv_final.dtype()),
            .sink_dtype = zml.kernel.triton.from(attn_sink.dtype()),
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
        .num_warps = @intCast(opts.num_warps),
        .num_stages = @intCast(opts.num_stages),
    });

    return out.output.reshape(q.shape());
}

pub fn sparseAttentionMLA(
    q: zml.Tensor,
    kv: zml.Tensor,
    attn_sink: zml.Tensor,
    topk: zml.Tensor,
    scale: ?f32,
    metadata: zml.attention.attention.Metadata,
    parameters: zml.attention.attention.Parameters,
) zml.Tensor {
    _ = metadata; // autofix
    return switch (parameters) {
        .vanilla => vanillaSparseAttention(q, kv, attn_sink, topk, scale),
        .attnd => @panic("Must be initialized manually"),
        else => tritonSparseAttention(q, kv, attn_sink, topk, scale),
    };
}

fn stride(v: i64) zml.Tensor {
    return zml.Tensor.constant(.{ .i64 = v }).reshape(.{1});
}

const Options = struct {
    num_warps: i32,
    num_stages: i32,
};
