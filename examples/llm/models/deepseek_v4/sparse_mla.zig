const std = @import("std");
const stdx = zml.stdx;

const zml = @import("zml");

const kernel = @import("sparse_mla_kernel.zig");

const log = std.log.scoped(.deepseek);

pub fn sparseAttentionMLA(
    q: zml.Tensor,
    kv: zml.Tensor,
    attn_sink: zml.Tensor,
    topk: zml.Tensor,
    scale: ?f32,
    attention_metadata: zml.attention.attention.Metadata,
    attention_parameters: zml.attention.attention.Parameters,
) zml.Tensor {
    _ = attention_metadata; // autofix
    _ = attention_parameters; // autofix
    // q: [batch, q, h=64, hd=512], kv: [batch, kv, hd=512], topk: [batch, seq, topk=128]
    // stdx.debug.assert();

    const q_final = q.merge(.{ .q = .{ .batch, .q } });
    const kv_final = kv.merge(.{ .kv = .{ .batch, .kv } }).insertAxes(.hd, .{.kv_hd});
    const topk_final = topk.merge(.{ .seq = .{ .batch, .seq } }).insertAxes(.topk, .{.kv_hd}).convert(.i32);
    const sink_final = attn_sink.convert(q.dtype());

    const q_heads = q_final.dim(.h);
    const q_dim = q_final.dim(.hd);
    const kv_heads = kv_final.dim(.kv_hd);
    const kv_dim = kv_final.dim(.hd);

    const seqlen = q_final.dim(.q);

    const out_shape: zml.Shape = .init(.{ .q = seqlen, .h = q_heads, .hd = kv_dim }, q.dtype());
    const softmax_shape: zml.Shape = .init(.{ seqlen, q_heads }, .f32);
    const max_logits_shape: zml.Shape = .init(.{ seqlen, q_heads }, .f32);

    const kv_group = @divExact(q_heads, kv_heads);

    const cfg: kernel.Cfg = .{
        .use_attn_sink = true,
        .kv_group_num = @intCast(kv_group),
        .index_topk = @intCast(topk_final.dim(.topk)),
        .sink_dtype = zml.kernel.triton.from(sink_final.dtype()),
    };

    const opts: Options = .{
        .num_warps = 4,
        .num_stages = 3,
    };

    const grid_h = std.math.divCeil(i64, q_heads, @min(16, kv_group)) catch unreachable;

    const out = kernel.Kernel.call(.{
        .q_buffer = q_final,
        .k_buffer = kv_final,
        .v_buffer = kv_final,
        .sink_ptr = sink_final,
        .indices_ptr = topk_final,
        .seq_q_ptr = zml.Tensor.constant(.{ .i64 = q_final.dim(.q) }).reshape(.{1}),
        .seq_kv_ptr = zml.Tensor.constant(.{ .i64 = kv_final.dim(.kv) }).reshape(.{1}),
        .h_q_ptr = zml.Tensor.constant(.{ .i64 = q_final.dim(.h) }).reshape(.{1}),
        .dim_qk_ptr = zml.Tensor.constant(.{ .i64 = q.dim(.hd) }).reshape(.{1}),
        .dim_v_ptr = zml.Tensor.constant(.{ .i64 = kv.dim(.hd) }).reshape(.{1}),
        .stride_q_token_ptr = stride(q_heads * q_dim),
        .stride_q_head_ptr = stride(q_dim),
        .stride_k_token_ptr = stride(kv_heads * kv_dim),
        .stride_k_head_ptr = stride(kv_dim),
        .stride_v_token_ptr = stride(kv_heads * kv_dim),
        .stride_v_head_ptr = stride(kv_dim),
        .stride_out_token_ptr = stride(q_heads * kv_dim),
        .stride_out_head_ptr = stride(kv_dim),
        .stride_lse_ptr = stride(q_heads),
        .stride_indices_token_ptr = stride(topk_final.dim(.kv_hd) * topk_final.dim(.topk)),
        .stride_indices_head_ptr = stride(topk.dim(.topk)),
        .sm_scale_ptr = zml.Tensor.constant(.{ .f32 = scale.? * std.math.log2e }).reshape(.{1}),
    }, .{
        .out = out_shape,
        .softmax_lse = softmax_shape,
        .max_logits = max_logits_shape,
    }, .{
        .cfg = cfg,
        .grid = .{ @intCast(seqlen), @intCast(grid_h), 1 },
        .num_warps = @intCast(opts.num_warps),
        .num_stages = @intCast(opts.num_stages),
    });

    return out.out.reshape(q.shape());
}

fn stride(v: i64) zml.Tensor {
    return zml.Tensor.constant(.{ .i64 = v }).reshape(.{1});
}

const Options = struct {
    num_warps: i32,
    num_stages: i32,
};
