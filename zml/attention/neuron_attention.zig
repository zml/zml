const std = @import("std");

const zml = @import("../zml.zig");
const stdx = zml.stdx;

const num_q_heads = 32;
const num_kv_heads = 8;
const head_dim = 64;
const q_heads_per_chunk = 8;
const kv_heads_per_chunk = 2;
const chunk_count = num_kv_heads / kv_heads_per_chunk;
const tkg_source = @embedFile("neuron_attention_tkg.py");
const forward_sample_source = @embedFile("neuron_attention_fwd_sample.py");

pub const Kernel = enum {
    decode_tkg,
    forward_sample,
};

pub const Parameters = struct {
    kernel: Kernel = .decode_tkg,
};

/// Llama 3.2 1B decode-only attention lowered to four NKI TKG custom calls.
///
/// The supported graph shape is the one exercised by
/// `//examples/neuron_nki:attention`: q={q=1,h=32,hd=64},
/// k/v={k=seq,h=8,hd=64}, and a scalar u32 token index.
pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, parameters: Parameters) zml.Tensor {
    return switch (parameters.kernel) {
        .decode_tkg => decodeTkg(q, k, v, token_index),
        .forward_sample => forwardSample(q, k, v),
    };
}

fn decodeTkg(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    assertDecodeContract(q, k, v, token_index);

    var outputs: [chunk_count]zml.Tensor = undefined;
    for (&outputs, 0..) |*output, chunk_idx| {
        const kv_chunk_start = @as(i64, @intCast(chunk_idx)) * kv_heads_per_chunk;
        const q_chunk_start = @as(i64, @intCast(chunk_idx)) * q_heads_per_chunk;
        output.* = attentionChunk(
            q.slice1d(.h, .{ .start = q_chunk_start, .end = q_chunk_start + q_heads_per_chunk }),
            k.slice1d(.h, .{ .start = kv_chunk_start, .end = kv_chunk_start + kv_heads_per_chunk }),
            v.slice1d(.h, .{ .start = kv_chunk_start, .end = kv_chunk_start + kv_heads_per_chunk }),
            token_index,
        );
    }

    return zml.Tensor.concatenate(outputs[0..], .h);
}

/// Full forward attention sample kernel adapted to the decode contract.
///
/// Unlike `decodeTkg`, this intentionally lowers the whole decode attention
/// operation as one custom call. The Python kernel owns the grouped-query
/// mapping from 32 query heads to 8 KV heads.
fn forwardSample(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor) zml.Tensor {
    assertDecodeContract(q, k, v, zml.Tensor.scalar(@as(u32, 0), .u32));

    const scale: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(q.dim(.hd)))));
    const q_sample = q.scale(scale).transpose(.{ .hd, .h, .q }).rename(.{ .hd = .d });
    const k_sample = k.transpose(.{ .hd, .h, .k }).rename(.{ .hd = .d });
    const v_sample = v.transpose(.{ .hd, .h, .k }).rename(.{ .hd = .d });

    return zml.ops.neuronNki(.{ q_sample, k_sample, v_sample }, .{zml.Shape.init(.{ .d = head_dim, .h = num_q_heads, .q = 1 }, q.dtype())}, .{
        .name = "attention_decode_sample",
        .entrypoint = "zml_attention_decode_sample",
        .source = forward_sample_source,
    })[0]
        .rename(.{ .d = .hd })
        .transpose(.{ .q, .h, .hd })
        .convert(q.dtype());
}

/// Lower one {h=8,kvh=2} grouped-query chunk to the layout expected by
/// `attention_tkg`, then restore the public {q,h,hd} tensor layout.
fn attentionChunk(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    const seq_len = k.dim(.k);
    const heads_per_kv = @divExact(q.dim(.h), k.dim(.h));
    const scale: f32 = @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(q.dim(.hd)))));

    const q_tkg = q
        .insertAxes(.q, .{.b})
        .splitAxis(.h, .{ .kvh = k.dim(.h), .hq = .auto })
        .scale(scale)
        .transpose(.{ .hd, .b, .kvh, .hq, .q })
        .merge(.{ .beff = .{ .b, .kvh } })
        .merge(.{ .tot = .{ .beff, .hq, .q } })
        .rename(.{ .hd = .d });

    const active_k = k.insertAxes(.k, .{.b}).dynamicSlice(.{ .k = zml.Tensor.DynSlice{ .start = token_index, .len = 1 } });
    const active_v = v.insertAxes(.k, .{.b}).dynamicSlice(.{ .k = zml.Tensor.DynSlice{ .start = token_index, .len = 1 } });

    const k_active_tkg = active_k
        .transpose(.{ .hd, .b, .h, .k })
        .merge(.{ .tot = .{ .b, .h, .k } })
        .rename(.{ .hd = .d });
    const v_active_tkg = active_v
        .transpose(.{ .b, .h, .k, .hd })
        .merge(.{ .beff = .{ .b, .h } })
        .rename(.{ .beff = .b, .hd = .d })
        .insertAxes(.k, .{.one});

    const k_prior_tkg = k
        .insertAxes(.k, .{.b})
        .transpose(.{ .b, .h, .hd, .k })
        .merge(.{ .beff = .{ .b, .h } })
        .rename(.{ .beff = .b, .hd = .d })
        .insertAxes(.d, .{.one});
    const v_prior_tkg = v
        .insertAxes(.k, .{.b})
        .transpose(.{ .b, .h, .k, .hd })
        .merge(.{ .beff = .{ .b, .h } })
        .rename(.{ .beff = .b, .hd = .d })
        .insertAxes(.k, .{.one});

    const mask_shape_u32 = zml.Shape.init(.{ .k = seq_len, .b = k.dim(.h), .hq = heads_per_kv, .q = 1 }, token_index.dtype());
    const mask_shape_u8 = zml.Shape.init(.{ .k = seq_len, .b = k.dim(.h), .hq = heads_per_kv, .q = 1 }, .u8);
    const prior_mask = zml.Tensor.iota(zml.Shape.init(.{ .k = seq_len }, .i32), .k)
        .convert(token_index.dtype())
        .insertAxes(.last, .{ .b, .hq, .q })
        .broad(mask_shape_u32)
        .cmp(.LE, token_index);
    const mask = prior_mask.select(
        zml.Tensor.constant(.{ .u8 = 1 }).broad(mask_shape_u8),
        zml.Tensor.constant(.{ .u8 = 0 }).broad(mask_shape_u8),
    );

    return zml.ops.neuronNki(
        .{ q_tkg, k_active_tkg, v_active_tkg, k_prior_tkg, v_prior_tkg, mask },
        .{zml.Shape.init(.{ .b = k.dim(.h), .hq = heads_per_kv, .d = q.dim(.hd), .q = 1 }, q.dtype())},
        .{
            .name = "attention_tkg",
            .entrypoint = "zml_attention_tkg",
            .source = tkg_source,
        },
    )[0]
        .rename(.{ .b = .kvh })
        .merge(.{ .h = .{ .kvh, .hq } })
        .rename(.{ .d = .hd })
        .transpose(.{ .q, .h, .hd })
        .convert(q.dtype());
}

fn assertDecodeContract(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) void {
    stdx.debug.assert(q.rank() == 3 and k.rank() == 3 and v.rank() == 3, "neuron attention expects q/k/v rank-3 tensors, got {f}, {f}, {f}", .{ q, k, v });
    stdx.debug.assert(q.dtype() == .bf16 and k.dtype() == .bf16 and v.dtype() == .bf16, "neuron attention expects bf16 q/k/v, got {}, {}, {}", .{ q.dtype(), k.dtype(), v.dtype() });
    stdx.debug.assert(q.dim(.q) == 1, "neuron attention only supports decode q=1, got {f}", .{q});
    stdx.debug.assert(q.dim(.h) == num_q_heads and k.dim(.h) == num_kv_heads and v.dim(.h) == num_kv_heads, "neuron attention expects Llama 3.2 1B heads, got {f}, {f}, {f}", .{ q, k, v });
    stdx.debug.assert(q.dim(.hd) == head_dim and k.dim(.hd) == head_dim and v.dim(.hd) == head_dim, "neuron attention expects hd=64, got {f}, {f}, {f}", .{ q, k, v });
    stdx.debug.assert(k.dim(.k) == v.dim(.k), "neuron attention expects k/v sequence lengths to match, got {f} and {f}", .{ k, v });
    stdx.debug.assert(token_index.rank() == 0 and token_index.dtype() == .u32, "neuron attention expects scalar u32 token_index, got {f}", .{token_index});
}
