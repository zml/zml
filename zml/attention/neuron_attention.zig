const std = @import("std");

const zml = @import("../zml.zig");
const stdx = zml.stdx;

const source = @embedFile("neuron_attention.py");
const max_tkg_q_heads_per_call = 8;

pub const Kernel = enum {
    decode_tkg,
    decode_inhouse,

    /// Compatibility alias for existing command lines.
    forward_sample,

    fn entrypoint(self: Kernel) []const u8 {
        return switch (self) {
            .decode_tkg => "decode_tkg",
            .decode_inhouse, .forward_sample => "decode_inhouse",
        };
    }
};

pub const Parameters = struct {
    kernel: Kernel = .decode_tkg,
};

/// Decode-only attention lowered to NKI custom calls.
pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, parameters: Parameters) zml.Tensor {
    assertDecodeContract(q, k, v, token_index);

    const heads_per_kv = @divExact(q.dim(.h), k.dim(.h));
    const kv_heads_per_call = std.math.clamp(@divTrunc(max_tkg_q_heads_per_call, heads_per_kv), 1, k.dim(.h));
    stdx.debug.assert(@mod(k.dim(.h), kv_heads_per_call) == 0, "neuron attention expects KV heads ({}) to split evenly into {}-head NKI calls", .{ k.dim(.h), kv_heads_per_call });

    const chunk_count: usize = @intCast(@divExact(k.dim(.h), kv_heads_per_call));
    var outputs: [32]zml.Tensor = undefined;
    stdx.debug.assert(chunk_count <= outputs.len, "neuron attention only supports up to {} NKI calls, got {}", .{ outputs.len, chunk_count });
    for (outputs[0..chunk_count], 0..) |*output, chunk_idx| {
        const kv_chunk_start = @as(i64, @intCast(chunk_idx)) * kv_heads_per_call;
        const q_chunk_start = kv_chunk_start * heads_per_kv;
        output.* = attentionCall(
            q.slice1d(.h, .{ .start = q_chunk_start, .end = q_chunk_start + kv_heads_per_call * heads_per_kv }),
            k.slice1d(.h, .{ .start = kv_chunk_start, .end = kv_chunk_start + kv_heads_per_call }),
            v.slice1d(.h, .{ .start = kv_chunk_start, .end = kv_chunk_start + kv_heads_per_call }),
            token_index,
            parameters.kernel,
        );
    }

    return zml.Tensor.concatenate(outputs[0..chunk_count], .h);
}

fn attentionCall(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, kernel: Kernel) zml.Tensor {
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
    const attn_mask_shape = zml.Shape.init(.{ .k = seq_len, .b = k.dim(.h), .hq = heads_per_kv, .q = 1 }, q.dtype());
    const attn_mask = prior_mask.select(
        zml.Tensor.constant(q.dtype().zero()).broad(attn_mask_shape),
        zml.Tensor.constant(q.dtype().minValue()).broad(attn_mask_shape),
    );

    return zml.ops.neuronNki(
        .{ q_tkg, k_active_tkg, v_active_tkg, k_prior_tkg, v_prior_tkg, mask, attn_mask },
        .{zml.Shape.init(.{ .b = k.dim(.h), .hq = heads_per_kv, .d = q.dim(.hd), .q = 1 }, q.dtype())},
        .{
            .name = "attention_decode",
            .entrypoint = kernel.entrypoint(),
            .source = source,
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
    stdx.debug.assert(k.dim(.h) == v.dim(.h), "neuron attention expects k/v head counts to match, got {f} and {f}", .{ k, v });
    stdx.debug.assert(q.dim(.hd) == k.dim(.hd) and k.dim(.hd) == v.dim(.hd), "neuron attention expects q/k/v head dimensions to match, got {f}, {f}, {f}", .{ q, k, v });
    stdx.debug.assert(k.dim(.k) == v.dim(.k), "neuron attention expects k/v sequence lengths to match, got {f} and {f}", .{ k, v });
    stdx.debug.assert(@mod(q.dim(.h), k.dim(.h)) == 0, "neuron attention expects query heads to be divisible by KV heads, got {f} and {f}", .{ q, k });
    stdx.debug.assert(token_index.rank() == 0 and token_index.dtype() == .u32, "neuron attention expects scalar u32 token_index, got {f}", .{token_index});
}
