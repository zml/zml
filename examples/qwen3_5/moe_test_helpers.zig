const zml = @import("zml");
const Tensor = zml.Tensor;
const qwen35 = @import("qwen3_5.zig");

pub fn cachePositionToTokenIndex(cache_position: Tensor) Tensor {
    return cache_position.withTags(.{ .s }).slice1d(.s, .{ .start = 0, .end = 1 }).squeeze(.s).convert(.u32);
}

pub fn emptyLayerKvCache(layer: qwen35.TransformerLayer, x: Tensor) qwen35.KvCache {
    return .{
        .layer_types = switch (layer.attn) {
            .self_attn => single_full_layer_types[0..],
            .linear_attn => single_linear_layer_types[0..],
        },
        .self_attn = switch (layer.attn) {
            .self_attn => |self_attn| emptySelfAttnCache(self_attn, x),
            .linear_attn => dummySelfAttnCache(x.dtype(), x.dim(.b), x.dim(.s)),
        },
        .gated_delta_net = switch (layer.attn) {
            .linear_attn => |linear_attn| emptyLinearAttnCache(linear_attn, x),
            .self_attn => dummyLinearAttnCache(x.dtype(), x.dim(.b)),
        },
    };
}

fn zeroTensor(shape: zml.Shape) Tensor {
    return Tensor.constant(shape.dtype().zero()).broad(shape);
}

fn emptySelfAttnCache(self_attn: qwen35.SelfAttn, x: Tensor) qwen35.KvCache.SelfAttnCache {
    const cache_shape = zml.Shape.init(.{
        .b = x.dim(.b),
        .layer = 1,
        .s = x.dim(.s),
        .h = self_attn.num_kv_heads,
        .hd = self_attn.head_dim,
    }, x.dtype());

    return .{
        .k = zeroTensor(cache_shape),
        .v = zeroTensor(cache_shape),
        .layer_index = Tensor.scalar(@as(u32, 0), .u32),
    };
}

fn emptyLinearAttnCache(linear_attn: qwen35.GatedDeltaNet, x: Tensor) qwen35.KvCache.GatedDeltaNetCache {
    const conv_dim = 2 * linear_attn.num_k_heads * linear_attn.head_k_dim + linear_attn.num_v_heads * linear_attn.head_v_dim;
    const left_pad = linear_attn.conv_kernel_size - 1;
    const conv_shape = zml.Shape.init(.{
        .b = x.dim(.b),
        .layer = 1,
        .s = left_pad,
        .mix = conv_dim,
    }, x.dtype());
    const recurrent_shape = zml.Shape.init(.{
        .b = x.dim(.b),
        .layer = 1,
        .vh = linear_attn.num_v_heads,
        .khd = linear_attn.head_k_dim,
        .vhd = linear_attn.head_v_dim,
    }, .f32);

    return .{
        .conv_state = zeroTensor(conv_shape),
        .recurrent_state = zeroTensor(recurrent_shape),
        .layer_index = Tensor.scalar(@as(u32, 0), .u32),
    };
}

const single_full_layer_types = [_]qwen35.Qwen35.LayerType{.full_attention};
const single_linear_layer_types = [_]qwen35.Qwen35.LayerType{.linear_attention};

fn dummySelfAttnCache(dtype: zml.DataType, batch_dim: i64, seq_len: i64) qwen35.KvCache.SelfAttnCache {
    const shape = zml.Shape.init(.{ .b = batch_dim, .layer = 1, .s = seq_len, .h = 1, .hd = 1 }, dtype);
    return .{
        .k = zeroTensor(shape),
        .v = zeroTensor(shape),
        .layer_index = Tensor.scalar(@as(u32, 0), .u32),
    };
}

fn dummyLinearAttnCache(conv_dtype: zml.DataType, batch_dim: i64) qwen35.KvCache.GatedDeltaNetCache {
    return .{
        .conv_state = zeroTensor(zml.Shape.init(.{ .b = batch_dim, .layer = 1, .s = 1, .mix = 1 }, conv_dtype)),
        .recurrent_state = zeroTensor(zml.Shape.init(.{ .b = batch_dim, .layer = 1, .vh = 1, .khd = 1, .vhd = 1 }, .f32)),
        .layer_index = Tensor.scalar(@as(u32, 0), .u32),
    };
}
