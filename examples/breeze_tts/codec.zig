// SPDX-License-Identifier: Apache-2.0
// Adapted from QwenLM/Qwen3-TTS (Alibaba Qwen Team) and Transformers Mimi.
const std = @import("std");
const zml = @import("zml");
const l = @import("layers.zig");
const T = zml.Tensor;
pub const Conv = struct {
    weight: T,
    bias: ?T,
    stride: i64,
    dilation: i64,
    transpose: bool,
    groups: i64,
    replicate: bool = false,
    pub fn init(s: l.View, stride: i64, dilation: i64, transpose: bool, groups: i64) Conv {
        return .{ .weight = s.createTensor("weight", .{ .cout, .cin, .kernel }, .replicated), .bias = s.maybeCreateTensor("bias", .{.d}, .replicated), .stride = stride, .dilation = dilation, .transpose = transpose, .groups = groups };
    }
    pub fn forward(self: Conv, input: T) T {
        var x = input.transpose(.{ .d, .s }).appendAxes(.{.batch}).transpose(.{ .batch, .d, .s });
        const kernel = self.weight.dim(.kernel);
        if (self.transpose) {
            x = x.conv1d(self.weight.reverse(.{.kernel}), .{ .lhs_dilation = self.stride, .padding = &.{ kernel - 1, self.stride - 1 }, .kernel_input_feature_dimension = 0, .kernel_output_feature_dimension = 1 });
        } else {
            const extra = @mod(-input.dim(.s), self.stride);
            const left = (kernel - 1) * self.dilation + 1 - self.stride;
            if (self.replicate) {
                const prefix = x.slice(.s, .{ .start = 0, .end = 1 }).broad(x.shape().setDim(.s, left));
                x = if (extra > 0) T.concatenate(&.{ prefix, x, x.slice(.s, .{ .start = x.dim(.s) - 1, .len = 1 }).broad(x.shape().setDim(.s, extra)) }, .s) else T.concatenate(&.{ prefix, x }, .s);
            }
            x = x.conv1d(self.weight, .{ .window_strides = self.stride, .rhs_dilation = self.dilation, .padding = &.{ if (self.replicate) 0 else left, if (self.replicate) 0 else extra }, .feature_group_count = self.groups });
        }
        x = x.withTags(.{ .batch, .d, .s }).squeeze(.batch).transpose(.{ .s, .d });
        if (self.bias) |b| x = x.add(b.broad(x.shape()));
        return x;
    }
};
fn gelu(x: T) T {
    // erf approximation, maximum absolute error below 1.5e-7.
    const a = x.scale(0.7071067811865476).abs();
    const t = T.scalar(1, .f32).div(a.scale(0.3275911).addConstant(1));
    const p = t.scale(1.061405429).addConstant(-1.453152027).mul(t).addConstant(1.421413741).mul(t).addConstant(-0.284496736).mul(t).addConstant(0.254829592).mul(t);
    const erf = p.mul(a.mul(a).negate().exp()).negate().addConstant(1);
    return x.scale(0.5).mul(x.cmp(.LT, T.scalar(0, x.dtype())).select(erf.negate(), erf).addConstant(1));
}
const Snake = struct {
    alpha: T,
    beta: T,
    fn init(s: l.View) Snake {
        return .{ .alpha = s.createTensor("alpha", .{.d}, .replicated), .beta = s.createTensor("beta", .{.d}, .replicated) };
    }
    fn forward(self: Snake, x: T) T {
        const wave = x.mul(self.alpha.exp().broad(x.shape())).sin();
        return x.add(wave.mul(wave).div(self.beta.exp().addConstant(1e-9).broad(x.shape())));
    }
};
const Residual = struct {
    a: Snake,
    b: Snake,
    c: Conv,
    d: Conv,
    fn init(s: l.View, dilation: i64) Residual {
        return .{ .a = .init(s.withPrefix("act1")), .b = .init(s.withPrefix("act2")), .c = .init(s.withPrefix("conv1.conv"), 1, dilation, false, 1), .d = .init(s.withPrefix("conv2.conv"), 1, 1, false, 1) };
    }
    fn forward(self: Residual, x: T) T {
        return x.add(self.d.forward(self.b.forward(self.c.forward(self.a.forward(x)))));
    }
};
const UpBlock = struct {
    snake: Snake,
    conv: Conv,
    residual: []Residual,
    fn init(a: std.mem.Allocator, s: l.View, rate: i64) !UpBlock {
        const v = s.withPrefix("block");
        const self: UpBlock = .{ .snake = .init(v.withLayer(0)), .conv = .init(v.withLayer(1).withPrefix("conv"), rate, 1, true, 1), .residual = try a.alloc(Residual, 3) };
        for (self.residual, [_]i64{ 1, 3, 9 }, 0..) |*r, d, i| r.* = .init(v.withLayer(i + 2), d);
        return self;
    }
    fn forward(self: UpBlock, x: T) T {
        var h = self.conv.forward(self.snake.forward(x));
        for (self.residual) |r| h = r.forward(h);
        return h;
    }
};
const ConvNext = struct {
    up: Conv,
    dw: Conv,
    norm: T,
    bias: T,
    p1: zml.nn.Linear,
    p2: zml.nn.Linear,
    gamma: T,
    fn init(s: l.View) ConvNext {
        const v = s.withLayer(1);
        return .{ .up = .init(s.withLayer(0).withPrefix("conv"), 2, 1, true, 1), .dw = .init(v.withPrefix("dwconv.conv"), 1, 1, false, 1024), .norm = v.createTensor("norm.weight", .{.d}, .replicated), .bias = v.createTensor("norm.bias", .{.d}, .replicated), .p1 = l.linear(v.withPrefix("pwconv1")), .p2 = l.linear(v.withPrefix("pwconv2")), .gamma = v.createTensor("gamma", .{.d}, .replicated) };
    }
    fn forward(self: ConvNext, x: T) T {
        const h = self.up.forward(x);
        const r = l.project(self.p2, gelu(l.project(self.p1, l.layerNorm(self.dw.forward(h), self.norm, self.bias, 1e-6))));
        return h.add(r.mul(self.gamma.broad(r.shape())));
    }
};
const Codebook = struct {
    sum: T,
    usage: T,
    fn init(s: l.View, encoder: bool) Codebook {
        return .{ .sum = s.createTensor(if (encoder) "embed_sum" else "embedding_sum", .{ .vocab, .d }, .replicated), .usage = s.createTensor("cluster_usage", .{.vocab}, .replicated) };
    }
    fn embedding(self: Codebook) T {
        return self.sum.div(self.usage.maximum(T.scalar(1e-5, .f32)).broad(self.sum.shape()));
    }
    fn decode(self: Codebook, ids: T) T {
        return self.embedding().gather(.{ .vocab = ids }, .{});
    }
    fn encode(self: Codebook, x: T) T {
        const e = self.embedding();
        const scores = x.dot(e, .{.d}).scale(2).sub(e.mul(e).sum(.d).squeeze(.d).broad(zml.Shape.init(.{ .s = x.dim(.s), .vocab = e.dim(.vocab) }, .f32)));
        return scores.argMax(.vocab).indices.squeeze(.vocab).convert(.u32);
    }
};
pub const Decoder = struct {
    books: []Codebook,
    first_proj: Conv,
    rest_proj: Conv,
    pre: Conv,
    input: zml.nn.Linear,
    output: zml.nn.Linear,
    norm: T,
    layers: []l.Layer,
    up: []ConvNext,
    start: Conv,
    blocks: []UpBlock,
    snake: Snake,
    end: Conv,
    pub fn init(a: std.mem.Allocator, s: l.View) !Decoder {
        const v = s.withPrefix("decoder");
        const self: Decoder = .{ .books = try a.alloc(Codebook, 16), .first_proj = .init(v.withPrefix("quantizer.rvq_first.output_proj"), 1, 1, false, 1), .rest_proj = .init(v.withPrefix("quantizer.rvq_rest.output_proj"), 1, 1, false, 1), .pre = .init(v.withPrefix("pre_conv.conv"), 1, 1, false, 1), .input = l.linear(v.withPrefix("pre_transformer.input_proj")), .output = l.linear(v.withPrefix("pre_transformer.output_proj")), .norm = v.createTensor("pre_transformer.norm.weight", .{.d}, .replicated), .layers = try a.alloc(l.Layer, 8), .up = try a.alloc(ConvNext, 2), .start = .init(v.withPrefix("decoder.0.conv"), 1, 1, false, 1), .blocks = try a.alloc(UpBlock, 4), .snake = .init(v.withPrefix("decoder.5")), .end = .init(v.withPrefix("decoder.6.conv"), 1, 1, false, 1) };
        for (self.books, 0..) |*b, i| b.* = .init(v.withPrefix(if (i == 0) "quantizer.rvq_first.vq.layers" else "quantizer.rvq_rest.vq.layers").withLayer(if (i == 0) 0 else i - 1).withPrefix("_codebook"), false);
        for (self.layers, 0..) |*b, i| b.* = .init(v.withPrefix("pre_transformer.layers").withLayer(i), .codec, i);
        for (self.up, 0..) |*b, i| b.* = .init(v.withPrefix("upsample").withLayer(i));
        for (self.blocks, [_]i64{ 8, 5, 4, 3 }, 0..) |*b, r, i| b.* = try .init(a, v.withPrefix("decoder").withLayer(i + 1), r);
        return self;
    }
    pub fn forward(self: Decoder, codes: T) T {
        const first = self.first_proj.forward(self.books[0].decode(codes.slice(.cb, .single(0))));
        var rest = self.books[1].decode(codes.slice(.cb, .single(1)));
        for (self.books[2..], 2..) |b, i| rest = rest.add(b.decode(codes.slice(.cb, .single(@intCast(i)))));
        var x = l.project(self.input, self.pre.forward(first.add(self.rest_proj.forward(rest))));
        for (self.layers) |b| x = b.forward(x);
        x = l.project(self.output, l.norm(x, self.norm, 1e-5, false));
        for (self.up) |b| x = b.forward(x);
        x = self.start.forward(x);
        for (self.blocks) |b| x = b.forward(x);
        return self.end.forward(self.snake.forward(x)).maximum(T.scalar(-1, .f32)).minimum(T.scalar(1, .f32)).reshape(.{ .samples = codes.dim(.s) * 1920 });
    }
};
const EncoderLayer = struct {
    q: zml.nn.Linear,
    k: zml.nn.Linear,
    v: zml.nn.Linear,
    o: zml.nn.Linear,
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,
    pre: T,
    pre_bias: T,
    post: T,
    post_bias: T,
    a_scale: T,
    f_scale: T,
    fn init(s: l.View) EncoderLayer {
        return .{ .q = l.linear(s.withPrefix("self_attn.q_proj")), .k = l.linear(s.withPrefix("self_attn.k_proj")), .v = l.linear(s.withPrefix("self_attn.v_proj")), .o = l.linear(s.withPrefix("self_attn.o_proj")), .fc1 = l.linear(s.withPrefix("mlp.fc1")), .fc2 = l.linear(s.withPrefix("mlp.fc2")), .pre = s.createTensor("input_layernorm.weight", .{.d}, .replicated), .pre_bias = s.createTensor("input_layernorm.bias", .{.d}, .replicated), .post = s.createTensor("post_attention_layernorm.weight", .{.d}, .replicated), .post_bias = s.createTensor("post_attention_layernorm.bias", .{.d}, .replicated), .a_scale = s.createTensor("self_attn_layer_scale.scale", .{.d}, .replicated), .f_scale = s.createTensor("mlp_layer_scale.scale", .{.d}, .replicated) };
    }
    fn forward(self: EncoderLayer, x: T) T {
        const n = l.layerNorm(x, self.pre, self.pre_bias, 1e-5);
        const pos = T.arange(.{ .end = x.dim(.s) }, .u32).withTags(.{.s});
        const q = zml.nn.rope(l.project(self.q, n).splitAxis(.d, .{ .h = 8, .hd = 64 }), pos, .{}).rename(.{ .s = .q });
        const k = zml.nn.rope(l.project(self.k, n).splitAxis(.d, .{ .h = 8, .hd = 64 }), pos, .{}).rename(.{ .s = .k });
        const v = l.project(self.v, n).splitAxis(.d, .{ .h = 8, .hd = 64 }).rename(.{ .s = .k });
        const a = l.project(self.o, zml.nn.sdpa(q, k, v, .{ .attn_mask = l.mask(x.dim(.s), x.dim(.s), T.scalar(0, .u32), 250, false, .f32) }).rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } }));
        const h = x.add(a.mul(self.a_scale.broad(a.shape())));
        const f = l.project(self.fc2, gelu(l.project(self.fc1, l.layerNorm(h, self.post, self.post_bias, 1e-5))));
        return h.add(f.mul(self.f_scale.broad(f.shape())));
    }
};
pub const Encoder = struct {
    start: Conv,
    residual1: []Conv,
    residual2: []Conv,
    downs: []Conv,
    end: Conv,
    layers: []EncoderLayer,
    downsample: Conv,
    first_proj: Conv,
    rest_proj: Conv,
    books: []Codebook,
    pub fn init(a: std.mem.Allocator, s: l.View) !Encoder {
        const v = s.withPrefix("encoder");
        const self: Encoder = .{
            .start = .init(v.withPrefix("encoder.layers.0.conv"), 1, 1, false, 1),
            .residual1 = try a.alloc(Conv, 4),
            .residual2 = try a.alloc(Conv, 4),
            .downs = try a.alloc(Conv, 4),
            .end = .init(v.withPrefix("encoder.layers.14.conv"), 1, 1, false, 1),
            .layers = try a.alloc(EncoderLayer, 8),
            .downsample = blk: {
                // Mimi uses replication padding for its final rate-conversion convolution.
                var conv = Conv.init(v.withPrefix("downsample.conv"), 2, 1, false, 1);
                conv.replicate = true;
                break :blk conv;
            },
            .first_proj = .init(v.withPrefix("quantizer.semantic_residual_vector_quantizer.input_proj"), 1, 1, false, 1),
            .rest_proj = .init(v.withPrefix("quantizer.acoustic_residual_vector_quantizer.input_proj"), 1, 1, false, 1),
            .books = try a.alloc(Codebook, 16),
        };
        for (0..4) |i| {
            const r = v.withPrefix("encoder.layers").withLayer(1 + 3 * i);
            self.residual1[i] = .init(r.withPrefix("block.1.conv"), 1, 1, false, 1);
            self.residual2[i] = .init(r.withPrefix("block.3.conv"), 1, 1, false, 1);
            self.downs[i] = .init(v.withPrefix("encoder.layers").withLayer(3 + 3 * i).withPrefix("conv"), ([_]i64{ 4, 5, 6, 8 })[i], 1, false, 1);
        }
        for (self.layers, 0..) |*b, i| b.* = .init(v.withPrefix("encoder_transformer.layers").withLayer(i));
        for (self.books, 0..) |*b, i| b.* = .init(v.withPrefix(if (i == 0) "quantizer.semantic_residual_vector_quantizer.layers" else "quantizer.acoustic_residual_vector_quantizer.layers").withLayer(if (i == 0) 0 else i - 1).withPrefix("codebook"), true);
        return self;
    }
    pub fn forward(self: Encoder, wave: T) struct { T, T } {
        var x = self.start.forward(wave.reshape(.{ .s = wave.dim(.samples), .d = 1 }));
        for (self.residual1, self.residual2, self.downs) |a, b, c| {
            x = x.add(b.forward(zml.nn.elu(a.forward(zml.nn.elu(x, 1)), 1)));
            x = c.forward(zml.nn.elu(x, 1));
        }
        x = self.end.forward(zml.nn.elu(x, 1));
        for (self.layers) |b| x = b.forward(x);
        x = self.downsample.forward(x);
        var codes: [16]T = undefined;
        codes[0] = self.books[0].encode(self.first_proj.forward(x)).appendAxes(.{.cb});
        var residual = self.rest_proj.forward(x);
        for (self.books[1..], 1..) |b, i| {
            const ids = b.encode(residual);
            codes[i] = ids.appendAxes(.{.cb});
            residual = residual.sub(b.decode(ids));
        }
        return .{ T.concatenate(&codes, .cb), x };
    }
};
