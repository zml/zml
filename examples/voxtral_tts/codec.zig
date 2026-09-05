// SPDX-License-Identifier: Apache-2.0
// Adapted from vLLM-Omni's Voxtral audio tokenizer; copyright contributors to the vLLM project.
// Native Zig/ZML implementation. See README.md for the pinned reference.
const std = @import("std");
const zml = @import("zml");
const layers = @import("layers.zig");
const Tensor = zml.Tensor;

pub const Conv = struct {
    magnitude: Tensor,
    direction: Tensor,

    pub fn init(store: layers.View) Conv {
        return .{
            .magnitude = store.createTensor("conv.parametrizations.weight.original0", .{ .cout, .cin, .kernel }, .replicated),
            .direction = store.createTensor("conv.parametrizations.weight.original1", .{ .cout, .cin, .kernel }, .replicated),
        };
    }

    pub fn weight(self: Conv) Tensor {
        const v = self.direction.convert(.f32);
        const magnitude = self.magnitude.convert(.f32);
        const lengths = v.mul(v).sum(.cin).sum(.kernel).sqrt();
        return v.mul(magnitude.div(lengths).broad(v.shape())).convert(self.direction.dtype());
    }

    pub fn forward(self: Conv, input: Tensor, transpose: bool, reflect: bool) Tensor {
        const weight_ = self.weight();
        var x = input.rename(.{ .s = .time, .d = .channels }).appendAxes(.{.batch}).transpose(.{ .batch, .channels, .time });
        const kernel = weight_.dim(.kernel);
        if (transpose) {
            // The Metal convolution path ignores window_reversal. Reverse the
            // taps explicitly so transposed convolution agrees on all targets.
            x = x.conv1d(weight_.reverse(.{.kernel}), .{
                .lhs_dilation = 2,
                .padding = &.{ kernel - 1, 1 },
                .kernel_input_feature_dimension = 0,
                .kernel_output_feature_dimension = 1,
            });
        } else {
            const padding = kernel - 1;
            const prefix = if (reflect)
                x.slice(.time, .{ .start = 1, .end = 1 + padding }).reverse(.{.time})
            else
                x.slice(.time, .{ .start = 0, .end = 1 }).broad(x.shape().setDim(.time, padding));
            x = Tensor.concatenate(&.{ prefix, x }, .time).conv1d(weight_, .{});
        }
        return x.withTags(.{ .batch, .channels, .time }).squeeze(.batch).transpose(.{ .time, .channels }).rename(.{ .time = .s, .channels = .d });
    }
};

pub const Codec = struct {
    usage: Tensor,
    embedding_sum: Tensor,
    convs: []Conv,
    blocks: []layers.Layer,
    output: Conv,

    pub fn init(allocator: std.mem.Allocator, store_: layers.View) !Codec {
        const store = store_.withPrefix("audio_tokenizer");
        const convs = try allocator.alloc(Conv, 4);
        errdefer allocator.free(convs);
        const self: Codec = .{
            .usage = store.createTensor("quantizer.semantic_codebook.cluster_usage", .{.vocab}, .replicated),
            .embedding_sum = store.createTensor("quantizer.semantic_codebook.embedding_sum", .{ .vocab, .d }, .replicated),
            .convs = convs,
            .blocks = try allocator.alloc(layers.Layer, 8),
            .output = .init(store.withPrefix("output_proj")),
        };
        for (self.convs, 0..) |*conv, i| {
            conv.* = .init(store.withPrefix("decoder_blocks").withLayer(2 * i));
            for (self.blocks[2 * i ..][0..2], 0..) |*block, j| block.* = .init(store.withPrefix("decoder_blocks").withLayer(2 * i + 1).withPrefix("layers").withLayer(j), true);
        }
        return self;
    }

    pub fn forward(self: Codec, codes: Tensor) Tensor {
        return self.stages(codes)[9].merge(.{ .samples = .{ .s, .d } }).convert(.f32);
    }

    pub fn stages(self: Codec, codes: Tensor) [10]Tensor {
        var trace: [10]Tensor = undefined;
        const ids = codes.convert(.i32).addConstant(-2);
        const embedding = self.embedding_sum.convert(.f32).div(self.usage.convert(.f32).maximum(Tensor.scalar(1e-5, .f32)).broad(self.embedding_sum.shape().withDtype(.f32))).convert(.bf16);
        const semantic = embedding.gather(.{ .vocab = ids.slice(.cb, .single(0)) }, .{});
        const acoustic = ids.slice(.cb, .{ .start = 1, .len = 36 }).convert(.f32).scale(0.1).addConstant(-1).convert(.bf16).rename(.{ .cb = .d });
        var h = Tensor.concatenate(&.{ semantic, acoustic }, .d);
        trace[0] = h;
        for (self.convs, 0..) |conv, i| {
            h = conv.forward(h, i != 0, false);
            trace[2 * i + 1] = h;
            const mask = alibiMask(h.dim(.s), @as(u32, 2) << @intCast(i));
            for (self.blocks[2 * i ..][0..2]) |block| h = block.forward(h, mask);
            trace[2 * i + 2] = h;
        }
        h = self.output.forward(h, false, true);
        trace[9] = h;
        return trace;
    }
};

pub fn alibiMask(length: i64, window: u32) Tensor {
    const shape = zml.Shape.init(.{ .h = 8, .q = length, .k = length }, .bf16);
    const q = Tensor.arange(.{ .end = length }, .i32).withTags(.{.q}).broad(shape.withDtype(.i32));
    const k = Tensor.arange(.{ .end = length }, .i32).withTags(.{.k}).broad(shape.withDtype(.i32));
    const relative = k.sub(q);
    const slopes = Tensor.constantTensor(.init(.{ .h = 8 }, .f32), std.mem.sliceAsBytes(&[_]f32{ 1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125 })).convert(.bf16);
    const bias = relative.convert(.bf16).mul(slopes.broad(shape));
    const causal = layers.causalMask(length, length, Tensor.scalar(0, .u32), window, .bf16);
    return bias.add(causal.broad(shape));
}
