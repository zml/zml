const std = @import("std");
const zml = @import("zml");

const Qwen3VLTextRMSNorm = @This();

weight: zml.Tensor,
/// Defaults to 1e-6
eps: f32,
tag: zml.Shape.Tag,

pub fn init(weight: zml.Tensor, tag: anytype) Qwen3VLTextRMSNorm {
    return Qwen3VLTextRMSNorm{ .weight = weight, .eps = 1e-6, .tag = zml.Shape.toTag(tag) };
}

pub fn unloadBuffers(self: *zml.Bufferized(Qwen3VLTextRMSNorm)) void {
    self.weight.deinit();
}

pub fn forward(self: Qwen3VLTextRMSNorm, x: zml.Tensor) zml.Tensor {
    const floats = x.convert(.f32);
    const variance = floats.pow(zml.Tensor.scalar(2, .f32)).mean(-1).broad(floats.shape());

    const rsqrt = (variance.add(zml.Tensor.scalar(self.eps, .f32))).rsqrt();
    return self.weight.rename(.{ .dim = .hd }).broad(floats.shape()).mul(floats.mul(rsqrt).convert(x.dtype()));
}
