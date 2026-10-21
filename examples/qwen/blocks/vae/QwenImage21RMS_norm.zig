const zml = @import("zml");
const std = @import("std");

pub const QwenImage21RMS_norm = @This();

gamma: zml.Tensor,
/// Defaults to 1e-6
eps: f32 = 1e-6,
tag: zml.Shape.Tag,

pub fn unloadBuffers(self: *zml.Bufferized(QwenImage21RMS_norm)) void {
    self.gamma.deinit();
}

pub fn forward(self: QwenImage21RMS_norm, x: zml.Tensor, out_dim: i64) zml.Tensor {
    const scale = zml.Tensor.scalar(@as(f32, @floatFromInt(out_dim)), .f32).broad(x.shape());

    const normalized = zml.nn.rmsNorm(x.convert(.f32), self.tag, self.eps);
    return normalized.mul(scale).mul(self.gamma.broad(normalized.shape())).convert(x.dtype());
}
