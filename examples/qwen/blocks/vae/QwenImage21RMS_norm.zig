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

pub fn forward(self: QwenImage21RMS_norm, x: zml.Tensor) zml.Tensor {
    const normalized = zml.nn.rmsNorm(x.convert(.f32), self.tag, self.eps);
    const gamma = self.gamma.squeeze(.kx).squeeze(.ky);
    return normalized.mul(gamma.broad(normalized.shape())).convert(x.dtype());
}
