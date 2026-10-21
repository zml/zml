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

pub fn forward(self: QwenImage21RMS_norm, x: zml.Tensor, out_dim: usize) zml.Tensor {
    _ = out_dim; // autofix

    const normalized = zml.nn.rmsNorm(x, self.tag, self.eps);
    _ = normalized; // autofix
    // return ()
}
