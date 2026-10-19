const zml = @import("zml");

// QwenImage21AdaLayerNormContinuous(
//   (silu): SiLU()
//   (linear): Linear(in_features=4096, out_features=4096, bias=False)
//   (norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=False, bias=False)
// )

pub const QwenImage21AdaLayerNormContinuous = @This();

linear: zml.nn.Linear,
norm: zml.nn.LayerNorm,

pub fn unloadBuffers(self: *zml.Bufferized(QwenImage21AdaLayerNormContinuous)) void {
    zml.nn.Linear.unloadBuffers(&self.linear);
}
