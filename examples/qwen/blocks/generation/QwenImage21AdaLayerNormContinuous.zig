const zml = @import("zml");

const select_modulation_rows = @import("./utils.zig").select_modulation_rows;

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

pub fn forward(
    self: QwenImage21AdaLayerNormContinuous,
    x: zml.Tensor,
    conditioning_embedding: zml.Tensor,
    target_token_mask: zml.Tensor,
) zml.Tensor {
    const scale = self.linear.forward(conditioning_embedding.silu().convert(x.dtype()), x.dtype());
    const modulated = select_modulation_rows(scale, target_token_mask);
    return self.norm.forward(x).mul(
        zml.Tensor.scalar(1, modulated.dtype()).broad(modulated.shape()).add(modulated),
    );
}
