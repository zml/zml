const zml = @import("zml");

const QwenImage21ZeroCenterRMSNorm = @import("./QwenImage21ZeroCenterRMSNorm.zig");

// QwenImage21TextProjection(
//   (text_norm): QwenImage21ZeroCenterRMSNorm()
//   (in_layer): Linear(in_features=4096, out_features=4096, bias=False)
//   (act): GELU(approximate='tanh')
//   (out_layer): Linear(in_features=4096, out_features=4096, bias=False)
// )

pub const QwenImage21TextProjection = @This();

text_norm: QwenImage21ZeroCenterRMSNorm,
in_layer: zml.nn.Linear,
out_layer: zml.nn.Linear,

pub fn unloadBuffers(self: *zml.Bufferized(QwenImage21TextProjection)) void {
    zml.nn.Linear.unloadBuffers(&self.in_layer);
    zml.nn.Linear.unloadBuffers(&self.out_layer);
}

// TODO: In BF16 there is a drift ~10% drift on the whole forward
pub fn forward(self: QwenImage21TextProjection, x: zml.Tensor) zml.Tensor {
    const text_norm = self.text_norm.forward(x);
    const in = self.in_layer.forward(text_norm, x.dtype()).convert(.f32).gelu().convert(x.dtype());
    return self.out_layer.forward(in.rename(.{ .dout = .d }), x.dtype());
}
