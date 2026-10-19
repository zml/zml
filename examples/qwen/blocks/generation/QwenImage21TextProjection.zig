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
