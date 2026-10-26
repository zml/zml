const std = @import("std");
const zml = @import("zml");

pub const QwenImage21SwiGLUFeedForward = @This();

gate_layer: zml.nn.Linear,
out: zml.nn.Linear,
proj: zml.nn.Linear,

pub fn unloadBuffers(self: *zml.Bufferized(QwenImage21SwiGLUFeedForward)) void {
    zml.nn.Linear.unloadBuffers(&self.gate_layer);
    zml.nn.Linear.unloadBuffers(&self.out);
    zml.nn.Linear.unloadBuffers(&self.proj);
}

pub fn forward(self: QwenImage21SwiGLUFeedForward, x: zml.Tensor) zml.Tensor {
    // It seems like I need to tag again after the silu(?)
    // 0.4286152 -> 0.43063986
    const left = self.gate_layer.forward(x, x.dtype()).convert(.f32).silu().convert(x.dtype()).withTags(.{ .bs, .d, .dup });
    const right = self.proj.forward(x, x.dtype());
    const dot = left.mul(right).withTags(.{ .bs, .dup, .d });
    return self.out.forward(dot, x.dtype());
}
