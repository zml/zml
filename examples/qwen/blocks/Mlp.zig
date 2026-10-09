const std = @import("std");
const zml = @import("zml");

pub const Mlp = @This();

gate_layer: zml.nn.Linear,
out: zml.nn.Linear,
proj: zml.nn.Linear,

pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
    zml.nn.Linear.unloadBuffers(&self.gate_layer);
    zml.nn.Linear.unloadBuffers(&self.gate_layer);
    zml.nn.Linear.unloadBuffers(&self.gate_layer);
}

pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
    // It seems like I need to tag again after the silu(?)
    const left = self.gate_layer.forward(x, x.dtype()).silu().withTags(.{ .bs, .d, .dup });
    const right = self.proj.forward(x, x.dtype());
    const dot = left.mul(right).withTags(.{ .bs, .dup, .d });
    return self.out.forward(dot, x.dtype());
}
