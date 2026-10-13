const std = @import("std");
const zml = @import("zml");

const Qwen3VLTextMLP = @This();

gate_proj: zml.nn.Linear,
up_proj: zml.nn.Linear,
down_proj: zml.nn.Linear,

pub fn unloadBuffers(self: *zml.Bufferized(Qwen3VLTextMLP)) void {
    zml.nn.Linear.unloadBuffers(&self.gate_proj);
    zml.nn.Linear.unloadBuffers(&self.up_proj);
    zml.nn.Linear.unloadBuffers(&self.down_proj);
}

pub fn forward(self: Qwen3VLTextMLP, x: zml.Tensor) zml.Tensor {
    // It seems like I need to tag again after the silu(?)
    const left = self.gate_proj.forward(x, x.dtype()).silu().withTags(.{ .bs, .d, .dup });
    const right = self.up_proj.forward(x, x.dtype());
    const dot = left.mul(right).withTags(.{ .bs, .dup, .d });
    return self.down_proj.forward(dot, x.dtype());
}
