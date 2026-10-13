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
