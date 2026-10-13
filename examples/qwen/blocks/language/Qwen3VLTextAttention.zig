const std = @import("std");
const zml = @import("zml");

const Qwen3VLTextRMSNorm = @import("./Qwen3VLTextRMSNorm.zig");

const Qwen3VLTextAttention = @This();

q_proj: zml.nn.Linear,
k_proj: zml.nn.Linear,
v_proj: zml.nn.Linear,
o_proj: zml.nn.Linear,
q_norm: Qwen3VLTextRMSNorm,
k_norm: Qwen3VLTextRMSNorm,

pub fn unloadBuffers(self: *zml.Bufferized(Qwen3VLTextAttention)) void {
    zml.nn.Linear.unloadBuffers(&self.q_proj);
    zml.nn.Linear.unloadBuffers(&self.k_proj);
    zml.nn.Linear.unloadBuffers(&self.v_proj);
    zml.nn.Linear.unloadBuffers(&self.o_proj);
    Qwen3VLTextRMSNorm.unloadBuffers(&self.q_norm);
    Qwen3VLTextRMSNorm.unloadBuffers(&self.k_norm);
}
