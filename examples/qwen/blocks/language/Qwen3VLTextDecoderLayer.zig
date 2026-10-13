const std = @import("std");
const zml = @import("zml");

const Qwen3VLTextAttention = @import("./Qwen3VLTextAttention.zig");
const Qwen3VLTextMLP = @import("./Qwen3VLTextMLP.zig");
const Qwen3VLTextRMSNorm = @import("./Qwen3VLTextRMSNorm.zig");

const Qwen3VLTextDecoderLayer = @This();

self_attn: Qwen3VLTextAttention,
mlp: Qwen3VLTextMLP,
input_layernorm: Qwen3VLTextRMSNorm,
post_attention_layernorm: Qwen3VLTextRMSNorm,

pub fn unloadBuffers(self: *zml.Bufferized(Qwen3VLTextDecoderLayer)) void {
    Qwen3VLTextAttention.unloadBuffers(&self.self_attn);
    Qwen3VLTextMLP.unloadBuffers(&self.mlp);
    Qwen3VLTextRMSNorm.unloadBuffers(&self.input_layernorm);
    Qwen3VLTextRMSNorm.unloadBuffers(&self.post_attention_layernorm);
}
