const std = @import("std");
const zml = @import("zml");

const Qwen3VLTextDecoderLayer = @import("./Qwen3VLTextDecoderLayer.zig");
const Qwen3VLTextRotaryEmbedding = @import("./Qwen3VLTextRotaryEmbedding.zig");

const Qwen3VLTextModel = @This();

pub const TEXT_DECODE_LAYER_COUNT = 36;

embedding: zml.nn.TokenEmbedding,
layers: []Qwen3VLTextDecoderLayer,
rotary_emb: Qwen3VLTextRotaryEmbedding,

pub fn unloadBuffers(self: *zml.Bufferized(Qwen3VLTextModel)) void {
    self.embedding.weight.deinit();
    for (self.layers) |*layer| {
        Qwen3VLTextDecoderLayer.unloadBuffers(layer);
    }
}
