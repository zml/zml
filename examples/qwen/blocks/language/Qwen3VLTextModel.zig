const std = @import("std");
const zml = @import("zml");

const CompileArgs = @import("../../compiler.zig").CompileArgs;

const Qwen3VLTextDecoderLayer = @import("./Qwen3VLTextDecoderLayer.zig");
const Qwen3VLTextRotaryEmbedding = @import("./Qwen3VLTextRotaryEmbedding.zig");

const Qwen3VLTextModel = @This();

embedding: zml.nn.TokenEmbedding,
layers: []Qwen3VLTextDecoderLayer,
rotary_emb: Qwen3VLTextRotaryEmbedding,

pub fn unloadBuffers(self: *zml.Bufferized(Qwen3VLTextModel)) void {
    self.embedding.weight.deinit();
    for (self.layers) |*layer| {
        Qwen3VLTextDecoderLayer.unloadBuffers(layer);
    }
}

pub fn forward(
    self: Qwen3VLTextModel,
    input_ids: zml.Tensor,
    args: CompileArgs,
) zml.Tensor {
    const x = self.embedding.forward(input_ids);
    const B = input_ids.dim(0);
    const S = input_ids.dim(1);

    const position_ids = zml.Tensor.arange(.{ .end = S }, .i64).withTags(.{.seq_len}).broad(.init(.{ .bs = B, .seq_len = S }, .i64));
    const rope_position_ids = position_ids.broad(.init(.{ .rope = 3, .bs = B, .seq_len = S }, .i64));

    const cos, const sin = self.rotary_emb.forward(x, rope_position_ids, args);

    var hidden_states = x;

    for (self.layers) |layer| {
        hidden_states = layer.forward(hidden_states, position_ids, cos, sin, args);
    }

    return hidden_states;
}
