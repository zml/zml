const std = @import("std");
const zml = @import("zml");

const CompileArgs = @import("../../compiler.zig").CompileArgs;

const Qwen3VLTextDecoderLayer = @import("./Qwen3VLTextDecoderLayer.zig");
const Qwen3VLTextRotaryEmbedding = @import("./Qwen3VLTextRotaryEmbedding.zig");

const Qwen3VLTextModel = @This();

// pub const TEXT_DECODE_LAYER_COUNT = 36;
pub const TEXT_DECODE_LAYER_COUNT = 1;

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
    _ = args; // autofix
    const x = self.embedding.forward(input_ids);
    const B, const S = &.{ input_ids.dim(0), input_ids.dim(1) };

    const position_ids = zml.Tensor.arange(.{ .end = S }, .i64).broad(.init(.{ .bs = B, .seq_len = S }, .i64));
    const rope_position_ids = position_ids.broad(.init(.{ .rope = 4, .bs = B, .seq_len = S }, .i64));

    const position_embeddings = self.rotary_emb.forward(x, rope_position_ids);
    _ = position_embeddings; // autofix

}
