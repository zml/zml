const std = @import("std");
const zml = @import("zml");
const CompileArgs = @import("../../compiler.zig").CompileArgs;

const Qwen3VLModel = @import("./Qwen3VLModel.zig");

const Qwen3VLForConditionalGeneration = @This();

// Qwen3VLForConditionalGeneration(
//   (model): Qwen3VLModel(
//     (visual): Qwen3VLVisionModel(
//       (patch_embed): Qwen3VLVisionPatchEmbed(
//         (proj): Conv3d(3, 1152, kernel_size=(2, 16, 16), stride=(2, 16, 16))
//       )
//       (pos_embed): Embedding(2304, 1152)
//       (rotary_pos_emb): Qwen3VLVisionRotaryEmbedding()
//       (blocks): ModuleList(
//         (0-26): 27 x Qwen3VLVisionBlock(
//           (norm1): LayerNorm((1152,), eps=1e-06, elementwise_affine=True, bias=True)
//           (norm2): LayerNorm((1152,), eps=1e-06, elementwise_affine=True, bias=True)
//           (attn): Qwen3VLVisionAttention(
//             (qkv): Linear(in_features=1152, out_features=3456, bias=True)
//             (proj): Linear(in_features=1152, out_features=1152, bias=True)
//           )
//           (mlp): Qwen3VLVisionMLP(
//             (linear_fc1): Linear(in_features=1152, out_features=4304, bias=True)
//             (linear_fc2): Linear(in_features=4304, out_features=1152, bias=True)
//             (act_fn): GELUTanh()
//           )
//         )
//       )
//       (merger): Qwen3VLVisionPatchMerger(
//         (norm): LayerNorm((1152,), eps=1e-06, elementwise_affine=True, bias=True)
//         (linear_fc1): Linear(in_features=4608, out_features=4608, bias=True)
//         (act_fn): GELU(approximate='none')
//         (linear_fc2): Linear(in_features=4608, out_features=4096, bias=True)
//       )
//       (deepstack_merger_list): ModuleList(
//         (0-2): 3 x Qwen3VLVisionPatchMerger(
//           (norm): LayerNorm((4608,), eps=1e-06, elementwise_affine=True, bias=True)
//           (linear_fc1): Linear(in_features=4608, out_features=4608, bias=True)
//           (act_fn): GELU(approximate='none')
//           (linear_fc2): Linear(in_features=4608, out_features=4096, bias=True)
//         )
//       )
//     )
//     (language_model): Qwen3VLTextModel(
//       (embed_tokens): Embedding(151936, 4096)
//       (layers): ModuleList(
//         (0-35): 36 x Qwen3VLTextDecoderLayer(
//           (self_attn): Qwen3VLTextAttention(
//             (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
//             (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
//             (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
//             (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
//             (q_norm): Qwen3VLTextRMSNorm((128,), eps=1e-06)
//             (k_norm): Qwen3VLTextRMSNorm((128,), eps=1e-06)
//           )
//           (mlp): Qwen3VLTextMLP(
//             (gate_proj): Linear(in_features=4096, out_features=12288, bias=False)
//             (up_proj): Linear(in_features=4096, out_features=12288, bias=False)
//             (down_proj): Linear(in_features=12288, out_features=4096, bias=False)
//             (act_fn): SiLUActivation()
//           )
//           (input_layernorm): Qwen3VLTextRMSNorm((4096,), eps=1e-06)
//           (post_attention_layernorm): Qwen3VLTextRMSNorm((4096,), eps=1e-06)
//         )
//       )
//       (norm): Qwen3VLTextRMSNorm((4096,), eps=1e-06)
//       (rotary_emb): Qwen3VLTextRotaryEmbedding()
//     )
//   )
//   (lm_head): Linear(in_features=4096, out_features=151936, bias=False)
// )

language_model: Qwen3VLModel,
lm_head: zml.nn.Linear,

pub fn load(
    self: *const Qwen3VLForConditionalGeneration,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *const zml.io.TensorStore,
) !zml.Bufferized(Qwen3VLForConditionalGeneration) {
    var buffers = try zml.mem.bufferize(allocator, Qwen3VLForConditionalGeneration, self);
    errdefer unloadBuffers(&buffers);

    var loader: zml.io.Loader = try .init(allocator, platform, .default);
    defer loader.deinit();

    try loader.load(io, Qwen3VLForConditionalGeneration, self, &buffers, store, &.{}, .{});
    try loader.await(io);

    return buffers;
}

pub fn unloadBuffers(self: *zml.Bufferized(Qwen3VLForConditionalGeneration)) void {
    Qwen3VLModel.unloadBuffers(&self.language_model);
}

pub fn forward(self: Qwen3VLForConditionalGeneration, input_ids: zml.Tensor, args: CompileArgs) zml.Tensor {
    const lang = self.language_model.forward(input_ids, args);
    return self.lm_head.forward(lang, lang.dtype());
}
