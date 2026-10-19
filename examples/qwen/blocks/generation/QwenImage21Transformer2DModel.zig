const zml = @import("zml");
const std = @import("std");

const QwenImage21Rope = @import("./QwenImage21Rope.zig");
const QwenImage21TimestepProjEmbeddings = @import("./QwenImage21TimestepProjEmbeddings.zig");
const QwenImage21TextProjection = @import("./QwenImage21TextProjection.zig");
const QwenImage21TransformerBlock = @import("./QwenImage21TransformerBlock.zig");
const QwenImage21AdaLayerNormContinuous = @import("./QwenImage21AdaLayerNormContinuous.zig");

// QwenImage21Transformer2DModel(
//   (pos_embed): QwenImage21Rope()
//   (time_text_embed): QwenImage21TimestepProjEmbeddings(
//     (time_proj): QwenImage21TemporalTimesteps()
//     (timestep_embedder): TimestepEmbedding(
//       (linear_1): Linear(in_features=256, out_features=4096, bias=False)
//       (act): SiLU()
//       (linear_2): Linear(in_features=4096, out_features=4096, bias=False)
//     )
//   )
//   (txt_in): QwenImage21TextProjection(
//     (text_norm): QwenImage21ZeroCenterRMSNorm()
//     (in_layer): Linear(in_features=4096, out_features=4096, bias=False)
//     (act): GELU(approximate='tanh')
//     (out_layer): Linear(in_features=4096, out_features=4096, bias=False)
//   )
//   (img_in): Linear(in_features=64, out_features=4096, bias=False)
//   (modulation): Sequential(
//     (0): SiLU()
//     (1): Linear(in_features=4096, out_features=16384, bias=False)
//   )
//   (transformer_blocks): ModuleList(
//     (0-31): 32 x QwenImage21TransformerBlock(
//       (img_norm1): LayerNorm((4096,), eps=1e-06, elementwise_affine=False, bias=False)
//       (attn): QwenImage21Attention(
//         (to_q): Linear(in_features=4096, out_features=4096, bias=False)
//         (to_k): Linear(in_features=4096, out_features=4096, bias=False)
//         (to_v): Linear(in_features=4096, out_features=4096, bias=False)
//         (to_out): ModuleList(
//           (0): Linear(in_features=4096, out_features=4096, bias=False)
//           (1): Dropout(p=0.0, inplace=False)
//         )
//         (norm_q): RMSNorm()
//         (norm_k): RMSNorm()
//       )
//       (img_norm2): LayerNorm((4096,), eps=1e-06, elementwise_affine=False, bias=False)
//       (img_mlp): QwenImage21SwiGLUFeedForward(
//         (proj): Linear(in_features=4096, out_features=12288, bias=False)
//         (out): Linear(in_features=12288, out_features=4096, bias=False)
//         (gate_layer): Linear(in_features=4096, out_features=12288, bias=False)
//         (activation_fn): SiLU()
//       )
//     )
//   )
//   (norm_out): QwenImage21AdaLayerNormContinuous(
//     (silu): SiLU()
//     (linear): Linear(in_features=4096, out_features=4096, bias=False)
//     (norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=False, bias=False)
//   )
//   (proj_out): Linear(in_features=4096, out_features=64, bias=False)
// )

const QwenImage21Transformer2DModel = @This();

pos_embed: QwenImage21Rope,
time_text_embed: QwenImage21TimestepProjEmbeddings,
txt_in: QwenImage21TextProjection,
img_in: zml.nn.Linear,
modulation: zml.nn.Linear,
transformer_blocks: []QwenImage21TransformerBlock,
norm_out: QwenImage21AdaLayerNormContinuous,
proj_out: zml.nn.Linear,

pub fn load(
    self: *const QwenImage21Transformer2DModel,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *const zml.io.TensorStore,
) !zml.Bufferized(QwenImage21Transformer2DModel) {
    var buffers = try zml.mem.bufferize(allocator, QwenImage21Transformer2DModel, self);
    errdefer unloadBuffers(&buffers);

    var loader: zml.io.Loader = try .init(allocator, platform, .default);
    defer loader.deinit();

    try loader.load(io, QwenImage21Transformer2DModel, self, &buffers, store, &.{}, .{});
    try loader.await(io);

    return buffers;
}

pub fn unloadBuffers(self: *zml.Bufferized(QwenImage21Transformer2DModel)) void {
    QwenImage21TimestepProjEmbeddings.unloadBuffers(&self.time_text_embed);
    QwenImage21TextProjection.unloadBuffers(&self.txt_in);
    zml.nn.Linear.unloadBuffers(&self.img_in);
    zml.nn.Linear.unloadBuffers(&self.modulation);
    for (self.transformer_blocks) |*block| {
        QwenImage21TransformerBlock.unloadBuffers(block);
    }
    QwenImage21AdaLayerNormContinuous.unloadBuffers(&self.norm_out);
    zml.nn.Linear.unloadBuffers(&self.proj_out);
}
