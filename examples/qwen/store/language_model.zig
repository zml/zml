const std = @import("std");
const log = std.log;

const zml = @import("zml");
const block = @import("../blocks.zig");

const LanguageModelStore = @This();

store: zml.io.TensorStore = undefined,
registry: zml.safetensors.TensorRegistry = undefined,
model: block.Qwen3VLModel,
model_buffer: zml.Bufferized(block.Qwen3VLModel),

// pub const TEXT_DECODE_LAYER_COUNT = 1;
pub const TEXT_DECODE_LAYER_COUNT = 36;

// From text_encoder/model-XXXXX-of-00002.safetensors
//  model.language_model.embed_tokens.weight                             (151936, 4096)        BF16
//  model.language_model.layers.0.input_layernorm.weight                 (4096,)               BF16
//  model.language_model.layers.0.mlp.down_proj.weight                   (4096, 12288)         BF16
//  model.language_model.layers.0.mlp.gate_proj.weight                   (12288, 4096)         BF16
//  model.language_model.layers.0.mlp.up_proj.weight                     (12288, 4096)         BF16
//  model.language_model.layers.0.post_attention_layernorm.weight        (4096,)               BF16
//  model.language_model.layers.0.self_attn.k_norm.weight                (128,)                BF16
//  model.language_model.layers.0.self_attn.k_proj.weight                (1024, 4096)          BF16
//  model.language_model.layers.0.self_attn.o_proj.weight                (4096, 4096)          BF16
//  model.language_model.layers.0.self_attn.q_norm.weight                (128,)                BF16
//  model.language_model.layers.0.self_attn.q_proj.weight                (4096, 4096)          BF16
//  model.language_model.layers.0.self_attn.v_proj.weight                (1024, 4096)          BF16
//  model.language_model.norm.weight                                     (4096,)               BF16

pub fn load_tensors(
    self: *LanguageModelStore,
    allocator: std.mem.Allocator,
    io: std.Io,
    meshes: []const *const zml.Mesh,
    folder: []const u8,
) !void {
    var buffer: [256]u8 = undefined;
    const model_registry_path = try std.fmt.bufPrint(&buffer, "{s}/{s}", .{ folder, "text_encoder/model.safetensors.index.json" });
    self.registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, model_registry_path);
    log.info("LanguageModel: Found {} tensors in {s}", .{ self.registry.tensors.count(), model_registry_path });

    self.store = .fromRegistry(allocator, &self.registry, meshes);
    const language_model_view = self.store.view().withPrefix("model.language_model");

    const embedding = zml.nn.TokenEmbedding{ .weight = language_model_view.createTensor("embed_tokens.weight", .{ .tokens, .d }, .model, .{ .d = .model }) };
    const layers: []block.Qwen3VLTextDecoderLayer =
        try allocator.alloc(block.Qwen3VLTextDecoderLayer, TEXT_DECODE_LAYER_COUNT);

    for (0..TEXT_DECODE_LAYER_COUNT) |i| {
        const layer_view = language_model_view.withPrefix("layers").withLayer(i);

        const attn_view = layer_view.withPrefix("self_attn");
        const self_attn: block.Qwen3VLTextAttention = .{
            .head_dim = 128,
            .k_norm = .init(attn_view.createTensor("k_norm.weight", .{.dim}, .replicated, .{}), .d),
            .k_proj = .init(attn_view.createTensor("k_proj.weight", .{ .dup, .d }, .model, .{ .dup = .model }), null, .d),
            .o_proj = .init(attn_view.createTensor("o_proj.weight", .{ .dup, .d }, .model, .{ .d = .model }), null, .d),
            .q_norm = .init(attn_view.createTensor("q_norm.weight", .{.dim}, .replicated, .{}), .d),
            .q_proj = .init(attn_view.createTensor("q_proj.weight", .{ .dup, .d }, .model, .{ .dup = .model }), null, .d),
            .v_proj = .init(attn_view.createTensor("v_proj.weight", .{ .dup, .d }, .model, .{ .dup = .model }), null, .d),
        };

        const mlp_view = layer_view.withPrefix("mlp");
        const mlp = block.Qwen3VLTextMLP{
            .down_proj = .init(mlp_view.createTensor("down_proj.weight", .{ .dout, .d }, .model, .{ .d = .model }), null, .d),
            .gate_proj = .init(mlp_view.createTensor("gate_proj.weight", .{ .dup, .d }, .model, .{ .dup = .model }), null, .d),
            .up_proj = .init(mlp_view.createTensor("up_proj.weight", .{ .dup, .d }, .model, .{ .dup = .model }), null, .d),
        };

        const input_layernorm = block.Qwen3VLTextRMSNorm.init(layer_view.createTensor("input_layernorm.weight", .{.dim}, .replicated, .{}), .d);
        const post_attention_layernorm = block.Qwen3VLTextRMSNorm.init(layer_view.createTensor("post_attention_layernorm.weight", .{.dim}, .replicated, .{}), .d);

        layers[i] = block.Qwen3VLTextDecoderLayer{
            .self_attn = self_attn,
            .mlp = mlp,
            .input_layernorm = input_layernorm,
            .post_attention_layernorm = post_attention_layernorm,
        };
    }

    const rotary_emb = block.Qwen3VLTextRotaryEmbedding.init();

    const language_model = block.Qwen3VLTextModel{ .embedding = embedding, .layers = layers, .rotary_emb = rotary_emb };
    self.model = block.Qwen3VLModel{ .language_model = language_model };
}

pub fn load_buffers(self: *LanguageModelStore, allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform) !void {
    self.model_buffer = try self.model.load(allocator, io, platform, &self.store);
}

pub fn deinit(self: *LanguageModelStore, allocator: std.mem.Allocator) void {
    block.Qwen3VLModel.unloadBuffers(&self.model_buffer);

    allocator.free(self.model.language_model.layers);
    allocator.free(self.model_buffer.language_model.layers);
    self.store.deinit();
    self.registry.deinit();
}
