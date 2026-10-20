const std = @import("std");
const log = std.log;

const zml = @import("zml");
const block = @import("../blocks.zig");

const TransformerStore = @This();

// const TRANSFORMER_BLOCK_COUNT: usize = 1;
const TRANSFORMER_BLOCK_COUNT: usize = 32;

store: zml.io.TensorStore = undefined,
registry: zml.safetensors.TensorRegistry = undefined,

model: block.QwenImage21Transformer2DModel,
model_buffer: zml.Bufferized(block.QwenImage21Transformer2DModel),

// From transformer/diffusion_pytorch_model-XXXXX-of-00002.safetensors
//  img_in.weight                                                        (4096, 64)            BF16
//  modulation.1.weight                                                  (16384, 4096)         BF16
//  time_text_embed.timestep_embedder.linear_1.weight                    (4096, 256)           BF16
//  time_text_embed.timestep_embedder.linear_2.weight                    (4096, 4096)          BF16
//  transformer_blocks.0.attn.norm_k.weight                              (128,)                BF16
//  transformer_blocks.0.attn.norm_q.weight                              (128,)                BF16
//  transformer_blocks.0.attn.to_k.weight                                (4096, 4096)          BF16
//  transformer_blocks.0.attn.to_out.0.weight                            (4096, 4096)          BF16
//  transformer_blocks.0.attn.to_q.weight                                (4096, 4096)          BF16
//  transformer_blocks.0.attn.to_v.weight                                (4096, 4096)          BF16
//  transformer_blocks.0.img_mlp.gate_layer.weight                       (12288, 4096)         BF16
//  transformer_blocks.0.img_mlp.out.weight                              (4096, 12288)         BF16
//  transformer_blocks.0.img_mlp.proj.weight                             (12288, 4096)         BF16
//  txt_in.in_layer.weight                                               (4096, 4096)          BF16
//  txt_in.out_layer.weight                                              (4096, 4096)          BF16
//  norm_out.linear.weight                                               (4096, 4096)          BF16
//  proj_out.weight                                                      (64, 4096)            BF16

pub fn load_tensors(
    self: *TransformerStore,
    allocator: std.mem.Allocator,
    io: std.Io,
    folder: []const u8,
) !void {
    var transformer_buffer: [256]u8 = undefined;
    const transformer_model_registry_path = try std.fmt.bufPrint(&transformer_buffer, "{s}/{s}", .{ folder, "transformer/diffusion_pytorch_model.safetensors.index.json" });
    self.registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, transformer_model_registry_path);
    log.info("Transformer: Found {} tensors in {s}", .{ self.registry.tensors.count(), transformer_model_registry_path });

    self.store = .fromRegistry(allocator, &self.registry);

    const pos_embed = block.QwenImage21Rope.init();

    const time_text_embed_view = self.store.view().withPrefix("time_text_embed.timestep_embedder");
    const time_text_embed = time_text_embed: {
        const time_proj = block.QwenImage21TemporalTimesteps.init();
        const timestep_embedder = block.TimestepEmbedding{
            .linear_1 = .init(time_text_embed_view.createTensor("linear_1.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .linear_2 = .init(time_text_embed_view.createTensor("linear_2.weight", .{ .dout, .d }, .{ .d = .model }), null, .d),
        };
        break :time_text_embed block.QwenImage21TimestepProjEmbeddings{
            .time_proj = time_proj,
            .timestep_embedder = timestep_embedder,
        };
    };

    const text_in_view = self.store.view().withPrefix("txt_in");
    const txt_in = block.QwenImage21TextProjection{
        .text_norm = .{ .weight = text_in_view.createTensor("text_norm.weight", .{.d}, .replicated), .eps = 1e-6 },
        .in_layer = .init(text_in_view.createTensor("in_layer.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
        .out_layer = .init(text_in_view.createTensor("out_layer.weight", .{ .dout, .d }, .{ .d = .model }), null, .d),
    };

    const img_in = zml.nn.Linear.init(self.store.view().createTensor("img_in.weight", .{ .dout, .d }, .replicated), null, .d);

    const modulation = zml.nn.Linear.init(self.store.view().createTensor("modulation.1.weight", .{ .img, .d }, .{ .img = .model }), null, .d);

    const transformer_blocks: []block.QwenImage21TransformerBlock =
        try allocator.alloc(block.QwenImage21TransformerBlock, TRANSFORMER_BLOCK_COUNT);

    const transformer_blocks_view = self.store.view().withPrefix("transformer_blocks");
    for (0..TRANSFORMER_BLOCK_COUNT) |i| {
        const tb_layer_view = transformer_blocks_view.withLayer(i);

        const attn_view = tb_layer_view.withPrefix("attn");
        const attn: block.QwenImage21Attention = .{
            .num_attention_heads = 32,
            .norm_k = .init(attn_view.createTensor("norm_k.weight", .{.dim}, .replicated), null, .hd),
            .norm_q = .init(attn_view.createTensor("norm_q.weight", .{.dim}, .replicated), null, .hd),
            .to_k = .init(attn_view.createTensor("to_k.weight", .{ .dup, .d }, .{ .dup = .model }), null, .d),
            .to_out = .init(attn_view.createTensor("to_out.0.weight", .{ .dup, .d }, .{ .d = .model }), null, .d),
            .to_q = .init(attn_view.createTensor("to_q.weight", .{ .dup, .d }, .{ .dup = .model }), null, .d),
            .to_v = .init(attn_view.createTensor("to_v.weight", .{ .dup, .d }, .{ .dup = .model }), null, .d),
        };

        const img_mlp_view = tb_layer_view.withPrefix("img_mlp");
        const img_mlp: block.QwenImage21SwiGLUFeedForward = .{
            .proj = .init(img_mlp_view.createTensor("proj.weight", .{ .dup, .d }, .{ .dup = .model }), null, .d),
            .out = .init(img_mlp_view.createTensor("out.weight", .{ .dout, .d }, .{ .d = .model }), null, .d),
            .gate_layer = .init(img_mlp_view.createTensor("gate_layer.weight", .{ .dup, .d }, .{ .dup = .model }), null, .d),
        };

        transformer_blocks[i] = .{
            .img_norm1 = .{ .eps = 1e-6 },
            .img_norm2 = .{ .eps = 1e-6 },
            .attn = attn,
            .img_mlp = img_mlp,
        };
    }

    const norm_out_view = self.store.view().withPrefix("norm_out");
    const norm_out = block.QwenImage21AdaLayerNormContinuous{
        .linear = .init(norm_out_view.createTensor("linear.weight", .{ .dup, .d }, .{ .dup = .model }), null, .d),
        .norm = .{ .eps = 1e-6 },
    };

    const proj_out_view = self.store.view();
    const proj_out = zml.nn.Linear.init(proj_out_view.createTensor("proj_out.weight", .{ .dout, .d }, .{ .d = .model }), null, .d);

    self.model = block.QwenImage21Transformer2DModel{
        .pos_embed = pos_embed,
        .time_text_embed = time_text_embed,
        .txt_in = txt_in,
        .img_in = img_in,
        .modulation = modulation,
        .transformer_blocks = transformer_blocks,
        .norm_out = norm_out,
        .proj_out = proj_out,
    };
}

pub fn load_buffers(self: *TransformerStore, allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform) !void {
    self.model_buffer = try self.model.load(allocator, io, platform, &self.store);
}

pub fn deinit(self: *TransformerStore, allocator: std.mem.Allocator) void {
    block.QwenImage21Transformer2DModel.unloadBuffers(&self.model_buffer);

    allocator.free(self.model.transformer_blocks);
    allocator.free(self.model_buffer.transformer_blocks);
    self.store.deinit();
    self.registry.deinit();
}
