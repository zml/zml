const std = @import("std");
const log = std.log;

const zml = @import("zml");
const block = @import("../blocks.zig");

const TransformerStore = @This();

store: zml.io.TensorStore = undefined,
registry: zml.safetensors.TensorRegistry = undefined,
transformer_blocks: []block.TransformerBlock = undefined,
transformer_block_buffers: []zml.Bufferized(block.TransformerBlock) = undefined,

// const TRANSFORMER_BLOCK_COUNT: usize = 32;
const TRANSFORMER_BLOCK_COUNT: usize = 1;

pub fn load(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    folder: []const u8,
) !TransformerStore {
    var transformer_buffer: [256]u8 = undefined;
    const transformer_model_registry_path = try std.fmt.bufPrint(&transformer_buffer, "{s}/{s}", .{ folder, "transformer/diffusion_pytorch_model.safetensors.index.json" });
    var transformer_model_registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, transformer_model_registry_path);
    log.info("Transformer: Found {} tensors in {s}", .{ transformer_model_registry.tensors.count(), transformer_model_registry_path });

    var store: zml.io.TensorStore = .fromRegistry(allocator, &transformer_model_registry);

    const transformer_blocks: []block.TransformerBlock =
        try allocator.alloc(block.TransformerBlock, TRANSFORMER_BLOCK_COUNT);

    const transformer_blocks_view = store.view().withPrefix("transformer_blocks");
    for (0..TRANSFORMER_BLOCK_COUNT) |i| {
        const tb_layer_view = transformer_blocks_view.withLayer(i);

        const attn_view = tb_layer_view.withPrefix("attn");
        const attn: block.Attn = .{
            .seq_len = 16415,
            .num_attention_heads = 128,
            .norm_k = .init(attn_view.createTensor("norm_k.weight", .{.dim}, .replicated), null, .d),
            .norm_q = .init(attn_view.createTensor("norm_q.weight", .{.dim}, .replicated), null, .d),
            .to_k = .init(attn_view.createTensor("to_k.weight", .{ .dup, .d }, .replicated), null, .d),
            .to_out = .init(attn_view.createTensor("to_out.0.weight", .{ .dup, .d }, .replicated), null, .d),
            .to_q = .init(attn_view.createTensor("to_q.weight", .{ .dup, .d }, .replicated), null, .d),
            .to_v = .init(attn_view.createTensor("to_v.weight", .{ .dup, .d }, .replicated), null, .d),
        };

        const img_mlp_view = tb_layer_view.withPrefix("img_mlp");
        const img_mlp: block.Mlp = .{
            .proj = .init(img_mlp_view.createTensor("proj.weight", .{ .dup, .d }, .replicated), null, .d),
            .out = .init(img_mlp_view.createTensor("out.weight", .{ .dout, .d }, .replicated), null, .d),
            .gate_layer = .init(img_mlp_view.createTensor("gate_layer.weight", .{ .dup, .d }, .replicated), null, .d),
        };

        transformer_blocks[i] = .{
            .img_norm1 = .{ .eps = 1e-6 },
            .img_norm2 = .{ .eps = 1e-6 },
            .attn = attn,
            .img_mlp = img_mlp,
        };
    }

    // Load buffers
    const transformer_block_buffers =
        try allocator.alloc(zml.Bufferized(block.TransformerBlock), TRANSFORMER_BLOCK_COUNT);

    for (0..TRANSFORMER_BLOCK_COUNT) |i| {
        log.info("Transfering weights....", .{});
        const start: std.Io.Timestamp = .now(io, .awake);
        defer log.info("✅ Transferred weights [{f}]", .{start.untilNow(io, .awake)});
        transformer_block_buffers[i] = try transformer_blocks[i].load(allocator, io, platform, &store);
    }

    return .{
        .store = store,
        .registry = transformer_model_registry,
        .transformer_blocks = transformer_blocks,
        .transformer_block_buffers = transformer_block_buffers,
    };
}

pub fn deinit(self: *TransformerStore, allocator: std.mem.Allocator) void {
    for (0..TRANSFORMER_BLOCK_COUNT) |i| {
        block.TransformerBlock.unloadBuffers(&self.transformer_block_buffers[i]);
    }

    allocator.free(self.transformer_block_buffers);
    allocator.free(self.transformer_blocks);
    self.store.deinit();
    self.registry.deinit();
}
