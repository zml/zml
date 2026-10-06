const std = @import("std");
const stdx = @import("stdx");
const log = std.log;

const zml = @import("zml");
const block = @import("./blocks.zig");

const CliArgs = struct {
    pub const help =
        \\ torch2zml --index=<path> --activations=<path>
    ;
    index: [:0]const u8,
    activations: [:0]const u8,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const cli_args: CliArgs = stdx.flags.parse(init.minimal.args, CliArgs);

    var activations_registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, cli_args.activations);
    defer activations_registry.deinit();
    log.info("Found {} activations in {s}", .{ activations_registry.tensors.count(), cli_args.activations });

    var model_registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, cli_args.index);
    defer model_registry.deinit();
    log.info("Found {} activations in {s}", .{ model_registry.tensors.count(), cli_args.index });

    var model_store: zml.io.TensorStore = .fromRegistry(allocator, &model_registry);
    defer model_store.deinit();

    // const TRANSFORMER_BLOCK_COUNT: usize = 32;
    const TRANSFORMER_BLOCK_COUNT: usize = 1;

    const transformer_blocks: []block.TransformerBlock =
        try allocator.alloc(block.TransformerBlock, TRANSFORMER_BLOCK_COUNT);
    defer allocator.free(transformer_blocks);

    const transformer_blocks_view = model_store.view().withPrefix("transformer_blocks");
    for (0..TRANSFORMER_BLOCK_COUNT) |i| {
        const tb_layer_view = transformer_blocks_view.withLayer(i);

        const attn_view = tb_layer_view.withPrefix("attn");
        const attn: block.Attn = .{
            .norm_k = .init(attn_view.createTensor("norm_k.weight", .{.dup}, .replicated), null, .d),
            .norm_q = .init(attn_view.createTensor("norm_q.weight", .{.dup}, .replicated), null, .d),
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

        transformer_blocks[i] = .{ .attn = attn, .img_mlp = img_mlp };
    }

    // Auto-select platform
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    // Load buffers
    const transformer_blocks_buffer =
        try allocator.alloc(zml.Bufferized(block.TransformerBlock), TRANSFORMER_BLOCK_COUNT);
    defer allocator.free(transformer_blocks_buffer);

    for (0..TRANSFORMER_BLOCK_COUNT) |i| {
        log.info("Transfering weights....", .{});
        const start: std.Io.Timestamp = .now(io, .awake);
        defer log.info("✅ Transferred weights [{f}]", .{start.untilNow(io, .awake)});
        transformer_blocks_buffer[i] = try transformer_blocks[i].load(init.arena.allocator(), io, platform, &model_store);
    }
    defer for (0..TRANSFORMER_BLOCK_COUNT) |i| {
        block.TransformerBlock.unloadBuffers(&transformer_blocks_buffer[i]);
    };

    var activations_store: zml.io.TensorStore = .fromRegistry(allocator, &activations_registry);
    defer activations_store.deinit();

    std.debug.print("\n\nStarting testing\n\n", .{});

    try zml.testing.testLayer(
        allocator,
        io,
        platform,
        Wrapper{ .mlp = transformer_blocks[0].img_mlp },
        .forward,
        &activations_store,
        "transformer.transformer_blocks.0.img_mlp",
        .{ .mlp = transformer_blocks_buffer[0].img_mlp },
        &.{},
        .{},
    );
}

// pos_embed -> <class 'diffusers.models.transformers.transformer_qwenimage21.QwenImage21Rope'>
// time_text_embed -> <class 'diffusers.models.transformers.transformer_qwenimage21.QwenImage21TimestepProjEmbeddings'>
// txt_in -> <class 'diffusers.models.transformers.transformer_qwenimage21.QwenImage21TextProjection'>
// img_in -> <class 'torch.nn.modules.linear.Linear'>
// modulation -> <class 'torch.nn.modules.container.Sequential'>
// transformer_blocks -> <class 'torch.nn.modules.container.ModuleList'>
// norm_out -> <class 'diffusers.models.transformers.transformer_qwenimage21.QwenImage21AdaLayerNormContinuous'>
// proj_out -> <class 'torch.nn.modules.linear.Linear'>

// QwenImage21TransformerBlock(
//   (img_norm1): LayerNorm((4096,), eps=1e-06, elementwise_affine=False, bias=False)
//   (attn): QwenImage21Attention(
//     (to_q): Linear(in_features=4096, out_features=4096, bias=False)
//     (to_k): Linear(in_features=4096, out_features=4096, bias=False)
//     (to_v): Linear(in_features=4096, out_features=4096, bias=False)
//     (to_out): ModuleList(
//       (0): Linear(in_features=4096, out_features=4096, bias=False)
//       (1): Dropout(p=0.0, inplace=False)
//     )
//     (norm_q): RMSNorm()
//     (norm_k): RMSNorm()
//   )
//   (img_norm2): LayerNorm((4096,), eps=1e-06, elementwise_affine=False, bias=False)
//   (img_mlp): QwenImage21SwiGLUFeedForward(
//     (proj): Linear(in_features=4096, out_features=12288, bias=False)
//     (out): Linear(in_features=12288, out_features=4096, bias=False)
//     (gate_layer): Linear(in_features=4096, out_features=12288, bias=False)
//     (activation_fn): SiLU()
//   )
// )

const Wrapper = struct {
    mlp: block.Mlp,

    pub fn forward(self: Wrapper, x: zml.Tensor) zml.Tensor {
        const tagged = x.withTags(.{ .bs, .dout, .d });
        return self.mlp.forward(tagged);
    }
};
