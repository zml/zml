const std = @import("std");
const stdx = @import("stdx");
const log = std.log;

const zml = @import("zml");

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

    const TRANSFORMER_BLOCK_COUNT: usize = 32;

    const transformer_blocks: []TransformerBlock =
        try allocator.alloc(TransformerBlock, TRANSFORMER_BLOCK_COUNT);
    defer allocator.free(transformer_blocks);

    const transformer_blocks_view = model_store.view().withPrefix("transformer_blocks");
    for (0..TRANSFORMER_BLOCK_COUNT) |i| {
        const tb_layer_view = transformer_blocks_view.withLayer(i);

        const attn_view = tb_layer_view.withPrefix("attn");
        const attn: Attn = .{
            .norm_k = .init(attn_view.createTensor("norm_k.weight", .{.dout}, .replicated), null, .d),
            .norm_q = .init(attn_view.createTensor("norm_q.weight", .{.dout}, .replicated), null, .d),
            .to_k = .init(attn_view.createTensor("to_k.weight", .{ .dout, .d }, .replicated), null, .d),
            .to_out = .{.init(attn_view.createTensor("to_out.0.weight", .{ .dout, .d }, .replicated), null, .d)},
            .to_q = .init(attn_view.createTensor("to_q.weight", .{ .dout, .d }, .replicated), null, .d),
            .to_v = .init(attn_view.createTensor("to_v.weight", .{ .dout, .d }, .replicated), null, .d),
        };

        const img_mlp_view = tb_layer_view.withPrefix("img_mlp");
        const img_mlp: Mlp = .{
            .gate_layer = .init(img_mlp_view.createTensor("gate_layer.weight", .{ .dout, .d }, .replicated), null, .dout),
            .out = .init(img_mlp_view.createTensor("out.weight", .{ .d, .dout }, .replicated), null, .d),
            .proj = .init(img_mlp_view.createTensor("proj.weight", .{ .dout, .d }, .replicated), null, .dout),
        };

        transformer_blocks[i] = .{ .attn = attn, .img_mlp = img_mlp };
    }
}

const TransformerBlock = struct {
    img_mlp: Mlp,
    attn: Attn,

    pub fn forward(self: TransformerBlock, x: zml.Tensor) zml.Tensor {
        const y = self.attn.forward(x);
        return self.img_mlp.forward(y);
    }
};

const Mlp = struct {
    gate_layer: zml.nn.Linear,
    out: zml.nn.Linear,
    proj: zml.nn.Linear,

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        _ = self; // autofix
        return x;
    }
};

const Attn = struct {
    norm_k: zml.nn.Linear,
    norm_q: zml.nn.Linear,
    to_k: zml.nn.Linear,
    to_out: [1]zml.nn.Linear,
    to_q: zml.nn.Linear,
    to_v: zml.nn.Linear,

    pub fn forward(self: Attn, x: zml.Tensor) zml.Tensor {
        _ = self; // autofix
        return x;
    }
};
