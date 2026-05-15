const std = @import("std");
const zml = @import("zml");

// --- 1. ACTIVATION NODE ---
// This replaces zml.nn.Linear/LayerNorm.
// Using `?zml.Tensor` (optional) prevents crashes if an MoE expert wasn't used!
pub const NodeActs = struct {
    input_0: ?zml.Tensor = null,
    output: ?zml.Tensor = null,
};

// --- 2. ACTIVATION BLUEPRINT ---
pub const Step3p5LayerActs = struct {
    input_layernorm: NodeActs,
    self_attn: struct {
        q_proj: NodeActs,
        k_proj: NodeActs,
        v_proj: NodeActs,
        o_proj: NodeActs,
        q_norm: NodeActs,
        k_norm: NodeActs,
    },
    post_attention_layernorm: NodeActs,
    moe: struct {
        gate: NodeActs,
        experts: []NodeActs,
    },
};

pub const Step3p5FlashActs = struct {
    embed_tokens: NodeActs,
    layers: []Step3p5LayerActs,
    norm: NodeActs,
    lm_head: NodeActs,
};

pub const Step3p5ModelActs = struct {
    model: Step3p5FlashActs,
};

const Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const up_proj = self.up_proj.forward(x);
        const gate = self.gate_proj.forward(x);

        // for later: clamp

        return self.down_proj.forward(gate.mul(up_proj));
    }
};

// intercept inputs before providing to our layer
pub const Wrapper = struct {
    mlp: Mlp,

    pub fn forward(wrapper: Wrapper, input: zml.Tensor) zml.Tensor {
        const tagged_input = input.withTags(.{ .b, .s, .d });
        return wrapper.mlp.forward(tagged_input);
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;

    var vfs = try zml.io.VFS.init(allocator, io);
    defer vfs.deinit();
    const vfs_io = vfs.io();

    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) return;

    // Load + resolve paths (the latter, because Zig 0.16.0 treats relative paths as maybes, which breaks GPU)
    const weights_path_raw = args[1];
    const activations_path_raw = args[2];
    const cwd = std.Io.Dir.cwd();
    const model_path = try cwd.realPathFileAlloc(io, weights_path_raw, arena);
    const activations_path = try cwd.realPathFileAlloc(io, activations_path_raw, arena);

    // Init registries. See line 150 for model_store, activation_store
    var model_registry = try zml.safetensors.TensorRegistry.fromPath(allocator, vfs_io, model_path);
    defer model_registry.deinit();

    var activations_registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, activations_path);
    defer activations_registry.deinit();

    // Connect to GPUs, load drivers, create memory map
    const platform = try zml.Platform.auto(allocator, vfs_io, .{});
    defer platform.deinit(allocator, vfs_io);

    // Find the max_layer and max_expert - via search highest indices in tensor names
    var max_layer: usize = 0;
    var max_expert: usize = 0;
    var it = model_registry.tensors.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (std.mem.indexOf(u8, name, "layers.")) |idx| {
            const rest = name[idx + 7 ..];
            var parts = std.mem.splitScalar(u8, rest, '.');
            if (parts.next()) |num_str| {
                const val = std.fmt.parseInt(usize, num_str, 10) catch continue;
                if (val >= max_layer) max_layer = val + 1;
            }
        }
        if (std.mem.indexOf(u8, name, "experts.")) |idx| {
            const rest = name[idx + 8 ..];
            var parts = std.mem.splitScalar(u8, rest, '.');
            if (parts.next()) |num_str| {
                const val = std.fmt.parseInt(usize, num_str, 10) catch continue;
                if (val >= max_expert) max_expert = val + 1;
            }
        }
    }

    std.log.info("max layers = {d} | max experts = {d}", .{ max_layer, max_expert });

    // Allocate max_layers * max_experts
    const layers = try allocator.alloc(Step3p5LayerActs, max_layer);
    defer allocator.free(layers);

    for (layers) |*layer| {
        layer.* = .{
            .input_layernorm = .{},
            .self_attn = .{ .q_proj = .{}, .k_proj = .{}, .v_proj = .{}, .o_proj = .{}, .q_norm = .{}, .k_norm = .{} },
            .post_attention_layernorm = .{},
            .moe = .{
                .gate = .{},
                .experts = try allocator.alloc(NodeActs, max_expert),
            },
        };
        // Initialize experts to nulls so the loader doesn't crash on empty memory
        @memset(layer.moe.experts, .{});
    }
    defer for (layers) |layer| allocator.free(layer.moe.experts);

    // Stores memory locations for all layers
    const model_blueprint = Step3p5FlashActs{
        .embed_tokens = .{},
        .layers = layers,
        .norm = .{},
        .lm_head = .{},
    };

    const root_blueprint = Step3p5ModelActs{ .model = model_blueprint };

    // Fetch actual data from registry (from safetensors file). In other words, a model input
    var model_store = zml.io.TensorStore.fromRegistry(allocator, &model_registry);
    defer model_store.deinit();

    var activation_store = zml.io.TensorStore.fromRegistry(allocator, &activations_registry);
    defer activation_store.deinit();

    // Structural mapping - map struct field names to safetensors parameter names
    // For example, `model.layers.0.self_attn.q_proj.input_0` in the safetensors file would map to: root_blueprint.layers[0].self_attn.q_proj
    const model = try zml.io.load(Step3p5ModelActs, &root_blueprint, arena, vfs_io, platform, &model_store, .{
        .parallelism = 1,
        .dma_chunks = 2,
        .dma_chunk_size = 1024 * 1024 * 128,
    });
    _ = model;

    std.log.info("all captured activations loaded", .{});

    // For Step 3.5 Flash, we have two types of layers:
    // - MLP layer; load `model.layers.i.mlp`
    // - MoE layer; load `model.layers.i.moe` and `model.layers.i.share_expert`

    const mlp_view = model_store.view().withPrefix("model.layers.0.mlp");

    // start with initialized version zand then change it to the pub const later
    const mlp: Mlp = .{
        .up_proj = .init(mlp_view.withPrefix("up_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }), mlp_view.withPrefix("up_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }), .d),
        .gate_proj = .init(mlp_view.withPrefix("gate_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }), mlp_view.withPrefix("gate_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }), .d),
        .down_proj = .init(mlp_view.withPrefix("down_proj").createTensor("weight", .{ .d, .dout }, .{ .d = .replicated, .dout = .replicated }), mlp_view.withPrefix("down_proj").maybeCreateTensor("bias", .{.d}, .{ .d = .replicated }), .dout),
    };

    // at this point, weights are on disk and layer is initialized

    const mlp_weights = try zml.io.load(Mlp, &mlp, allocator, vfs_io, platform, &model_store, .auto);
    try zml.testing.testLayer(arena, vfs_io, platform, Wrapper{ .mlp = mlp }, .forward, activation_store.view(), "model.layers.0.mlp", .{ .mlp = mlp_weights }, &.{}, .{});
}
