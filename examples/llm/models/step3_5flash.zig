const std = @import("std");
const zml = @import("zml");
const common = @import("common.zig");

pub const Config = struct {
    architectures: []const []const u8 = &.{},
    model_type: []const u8,

    auto_map: ?AutoMap = null,

    rope_scaling: RopeScaling,
    yarn_only_types: []const []const u8 = &.{},

    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    max_seq_len: u32,
    vocab_size: u32,

    torch_dtype: []const u8 = "bfloat16",

    use_qk_norm: bool = false,

    moe_layers_enum: []const u8 = "",
    num_attention_heads: u32,
    num_attention_groups: u32,
    head_dim: u32,

    use_moe: bool = false,
    moe_num_experts: u32 = 0,
    moe_top_k: u32 = 0,
    moe_intermediate_size: u32 = 0,
    share_expert_dim: u32 = 0,
    moe_layer_offset: u32 = 0,
    moe_every_n_layer: u32 = 1,
    norm_expert_weight: bool = false,
    moe_router_activation: []const u8 = "sigmoid",
    moe_router_scaling_factor: f32 = 1.0,

    att_impl_type: []const u8 = "GQA",
    tie_word_embeddings: bool = false,

    rope_theta: []const f32,

    use_head_wise_attn_gate: bool = false,
    sliding_window: u32 = 0,

    use_moe_router_bias: bool = false,
    need_fp32_gate: bool = false,
    sink: bool = false,

    layer_types: []const []const u8 = &.{},
    use_rope_layers: []const u32 = &.{},

    num_nextn_predict_layers: u32 = 0,
    partial_rotary_factors: []const f32 = &.{},

    attention_other_setting: ?AttentionOtherSetting = null,

    swiglu_limits: []const f32 = &.{},
    swiglu_limits_shared: []const f32 = &.{},

    zero_centered: bool = false,
    max_position_embeddings: u32,

    pub const AutoMap = struct {
        AutoConfig: []const u8,
        AutoModelForCausalLM: []const u8,
    };

    pub const RopeScaling = struct {
        rope_type: []const u8,
        factor: f32,
        original_max_position_embeddings: u32,
        low_freq_factor: f32,
        high_freq_factor: f32,
    };

    pub const AttentionOtherSetting = struct {
        attention_type: []const u8,
        num_attention_heads: u32,
        num_attention_groups: u32,
        head_dim: u32,
        true_head_dim: u32,
    };

    pub fn numKeyValueHeads(self: Config) u32 {
        return self.num_attention_groups;
    }
};

// Multilayer Perceptron
const Mlp = struct {
    up_proj: zml.nn.Linear, // (dim -> hidden_dim)
    gate_proj: zml.nn.Linear, // (dim -> hidden_dim)
    down_proj: zml.nn.Linear, // (hidden_dim -> dim)
    limit: ?i32,

    pub fn init(self: Mlp, store: zml.io.TensorStore.View, swiglu_limit: ?i32) Mlp {
        self.limit = swiglu_limit;
        return .{
            .up_proj = .init(store.createTensor("up_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .gate_proj = .init(store.createTensor("gate_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .down_proj = .init(store.createTensor("down_proj.weight", .{ .dout, .d }, .{ .d = .model }), null, .d),
        };
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        // Add tags to input before providing to our layer.
        const input = x.withTags(.{ .b, .s, .d });

        var up_proj = self.up_proj.forward(input);
        var gate = self.gate_proj.forward(input);
        gate = gate.silu();

        // Step 3.5 Flash clamps gate projection asymmetrically
        if (self.limit) |limit| {
            if (limit != 0) {
                const lim_f = @as(f32, @floatFromInt(limit));
                const max_t = zml.Tensor.scalar(lim_f, gate.dtype());
                const min_t = zml.Tensor.scalar(-lim_f, gate.dtype());

                // Step 3.5 Flash has asymmetric clamping of gate projection
                gate = gate.minimum(max_t);
                up_proj = up_proj.clamp(min_t, max_t);
            }
        }
        return self.down_proj.forward(gate.mul(up_proj));
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;

    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 4) return;

    // Load + resolve paths (Zig 0.16.0 treats relative paths as maybes, which breaks GPU)
    const weights_path_raw = args[1];
    const cwd = std.Io.Dir.cwd();
<<<<<<< HEAD
    const absolute_path = try cwd.realPathFileAlloc(io, weights_path_raw, arena);

<<<<<<< HEAD
    const platform = try zml.Platform.auto(allocator, vfs_io, .{});
    defer platform.deinit(allocator, vfs_io);

    var registry = try zml.safetensors.TensorRegistry.fromPath(allocator, vfs_io, absolute_path);
    defer registry.deinit();

    // Find the max_layer and max_expert - via search highest indices in tensor names
    var max_layer: usize = 0;
    var max_expert: usize = 0;
    var it = registry.tensors.iterator();
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
    var store = zml.io.TensorStore.fromRegistry(allocator, &registry);

    // Structural mapping - map struct field names to safetensors parameter names
    // For example, `model.layers.0.self_attn.q_proj.input_0` in the safetensors file would map to: root_blueprint.layers[0].self_attn.q_proj
    const model = try zml.io.load(Step3p5ModelActs, &root_blueprint, arena, vfs_io, platform, &store, .{
        .parallelism = 1,
        .dma_chunks = 2,
        .dma_chunk_size = 1024 * 1024 * 128,
    });
    _ = model;

    std.log.info("all captured activations loaded", .{});
    const activations_path = try cwd.realPathFileAlloc(io, activations_path_raw, arena);

    // Registries
    const repo = try zml.safetensors.resolveModelRepo(io, weights_path_raw);
    var model_registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer model_registry.deinit();
    var activations_registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, activations_path);
    defer activations_registry.deinit();

    // Connect to GPUs, load drivers, create memory map
    const platform = try zml.Platform.auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    // Stores
    var model_store = zml.io.TensorStore.fromRegistry(allocator, &model_registry);
    defer model_store.deinit();

    var activation_store = zml.io.TensorStore.fromRegistry(allocator, &activations_registry);
    defer activation_store.deinit();

    std.log.info("All captured activations loaded", .{});

    // Load config
    const config_path_raw = args[3];

    const dir = try std.Io.Dir.openDir(.cwd(), io, config_path_raw, .{});
    defer dir.close(io);
    const parsed = try common.parseConfig(Config, allocator, io, dir);
    defer parsed.deinit();

    ////////////////////////////////////////////////////////////////
    const current_layer = "model.layers.2.mlp";
    const swiglu_limit = 0;
    ////////////////////////////////////////////////////////////////
    const mlp_view = model_store.view().withPrefix(current_layer);

    // start with initialized version zand then change it to the pub const later
    const mlp: Mlp = .{
        .up_proj = .init(mlp_view.withPrefix("up_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }), mlp_view.withPrefix("up_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }), .d),
        .gate_proj = .init(mlp_view.withPrefix("gate_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }), mlp_view.withPrefix("gate_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }), .d),
        .down_proj = .init(mlp_view.withPrefix("down_proj").createTensor("weight", .{ .d, .dout }, .{ .d = .replicated, .dout = .replicated }), mlp_view.withPrefix("down_proj").maybeCreateTensor("bias", .{.d}, .{ .d = .replicated }), .dout),
        .limit = swiglu_limit,
    };

    // at this point, weights are on disk and layer is initialized
    const mlp_weights = try zml.io.load(Mlp, &mlp, allocator, io, platform, &model_store, .auto);
    try zml.testing.testLayer(arena, io, platform, mlp, .forward, activation_store.view(), current_layer, mlp_weights, &.{}, .{});
}
