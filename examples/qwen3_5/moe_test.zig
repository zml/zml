const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;
const stdx = zml.stdx;

const qwen35 = @import("qwen3_5.zig");
const Qwen35 = qwen35.Qwen35;
const helpers = @import("moe_test_helpers.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    activations: []const u8,
    moe_backend: ?[]const u8 = null,

    pub const help =
        \\Use moe_test --model=<path> --activations=<path>
        \\
        \\Validate the Qwen3.5 MoE implementation against activation fixtures.
        \\
        \\Options:
        \\  --model=<path>            Path to the model repository
        \\  --activations=<path>      Path to activation safetensors
        \\  --moe-backend=<name>      MoE backend (triton); default is auto
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = stdx.flags.parse(init.minimal.args, Args);

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const parsed_config = try parseConfig(allocator, io, repo);
    defer parsed_config.deinit();
    const config = parsed_config.value;
    if (try config.kind() != .qwen3_5_moe) {
        return error.UnsupportedModelType;
    }

    const gen_options: Qwen35.GenOptions = .{ .max_seq_len = 1 };
    var qwen_model: Qwen35 = try .init(allocator, store.view(), config, gen_options);
    defer qwen_model.deinit(allocator);

    const dtype = qwen_model.text_model.embed_tokens.weight.dtype();
    const moe_backend: zml.moe.Backend = if (args.moe_backend) |name| b: {
        if (std.mem.eql(u8, name, "triton")) break :b .triton;
        std.log.err("Unknown MoE backend: {s}", .{name});
        return error.UnsupportedMoeBackend;
    } else zml.moe.Backend.auto(platform, dtype) catch |err| {
        if (err == error.UnimplementedMoEBackend) {
            std.log.err("No MoE backend available. Enable CUDA (e.g. --@zml//platforms:cuda=true) or pass --moe-backend=triton.", .{});
        }
        return err;
    };
    try moe_backend.load(allocator);
    try moe_backend.register(platform);

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    var qwen35_buffers = try qwen_model.load(allocator, io, platform, &store, &.{replicated_sharding}, &progress);
    defer Qwen35.unloadBuffers(&qwen35_buffers, allocator);
    progress.end();

    try run(allocator, io, platform, args.activations, qwen_model, qwen35_buffers, replicated_sharding);
}

fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activations_path: []const u8,
    qwen_model: Qwen35,
    qwen35_buffers: zml.Bufferized(Qwen35),
    sharding: zml.sharding.Sharding,
) !void {
    var activations_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, activations_path);
    defer activations_registry.deinit();

    var activations_store: zml.io.TensorStore = .fromRegistry(allocator, &activations_registry);
    defer activations_store.deinit();

    const comp_opts: zml.testing.CompareOpts = .{
        .absolute_tolerance = 1e-2,
        .relative_tolerance = 1e-2,
        .minimum_close_fraction = 0.999,
    };
    const shardings = &.{sharding};

    const EmbedTokensHarness = struct {
        embedTokens: zml.nn.TokenEmbedding,
        pub fn forward(self: @This(), inputIds: Tensor) Tensor {
            return self.embedTokens.weight
                .gather(.{ .voc = inputIds.withTags(.{ .b, .s }) }, .{});
        }
    };

    const embedTokensHarness: EmbedTokensHarness = .{
        .embedTokens = qwen_model.text_model.embed_tokens,
    };
    const embedTokensHarnessBuffers: zml.Bufferized(EmbedTokensHarness) = .{
        .embedTokens = qwen35_buffers.text_model.embed_tokens,
    };
    try zml.testing.testLayer(
        allocator,
        io,
        platform,
        embedTokensHarness,
        .forward,
        activations_store.view(),
        "model.model.embed_tokens",
        embedTokensHarnessBuffers,
        shardings,
        comp_opts,
    );

    const LinearBSDHarness = struct {
        linear: zml.nn.Linear,
        pub fn forward(self: @This(), x: Tensor) Tensor {
            return self.linear.forward(x.withTags(.{ .b, .s, .d }));
        }
    };

    const LinearSDHarness = struct {
        linear: zml.nn.Linear,
        pub fn forward(self: @This(), x: Tensor) Tensor {
            return self.linear.forward(x.withTags(.{ .s, .d }));
        }
    };

    const RmsNormBSDHarness = struct {
        norm: qwen35.RmsNorm,
        pub fn forward(self: @This(), x: Tensor) Tensor {
            return self.norm.forward(x.withTags(.{ .b, .s, .d }));
        }
    };

    // const RmsNormSDHarness = struct {
    //     norm: qwen35.RmsNorm,
    //     pub fn forward(self: @This(), x: Tensor) Tensor {
    //         return self.norm.forward(x.withTags(.{ .s, .d }));
    //     }
    // };

    const RmsNormGatedSDHarness = struct {
        norm: qwen35.RmsNormGated,
        pub fn forward(self: @This(), x: Tensor, gate: Tensor) Tensor {
            return self.norm.forward(x.withTags(.{ .s, .d }), gate.withTags(.{ .s, .d }));
        }
    };

    const MlpSDHarness = struct {
        mlp: qwen35.Mlp,
        pub fn forward(self: @This(), x: Tensor) Tensor {
            return self.mlp.forward(x.withTags(.{ .s, .d }));
        }
    };

    const MoeHarness = struct {
        moe: qwen35.Moe,
        config: Qwen35.Config,
        pub fn forward(self: @This(), x: Tensor) Tensor {
            return self.moe.forward(x.withTags(.{ .b, .s, .d }), self.config);
        }
    };

    const MoeGateHarness = struct {
        router: zml.nn.Linear,
        num_experts_per_tok: u32,

        pub fn forward(self: @This(), x: Tensor) struct { Tensor, Tensor, Tensor } {
            const hidden_states = x.withTags(.{ .s, .d });
            const router_logits = self.router.forward(hidden_states).convert(.f32);
            const router_probs = router_logits.softmax(.expert);
            const routing = router_probs.topK(.{ .top_expert = .expert }, self.num_experts_per_tok, .{});
            const router_scores = routing.values.div(routing.values.sum(.top_expert).broad(routing.values.shape()));
            const router_indices = routing.indices.convert(.i64);
            std.log.info("router_probs: {f}", .{router_probs});
            std.log.info("router_scores: {f}", .{router_scores});
            std.log.info("router_indices: {f}", .{router_indices});
            return .{ router_probs, router_scores, router_indices };
        }
    };
    _ = MoeGateHarness; // autofix

    const MoeExpertsHarness = struct {
        gate_up_proj: Tensor,
        down_proj: Tensor,

        pub fn forward(self: @This(), hidden_states: Tensor, top_k_index: Tensor, top_k_weights: Tensor) Tensor {
            const seq_len = hidden_states.dim(0);
            const hidden_dim = hidden_states.dim(1);
            const num_experts_per_tok = top_k_index.dim(1);

            const hidden_states_bsd = hidden_states.withTags(.{ .s, .d }).reshape(.{ .b = 1, .s = seq_len, .d = hidden_dim });
            const top_k_index_bst = top_k_index.withTags(.{ .s, .top_expert }).reshape(.{ .b = 1, .s = seq_len, .top_expert = num_experts_per_tok }).convert(.i32);
            const top_k_weights_bst = top_k_weights.withTags(.{ .s, .top_expert }).reshape(.{ .b = 1, .s = seq_len, .top_expert = num_experts_per_tok });

            const experts_out = zml.general_triton_moe.fusedExpertsImpl(
                hidden_states_bsd,
                self.gate_up_proj,
                self.down_proj,
                top_k_weights_bst,
                top_k_index_bst,
                .{},
            ) catch |err| stdx.debug.panic("experts harness failed: {}", .{err});

            return experts_out.reshape(.{ .s = seq_len, .d = hidden_dim }).convert(hidden_states.dtype());
        }
    };

    const LayerOutHarness = struct {
        layer: qwen35.TransformerLayer,
        config: Qwen35.Config,
        pub fn forward(
            self: @This(),
            x: Tensor,
            cos: Tensor,
            sin: Tensor,
            position_ids: Tensor,
            cache_position: Tensor,
        ) Tensor {
            _ = cos;
            _ = sin;
            _ = position_ids;
            const token_index = helpers.cachePositionToTokenIndex(cache_position);
            const tagged_x = x.withTags(.{ .b, .s, .d });
            const kv_cache = helpers.emptyLayerKvCache(self.layer, tagged_x);
            const output, _ = self.layer.forward(tagged_x, token_index, kv_cache.atLayer(0), self.config);
            return output;
        }
    };

    const layers_to_test = .{0};
    for (qwen_model.text_model.layers, qwen35_buffers.text_model.layers, 0..) |layer, layer_buffers, layer_index| {
        const should_test = inline for (layers_to_test) |test_idx| {
            if (layer_index == test_idx) break true;
        } else false;
        if (!should_test) continue;

        var name_buf: [128]u8 = undefined;
        const layer_out_name = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}", .{layer_index});
        const layerOutHarness: LayerOutHarness = .{ .layer = layer, .config = qwen_model.config };
        const layerOutHarnessBuffers: zml.Bufferized(LayerOutHarness) = .{ .layer = layer_buffers };
        try zml.testing.testLayer(
            allocator,
            io,
            platform,
            layerOutHarness,
            .forward,
            activations_store.view(),
            layer_out_name,
            layerOutHarnessBuffers,
            shardings,
            comp_opts,
        );

        const input_layernorm_name = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.input_layernorm", .{layer_index});
        const inputLayerNormHarness: RmsNormBSDHarness = .{ .norm = layer.input_layernorm };
        const inputLayerNormHarnessBuffers: zml.Bufferized(RmsNormBSDHarness) = .{ .norm = layer_buffers.input_layernorm };
        try zml.testing.testLayer(
            allocator,
            io,
            platform,
            inputLayerNormHarness,
            .forward,
            activations_store.view(),
            input_layernorm_name,
            inputLayerNormHarnessBuffers,
            shardings,
            comp_opts,
        );

        switch (layer.attn) {
            .self_attn => {},
            .linear_attn => |linear_attn| {
                const linear_attn_buffers = switch (layer_buffers.attn) {
                    .linear_attn => |buffered_linear_attn| buffered_linear_attn,
                    else => unreachable,
                };

                const in_proj_qkv_name = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.linear_attn.in_proj_qkv", .{layer_index});
                const inProjQkvHarness: LinearBSDHarness = .{ .linear = linear_attn.in_proj_qkv };
                const inProjQkvHarnessBuffers: zml.Bufferized(LinearBSDHarness) = .{ .linear = linear_attn_buffers.in_proj_qkv };
                try zml.testing.testLayer(allocator, io, platform, inProjQkvHarness, .forward, activations_store.view(), in_proj_qkv_name, inProjQkvHarnessBuffers, shardings, comp_opts);

                const in_proj_z_name = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.linear_attn.in_proj_z", .{layer_index});
                const inProjZHarness: LinearBSDHarness = .{ .linear = linear_attn.in_proj_z };
                const inProjZHarnessBuffers: zml.Bufferized(LinearBSDHarness) = .{ .linear = linear_attn_buffers.in_proj_z };
                try zml.testing.testLayer(allocator, io, platform, inProjZHarness, .forward, activations_store.view(), in_proj_z_name, inProjZHarnessBuffers, shardings, comp_opts);

                const in_proj_b_name = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.linear_attn.in_proj_b", .{layer_index});
                const inProjBHarness: LinearBSDHarness = .{ .linear = linear_attn.in_proj_b };
                const inProjBHarnessBuffers: zml.Bufferized(LinearBSDHarness) = .{ .linear = linear_attn_buffers.in_proj_b };
                try zml.testing.testLayer(allocator, io, platform, inProjBHarness, .forward, activations_store.view(), in_proj_b_name, inProjBHarnessBuffers, shardings, comp_opts);

                const in_proj_a_name = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.linear_attn.in_proj_a", .{layer_index});
                const inProjAHarness: LinearBSDHarness = .{ .linear = linear_attn.in_proj_a };
                const inProjAHarnessBuffers: zml.Bufferized(LinearBSDHarness) = .{ .linear = linear_attn_buffers.in_proj_a };
                try zml.testing.testLayer(allocator, io, platform, inProjAHarness, .forward, activations_store.view(), in_proj_a_name, inProjAHarnessBuffers, shardings, comp_opts);

                const out_proj_name = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.linear_attn.out_proj", .{layer_index});
                const outProjHarness: LinearBSDHarness = .{ .linear = linear_attn.out_proj };
                const outProjHarnessBuffers: zml.Bufferized(LinearBSDHarness) = .{ .linear = linear_attn_buffers.out_proj };
                try zml.testing.testLayer(allocator, io, platform, outProjHarness, .forward, activations_store.view(), out_proj_name, outProjHarnessBuffers, shardings, comp_opts);

                const norm_name = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.linear_attn.norm", .{layer_index});
                const normHarness: RmsNormGatedSDHarness = .{ .norm = linear_attn.norm };
                const normHarnessBuffers: zml.Bufferized(RmsNormGatedSDHarness) = .{ .norm = linear_attn_buffers.norm };
                try zml.testing.testLayer(allocator, io, platform, normHarness, .forward, activations_store.view(), norm_name, normHarnessBuffers, shardings, comp_opts);
            },
        }

        const post_attention_layernorm_name = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.post_attention_layernorm", .{layer_index});
        const postAttentionLayerNormHarness: RmsNormBSDHarness = .{ .norm = layer.post_attention_layernorm };
        const postAttentionLayerNormHarnessBuffers: zml.Bufferized(RmsNormBSDHarness) = .{ .norm = layer_buffers.post_attention_layernorm };
        try zml.testing.testLayer(
            allocator,
            io,
            platform,
            postAttentionLayerNormHarness,
            .forward,
            activations_store.view(),
            post_attention_layernorm_name,
            postAttentionLayerNormHarnessBuffers,
            shardings,
            comp_opts,
        );

        const moe_name = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.mlp", .{layer_index});
        const moe = switch (layer.ffn) {
            .sparse => |moe_layer| moe_layer,
            .dense => return error.UnexpectedDenseFfn,
        };
        const moe_buffers = switch (layer_buffers.ffn) {
            .sparse => |buffered_moe| buffered_moe,
            .dense => return error.UnexpectedDenseFfn,
        };
        const moeHarness: MoeHarness = .{ .moe = moe, .config = qwen_model.config };
        const moeHarnessBuffers: zml.Bufferized(MoeHarness) = .{ .moe = moe_buffers };

        // const moeGateName = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.mlp.gate", .{layer_index});
        // const moeGateHarness: MoeGateHarness = .{
        //     .router = moe.router,
        //     .num_experts_per_tok = qwen_model.config.text_config.num_experts_per_tok.?,
        // };
        // const moeGateHarnessBuffers: zml.Bufferized(MoeGateHarness) = .{
        //     .router = moe_buffers.router,
        // };
        // try zml.testing.testLayer(
        //     allocator,
        //     io,
        //     platform,
        //     moeGateHarness,
        //     .forward,
        //     activations_store.view(),
        //     moeGateName,
        //     moeGateHarnessBuffers,
        //     shardings,
        //     comp_opts,
        // );

        const moeExpertsName = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.mlp.experts", .{layer_index});
        const moeExpertsHarness: MoeExpertsHarness = .{
            .gate_up_proj = moe.gate_up_proj,
            .down_proj = moe.down_proj,
        };
        const moeExpertsHarnessBuffers: zml.Bufferized(MoeExpertsHarness) = .{
            .gate_up_proj = moe_buffers.gate_up_proj,
            .down_proj = moe_buffers.down_proj,
        };
        try zml.testing.testLayer(
            allocator,
            io,
            platform,
            moeExpertsHarness,
            .forward,
            activations_store.view(),
            moeExpertsName,
            moeExpertsHarnessBuffers,
            shardings,
            comp_opts,
        );

        const compare_opts: zml.testing.CompareOpts = .{
            .absolute_tolerance = 5e-2,
            .relative_tolerance = 5e-2,
            .minimum_close_fraction = 0.999,
        };

        try zml.testing.testLayer(
            allocator,
            io,
            platform,
            moeHarness,
            .forward,
            activations_store.view(),
            moe_name,
            moeHarnessBuffers,
            shardings,
            compare_opts,
        );

        const shared_expert_name = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.mlp.shared_expert", .{layer_index});
        const sharedExpertHarness: MlpSDHarness = .{ .mlp = moe.shared_expert };
        const sharedExpertHarnessBuffers: zml.Bufferized(MlpSDHarness) = .{ .mlp = moe_buffers.shared_expert };
        try zml.testing.testLayer(
            allocator,
            io,
            platform,
            sharedExpertHarness,
            .forward,
            activations_store.view(),
            shared_expert_name,
            sharedExpertHarnessBuffers,
            shardings,
            comp_opts,
        );

        const shared_expert_gate_name = try std.fmt.bufPrint(&name_buf, "model.model.layers.{d}.mlp.shared_expert_gate", .{layer_index});
        const sharedExpertGateHarness: LinearSDHarness = .{ .linear = moe.shared_expert_gate };
        const sharedExpertGateHarnessBuffers: zml.Bufferized(LinearSDHarness) = .{ .linear = moe_buffers.shared_expert_gate };
        try zml.testing.testLayer(
            allocator,
            io,
            platform,
            sharedExpertGateHarness,
            .forward,
            activations_store.view(),
            shared_expert_gate_name,
            sharedExpertGateHarnessBuffers,
            shardings,
            comp_opts,
        );
    }
}

fn parseConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(qwen35.Qwen35.Config) {
    const parsed_config = blk: {
        const config_json_file = try dir.openFile(io, "config.json", .{});
        defer config_json_file.close(io);
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(io, &config_json_buffer);
        var reader: std.json.Reader = .init(allocator, &config_reader.interface);
        defer reader.deinit();
        break :blk try std.json.parseFromTokenSource(qwen35.Qwen35.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
    };
    errdefer parsed_config.deinit();

    return parsed_config;
}
