const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const common = @import("../common.zig");

const log = std.log.scoped(.glm_moe_dsa);

pub const IndexerType = enum { full, shared };
pub const MlpLayerType = enum { dense, sparse };

pub const RopeParameters = struct {
    rope_theta: f32,
    rope_type: []const u8,
};

pub const Config = struct {
    attention_bias: bool,
    dtype: []const u8,
    first_k_dense_replace: u32,
    hidden_size: i64,
    index_head_dim: i64,
    index_n_heads: i64,
    index_topk: u32,
    indexer_types: []const IndexerType,
    intermediate_size: i64,
    kv_lora_rank: i64,
    max_position_embeddings: i64,
    mlp_layer_types: []const MlpLayerType,
    model_type: []const u8,
    moe_intermediate_size: i64,
    n_group: u32,
    n_routed_experts: u32,
    n_shared_experts: u32,
    norm_topk_prob: bool,
    num_attention_heads: i64,
    num_experts_per_tok: u32,
    num_hidden_layers: u32,
    q_lora_rank: i64,
    qk_nope_head_dim: i64,
    qk_rope_head_dim: i64,
    rms_norm_eps: f32,
    rope_parameters: RopeParameters,
    routed_scaling_factor: f32,
    topk_group: u32,
    v_head_dim: i64,
    vocab_size: i64,
};

pub const InitOptions = struct {
    layer_limit: ?usize = null,
    index_topk_override: ?u32 = null,
};

pub const LoadedModel = struct {
    inner: Model,
    parsed_config: std.json.Parsed(Config),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
        options: InitOptions,
    ) !LoadedModel {
        const parsed_config = try common.parseConfig(Config, allocator, io, repo);
        errdefer parsed_config.deinit();

        return .{
            .inner = try .init(allocator, store, parsed_config.value, options),
            .parsed_config = parsed_config,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        self.parsed_config.deinit();
    }

    pub fn loadBuffers(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: common.Shardings,
    ) !Buffers {
        return self.inner.loadBuffers(allocator, io, platform, store, progress, shardings);
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        _ = self;
        Model.unloadBuffers(buffers, allocator);
    }
};

pub const Buffers = zml.Bufferized(Model);

pub const Model = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []DecoderLayer,
    norm: RmsNorm,
    lm_head: zml.nn.Linear,
    config: Config,
    index_topk: u32,

    pub fn init(
        allocator: std.mem.Allocator,
        root_store: zml.io.TensorStore.View,
        config: Config,
        options: InitOptions,
    ) !Model {
        stdx.debug.assert(std.mem.eql(u8, config.model_type, "glm_moe_dsa"), "Expected glm_moe_dsa config, got {s}", .{config.model_type});
        stdx.debug.assert(config.indexer_types.len == config.num_hidden_layers, "indexer_types length must match num_hidden_layers", .{});
        stdx.debug.assert(config.mlp_layer_types.len == config.num_hidden_layers, "mlp_layer_types length must match num_hidden_layers", .{});
        stdx.debug.assert(config.n_group == 1 and config.topk_group == 1, "GLM-5.2 expects a single expert routing group", .{});
        stdx.debug.assert(config.n_shared_experts == 1, "Only the GLM-5.2 single shared expert layout is supported", .{});

        const layer_count = options.layer_limit orelse @as(usize, @intCast(config.num_hidden_layers));
        stdx.debug.assert(layer_count <= config.num_hidden_layers, "layer_limit {} exceeds num_hidden_layers {}", .{ layer_count, config.num_hidden_layers });
        const index_topk = options.index_topk_override orelse config.index_topk;

        const model_store = root_store.withPrefix("model");
        const layers = try allocator.alloc(DecoderLayer, layer_count);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| {
            errdefer for (layers[0..i]) |previous| previous.deinit(allocator);
            layer.* = try .init(
                allocator,
                model_store.withPrefix("layers").withLayer(i),
                config,
                config.indexer_types[i],
                config.mlp_layer_types[i],
                index_topk,
            );
        }

        return .{
            .embed_tokens = .{ .weight = model_store.withPrefix("embed_tokens").createTensor("weight", .{ .voc, .d }, .{ .voc = .model, .d = .replicated }) },
            .layers = layers,
            .norm = .init(model_store.withPrefix("norm"), config.rms_norm_eps, .d),
            .lm_head = .init(root_store.withPrefix("lm_head").createTensor("weight", .{ .voc, .d }, .{ .voc = .model, .d = .replicated }), null, .d),
            .config = config,
            .index_topk = index_topk,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        for (self.layers) |layer| layer.deinit(allocator);
        allocator.free(self.layers);
    }

    pub fn loadBuffers(
        self: Model,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: common.Shardings,
    ) !Buffers {
        var buffers = try zml.mem.bufferize(allocator, Model, &self);
        errdefer Model.unloadBuffers(&buffers, allocator);

        var loader: zml.io.Loader = try .init(allocator, platform, .{
            .dma_chunks = 32,
            .dma_chunk_size = 256 * zml.MiB,
            .parallelism = 16,
        });
        defer loader.deinit();

        const all_shardings = shardings.all();
        loader.load(io, zml.nn.TokenEmbedding, &self.embed_tokens, &buffers.embed_tokens, store, &all_shardings, .{ .progress = progress });
        loader.load(io, RmsNorm, &self.norm, &buffers.norm, store, &all_shardings, .{ .progress = progress });
        loader.load(io, zml.nn.Linear, &self.lm_head, &buffers.lm_head, store, &all_shardings, .{ .progress = progress });

        for (self.layers, buffers.layers) |*layer, *layer_buffers| {
            loader.load(io, RmsNorm, &layer.input_layernorm, &layer_buffers.input_layernorm, store, &all_shardings, .{ .progress = progress });
            loader.load(io, Attention, &layer.self_attn, &layer_buffers.self_attn, store, &all_shardings, .{ .progress = progress });
            loader.load(io, RmsNorm, &layer.post_attention_layernorm, &layer_buffers.post_attention_layernorm, store, &all_shardings, .{ .progress = progress });
            switch (layer.feed_forward) {
                .dense => |*dense| loader.load(io, DenseMlp, dense, &layer_buffers.feed_forward.dense, store, &all_shardings, .{ .progress = progress }),
                .sparse => |*moe| {
                    loader.load(io, Router, &moe.gate, &layer_buffers.feed_forward.sparse.gate, store, &all_shardings, .{ .progress = progress });
                    loader.load(io, DenseMlp, &moe.shared_experts, &layer_buffers.feed_forward.sparse.shared_experts, store, &all_shardings, .{ .progress = progress });
                },
            }
        }
        try loader.await(io);

        // Hub checkpoints store each expert projection separately. Pack those tensors once at
        // load time into the layout consumed by zml.moe, with the expert axis sharded.
        for (self.layers, buffers.layers) |layer, *layer_buffers| switch (layer.feed_forward) {
            .dense => {},
            .sparse => |moe| {
                try loadPackedExperts(
                    allocator,
                    io,
                    platform,
                    &loader,
                    store,
                    &all_shardings,
                    progress,
                    moe.experts,
                    &layer_buffers.feed_forward.sparse.experts,
                );
            },
        };

        return buffers;
    }

    pub fn unloadBuffers(self: *Buffers, allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| DecoderLayer.unloadBuffers(layer, allocator);
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
        self.lm_head.weight.deinit();
        if (self.lm_head.bias) |*bias| bias.deinit();
    }

    pub fn forward(
        self: Model,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        cache_: Cache,
        moe_metadata: zml.moe.Metadata,
        moe_parameters: zml.moe.Parameters,
    ) struct { zml.Tensor, Cache } {
        var hidden = self.embed_tokens.forward(tokens.withPartialTags(.{ .b, .s }));
        var cache = cache_;
        var previous_topk: ?zml.Tensor = null;

        for (self.layers, 0..) |layer, layer_index| {
            hidden, cache, previous_topk = layer.forward(
                hidden,
                token_index,
                cache,
                zml.Tensor.scalar(@as(u32, @intCast(layer_index)), .u32),
                previous_topk,
                moe_metadata,
                moe_parameters,
            );
        }

        hidden = self.norm.forward(hidden);
        const logits = self.lm_head.forward(hidden).withPartialTags(.{ .b, .s, .voc });
        return .{ logits, cache.reuseBuffer(cache_) };
    }
};

pub const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32,
    axis: zml.Shape.Tag,

    pub fn init(store: zml.io.TensorStore.View, eps: f32, axis: anytype) RmsNorm {
        return .{
            .weight = store.createTensor("weight", .{axis}, .replicated),
            .eps = eps,
            .axis = zml.Shape.toTag(axis),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        return zml.nn.rmsNorm(input, self.axis, self.eps).mul(self.weight.convert(input.dtype()).broad(input.shape()));
    }
};

pub const DenseMlp = struct {
    gate_proj: zml.nn.Linear,
    up_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) DenseMlp {
        return .{
            .gate_proj = linear(store.withPrefix("gate_proj"), .{ .dout, .d }, .{ .dout = .model, .d = .replicated }, .d),
            .up_proj = linear(store.withPrefix("up_proj"), .{ .dout, .d }, .{ .dout = .model, .d = .replicated }, .d),
            .down_proj = linear(store.withPrefix("down_proj"), .{ .dout, .d }, .{ .dout = .replicated, .d = .model }, .d),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(DenseMlp)) void {
        unloadLinear(&self.gate_proj);
        unloadLinear(&self.up_proj);
        unloadLinear(&self.down_proj);
    }

    pub fn forward(self: DenseMlp, input: zml.Tensor) zml.Tensor {
        const hidden = self.gate_proj.forward(input).silu().mul(self.up_proj.forward(input)).rename(.{ .dout = .d });
        return self.down_proj.forward(hidden).rename(.{ .dout = .d });
    }
};

pub const Router = struct {
    weight: zml.Tensor,
    correction_bias: zml.Tensor,
    num_experts_per_tok: u32,
    routed_scaling_factor: f32,
    norm_topk_prob: bool,

    pub fn init(store: zml.io.TensorStore.View, config: Config) Router {
        return .{
            .weight = store.createTensor("weight", .{ .expert, .d }, .{ .expert = .replicated, .d = .replicated }),
            .correction_bias = store.createTensor("e_score_correction_bias", .{.expert}, .replicated),
            .num_experts_per_tok = config.num_experts_per_tok,
            .routed_scaling_factor = config.routed_scaling_factor,
            .norm_topk_prob = config.norm_topk_prob,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Router)) void {
        self.weight.deinit();
        self.correction_bias.deinit();
    }

    pub fn forward(self: Router, input: zml.Tensor) struct { zml.Tensor, zml.Tensor, zml.Tensor } {
        const logits = input.convert(.f32).dot(self.weight.convert(.f32), .d);
        const scores = logits.sigmoid();
        const scores_for_choice = scores.add(self.correction_bias.convert(.f32).broad(scores.shape()));
        const selected = scores_for_choice.topK(.{ .top_expert = .expert }, self.num_experts_per_tok, .{});
        const indices = selected.indices.convert(.i32);
        var weights = scores.gather(.{ .expert = indices }, .{});
        if (self.norm_topk_prob) {
            weights = weights.div(weights.sum(.top_expert).addConstant(1e-20).broad(weights.shape()));
        }
        weights = weights.scale(self.routed_scaling_factor);
        return .{ logits, weights, indices };
    }
};

pub const Experts = struct {
    gate_up_proj: zml.Tensor,
    down_proj: zml.Tensor,
    num_experts: usize,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Experts {
        const expert_count: usize = @intCast(config.n_routed_experts);
        var gate_up_names = try allocator.alloc([]const u8, 2 * expert_count);
        defer {
            for (gate_up_names) |name| allocator.free(name);
            allocator.free(gate_up_names);
        }
        var down_names = try allocator.alloc([]const u8, expert_count);
        defer {
            for (down_names) |name| allocator.free(name);
            allocator.free(down_names);
        }

        for (0..expert_count) |i| {
            gate_up_names[2 * i] = try std.fmt.allocPrint(allocator, "{d}.gate_proj.weight", .{i});
            gate_up_names[2 * i + 1] = try std.fmt.allocPrint(allocator, "{d}.up_proj.weight", .{i});
            down_names[i] = try std.fmt.allocPrint(allocator, "{d}.down_proj.weight", .{i});
        }

        const dtype = store.getShape(gate_up_names[0]).?.dtype();
        const gate_up_shape = zml.Shape.init(.{
            .expert = expert_count,
            .dout = 2 * config.moe_intermediate_size,
            .d = config.hidden_size,
        }, dtype).withPartitioning(.{ .expert = .experts, .dout = .replicated, .d = .replicated });
        const down_shape = zml.Shape.init(.{
            .expert = expert_count,
            .dout = config.hidden_size,
            .d = config.moe_intermediate_size,
        }, dtype).withPartitioning(.{ .expert = .experts, .dout = .replicated, .d = .replicated });

        return .{
            .gate_up_proj = store.maybeCreateBinding(gate_up_names, gate_up_shape) orelse return error.MissingExpertWeights,
            .down_proj = store.maybeCreateBinding(down_names, down_shape) orelse return error.MissingExpertWeights,
            .num_experts = expert_count,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Experts)) void {
        self.gate_up_proj.deinit();
        self.down_proj.deinit();
    }

    pub fn forward(
        self: Experts,
        input: zml.Tensor,
        indices: zml.Tensor,
        weights: zml.Tensor,
        metadata: zml.moe.Metadata,
        parameters: zml.moe.Parameters,
    ) zml.Tensor {
        return zml.moe.forwardMoe(
            input,
            indices,
            weights,
            self.gate_up_proj,
            null,
            null,
            self.down_proj,
            null,
            null,
            metadata,
            parameters,
        ) catch |err| stdx.debug.panic("GLM MoE backend failed: {}", .{err});
    }
};

pub const Moe = struct {
    experts: Experts,
    gate: Router,
    shared_experts: DenseMlp,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Moe {
        return .{
            .experts = try .init(allocator, store.withPrefix("experts"), config),
            .gate = .init(store.withPrefix("gate"), config),
            .shared_experts = .init(store.withPrefix("shared_experts")),
        };
    }

    pub fn deinit(self: Moe, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Moe), allocator: std.mem.Allocator) void {
        _ = allocator;
        Experts.unloadBuffers(&self.experts);
        Router.unloadBuffers(&self.gate);
        DenseMlp.unloadBuffers(&self.shared_experts);
    }

    pub fn forward(
        self: Moe,
        input: zml.Tensor,
        metadata: zml.moe.Metadata,
        parameters: zml.moe.Parameters,
    ) zml.Tensor {
        _, const weights, const indices = self.gate.forward(input);
        const routed = self.experts.forward(input, indices, weights, metadata, parameters);
        return routed.add(self.shared_experts.forward(input)).withPartitioning(.{ .d = .replicated });
    }
};

pub const FeedForward = union(MlpLayerType) {
    dense: DenseMlp,
    sparse: Moe,

    pub fn init(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: Config,
        kind: MlpLayerType,
    ) !FeedForward {
        return switch (kind) {
            .dense => .{ .dense = .init(store) },
            .sparse => .{ .sparse = try .init(allocator, store, config) },
        };
    }

    pub fn deinit(self: FeedForward, allocator: std.mem.Allocator) void {
        switch (self) {
            .dense => {},
            .sparse => |moe| moe.deinit(allocator),
        }
    }

    pub fn unloadBuffers(self: *zml.Bufferized(FeedForward), allocator: std.mem.Allocator) void {
        switch (self.*) {
            .dense => |*dense| DenseMlp.unloadBuffers(dense),
            .sparse => |*moe| Moe.unloadBuffers(moe, allocator),
        }
    }

    pub fn forward(
        self: FeedForward,
        input: zml.Tensor,
        metadata: zml.moe.Metadata,
        parameters: zml.moe.Parameters,
    ) zml.Tensor {
        return switch (self) {
            .dense => |dense| dense.forward(input),
            .sparse => |moe| moe.forward(input, metadata, parameters),
        };
    }
};

pub const Indexer = struct {
    wq_b: zml.nn.Linear,
    wk: zml.nn.Linear,
    k_norm: zml.nn.LayerNorm,
    weights_proj: zml.nn.Linear,
    num_heads: i64,
    head_dim: i64,
    rope_head_dim: i64,
    topk: u32,
    rope_theta: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config, topk: u32) Indexer {
        return .{
            .wq_b = linear(store.withPrefix("wq_b"), .{ .dout, .d }, .{ .dout = .model, .d = .replicated }, .d),
            .wk = linear(store.withPrefix("wk"), .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }, .d),
            .k_norm = .{
                .weight = store.withPrefix("k_norm").createTensor("weight", .{.d}, .replicated),
                .bias = store.withPrefix("k_norm").createTensor("bias", .{.d}, .replicated),
                .eps = 1e-6,
            },
            .weights_proj = linear(store.withPrefix("weights_proj"), .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }, .d),
            .num_heads = config.index_n_heads,
            .head_dim = config.index_head_dim,
            .rope_head_dim = config.qk_rope_head_dim,
            .topk = topk,
            .rope_theta = config.rope_parameters.rope_theta,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Indexer)) void {
        unloadLinear(&self.wq_b);
        unloadLinear(&self.wk);
        self.k_norm.weight.deinit();
        if (self.k_norm.bias) |*bias| bias.deinit();
        unloadLinear(&self.weights_proj);
    }

    pub fn forward(
        self: Indexer,
        hidden: zml.Tensor,
        q_resid: zml.Tensor,
        positions: zml.Tensor,
        cache_: Cache,
        layer_index: zml.Tensor,
    ) struct { zml.Tensor, Cache } {
        var q = self.wq_b.forward(q_resid.rename(.{ .q_lora = .d }))
            .splitAxis(.dout, .{ .ih = self.num_heads, .ihd = self.head_dim });
        var k = self.wk.forward(hidden).rename(.{ .dout = .d });
        k = self.k_norm.forward(k).rename(.{ .d = .ihd });

        const q_rot = q.slice1d(.ihd, .{ .end = self.rope_head_dim }).rename(.{ .ihd = .hd });
        const q_pass = q.slice1d(.ihd, .{ .start = self.rope_head_dim });
        const k_rot = k.slice1d(.ihd, .{ .end = self.rope_head_dim }).rename(.{ .ihd = .hd });
        const k_pass = k.slice1d(.ihd, .{ .start = self.rope_head_dim });
        q = zml.Tensor.concatenate(&.{
            applyInterleavedRope(q_rot, positions, self.rope_theta).rename(.{ .hd = .ihd }),
            q_pass,
        }, .ihd);
        k = zml.Tensor.concatenate(&.{
            applyInterleavedRope(k_rot, positions, self.rope_theta).rename(.{ .hd = .ihd }),
            k_pass,
        }, .ihd);

        const cache = cache_.updateIndexer(k, positions, layer_index);
        const all_k = cache.indexerKeys(layer_index).convert(.f32);
        var scores = q.convert(.f32).dot(all_k, .ihd).scale(1.0 / std.math.sqrt(@as(f32, @floatFromInt(self.head_dim)))).relu();
        const head_weights = self.weights_proj.forward(hidden).rename(.{ .dout = .ih }).convert(.f32)
            .scale(1.0 / std.math.sqrt(@as(f32, @floatFromInt(self.num_heads))));
        scores = scores.mul(head_weights.broad(scores.shape())).sum(.ih).squeeze(.ih);
        scores = applyCausalMask(scores, positions, .k);
        const topk = scores.topK(.{ .topk = .k }, self.topk, .{}).indices.convert(.i32);
        return .{ topk, cache };
    }
};

pub const Attention = struct {
    q_a_proj: zml.nn.Linear,
    q_a_layernorm: RmsNorm,
    q_b_proj: zml.nn.Linear,
    kv_a_proj_with_mqa: zml.nn.Linear,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,
    indexer: ?Indexer,
    num_heads: i64,
    q_lora_rank: i64,
    kv_lora_rank: i64,
    qk_nope_head_dim: i64,
    qk_rope_head_dim: i64,
    v_head_dim: i64,
    scale: f32,
    rope_theta: f32,

    pub fn init(
        store: zml.io.TensorStore.View,
        config: Config,
        indexer_type: IndexerType,
        topk: u32,
    ) Attention {
        return .{
            .q_a_proj = linear(store.withPrefix("q_a_proj"), .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }, .d),
            .q_a_layernorm = .init(store.withPrefix("q_a_layernorm"), 1e-6, .q_lora),
            .q_b_proj = linear(store.withPrefix("q_b_proj"), .{ .dout, .d }, .{ .dout = .model, .d = .replicated }, .d),
            .kv_a_proj_with_mqa = linear(store.withPrefix("kv_a_proj_with_mqa"), .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }, .d),
            .kv_a_layernorm = .init(store.withPrefix("kv_a_layernorm"), 1e-6, .kv_lora),
            .kv_b_proj = linear(store.withPrefix("kv_b_proj"), .{ .dout, .d }, .{ .dout = .model, .d = .replicated }, .d),
            .o_proj = linear(store.withPrefix("o_proj"), .{ .dout, .d }, .{ .dout = .replicated, .d = .model }, .d),
            .indexer = if (indexer_type == .full) .init(store.withPrefix("indexer"), config, topk) else null,
            .num_heads = config.num_attention_heads,
            .q_lora_rank = config.q_lora_rank,
            .kv_lora_rank = config.kv_lora_rank,
            .qk_nope_head_dim = config.qk_nope_head_dim,
            .qk_rope_head_dim = config.qk_rope_head_dim,
            .v_head_dim = config.v_head_dim,
            .scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(config.qk_nope_head_dim + config.qk_rope_head_dim))),
            .rope_theta = config.rope_parameters.rope_theta,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Attention)) void {
        unloadLinear(&self.q_a_proj);
        RmsNorm.unloadBuffers(&self.q_a_layernorm);
        unloadLinear(&self.q_b_proj);
        unloadLinear(&self.kv_a_proj_with_mqa);
        RmsNorm.unloadBuffers(&self.kv_a_layernorm);
        unloadLinear(&self.kv_b_proj);
        unloadLinear(&self.o_proj);
        if (self.indexer) |*indexer| Indexer.unloadBuffers(indexer);
    }

    pub fn forward(
        self: Attention,
        hidden: zml.Tensor,
        token_index: zml.Tensor,
        cache_: Cache,
        layer_index: zml.Tensor,
        previous_topk: ?zml.Tensor,
    ) struct { zml.Tensor, Cache, zml.Tensor } {
        const positions = zml.Tensor.arange(.{ .end = hidden.dim(.s) }, token_index.dtype()).withTags(.{.s})
            .add(token_index.broad(zml.Shape.init(.{ .s = hidden.dim(.s) }, token_index.dtype())));

        const q_a = self.q_a_proj.forward(hidden).rename(.{ .dout = .q_lora });
        const q_resid = self.q_a_layernorm.forward(q_a);
        var q = self.q_b_proj.forward(q_resid.rename(.{ .q_lora = .d }))
            .splitAxis(.dout, .{ .h = self.num_heads, .hd = self.qk_nope_head_dim + self.qk_rope_head_dim });
        const q_pass = q.slice1d(.hd, .{ .end = self.qk_nope_head_dim });
        const q_rot = q.slice1d(.hd, .{ .start = self.qk_nope_head_dim });
        q = zml.Tensor.concatenate(&.{ q_pass, applyInterleavedRope(q_rot, positions, self.rope_theta) }, .hd)
            .withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });

        const compressed_kv = self.kv_a_proj_with_mqa.forward(hidden);
        var kv_pass = compressed_kv.slice1d(.dout, .{ .end = self.kv_lora_rank }).rename(.{ .dout = .kv_lora });
        var k_rot = compressed_kv.slice1d(.dout, .{ .start = self.kv_lora_rank }).rename(.{ .dout = .hd });
        kv_pass = self.kv_a_layernorm.forward(kv_pass);
        k_rot = applyInterleavedRope(k_rot, positions, self.rope_theta);

        var expanded = self.kv_b_proj.forward(kv_pass.rename(.{ .kv_lora = .d }))
            .splitAxis(.dout, .{ .h = self.num_heads, .mixed = self.qk_nope_head_dim + self.v_head_dim });
        const k_pass = expanded.slice1d(.mixed, .{ .end = self.qk_nope_head_dim }).rename(.{ .mixed = .hd });
        var value = expanded.slice1d(.mixed, .{ .start = self.qk_nope_head_dim }).rename(.{ .mixed = .hd });
        const repeated_k_rot = k_rot.insertAxes(.hd, .{.h}).broad(k_pass.shape().setDim(.hd, self.qk_rope_head_dim));
        var key = zml.Tensor.concatenate(&.{ k_pass, repeated_k_rot }, .hd);
        key = key.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        value = value.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });

        var cache = cache_.updateKv(key, value, positions, layer_index);
        const topk = if (self.indexer) |indexer| blk: {
            const result = indexer.forward(hidden, q_resid, positions, cache, layer_index);
            cache = result[1];
            break :blk result[0];
        } else previous_topk orelse stdx.debug.panic("Shared DSA layer requires previous top-k indices", .{});

        const all_key = cache.keys(layer_index);
        const all_value = cache.values(layer_index);
        const selected_key = all_key.gather(.{ .k = topk }, .{});
        const selected_value = all_value.gather(.{ .k = topk }, .{});
        var scores = q.convert(.f32).dot(selected_key.convert(.f32), .hd).scale(self.scale);
        const valid = topk.cmp(.LE, positions.convert(.i32).broad(topk.shape()));
        scores = zml.Tensor.select(valid.insertAxes(.topk, .{.h}).broad(scores.shape()), scores, zml.Tensor.constant(scores.dtype().minValue()).broad(scores.shape()));
        const weights = scores.softmax(.topk).convert(value.dtype());
        const attention = weights.dot(selected_value, .topk)
            .merge(.{ .d = .{ .h, .hd } })
            .withPartitioning(.{ .d = .model });
        const output = self.o_proj.forward(attention).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
        return .{ output, cache, topk };
    }
};

pub const DecoderLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    feed_forward: FeedForward,

    pub fn init(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: Config,
        indexer_type: IndexerType,
        mlp_type: MlpLayerType,
        index_topk: u32,
    ) !DecoderLayer {
        return .{
            .input_layernorm = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps, .d),
            .self_attn = .init(store.withPrefix("self_attn"), config, indexer_type, index_topk),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps, .d),
            .feed_forward = try .init(allocator, store.withPrefix("mlp"), config, mlp_type),
        };
    }

    pub fn deinit(self: DecoderLayer, allocator: std.mem.Allocator) void {
        self.feed_forward.deinit(allocator);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(DecoderLayer), allocator: std.mem.Allocator) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        Attention.unloadBuffers(&self.self_attn);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
        FeedForward.unloadBuffers(&self.feed_forward, allocator);
    }

    pub fn forward(
        self: DecoderLayer,
        hidden_: zml.Tensor,
        token_index: zml.Tensor,
        cache: Cache,
        layer_index: zml.Tensor,
        previous_topk: ?zml.Tensor,
        moe_metadata: zml.moe.Metadata,
        moe_parameters: zml.moe.Parameters,
    ) struct { zml.Tensor, Cache, ?zml.Tensor } {
        const hidden = hidden_.withPartitioning(.{ .d = .replicated });
        const attention, const updated_cache, const topk = self.self_attn.forward(
            self.input_layernorm.forward(hidden),
            token_index,
            cache,
            layer_index,
            previous_topk,
        );
        const after_attention = hidden.add(attention).withPartitioning(.{ .d = .replicated });
        const mlp = self.feed_forward.forward(
            self.post_attention_layernorm.forward(after_attention),
            moe_metadata,
            moe_parameters,
        );
        return .{ after_attention.add(mlp).withPartitioning(.{ .d = .replicated }), updated_cache, topk };
    }
};

pub const Cache = struct {
    k: zml.Tensor,
    v: zml.Tensor,
    indexer_k: zml.Tensor,

    pub fn init(
        layer_count: usize,
        batch_size: i64,
        max_seq_len: i64,
        config: Config,
        dtype: zml.DataType,
    ) Cache {
        const kv_shape = zml.Shape.init(.{
            .layer = layer_count,
            .b = batch_size,
            .h = config.num_attention_heads,
            .k = max_seq_len,
            .hd = config.qk_nope_head_dim + config.qk_rope_head_dim,
        }, dtype).withPartitioning(.{ .h = .model });
        const indexer_shape = zml.Shape.init(.{
            .layer = layer_count,
            .b = batch_size,
            .k = max_seq_len,
            .ihd = config.index_head_dim,
        }, dtype).withReplicatedPartitioning();
        return .{
            .k = .fromShape(kv_shape),
            .v = .fromShape(kv_shape.setDim(.hd, config.v_head_dim)),
            .indexer_k = .fromShape(indexer_shape),
        };
    }

    pub fn initBuffers(self: Cache, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(Cache) {
        return .{
            .k = try .uninitialized(io, platform, self.k.shape(), sharding, .{}),
            .v = try .uninitialized(io, platform, self.v.shape(), sharding, .{}),
            .indexer_k = try .uninitialized(io, platform, self.indexer_k.shape(), sharding, .{}),
        };
    }

    pub fn deinitBuffers(self: *zml.Bufferized(Cache)) void {
        self.k.deinit();
        self.v.deinit();
        self.indexer_k.deinit();
    }

    pub fn reuseBuffer(self: Cache, other: Cache) Cache {
        return .{
            .k = self.k.reuseBuffer(other.k),
            .v = self.v.reuseBuffer(other.v),
            .indexer_k = self.indexer_k.reuseBuffer(other.indexer_k),
        };
    }

    pub fn keys(self: Cache, layer_index: zml.Tensor) zml.Tensor {
        return self.k.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn values(self: Cache, layer_index: zml.Tensor) zml.Tensor {
        return self.v.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn indexerKeys(self: Cache, layer_index: zml.Tensor) zml.Tensor {
        return self.indexer_k.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn updateKv(
        self: Cache,
        new_k: zml.Tensor,
        new_v: zml.Tensor,
        positions: zml.Tensor,
        layer_index: zml.Tensor,
    ) Cache {
        const layer = layer_index.broad(positions.shape());
        return .{
            .k = self.k.scatterSlices(.{ .layer = layer, .k = positions }, new_k, .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.k),
            .v = self.v.scatterSlices(.{ .layer = layer, .k = positions }, new_v, .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.v),
            .indexer_k = self.indexer_k,
        };
    }

    pub fn updateIndexer(
        self: Cache,
        new_k: zml.Tensor,
        positions: zml.Tensor,
        layer_index: zml.Tensor,
    ) Cache {
        const layer = layer_index.broad(positions.shape());
        return .{
            .k = self.k,
            .v = self.v,
            .indexer_k = self.indexer_k.scatterSlices(.{ .layer = layer, .k = positions }, new_k, .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.indexer_k),
        };
    }
};

fn linear(store: zml.io.TensorStore.View, tagz: anytype, partitioning: anytype, contract: anytype) zml.nn.Linear {
    return .init(store.createTensor("weight", tagz, partitioning), store.maybeCreateTensor("bias", .{.dout}, partitioning), contract);
}

fn unloadLinear(value: *zml.Bufferized(zml.nn.Linear)) void {
    value.weight.deinit();
    if (value.bias) |*bias| bias.deinit();
}

fn applyInterleavedRope(input: zml.Tensor, positions: zml.Tensor, theta: f32) zml.Tensor {
    const rotated = zml.nn.rope(input, positions, .{
        .layout = .interleaved,
        .scaling = .{ .default = .{ .rope_theta = theta } },
    });
    // Transformers consumes interleaved pairs but returns the rotated real half followed by
    // the rotated imaginary half.
    return zml.Tensor.concatenate(&.{
        rotated.slice1d(.hd, .{ .start = 0, .step = 2 }),
        rotated.slice1d(.hd, .{ .start = 1, .step = 2 }),
    }, .hd);
}

fn applyCausalMask(scores: zml.Tensor, positions: zml.Tensor, key_tag: anytype) zml.Tensor {
    const key_positions = zml.Tensor.arange(.{ .end = scores.dim(key_tag) }, positions.dtype()).withTags(.{key_tag});
    const valid = key_positions.broad(scores.shape()).cmp(.LE, positions.broad(scores.shape()));
    return zml.Tensor.select(valid, scores, zml.Tensor.constant(scores.dtype().minValue()).broad(scores.shape()));
}

fn sourceInputs(allocator: std.mem.Allocator, store: *const zml.io.TensorStore, tensor: zml.Tensor) ![]zml.Tensor {
    const sources = store.getSourcesById(tensor.id) orelse return error.MissingExpertWeights;
    const inputs = try allocator.alloc(zml.Tensor, sources.len);
    for (sources, inputs) |source, *input| input.* = .fromShape(source.shape.withReplicatedPartitioning());
    return inputs;
}

fn packGateUp(inputs: []const zml.Tensor) zml.Tensor {
    const allocator = zml.module.CompilationContext.current().allocator;
    const expert_count = @divExact(inputs.len, 2);
    const experts = allocator.alloc(zml.Tensor, expert_count) catch @panic("OOM");
    defer allocator.free(experts);
    for (experts, 0..) |*expert, i| {
        expert.* = zml.Tensor.concatenate(&.{ inputs[2 * i], inputs[2 * i + 1] }, 0).insertAxes(0, .{.expert});
    }
    return zml.Tensor.concatenate(experts, .expert)
        .withTags(.{ .expert, .dout, .d })
        .withPartitioning(.{ .expert = .experts, .dout = .replicated, .d = .replicated });
}

fn packDown(inputs: []const zml.Tensor) zml.Tensor {
    const allocator = zml.module.CompilationContext.current().allocator;
    const experts = allocator.alloc(zml.Tensor, inputs.len) catch @panic("OOM");
    defer allocator.free(experts);
    for (inputs, experts) |input, *expert| expert.* = input.insertAxes(0, .{.expert});
    return zml.Tensor.concatenate(experts, .expert)
        .withTags(.{ .expert, .dout, .d })
        .withPartitioning(.{ .expert = .experts, .dout = .replicated, .d = .replicated });
}

fn loadPackedExperts(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    loader: *zml.io.Loader,
    store: *const zml.io.TensorStore,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
    experts: Experts,
    buffers: *zml.Bufferized(Experts),
) !void {
    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const gate_up_inputs = try sourceInputs(allocator, store, experts.gate_up_proj);
    defer allocator.free(gate_up_inputs);
    const gate_up_exe = try platform.compileFn(allocator, io, packGateUp, .{gate_up_inputs}, .{ .shardings = shardings });
    defer gate_up_exe.deinit();
    try loader.loadExecute(arena, io, experts.gate_up_proj, &buffers.gate_up_proj, store, shardings, &gate_up_exe, .{ .progress = progress });

    const down_inputs = try sourceInputs(allocator, store, experts.down_proj);
    defer allocator.free(down_inputs);
    const down_exe = try platform.compileFn(allocator, io, packDown, .{down_inputs}, .{ .shardings = shardings });
    defer down_exe.deinit();
    try loader.loadExecute(arena, io, experts.down_proj, &buffers.down_proj, store, shardings, &down_exe, .{ .progress = progress });
}
