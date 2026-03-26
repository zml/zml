const std = @import("std");
const testing = std.testing;

const zml = @import("zml");
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;
const stdx = zml.stdx;

const log = std.log.scoped(.qwen3_5);

pub const Qwen35 = struct {
    pub const Config = struct {
        model_type: []const u8,
        text_config: TextConfig,

        pub const ModelKind = enum {
            qwen3_5,
            qwen3_5_moe,
        };

        pub fn kind(self: Config) !ModelKind {
            if (std.mem.eql(u8, self.model_type, "qwen3_5")) return .qwen3_5;
            if (std.mem.eql(u8, self.model_type, "qwen3_5_moe")) return .qwen3_5_moe;
            return error.UnsupportedModelType;
        }
    };

    pub const TextConfig = struct {
        // General
        num_hidden_layers: i64,
        layer_types: []const LayerType,
        hidden_size: i64,
        max_position_embeddings: i64,
        rms_norm_eps: f32,
        // Self attention
        head_dim: i64,
        num_attention_heads: i64,
        num_key_value_heads: i64,
        rope_parameters: RopeParameters,
        // Linear attention
        linear_conv_kernel_dim: i64,
        linear_key_head_dim: i64,
        linear_num_key_heads: i64,
        linear_num_value_heads: i64,
        linear_value_head_dim: i64,
        // MoE
        num_experts: ?i64 = null,
        num_experts_per_tok: ?u32 = null,
    };

    // Each layer uses either: full attention (SelfAttn) or linear attention (GatedDeltaNet).
    pub const LayerType = enum {
        linear_attention,
        full_attention,
    };

    pub const RopeParameters = struct {
        mrope_section: [3]i64,
        partial_rotary_factor: f32,
        rope_theta: f32,
    };

    pub const GenOptions = struct { sampling_strategy: zml.nn.SamplingStrategy = .{}, max_seq_len: i64 };

    pub const SpecialTokens = struct {
        im_start_token_id: u32,
        im_end_token_id: u32,
        end_of_text_token_id: u32,
    };

    text_model: TextModel,

    config: Config,
    special_tokens: SpecialTokens = .{
        .im_start_token_id = 248045,
        .im_end_token_id = 248046,
        .end_of_text_token_id = 248044,
    },

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, gen_options: GenOptions) !Qwen35 {
        // For some Qwen3.5 versions, the output projection lm_head has a standalone weight tensor, while for others it's the same as the input embedding layer
        return .{
            .text_model = try .init(allocator, store.withPrefix("model.language_model"), config, gen_options),

            .config = config,
        };
    }

    pub fn deinit(self: Qwen35, allocator: std.mem.Allocator) void {
        self.text_model.deinit(allocator);
    }

    pub fn load(
        self: *const Qwen35,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(Qwen35) {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);
        var total_bytes: usize = 0;
        defer {
            const took = now.untilNow(io, .awake);
            const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
            log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });
        }
        return zml.io.load(Qwen35, self, allocator, io, platform, store, .{
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = progress,
            .shardings = shardings,
            .parallelism = 16,
            .total_bytes = &total_bytes,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Qwen35), allocator: std.mem.Allocator) void {
        TextModel.unloadBuffers(&self.text_model, allocator);
    }

    pub fn forward(
        self: Qwen35,
        tokens_: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        rng: Tensor.Rng,
    ) struct { Tensor, KvCache, Tensor.Rng } {
        const tokens = tokens_.withPartialTags(.{.s});
        const new_tokens, const updated_kv_cache, const new_rng = self.text_model.forward(tokens, token_index, kv_cache, self.config, rng);
        return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), updated_kv_cache, new_rng };
    }
};

//========================Text model========================

pub const TextModel = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    lm_head: zml.nn.Linear,
    norm: RmsNorm,
    gen_options: Qwen35.GenOptions,

    pub fn sampleTokens(
        self: TextModel,
        out: Tensor,
        rng: Tensor.Rng,
    ) struct { Tensor, Tensor.Rng } {
        const x = self.norm.forward(out);
        const logits = self.lm_head.forward(x.withPartialTags(.{.d})).rename(.{ .dout = .voc });
        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, self.gen_options.sampling_strategy, rng);
        return .{ next_tokens.convert(.u32), new_rng };
    }

    pub fn init(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: Qwen35.Config,
        gen_options: Qwen35.GenOptions,
    ) !TextModel {
        const lm_head_prefix = if (store.root().hasKey("lm_head.weight")) "lm_head" else "model.language_model.embed_tokens";

        const layers = try allocator.alloc(TransformerLayer, @intCast(config.text_config.num_hidden_layers));
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = try .init(store.withPrefix("layers").withLayer(i), config, i);
        }

        return .{
            .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, null) },
            .layers = layers,
            .lm_head = .init(store.root().withPrefix(lm_head_prefix).createTensor("weight", .{ .dout, .d }, null), null, .d),
            .norm = RmsNorm.init(store.withPrefix("norm"), config.text_config.rms_norm_eps),
            .gen_options = gen_options,
        };
    }

    pub fn deinit(self: TextModel, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TextModel), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }

        allocator.free(self.layers);
        self.lm_head.weight.deinit();
        RmsNorm.unloadBuffers(&self.norm);
    }

    pub fn forward(
        self: TextModel,
        tokens: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        config: Qwen35.Config,
        rng: Tensor.Rng,
    ) struct { Tensor, KvCache, Tensor.Rng } {
        var hidden_states = self.embed_tokens.forward(tokens);

        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden_states, updated_kv_cache = layer.forward(hidden_states, token_index, updated_kv_cache.atLayer(i), config);
        }

        const new_tokens, const new_rng = self.sampleTokens(hidden_states, rng);
        return .{ new_tokens, updated_kv_cache.reuseBuffer(kv_cache), new_rng };
    }
};

pub const TransformerLayer = struct {
    const Attn = union(enum) {
        self_attn: SelfAttn,
        linear_attn: GatedDeltaNet,
    };

    const Ffn = union(enum) {
        dense: Mlp,
        sparse: Moe,
    };

    input_layernorm: RmsNorm,
    attn: Attn,
    ffn: Ffn,
    post_attention_layernorm: RmsNorm,

    pub fn init(store: zml.io.TensorStore.View, config: Qwen35.Config, layer_index: usize) !TransformerLayer {
        const is_full_attention = config.text_config.layer_types[layer_index] == .full_attention;
        return .{
            .input_layernorm = RmsNorm.init(store.withPrefix("input_layernorm"), config.text_config.rms_norm_eps),
            .attn = if (is_full_attention)
                .{ .self_attn = try .init(store.withPrefix("self_attn"), config) }
            else
                .{ .linear_attn = .init(store.withPrefix("linear_attn"), config) },
            .ffn = switch (try config.kind()) {
                .qwen3_5 => .{ .dense = Mlp.init(store.withPrefix("mlp")) },
                .qwen3_5_moe => .{ .sparse = try Moe.init(store.withPrefix("mlp"), config) },
            },
            .post_attention_layernorm = RmsNorm.init(store.withPrefix("post_attention_layernorm"), config.text_config.rms_norm_eps),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        switch (self.attn) {
            .self_attn => |*self_attn| SelfAttn.unloadBuffers(self_attn),
            .linear_attn => |*linear_attn| GatedDeltaNet.unloadBuffers(linear_attn),
        }
        switch (self.ffn) {
            .dense => |*dense| Mlp.unloadBuffers(dense),
            .sparse => |*sparse| Moe.unloadBuffers(sparse),
        }
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
    }

    pub fn forwardSelfAttn(
        self: TransformerLayer,
        x0: Tensor,
        token_index: Tensor,
        kv_cache: KvCache.SelfAttnCache,
        config: Qwen35.Config,
    ) struct { Tensor, KvCache.SelfAttnCache } {
        const residual0 = x0;
        const normalized_x0 = self.input_layernorm.forward(x0);

        const self_attn = switch (self.attn) {
            .self_attn => |self_attn| self_attn,
            .linear_attn => unreachable,
        };
        const attention_output, const updated_kv_cache = self_attn.forward(normalized_x0, token_index, kv_cache);

        const x1 = attention_output.add(residual0);
        const residual1 = x1;
        const normalized_hidden = self.post_attention_layernorm.forward(x1);

        const mlp_output = switch (self.ffn) {
            .dense => |*dense| dense.forward(normalized_hidden),
            .sparse => |*sparse| sparse.forward(normalized_hidden, config),
        };

        return .{ mlp_output.add(residual1), updated_kv_cache };
    }

    pub fn forwardLinearAttn(
        self: TransformerLayer,
        x0: Tensor,
        token_index: Tensor,
        kv_cache: KvCache.GatedDeltaNetCache,
        config: Qwen35.Config,
    ) struct { Tensor, KvCache.GatedDeltaNetCache } {
        _ = token_index;

        const residual0 = x0;
        const normalized_x0 = self.input_layernorm.forward(x0);

        const linear_attn = switch (self.attn) {
            .linear_attn => |linear_attn| linear_attn,
            .self_attn => unreachable,
        };
        const attention_output, const updated_kv_cache = linear_attn.forward(normalized_x0, kv_cache);

        const x1 = attention_output.add(residual0);
        const residual1 = x1;
        const normalized_hidden = self.post_attention_layernorm.forward(x1);

        const mlp_output = switch (self.ffn) {
            .dense => |*dense| dense.forward(normalized_hidden),
            .sparse => |*sparse| sparse.forward(normalized_hidden, config),
        };

        return .{ mlp_output.add(residual1), updated_kv_cache };
    }

    pub fn forward(
        self: TransformerLayer,
        x0: Tensor,
        token_index: Tensor,
        kv_cache: KvCache.LayerView,
        config: Qwen35.Config,
    ) struct { Tensor, KvCache } {
        const residual0 = x0;
        const normalized_x0 = self.input_layernorm.forward(x0);

        var attention_output: Tensor = undefined;
        var updated_kv_cache: KvCache = kv_cache.parent;
        switch (self.attn) {
            .self_attn => |*self_attn| {
                const result = self_attn.forward(normalized_x0, token_index, kv_cache.cache.self_attn);
                attention_output = result[0];
                updated_kv_cache.self_attn = result[1];
            },
            .linear_attn => |*linear_attn| {
                const result = linear_attn.forward(normalized_x0, kv_cache.cache.linear_attn);
                attention_output = result[0];
                updated_kv_cache.gated_delta_net = result[1];
            },
        }

        const x1 = attention_output.add(residual0);
        const residual1 = x1;
        const normalized_hidden = self.post_attention_layernorm.forward(x1);

        const mlp_output = switch (self.ffn) {
            .dense => |*dense| dense.forward(normalized_hidden),
            .sparse => |*sparse| sparse.forward(normalized_hidden, config),
        };

        return .{ mlp_output.add(residual1), updated_kv_cache };
    }
};

pub const Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .up_proj = .init(store.withPrefix("up_proj").createTensor("weight", .{ .dout, .d }, null), store.withPrefix("up_proj").maybeCreateTensor("bias", .{.dout}, null), .d),
            .gate_proj = .init(store.withPrefix("gate_proj").createTensor("weight", .{ .dout, .d }, null), store.withPrefix("gate_proj").maybeCreateTensor("bias", .{.dout}, null), .d),
            .down_proj = .init(store.withPrefix("down_proj").createTensor("weight", .{ .d, .dout }, null), store.withPrefix("down_proj").maybeCreateTensor("bias", .{.d}, null), .dout),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        self.up_proj.weight.deinit();
        if (self.up_proj.bias) |*bias| bias.deinit();
        self.gate_proj.weight.deinit();
        if (self.gate_proj.bias) |*bias| bias.deinit();
        self.down_proj.weight.deinit();
        if (self.down_proj.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: Mlp, x: Tensor) Tensor {
        const up_projed = self.up_proj.forward(x);
        const gate = self.gate_proj.forward(x);
        const hidden = gate.silu().mul(up_projed);

        const output = self.down_proj.forward(hidden);
        return output;
    }
};

const Router = struct {
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

pub const Moe = struct {
    shared_expert: Mlp,
    shared_expert_gate: zml.nn.Linear,
    gate_up_proj: Tensor,
    down_proj: Tensor,
    router: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View, config: Qwen35.Config) !Moe {
        _ = config; // autofix
        return .{
            .shared_expert = Mlp.init(store.withPrefix("shared_expert")),
            .shared_expert_gate = .init(store.withPrefix("shared_expert_gate").createTensor("weight", .{ .dout, .d }, null), store.withPrefix("shared_expert_gate").maybeCreateTensor("bias", .{.out}, null), .d),
            .gate_up_proj = store.withPrefix("experts").createTensor("gate_up_proj", .{ .expert, .dout, .d }, null),
            .down_proj = store.withPrefix("experts").createTensor("down_proj", .{ .expert, .d, .dout }, null),
            .router = .init(store.withPrefix("gate").createTensor("weight", .{ .expert, .d }, null), null, .d),
        };
    }

    pub fn forward(self: Moe, x: Tensor, config: Qwen35.Config) Tensor {
        const moe_metadata: zml.moe.Metadata = .init(.fromBackend(.triton));
        const moe_parameters: zml.moe.Parameters = .init(.fromBackend(.triton));
        const num_experts_per_tok: u32 = config.text_config.num_experts_per_tok.?;

        // const down_weight = self.down_proj.weight.rename(.{ .d = .out, .dout = .d });
        // const down_bias = if (self.down_proj.bias) |b| b.rename(.{ .d = .out }) else null;

        const router_logits = self.router.forward(x).convert(.f32);
        // Match HF: softmax over all experts, then top-k and renormalize.
        const router_probs = router_logits.softmax(.expert);
        const routing = router_probs.topK(.{ .top_expert = .expert }, num_experts_per_tok, .{});
        const topk_ids = routing.indices.convert(.i32);
        var topk_weights = routing.values;
        topk_weights = topk_weights.div(topk_weights.sum(.top_expert).broad(topk_weights.shape())).convert(router_logits.dtype());

        const routed = zml.moe.moe(
            x,
            null,
            topk_ids,
            topk_weights,
            self.gate_up_proj,
            null,
            null,
            self.down_proj,
            null,
            null,
            num_experts_per_tok,
            moe_metadata,
            moe_parameters,
        ) catch |err| stdx.debug.panic("moe backend failed: {}", .{err});

        const shared_gate = self.shared_expert_gate.forward(x).sigmoid().broad(x.shape());
        const shared = self.shared_expert.forward(x).mul(shared_gate);

        return routed.convert(shared.dtype()).reshape(shared.shape()).add(shared);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Moe)) void {
        Mlp.unloadBuffers(&self.shared_expert);
        self.shared_expert_gate.weight.deinit();
        if (self.shared_expert_gate.bias) |*bias| bias.deinit();
        self.gate_up_proj.deinit();
        self.down_proj.deinit();
        self.router.weight.deinit();
        if (self.router.bias) |*bias| bias.deinit();
    }
};

pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,

    q_norm: RmsNorm,
    k_norm: RmsNorm,

    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    rotary_dim: i64,
    rotary_embed: TextRotaryEmbedding,
    o_proj: zml.nn.Linear,

    fn initProj(store: zml.io.TensorStore.View) zml.nn.Linear {
        return .init(store.createTensor("weight", .{ .dout, .d }, null), store.maybeCreateTensor("bias", .{.dout}, null), .d);
    }

    pub fn init(store: zml.io.TensorStore.View, config: Qwen35.Config) !SelfAttn {
        const rotary_dim: i64 = @intFromFloat(@as(f32, @floatFromInt(config.text_config.head_dim)) *
            config.text_config.rope_parameters.partial_rotary_factor);
        return .{
            .q_proj = initProj(store.withPrefix("q_proj")),
            .k_proj = initProj(store.withPrefix("k_proj")),
            .v_proj = initProj(store.withPrefix("v_proj")),
            .o_proj = initProj(store.withPrefix("o_proj")),
            .q_norm = RmsNorm.init(store.withPrefix("q_norm"), config.text_config.rms_norm_eps),
            .k_norm = RmsNorm.init(store.withPrefix("k_norm"), config.text_config.rms_norm_eps),
            .num_heads = config.text_config.num_attention_heads,
            .num_kv_heads = config.text_config.num_key_value_heads,
            .head_dim = config.text_config.head_dim,
            .rotary_dim = rotary_dim,
            .rotary_embed = .init(rotary_dim, config.text_config.rope_parameters.rope_theta, config.text_config.rope_parameters.mrope_section),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SelfAttn)) void {
        self.q_proj.weight.deinit();
        if (self.q_proj.bias) |*bias| bias.deinit();
        self.k_proj.weight.deinit();
        if (self.k_proj.bias) |*bias| bias.deinit();
        self.v_proj.weight.deinit();
        if (self.v_proj.bias) |*bias| bias.deinit();
        self.o_proj.weight.deinit();
        if (self.o_proj.bias) |*bias| bias.deinit();
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    fn projectQAndGate(self: SelfAttn, x: Tensor) struct { Tensor, Tensor } {
        const q_proj = self.q_proj.forward(x).splitAxis(.dout, .{ .h = self.num_heads, .hd = 2 * self.head_dim });
        const q, var gate = q_proj.chunkExact(.hd, 2);
        gate = gate.merge(.{ .d_out_proj = .{ .h, .hd } });
        return .{ q, gate };
    }

    fn projectKV(self: SelfAttn, x: Tensor) struct { Tensor, Tensor } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;
        const k = self.k_proj.forward(x).splitAxis(.dout, .{ .h = num_kv_heads, .hd = self.head_dim });
        const v = self.v_proj.forward(x).splitAxis(.dout, .{ .h = num_kv_heads, .hd = self.head_dim });
        return .{ k, v };
    }

    pub fn forward(
        self: SelfAttn,
        x: Tensor,
        token_index: Tensor,
        kv_cache: KvCache.SelfAttnCache,
    ) struct { Tensor, KvCache.SelfAttnCache } {
        var q, const gate = self.projectQAndGate(x);
        var k, var v = self.projectKV(x);
        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        const dtype = q.dtype();
        const position_ids = Tensor.arange(.{ .end = x.dim(.s) }, .i64)
            .withTags(.{.s}).insertAxes(.s, .{.b}).broad(zml.Shape.init(.{ .b = x.dim(.b), .s = x.dim(.s) }, .i64))
            .add(token_index.convert(.i64).broad(zml.Shape.init(.{ .b = x.dim(.b), .s = x.dim(.s) }, .i64)));

        const cos, const sin = self.rotary_embed.getCosAndSin(position_ids, dtype);
        q = self.rotary_embed.applyRope(q, cos, sin);
        k = self.rotary_embed.applyRope(k, cos, sin);

        const new_kv_cache = kv_cache.update(k, v, token_index.convert(.u32));
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);

        const attn_output = zml.attention.attention.attention(
            q.rename(.{ .s = .q }),
            k.rename(.{ .s = .k }),
            v.rename(.{ .s = .k }),
            token_index,
            zml.attention.attention.Metadata.init(.fromBackend(.vanilla, x.dim(.s), self.num_heads)),
            zml.attention.attention.Parameters.init(.fromBackend(.vanilla)),
        ).rename(.{ .q = .s }).merge(.{ .d_out_proj = .{ .h, .hd } });

        const gated_output = attn_output.mul(gate.sigmoid());
        const projected_output = self.o_proj.forward(gated_output.rename(.{ .d_out_proj = .d })).rename(.{ .dout = .d });

        return .{ projected_output, new_kv_cache };
    }
};

pub const TextRotaryEmbedding = struct {
    rope_opts: zml.nn.RopeOpts,
    rotary_dim: i64,
    mrope_section: [3]i64,

    pub fn init(rotary_dim: i64, theta: f32, mrope_section: [3]i64) TextRotaryEmbedding {
        return .{
            .rope_opts = .{
                .layout = .sequential,
                .scaling = .{ .default = .{ .rope_theta = theta } },
            },
            .rotary_dim = rotary_dim,
            .mrope_section = mrope_section,
        };
    }

    pub fn getCosAndSin(self: TextRotaryEmbedding, position_ids: Tensor, dtype: zml.DataType) struct { Tensor, Tensor } {
        const inv_freq = zml.nn.invFreq(self.rotary_dim, self.rope_opts).withTags(.{.hd});

        const freqs_t = position_ids.convert(.f32).outer(inv_freq);

        const emb = Tensor.concatenate(&.{ freqs_t, freqs_t }, -1);
        const cos = emb.cos().convert(dtype);
        const sin = emb.sin().convert(dtype);

        return .{ cos, sin };
    }

    pub fn getCosAndSinInterleaved(self: TextRotaryEmbedding, position_ids: Tensor, dtype: zml.DataType) struct { Tensor, Tensor } { // To be used later in image extension
        const stacked_position_ids = Tensor.stack(&.{ position_ids, position_ids, position_ids }, 0, .g).convert(.f32);
        const inv_freq = zml.nn.invFreq(self.rotary_dim, self.rope_opts).withTags(.{.hd});

        var freqs = stacked_position_ids.outer(inv_freq);
        var freqs_t, var freqs_h, var freqs_w = freqs.chunkExact(.g, 3);
        freqs_t = freqs_t.squeeze(.g);
        freqs_h = freqs_h.squeeze(.g);
        freqs_w = freqs_w.squeeze(.g);

        const h_indices = Tensor.iota(zml.Shape.init(.{ .h = self.mrope_section[1] }, .i32), .h).scale(3).addConstant(1);
        const w_indices = Tensor.iota(zml.Shape.init(.{ .h = self.mrope_section[2] }, .i32), .h).scale(3).addConstant(2);

        const h_input = freqs_h.gather(.{ .dh = h_indices }, .{ .indices_are_sorted = true });
        const w_input = freqs_w.gather(.{ .dh = w_indices }, .{ .indices_are_sorted = true });
        freqs_t = freqs_t.scatterSlices(.{ .dh = h_indices }, h_input, .{ .update_fn = zml.Tensor.ScatterOpts.override });
        freqs = freqs_t.scatterSlices(.{ .dh = w_indices }, w_input, .{ .update_fn = zml.Tensor.ScatterOpts.override });

        const emb = Tensor.concatenate(&.{ freqs, freqs }, -1);
        const cos = emb.cos().convert(dtype);
        const sin = emb.sin().convert(dtype);

        return .{ cos, sin };
    }

    fn rotateHalf(x: Tensor) Tensor {
        const half_dim = @divExact(x.dim(-1), 2);
        const x1 = x.slice1d(-1, .{ .start = 0, .end = half_dim });
        const x2 = x.slice1d(-1, .{ .start = half_dim, .end = x.dim(-1) });
        return Tensor.concatenate(&.{ x2.negate(), x1 }, -1);
    }

    pub fn applyRope(self: TextRotaryEmbedding, x: Tensor, cos: Tensor, sin: Tensor) Tensor {
        const x_rot = x.slice1d(-1, .{ .start = 0, .end = self.rotary_dim });
        const x_pass = x.slice1d(-1, .{ .start = self.rotary_dim, .end = x.dim(-1) });

        const cos_x = cos.insertAxes(.hd, .{.h}).broad(x_rot.shape());
        const sin_x = sin.insertAxes(.hd, .{.h}).broad(x_rot.shape());

        const rotated = x_rot.mul(cos_x).add(rotateHalf(x_rot).mul(sin_x));

        return Tensor.concatenate(&.{ rotated, x_pass }, -1);
    }
};

pub const GatedDeltaNet = struct {
    in_proj_qkv: zml.nn.Linear,
    in_proj_z: zml.nn.Linear,
    in_proj_b: zml.nn.Linear,
    in_proj_a: zml.nn.Linear,
    out_proj: zml.nn.Linear,
    conv1d_weight: Tensor,
    dt_bias: Tensor,
    aLog: Tensor,
    norm: RmsNormGated,

    num_k_heads: i64,
    num_v_heads: i64,
    qk_head_repetition: i64,
    head_k_dim: i64,
    head_v_dim: i64,
    conv_kernel_size: i64,

    fn initProj(store: zml.io.TensorStore.View) zml.nn.Linear {
        return .init(store.createTensor("weight", .{ .dout, .d }, null), null, .d);
    }

    pub fn init(store: zml.io.TensorStore.View, config: Qwen35.Config) GatedDeltaNet {
        const qk_head_repetition =
            @divExact(config.text_config.linear_num_value_heads, config.text_config.linear_num_key_heads);
        return .{
            .in_proj_qkv = initProj(store.withPrefix("in_proj_qkv")),
            .in_proj_z = initProj(store.withPrefix("in_proj_z")),
            .in_proj_b = initProj(store.withPrefix("in_proj_b")),
            .in_proj_a = initProj(store.withPrefix("in_proj_a")),
            .out_proj = initProj(store.withPrefix("out_proj")),
            .conv1d_weight = store.withPrefix("conv1d").createTensor("weight", .{ .out, .in, .kernel_size }, null),
            .dt_bias = store.createTensor("dt_bias", .{.vh}, null),
            .aLog = store.createTensor("A_log", .{.vh}, null),
            .norm = RmsNormGated.init(store.withPrefix("norm"), config.text_config.rms_norm_eps),
            .num_k_heads = config.text_config.linear_num_key_heads,
            .num_v_heads = config.text_config.linear_num_value_heads,
            .qk_head_repetition = qk_head_repetition,
            .head_k_dim = config.text_config.linear_key_head_dim,
            .head_v_dim = config.text_config.linear_value_head_dim,
            .conv_kernel_size = config.text_config.linear_conv_kernel_dim,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(GatedDeltaNet)) void {
        self.in_proj_qkv.weight.deinit();
        self.in_proj_z.weight.deinit();
        self.in_proj_b.weight.deinit();
        self.in_proj_a.weight.deinit();
        self.out_proj.weight.deinit();
        self.conv1d_weight.deinit();
        self.dt_bias.deinit();
        self.aLog.deinit();
        RmsNormGated.unloadBuffers(&self.norm);
    }

    fn recurrent_gated_delta_rule(query: Tensor, key: Tensor, value: Tensor, g: Tensor, beta: Tensor, initial_state: ?Tensor) struct { Tensor, Tensor } {
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(query.dim(.khd))));
        const query_norm = zml.nn.normalizeL2(query.rename(.{ .kh = .vh }), 1e-6);
        const key_norm = zml.nn.normalizeL2(key.rename(.{ .kh = .vh }), 1e-6);

        const query_f32, const key_f32, const value_f32, const alpha_f32, const beta_f32 = .{
            query_norm.convert(.f32).scale(scale).rename(.{ .vh = .h, .khd = .k }),
            key_norm.convert(.f32).rename(.{ .vh = .h, .khd = .k }),
            value.convert(.f32).rename(.{ .vh = .h, .vhd = .v }),
            g.convert(.f32).exp().rename(.{ .vh = .h }),
            beta.convert(.f32).rename(.{ .vh = .h }),
        };

        const initial_recurrent_state = if (initial_state) |state|
            state.convert(.f32).transpose(.{ .b, .vh, .vhd, .khd }).rename(.{ .vh = .h, .vhd = .v, .khd = .k })
        else
            Tensor.constant(zml.DataType.zero(.f32)).broad(zml.Shape.init(.{
                .b = value.dim(.b),
                .h = value.dim(.vh),
                .v = value.dim(.vhd),
                .k = query.dim(.khd),
            }, .f32));

        const result = zml.nn.GatedDeltaNet.forward(
            query_f32,
            key_f32,
            value_f32,
            alpha_f32,
            beta_f32,
            .{ .s = initial_recurrent_state },
        );

        return .{
            result.outputs.rename(.{ .h = .vh, .v = .vhd }).convert(query.dtype()),
            result.state.s.transpose(.{ .b, .h, .k, .v }).rename(.{ .h = .vh, .k = .khd, .v = .vhd }),
        };
    }

    fn buildUpdatedConvState(input: Tensor, left_pad: i64) Tensor {
        const copy_len = @min(input.dim(.s), left_pad);
        const tail = input.slice1d(.s, .{ .start = input.dim(.s) - copy_len, .end = input.dim(.s) });
        if (copy_len == left_pad) return tail;

        const padding_shape = zml.Shape.init(.{ .b = input.dim(.b), .s = left_pad - copy_len, .mix = input.dim(.mix) }, input.dtype());
        const padding = Tensor.constant(input.dtype().zero()).broad(padding_shape);
        return Tensor.concatenate(&.{ padding, tail }, .s);
    }

    pub fn forward(self: GatedDeltaNet, x: Tensor, cache: KvCache.GatedDeltaNetCache) struct { Tensor, KvCache.GatedDeltaNetCache } {
        const key_dim = self.num_k_heads * self.head_k_dim;
        const value_dim = self.num_v_heads * self.head_v_dim;
        const conv_dim = 2 * key_dim + value_dim;
        const left_pad = self.conv_kernel_size - 1;

        const projected_qkv = self.in_proj_qkv.forward(x).rename(.{ .dout = .mix });
        const use_cached_state = x.dim(.s) == 1 and left_pad > 0;
        const conv_input = if (use_cached_state)
            Tensor.concatenate(&.{ cache.convState(), projected_qkv }, .s)
        else
            projected_qkv;

        const kernel = self.conv1d_weight;
        var mixed_qkv = Tensor.conv1d(
            conv_input,
            kernel,
            .{
                .padding = &.{ left_pad, 0 },
                .input_batch_dimension = 0,
                .input_feature_dimension = 2,
                .input_spatial_dimensions = 1,
                .kernel_output_feature_dimension = 0,
                .kernel_input_feature_dimension = 1,
                .kernel_spatial_dimensions = 2,
                .output_batch_dimension = 0,
                .output_feature_dimension = 2,
                .output_spatial_dimensions = 1,
                .feature_group_count = conv_dim,
            },
        )
            .silu();

        if (use_cached_state) {
            mixed_qkv = mixed_qkv.slice1d(.s, .{ .start = mixed_qkv.dim(.s) - 1, .end = mixed_qkv.dim(.s) });
        }

        const z = self.in_proj_z.forward(x).splitAxis(.dout, .{ .vh = self.num_v_heads, .vhd = self.head_v_dim });
        const b = self.in_proj_b.forward(x).rename(.{ .dout = .vh });
        const a = self.in_proj_a.forward(x).rename(.{ .dout = .vh });

        const query = mixed_qkv
            .slice1d(.mix, .{ .start = 0, .end = key_dim })
            .splitAxis(.mix, .{ .kh = self.num_k_heads, .khd = self.head_k_dim });
        const key = mixed_qkv
            .slice1d(.mix, .{ .start = key_dim, .end = 2 * key_dim })
            .splitAxis(.mix, .{ .kh = self.num_k_heads, .khd = self.head_k_dim });
        const value = mixed_qkv
            .slice1d(.mix, .{ .start = 2 * key_dim, .end = 2 * key_dim + value_dim })
            .splitAxis(.mix, .{ .vh = self.num_v_heads, .vhd = self.head_v_dim });

        const beta = b.sigmoid();
        const aLog_type = self.aLog.dtype();
        const g = self.aLog.broad(a.shape()).exp().mul(softplus(a.convert(aLog_type).add(self.dt_bias.convert(aLog_type).broad(a.shape())))).negate();

        const query_for_rule = if (self.qk_head_repetition == 1) query else query.stutter1d(@intCast(query.axis(.kh)), @intCast(self.qk_head_repetition));
        const key_for_rule = if (self.qk_head_repetition == 1) key else key.stutter1d(@intCast(key.axis(.kh)), @intCast(self.qk_head_repetition));

        const core_attn_out, const last_recurrent_state = recurrent_gated_delta_rule(
            query_for_rule,
            key_for_rule,
            value,
            g,
            beta,
            if (use_cached_state) cache.recurrentState() else null,
        );

        const core_attn_out_normed = self.norm
            .forward(
                core_attn_out.rename(.{ .vhd = .d }),
                z.rename(.{ .vhd = .d }),
            )
            .rename(.{ .d = .vhd });

        const output = self.out_proj.forward(core_attn_out_normed.merge(.{ .d = .{ .vh, .vhd } })).rename(.{ .dout = .d });
        const updated_cache = cache.update(
            buildUpdatedConvState(conv_input, left_pad),
            last_recurrent_state,
        );
        return .{ output, updated_cache };
    }
};

pub const RmsNorm = struct {
    weight: Tensor,
    eps: f32 = 1e-6,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{ .weight = store.createTensor("weight", .{.d}, null), .eps = eps };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNorm, x: Tensor) Tensor {
        const x_f32 = x.convert(.f32);
        const weight_f32 = self.weight.convert(.f32);

        const normalized = zml.nn.rmsNorm(x_f32, .d, self.eps);
        return normalized.mul(weight_f32.broad(x.shape())).add(normalized).convert(x.dtype());
    }
};

pub const RmsNormGated = struct {
    weight: Tensor,
    eps: f32 = 1e-6,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNormGated {
        return .{ .weight = store.createTensor("weight", .{.d}, null), .eps = eps };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNormGated)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNormGated, x: Tensor, gate: Tensor) Tensor {
        const x_f32 = x.convert(.f32);
        const gate_f32 = gate.convert(.f32);

        const normalized = zml.nn.rmsNorm(x_f32, .d, self.eps);
        const output = normalized.mul(self.weight.broad(x.shape()));

        const gated_output = output.mul(gate_f32.silu());
        return gated_output.convert(x.dtype());
    }
};

pub const KvCache = struct {
    layer_types: []const Qwen35.LayerType,
    self_attn: SelfAttnCache,
    gated_delta_net: GatedDeltaNetCache,

    pub const SelfAttnCache = struct {
        k: Tensor,
        v: Tensor,
        layer_index: Tensor,

        pub fn init(config: Qwen35.Config, batch_dim: i64, max_seq_len: i64, dtype: zml.DataType) SelfAttnCache {
            const num_self_attn_layers = countLayers(config.text_config.layer_types, .full_attention);
            const kv_shape = zml.Shape.init(.{
                .b = batch_dim,
                .layer = num_self_attn_layers,
                .s = max_seq_len,
                .h = config.text_config.num_key_value_heads,
                .hd = config.text_config.head_dim,
            }, dtype);
            return .{
                .k = .fromShape(kv_shape),
                .v = .fromShape(kv_shape),
                .layer_index = .init(.{}, .u32),
            };
        }

        pub fn initBuffer(self: SelfAttnCache, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(SelfAttnCache) {
            const sharding = try zml.sharding.replicatedSharding(platform);
            return .{
                .k = try zml.Buffer.uninitialized(io, platform, self.k.shape(), sharding, .{}),
                .v = try zml.Buffer.uninitialized(io, platform, self.v.shape(), sharding, .{}),
                .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32, sharding),
            };
        }

        pub fn deinitBuffer(self: *zml.Bufferized(SelfAttnCache)) void {
            self.k.deinit();
            self.v.deinit();
            self.layer_index.deinit();
        }

        pub fn keys(self: SelfAttnCache) Tensor {
            return self.k.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
        }

        pub fn values(self: SelfAttnCache) Tensor {
            return self.v.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
        }

        pub fn update(self: SelfAttnCache, new_k: Tensor, new_v: Tensor, token_index: ?Tensor) SelfAttnCache {
            const k_shape = self.k.shape().drop(.layer);
            var layer = self.layer_index;
            layer = if (token_index) |idx| layer.broad(idx.shape()) else layer;

            return if (token_index) |idx| .{
                .k = self.k.scatterSlices(
                    .{ .layer = layer, .s = idx },
                    new_k.convert(self.k.dtype()).transpose(k_shape),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(self.k),
                .v = self.v.scatterSlices(
                    .{ .layer = layer, .s = idx },
                    new_v.convert(self.v.dtype()).transpose(k_shape),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(self.v),
                .layer_index = self.layer_index,
            } else .{
                .k = self.k.scatterSlices(
                    .{ .layer = layer },
                    new_k.convert(self.k.dtype()).transpose(k_shape),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(self.k),
                .v = self.v.scatterSlices(
                    .{ .layer = layer },
                    new_v.convert(self.v.dtype()).transpose(k_shape),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(self.v),
                .layer_index = self.layer_index,
            };
        }

        pub fn atLayer(self: SelfAttnCache, layer_index: usize) SelfAttnCache {
            return .{
                .k = self.k,
                .v = self.v,
                .layer_index = Tensor.scalar(layer_index, .u32),
            };
        }

        pub fn reuseBuffer(self: SelfAttnCache, other: SelfAttnCache) SelfAttnCache {
            return .{
                .k = self.k.reuseBuffer(other.k),
                .v = self.v.reuseBuffer(other.v),
                .layer_index = self.layer_index.reuseBuffer(other.layer_index),
            };
        }
    };

    pub const GatedDeltaNetCache = struct {
        conv_state: Tensor,
        recurrent_state: Tensor,
        layer_index: Tensor,

        pub fn init(config: Qwen35.Config, batch_dim: i64, conv_dtype: zml.DataType, recurrent_dtype: zml.DataType) GatedDeltaNetCache {
            const num_linear_attn_layers = countLayers(config.text_config.layer_types, .linear_attention);
            const conv_dim = 2 * config.text_config.linear_num_key_heads * config.text_config.linear_key_head_dim + config.text_config.linear_num_value_heads * config.text_config.linear_value_head_dim;
            const conv_state_shape = zml.Shape.init(.{
                .b = batch_dim,
                .layer = num_linear_attn_layers,
                .s = config.text_config.linear_conv_kernel_dim - 1,
                .mix = conv_dim,
            }, conv_dtype);
            const recurrent_state_shape = zml.Shape.init(.{
                .b = batch_dim,
                .layer = num_linear_attn_layers,
                .vh = config.text_config.linear_num_value_heads,
                .khd = config.text_config.linear_key_head_dim,
                .vhd = config.text_config.linear_value_head_dim,
            }, recurrent_dtype);
            return .{
                .conv_state = .fromShape(conv_state_shape),
                .recurrent_state = .fromShape(recurrent_state_shape),
                .layer_index = .init(.{}, .u32),
            };
        }

        pub fn initBuffer(self: GatedDeltaNetCache, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(GatedDeltaNetCache) {
            const sharding = try zml.sharding.replicatedSharding(platform);
            return .{
                .conv_state = try zml.Buffer.uninitialized(io, platform, self.conv_state.shape(), sharding, .{}),
                .recurrent_state = try zml.Buffer.uninitialized(io, platform, self.recurrent_state.shape(), sharding, .{}),
                .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32, sharding),
            };
        }

        pub fn deinitBuffer(self: *zml.Bufferized(GatedDeltaNetCache)) void {
            self.conv_state.deinit();
            self.recurrent_state.deinit();
            self.layer_index.deinit();
        }

        pub fn convState(self: GatedDeltaNetCache) Tensor {
            return self.conv_state.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
        }

        pub fn recurrentState(self: GatedDeltaNetCache) Tensor {
            return self.recurrent_state.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
        }

        pub fn update(self: GatedDeltaNetCache, new_conv_state: ?Tensor, new_recurrent_state: ?Tensor) GatedDeltaNetCache {
            const conv_state = if (new_conv_state) |state|
                self.conv_state.scatterSlices(
                    .{ .layer = self.layer_index },
                    state.convert(self.conv_state.dtype()).transpose(self.conv_state.shape().drop(.layer)),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(self.conv_state)
            else
                self.conv_state;

            const recurrent_state = if (new_recurrent_state) |state|
                self.recurrent_state.scatterSlices(
                    .{ .layer = self.layer_index },
                    state.convert(self.recurrent_state.dtype()).transpose(self.recurrent_state.shape().drop(.layer)),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(self.recurrent_state)
            else
                self.recurrent_state;

            return .{
                .conv_state = conv_state,
                .recurrent_state = recurrent_state,
                .layer_index = self.layer_index,
            };
        }

        pub fn atLayer(self: GatedDeltaNetCache, layer_index: usize) GatedDeltaNetCache {
            return .{
                .conv_state = self.conv_state,
                .recurrent_state = self.recurrent_state,
                .layer_index = Tensor.scalar(layer_index, .u32),
            };
        }

        pub fn reuseBuffer(self: GatedDeltaNetCache, other: GatedDeltaNetCache) GatedDeltaNetCache {
            return .{
                .conv_state = self.conv_state.reuseBuffer(other.conv_state),
                .recurrent_state = self.recurrent_state.reuseBuffer(other.recurrent_state),
                .layer_index = self.layer_index.reuseBuffer(other.layer_index),
            };
        }
    };

    pub fn init(
        config: Qwen35.Config,
        batch_dim: i64,
        max_seq_len: i64,
        cache_dtype: zml.DataType,
        recurrent_dtype: zml.DataType,
    ) KvCache {
        return .{
            .layer_types = config.text_config.layer_types,
            .self_attn = SelfAttnCache.init(config, batch_dim, max_seq_len, cache_dtype),
            .gated_delta_net = GatedDeltaNetCache.init(config, batch_dim, cache_dtype, recurrent_dtype),
        };
    }

    pub fn initBuffer(self: KvCache, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(KvCache) {
        return .{
            .self_attn = try self.self_attn.initBuffer(io, platform),
            .gated_delta_net = try self.gated_delta_net.initBuffer(io, platform),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(KvCache)) void {
        SelfAttnCache.deinitBuffer(&self.self_attn);
        GatedDeltaNetCache.deinitBuffer(&self.gated_delta_net);
    }

    pub const LayerView = struct {
        parent: KvCache,
        cache: union(enum) {
            self_attn: SelfAttnCache,
            linear_attn: GatedDeltaNetCache,
        },
    };

    pub fn atLayer(self: KvCache, layer_index: usize) LayerView {
        return switch (getDenseIndex(self.layer_types, layer_index)) {
            .full_attention => |dense_index| .{
                .parent = self,
                .cache = .{ .self_attn = self.self_attn.atLayer(dense_index.layer_dense_index) },
            },
            .linear_attention => |dense_index| .{
                .parent = self,
                .cache = .{ .linear_attn = self.gated_delta_net.atLayer(dense_index.layer_dense_index) },
            },
        };
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{
            .layer_types = self.layer_types,
            .self_attn = self.self_attn.reuseBuffer(other.self_attn),
            .gated_delta_net = self.gated_delta_net.reuseBuffer(other.gated_delta_net),
        };
    }

    fn countLayers(layer_types: []const Qwen35.LayerType, layer_type: Qwen35.LayerType) i64 {
        var count: i64 = 0;
        for (layer_types) |registered_layer_type| {
            if (registered_layer_type == layer_type) count += 1;
        }
        return count;
    }

    fn getDenseIndex(layer_types: []const Qwen35.LayerType, layer_index: usize) union(enum) {
        full_attention: struct { layer_dense_index: usize },
        linear_attention: struct { layer_dense_index: usize },
    } {
        var self_attn_layer_index: usize = 0;
        var linear_attn_layer_index: usize = 0;
        for (layer_types[0..layer_index]) |layer_type| {
            switch (layer_type) {
                .full_attention => self_attn_layer_index += 1,
                .linear_attention => linear_attn_layer_index += 1,
            }
        }
        return switch (layer_types[layer_index]) {
            .full_attention => .{ .full_attention = .{ .layer_dense_index = self_attn_layer_index } },
            .linear_attention => .{ .linear_attention = .{ .layer_dense_index = linear_attn_layer_index } },
        };
    }
};

//========================Utils========================

fn softplus(x: Tensor) Tensor {
    return x.exp().addConstant(1).log();
}
