const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

pub const SamplingConfig = struct {
    topk: u32 = 1,
    temperature: f32 = 0.0,
};

pub const Config = struct {
    model_type: []const u8 = "",
    vocab_size: u32,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    head_dim: u32,
    num_key_value_heads: u32 = 0,
    intermediate_size: u32 = 0,
    hidden_act: []const u8 = "silu",
    attention_bias: bool = false,
    q_lora_rank: u32 = 0,
    qk_rope_head_dim: u32 = 0,
    o_lora_rank: u32 = 0,
    o_groups: u32 = 0,
    n_routed_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    n_shared_experts: ?u32 = null,
    norm_topk_prob: bool = true,
    routed_scaling_factor: f32 = 1.0,
    scoring_func: []const u8 = "sqrtsoftplus",
    rms_norm_eps: f32,
    rope_theta: f32 = 1_000_000,
    rope_parameters: RopeParameters = .{},
    rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = .{} },
    hc_mult: u32 = 1,
    hc_eps: f32 = 1e-6,
    hc_sinkhorn_iters: u32 = 20,
    dspark_target_layer_ids: []const u32 = &.{},
    target_layer_ids: []const u32 = &.{},
    n_mtp_layers: ?u32 = null,
    dspark_markov_rank: u32 = 0,
    markov_rank: u32 = 0,
    draft_vocab_size: ?u32 = null,
    mask_token_id: ?u32 = null,

    pub fn isQwen3(self: Config) bool {
        return std.mem.eql(u8, self.model_type, "qwen3");
    }

    pub fn qwenTargetLayerIds(self: Config) []const u32 {
        if (self.target_layer_ids.len > 0) return self.target_layer_ids;
        return self.dspark_target_layer_ids;
    }
};

pub const RopeParameters = struct {
    rope_theta: f32 = 1_000_000,
    partial_rotary_factor: f32 = 1.0,
};

pub const Buffers = zml.Bufferized(Model);

pub const Model = union(enum) {
    deepseek: DeepSeekModel,
    qwen3: Qwen3Model,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Model {
        if (config.isQwen3()) {
            return .{ .qwen3 = try .init(allocator, store, config) };
        }
        return .{ .deepseek = try .init(allocator, store, config) };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        switch (self) {
            .deepseek => |model| model.deinit(allocator),
            .qwen3 => |model| model.deinit(allocator),
        }
    }

    pub fn unloadBuffers(self: *Buffers, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .deepseek => |*buffers| DeepSeekModel.unloadBuffers(buffers, allocator),
            .qwen3 => |*buffers| Qwen3Model.unloadBuffers(buffers, allocator),
        }
    }

    pub fn blockPadToken(self: Model) u32 {
        return switch (self) {
            .deepseek => 0,
            .qwen3 => |model| model.mask_token_id orelse 0,
        };
    }

    pub fn targetAuxWidth(self: Model) u32 {
        return switch (self) {
            .deepseek => |model| model.target_aux_width,
            .qwen3 => |model| model.target_aux_width,
        };
    }

    pub fn auxHiddenDtype(self: Model) zml.DataType {
        return switch (self) {
            .deepseek => |model| model.layers[0].ffn.experts[0].w1.weight.dtype(),
            .qwen3 => |model| model.fc.weight.dtype(),
        };
    }

    pub fn loadBuffers(
        self: *const Model,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
    ) !Buffers {
        return zml.io.load(Model, self, allocator, io, platform, store, .{
            .parallelism = 16,
            .shardings = shardings,
            .dma_chunks = 32,
            .dma_chunk_size = 256 * zml.MiB,
        });
    }

    pub fn draftLogits(self: Model, hidden: zml.Tensor, previous_tokens: zml.Tensor) zml.Tensor {
        return switch (self) {
            .deepseek => |model| model.draftLogits(hidden, previous_tokens),
            .qwen3 => |model| model.draftLogits(hidden, previous_tokens),
        };
    }

    pub fn sample(self: Model, logits: zml.Tensor, sampling: SamplingConfig, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        _ = self;
        const topk: u32 = if (sampling.temperature < 0.00001) 1 else if (sampling.topk == 0) @intCast(logits.dim(.voc)) else sampling.topk;
        const temperature: f32 = if (sampling.temperature < 0.00001) 1.0 else sampling.temperature;
        return zml.nn.sampleTokens(logits, .{ .topk = topk, .temperature = temperature }, rng);
    }
};

pub const DeepSeekModel = struct {
    const Self = @This();

    embed_tokens: zml.nn.TokenEmbedding,
    main_proj: zml.nn.Linear,
    main_norm: RmsNorm,
    layers: []DecoderLayer,
    norm: RmsNorm,
    hc_head_fn: zml.Tensor,
    hc_head_base: zml.Tensor,
    hc_head_scale: zml.Tensor,
    lm_head: zml.nn.Linear,
    markov_head: MarkovHead,
    hc_mult: u32,
    hc_eps: f32,
    rms_norm_eps: f32,
    target_aux_width: u32,
    draft_vocab_size: u32,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Self {
        const num_dspark_layers = config.n_mtp_layers orelse 3;
        const mtp_store = store.withPrefix("mtp");

        const layers = try allocator.alloc(DecoderLayer, num_dspark_layers);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| {
            errdefer for (layers[0..i]) |previous| previous.deinit(allocator);
            layer.* = try .init(allocator, mtp_store.withLayer(i), config, i);
        }

        const head_store = mtp_store.withLayer(num_dspark_layers - 1);
        const target_aux_width: u32 = @intCast(config.hidden_size * config.dspark_target_layer_ids.len);
        const draft_vocab_size = config.draft_vocab_size orelse config.vocab_size;
        if (draft_vocab_size != config.vocab_size) return error.UnsupportedDraftVocabRemap;

        return .{
            .embed_tokens = .{ .weight = store.createTensor("model.embed_tokens.weight", .{ .voc, .d }, .{ .voc = .replicated, .d = .model }) },
            .main_proj = .init(
                mtp_store.withLayer(0).withPrefix("main_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                null,
                .d,
            ),
            .main_norm = .init(mtp_store.withLayer(0).withPrefix("main_norm"), config.rms_norm_eps),
            .layers = layers,
            .norm = .init(head_store.withPrefix("norm"), config.rms_norm_eps),
            .hc_head_fn = head_store.createTensor("hc_head_fn", .{ .hc, .hcd }, .{ .hc = .replicated, .hcd = .replicated }),
            .hc_head_base = head_store.createTensor("hc_head_base", .{.hc}, .replicated),
            .hc_head_scale = head_store.createTensor("hc_head_scale", .{.scale}, .replicated),
            .lm_head = .init(store.createTensor("lm_head.weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }), null, .d),
            .markov_head = .init(head_store.withPrefix("markov_head"), config),
            .hc_mult = config.hc_mult,
            .hc_eps = config.hc_eps,
            .rms_norm_eps = config.rms_norm_eps,
            .target_aux_width = target_aux_width,
            .draft_vocab_size = draft_vocab_size,
        };
    }

    pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
        for (self.layers) |layer| layer.deinit(allocator);
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Self), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        self.main_proj.weight.deinit();
        RmsNorm.unloadBuffers(&self.main_norm);
        for (self.layers) |*layer| DecoderLayer.unloadBuffers(layer, allocator);
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
        self.hc_head_fn.deinit();
        self.hc_head_base.deinit();
        self.hc_head_scale.deinit();
        self.lm_head.weight.deinit();
        MarkovHead.unloadBuffers(&self.markov_head);
    }

    pub fn loadBuffers(
        self: *const Self,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
    ) !zml.Bufferized(Self) {
        return zml.io.load(Self, self, allocator, io, platform, store, .{
            .parallelism = 16,
            .shardings = shardings,
            .dma_chunks = 32,
            .dma_chunk_size = 256 * zml.MiB,
        });
    }

    pub fn forward(
        self: Self,
        input_ids_: zml.Tensor,
        positions_: zml.Tensor,
        aux_hidden_states_: zml.Tensor,
        moe_metadata: zml.moe.Metadata,
        moe_parameters: zml.moe.Parameters,
    ) zml.Tensor {
        const input_ids = input_ids_.withPartialTags(.{.s});
        const positions = positions_.withPartialTags(.{.s});
        const aux_hidden_states = aux_hidden_states_.withPartialTags(.{ .s, .d });
        const main_x = self.main_norm.forward(
            self.main_proj.forward(aux_hidden_states.convert(self.main_proj.weight.dtype()))
                .rename(.{ .dout = .d }),
        );
        _ = main_x;

        const embeds = self.embed_tokens.forward(input_ids)
            .withPartialTags(.{ .s, .d })
            .convert(self.norm.weight.dtype());
        var hidden_hc = embeds.insertAxes(.d, .{.hc}).broad(.init(.{
            .s = embeds.dim(.s),
            .hc = self.hc_mult,
            .d = embeds.dim(.d),
        }, embeds.dtype()));

        var x: zml.Tensor = undefined;
        var state: ?MhcState = null;
        for (self.layers) |layer| {
            x, state = layer.forward(hidden_hc, x, state, positions, input_ids, moe_metadata, moe_parameters);
            hidden_hc = undefined;
        }

        const final_state = state orelse @panic("DSpark model requires at least one draft layer");
        const head_residual = mhcPost(x, final_state.residual, final_state.post_mix, final_state.res_mix);
        return self.hcHead(head_residual);
    }

    pub fn logits(self: Self, hidden_: zml.Tensor) zml.Tensor {
        const hidden = self.norm.forward(hidden_.withPartialTags(.{ .s, .d }));
        return self.lm_head.forward(hidden.convert(self.lm_head.weight.dtype())).rename(.{ .dout = .voc });
    }

    pub fn draftLogits(self: Self, hidden: zml.Tensor, previous_tokens: zml.Tensor) zml.Tensor {
        const base_logits = self.logits(hidden);
        const markov = self.markov_head.bias(previous_tokens).convert(base_logits.dtype());
        return base_logits.add(markov);
    }

    pub fn sample(self: Self, logits_: zml.Tensor, sampling: SamplingConfig, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        _ = self;
        const topk: u32 = if (sampling.temperature < 0.00001) 1 else if (sampling.topk == 0) @intCast(logits_.dim(.voc)) else sampling.topk;
        const temperature: f32 = if (sampling.temperature < 0.00001) 1.0 else sampling.temperature;
        return zml.nn.sampleTokens(logits_, .{ .topk = topk, .temperature = temperature }, rng);
    }

    fn hcHead(self: Self, residual_: zml.Tensor) zml.Tensor {
        const residual = residual_.withPartialTags(.{ .s, .hc, .d });
        const flat = residual.convert(.f32).merge(.{ .hcd = .{ .hc, .d } });
        const normalized = flat.mul(flat.mul(flat).mean(.hcd).addConstant(self.rms_norm_eps).rsqrt().broad(flat.shape()));
        const fn_ = self.hc_head_fn.withTags(.{ .hc, .hcd }).convert(.f32);
        const scale = scalarAt(self.hc_head_scale.withTags(.{.scale}), .scale, 0);
        const pre_mix = normalized.dot(fn_, .hcd)
            .mul(scale.broad(.init(.{ .s = residual.dim(.s), .hc = self.hc_mult }, .f32)))
            .add(self.hc_head_base.withTags(.{.hc}).convert(.f32).broad(.init(.{ .s = residual.dim(.s), .hc = self.hc_mult }, .f32)))
            .sigmoid()
            .addConstant(self.hc_eps);
        return residual.convert(.f32)
            .mul(pre_mix.insertAxes(.hc, .{.d}).broad(residual.shape()))
            .sum(.hc)
            .squeeze(.hc)
            .convert(residual.dtype());
    }
};

pub const Qwen3Model = struct {
    const Self = @This();

    embed_tokens: zml.nn.TokenEmbedding,
    fc: zml.nn.Linear,
    hidden_norm: RmsNorm,
    layers: []Qwen3DecoderLayer,
    norm: RmsNorm,
    lm_head: zml.nn.Linear,
    markov_head: MarkovHead,
    target_aux_width: u32,
    mask_token_id: ?u32,
    draft_vocab_size: u32,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Self {
        if (!std.mem.eql(u8, config.hidden_act, "silu")) return error.UnsupportedQwen3Activation;
        const layers = try allocator.alloc(Qwen3DecoderLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| {
            layer.* = try .init(store.withPrefix("layers").withLayer(i), config);
        }

        const target_layer_ids = config.qwenTargetLayerIds();
        const target_aux_width: u32 = @intCast(config.hidden_size * target_layer_ids.len);
        const draft_vocab_size = config.draft_vocab_size orelse config.vocab_size;
        if (draft_vocab_size != config.vocab_size) return error.UnsupportedDraftVocabRemap;

        return .{
            .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, .{ .voc = .replicated, .d = .model }) },
            .fc = .init(
                store.withPrefix("fc").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                null,
                .d,
            ),
            .hidden_norm = .init(store.withPrefix("hidden_norm"), config.rms_norm_eps),
            .layers = layers,
            .norm = .init(store.withPrefix("norm"), config.rms_norm_eps),
            .lm_head = .init(store.createTensor("lm_head.weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }), null, .d),
            .markov_head = .init(store.withPrefix("markov_head"), config),
            .target_aux_width = target_aux_width,
            .mask_token_id = config.mask_token_id,
            .draft_vocab_size = draft_vocab_size,
        };
    }

    pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Self), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        unloadLinear(&self.fc);
        RmsNorm.unloadBuffers(&self.hidden_norm);
        for (self.layers) |*layer| Qwen3DecoderLayer.unloadBuffers(layer);
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
        self.lm_head.weight.deinit();
        MarkovHead.unloadBuffers(&self.markov_head);
    }

    pub fn loadBuffers(
        self: *const Self,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
    ) !zml.Bufferized(Self) {
        return zml.io.load(Self, self, allocator, io, platform, store, .{
            .parallelism = 16,
            .shardings = shardings,
            .dma_chunks = 32,
            .dma_chunk_size = 256 * zml.MiB,
        });
    }

    pub fn forward(self: Self, input_ids_: zml.Tensor, positions_: zml.Tensor, aux_hidden_states_: zml.Tensor) zml.Tensor {
        const input_ids = input_ids_.withPartialTags(.{.s});
        const positions = positions_.withPartialTags(.{.s});
        const aux_hidden_states = aux_hidden_states_.withPartialTags(.{ .s, .d });
        const context_states = self.fc.forward(aux_hidden_states.convert(self.fc.weight.dtype())).rename(.{ .dout = .d });
        _ = self.hidden_norm.forward(context_states);

        var hidden = self.embed_tokens.forward(input_ids)
            .withPartialTags(.{ .s, .d })
            .convert(self.norm.weight.dtype());
        for (self.layers) |layer| {
            hidden = layer.forward(hidden, positions);
        }
        return self.norm.forward(hidden);
    }

    pub fn logits(self: Self, hidden_: zml.Tensor) zml.Tensor {
        const hidden = hidden_.withPartialTags(.{ .s, .d });
        return self.lm_head.forward(hidden.convert(self.lm_head.weight.dtype())).rename(.{ .dout = .voc });
    }

    pub fn draftLogits(self: Self, hidden: zml.Tensor, previous_tokens: zml.Tensor) zml.Tensor {
        const base_logits = self.logits(hidden);
        const markov = self.markov_head.bias(previous_tokens).convert(base_logits.dtype());
        return base_logits.add(markov);
    }
};

pub const Qwen3DecoderLayer = struct {
    input_layernorm: RmsNorm,
    attn: Qwen3Attention,
    mlp: Qwen3Mlp,
    post_attention_layernorm: RmsNorm,

    pub fn init(store: zml.io.TensorStore.View, config: Config) !Qwen3DecoderLayer {
        return .{
            .input_layernorm = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps),
            .attn = .init(store.withPrefix("self_attn"), config),
            .mlp = .init(store.withPrefix("mlp")),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Qwen3DecoderLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        Qwen3Attention.unloadBuffers(&self.attn);
        Qwen3Mlp.unloadBuffers(&self.mlp);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
    }

    pub fn forward(self: Qwen3DecoderLayer, x0_: zml.Tensor, positions: zml.Tensor) zml.Tensor {
        const x0 = x0_.withPartialTags(.{ .s, .d }).withPartitioning(.{ .d = .replicated });
        const attn_out = self.attn.forward(self.input_layernorm.forward(x0), positions);
        const x1 = x0.add(attn_out).withPartitioning(.{ .d = .replicated });
        const mlp_out = self.mlp.forward(self.post_attention_layernorm.forward(x1)).withPartitioning(.{ .d = .replicated });
        return x1.add(mlp_out).withPartitioning(.{ .d = .replicated });
    }
};

pub const Qwen3Attention = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    o_proj: zml.nn.Linear,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    rope_opts: zml.nn.RopeOpts,

    pub fn init(store: zml.io.TensorStore.View, config: Config) Qwen3Attention {
        const num_kv_heads = if (config.num_key_value_heads == 0) config.num_attention_heads else config.num_key_value_heads;
        const rotary_dim: u32 = @intFromFloat(@as(f32, @floatFromInt(config.head_dim)) * config.rope_parameters.partial_rotary_factor);
        return .{
            .q_proj = qwenLinear(store.withPrefix("q_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .k_proj = qwenLinear(store.withPrefix("k_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .v_proj = qwenLinear(store.withPrefix("v_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .q_norm = .init(store.withPrefix("q_norm"), config.rms_norm_eps),
            .k_norm = .init(store.withPrefix("k_norm"), config.rms_norm_eps),
            .o_proj = qwenLinear(store.withPrefix("o_proj"), .{ .dout = .replicated, .d = .model }, .{ .dout = .replicated }),
            .num_heads = config.num_attention_heads,
            .num_kv_heads = num_kv_heads,
            .head_dim = config.head_dim,
            .rotary_dim = rotary_dim,
            .rope_opts = .{
                .layout = .sequential,
                .scaling = .{ .default = .{ .rope_theta = config.rope_parameters.rope_theta } },
            },
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Qwen3Attention)) void {
        unloadLinear(&self.q_proj);
        unloadLinear(&self.k_proj);
        unloadLinear(&self.v_proj);
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
        unloadLinear(&self.o_proj);
    }

    pub fn forward(self: Qwen3Attention, x_: zml.Tensor, positions_: zml.Tensor) zml.Tensor {
        const x = x_.withPartialTags(.{ .s, .d }).withPartitioning(.{ .d = .replicated });
        const positions = positions_.withPartialTags(.{.s});

        var q = self.q_proj.forward(x.convert(self.q_proj.weight.dtype()))
            .splitAxis(.dout, .{ .h = self.num_heads, .hd = self.head_dim })
            .withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        var k = self.k_proj.forward(x.convert(self.k_proj.weight.dtype()))
            .splitAxis(.dout, .{ .h = self.num_kv_heads, .hd = self.head_dim });
        var v = self.v_proj.forward(x.convert(self.v_proj.weight.dtype()))
            .splitAxis(.dout, .{ .h = self.num_kv_heads, .hd = self.head_dim });

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        q = applyLeadingRope(q, positions, self.rotary_dim, self.rope_opts)
            .withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        k = applyLeadingRope(k, positions, self.rotary_dim, self.rope_opts);

        const kv_repeat = @divExact(self.num_heads, self.num_kv_heads);
        if (kv_repeat != 1) {
            k = k.stutter1d(k.axis(.h), kv_repeat);
            v = v.stutter1d(v.axis(.h), kv_repeat);
        }
        k = k.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });

        const attn = zml.nn.sdpa(
            q.rename(.{ .s = .q }).convert(.f32),
            k.rename(.{ .s = .k }).convert(.f32),
            v.rename(.{ .s = .k }).convert(.f32),
            .{},
        )
            .rename(.{ .q = .s })
            .withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated })
            .convert(self.o_proj.weight.dtype())
            .merge(.{ .d = .{ .h, .hd } });

        return self.o_proj.forward(attn)
            .rename(.{ .dout = .d })
            .withPartitioning(.{ .d = .replicated });
    }
};

pub const Qwen3Mlp = struct {
    gate_proj: zml.nn.Linear,
    up_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Qwen3Mlp {
        return .{
            .gate_proj = qwenLinear(store.withPrefix("gate_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .up_proj = qwenLinear(store.withPrefix("up_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .down_proj = qwenLinear(store.withPrefix("down_proj"), .{ .dout = .replicated, .d = .model }, .{ .dout = .replicated }),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Qwen3Mlp)) void {
        unloadLinear(&self.gate_proj);
        unloadLinear(&self.up_proj);
        unloadLinear(&self.down_proj);
    }

    pub fn forward(self: Qwen3Mlp, x: zml.Tensor) zml.Tensor {
        const gate = self.gate_proj.forward(x.convert(self.gate_proj.weight.dtype()));
        const up = self.up_proj.forward(x.convert(self.up_proj.weight.dtype()));
        const hidden = gate.silu().mul(up).rename(.{ .dout = .d });
        return self.down_proj.forward(hidden.convert(self.down_proj.weight.dtype())).rename(.{ .dout = .d });
    }
};

pub const MarkovHead = struct {
    markov_w1: zml.nn.TokenEmbedding,
    markov_w2: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View, config: Config) MarkovHead {
        _ = config;
        return .{
            .markov_w1 = .{ .weight = store.withPrefix("markov_w1").createTensor("weight", .{ .voc, .rank }, .{ .voc = .replicated, .rank = .model }) },
            .markov_w2 = store.withPrefix("markov_w2").createTensor("weight", .{ .voc, .rank }, .{ .voc = .model, .rank = .replicated }),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(MarkovHead)) void {
        self.markov_w1.weight.deinit();
        self.markov_w2.deinit();
    }

    pub fn bias(self: MarkovHead, token_ids: zml.Tensor) zml.Tensor {
        const embed = self.markov_w1.forward(token_ids).withPartialTags(.{ .s, .rank });
        return embed.dot(self.markov_w2.withTags(.{ .voc, .rank }), .rank);
    }
};

const MhcState = struct {
    residual: zml.Tensor,
    post_mix: zml.Tensor,
    res_mix: zml.Tensor,
};

const MhcWeights = struct {
    fn_: zml.Tensor,
    base: zml.Tensor,
    scale: zml.Tensor,
};

pub const DecoderLayer = struct {
    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    attn: Attention,
    ffn: Moe,
    hc_attn: MhcWeights,
    hc_ffn: MhcWeights,
    hc_mult: u32,
    hc_eps: f32,
    hc_sinkhorn_iters: u32,
    rms_norm_eps: f32,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, layer_index: usize) !DecoderLayer {
        return .{
            .attn_norm = .init(store.withPrefix("attn_norm"), config.rms_norm_eps),
            .ffn_norm = .init(store.withPrefix("ffn_norm"), config.rms_norm_eps),
            .attn = .init(store.withPrefix("attn"), config),
            .ffn = try .init(allocator, store.withPrefix("ffn"), config, config.num_hidden_layers + @as(u32, @intCast(layer_index))),
            .hc_attn = .{
                .fn_ = store.createTensor("hc_attn_fn", .{ .mix, .hcd }, .replicated),
                .base = store.createTensor("hc_attn_base", .{.mix}, .replicated),
                .scale = store.createTensor("hc_attn_scale", .{.scale}, .replicated),
            },
            .hc_ffn = .{
                .fn_ = store.createTensor("hc_ffn_fn", .{ .mix, .hcd }, .replicated),
                .base = store.createTensor("hc_ffn_base", .{.mix}, .replicated),
                .scale = store.createTensor("hc_ffn_scale", .{.scale}, .replicated),
            },
            .hc_mult = config.hc_mult,
            .hc_eps = config.hc_eps,
            .hc_sinkhorn_iters = config.hc_sinkhorn_iters,
            .rms_norm_eps = config.rms_norm_eps,
        };
    }

    pub fn deinit(self: DecoderLayer, allocator: std.mem.Allocator) void {
        self.ffn.deinit(allocator);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(DecoderLayer), allocator: std.mem.Allocator) void {
        RmsNorm.unloadBuffers(&self.attn_norm);
        RmsNorm.unloadBuffers(&self.ffn_norm);
        Attention.unloadBuffers(&self.attn);
        Moe.unloadBuffers(&self.ffn, allocator);
        unloadMhcWeights(&self.hc_attn);
        unloadMhcWeights(&self.hc_ffn);
    }

    pub fn forward(
        self: DecoderLayer,
        first_hidden_hc: zml.Tensor,
        previous_x: zml.Tensor,
        previous_state: ?MhcState,
        positions: zml.Tensor,
        input_ids: zml.Tensor,
        moe_metadata: zml.moe.Metadata,
        moe_parameters: zml.moe.Parameters,
    ) struct { zml.Tensor, MhcState } {
        const attn_residual = if (previous_state) |state|
            mhcPost(previous_x, state.residual, state.post_mix, state.res_mix)
        else
            first_hidden_hc.withPartialTags(.{ .s, .hc, .d });

        const attn_pre = mhcPre(attn_residual, self.hc_attn, self.attn_norm, self.rms_norm_eps, self.hc_eps, self.hc_eps, 2.0, self.hc_sinkhorn_iters);
        const attn_out = self.attn.forward(attn_pre.layer_input, positions);

        const ffn_residual = mhcPost(attn_out, attn_residual, attn_pre.post_mix, attn_pre.res_mix);
        const ffn_pre = mhcPre(ffn_residual, self.hc_ffn, self.ffn_norm, self.rms_norm_eps, self.hc_eps, self.hc_eps, 2.0, self.hc_sinkhorn_iters);
        const ffn_out = self.ffn.forward(ffn_pre.layer_input, input_ids, moe_metadata, moe_parameters);

        return .{
            ffn_out.withPartialTags(.{ .s, .d }),
            .{
                .residual = ffn_residual,
                .post_mix = ffn_pre.post_mix,
                .res_mix = ffn_pre.res_mix,
            },
        };
    }
};

fn unloadMhcWeights(weights: *zml.Bufferized(MhcWeights)) void {
    weights.fn_.deinit();
    weights.base.deinit();
    weights.scale.deinit();
}

const MhcPreOutput = struct {
    post_mix: zml.Tensor,
    res_mix: zml.Tensor,
    layer_input: zml.Tensor,
};

fn mhcPre(
    residual_: zml.Tensor,
    weights: MhcWeights,
    norm: RmsNorm,
    rms_eps: f32,
    hc_pre_eps: f32,
    hc_sinkhorn_eps: f32,
    hc_post_alpha: f32,
    sinkhorn_repeat: u32,
) MhcPreOutput {
    const residual = residual_.withPartialTags(.{ .s, .hc, .d });
    const hc_mult = residual.dim(.hc);
    const flat = residual.convert(.f32).merge(.{ .hcd = .{ .hc, .d } });
    const fn_ = weights.fn_.withTags(.{ .mix, .hcd }).convert(.f32);
    const sqrsum = flat.mul(flat).sum(.hcd).squeeze(.hcd);
    const norm_scale = sqrsum.divByConst(flat.dim(.hcd)).addConstant(rms_eps).rsqrt();
    const mixes = flat.dot(fn_, .hcd).mul(norm_scale.broad(.init(.{
        .s = residual.dim(.s),
        .mix = fn_.dim(.mix),
    }, .f32)));

    const scale = weights.scale.withTags(.{.scale}).convert(.f32);
    const base = weights.base.withTags(.{.mix}).convert(.f32);

    const pre_logits = mixes.slice1d(.mix, .{ .start = 0, .end = hc_mult }).rename(.{ .mix = .hc })
        .mul(scalarAt(scale, .scale, 0).broad(.init(.{ .s = residual.dim(.s), .hc = hc_mult }, .f32)))
        .add(base.slice1d(.mix, .{ .start = 0, .end = hc_mult }).rename(.{ .mix = .hc }).broad(.init(.{ .s = residual.dim(.s), .hc = hc_mult }, .f32)));
    const pre_mix = pre_logits.sigmoid().addConstant(hc_pre_eps);

    const post_logits = mixes.slice1d(.mix, .{ .start = hc_mult, .end = 2 * hc_mult }).rename(.{ .mix = .hc })
        .mul(scalarAt(scale, .scale, 1).broad(.init(.{ .s = residual.dim(.s), .hc = hc_mult }, .f32)))
        .add(base.slice1d(.mix, .{ .start = hc_mult, .end = 2 * hc_mult }).rename(.{ .mix = .hc }).broad(.init(.{ .s = residual.dim(.s), .hc = hc_mult }, .f32)));
    const post_mix = post_logits.sigmoid().scale(hc_post_alpha);

    const res_start = 2 * hc_mult;
    const res_logits = mixes.slice1d(.mix, .{ .start = res_start, .end = res_start + hc_mult * hc_mult })
        .mul(scalarAt(scale, .scale, 2).broad(.init(.{ .s = residual.dim(.s), .mix = hc_mult * hc_mult }, .f32)))
        .add(base.slice1d(.mix, .{ .start = res_start, .end = res_start + hc_mult * hc_mult }).broad(.init(.{ .s = residual.dim(.s), .mix = hc_mult * hc_mult }, .f32)))
        .reshape(.{ .s = residual.dim(.s), .hc_in = hc_mult, .hc_out = hc_mult });
    var res_mix = res_logits.softmax(.hc_out).addConstant(hc_sinkhorn_eps);
    res_mix = res_mix.div(res_mix.sum(.hc_in).addConstant(hc_sinkhorn_eps).broad(res_mix.shape()));
    var i: u32 = 1;
    while (i < sinkhorn_repeat) : (i += 1) {
        res_mix = res_mix.div(res_mix.sum(.hc_out).addConstant(hc_sinkhorn_eps).broad(res_mix.shape()));
        res_mix = res_mix.div(res_mix.sum(.hc_in).addConstant(hc_sinkhorn_eps).broad(res_mix.shape()));
    }

    const mixed = residual.convert(.f32)
        .mul(pre_mix.insertAxes(.hc, .{.d}).broad(residual.shape()))
        .sum(.hc)
        .squeeze(.hc)
        .convert(residual.dtype());
    return .{
        .post_mix = post_mix,
        .res_mix = res_mix,
        .layer_input = norm.forward(mixed),
    };
}

fn mhcPost(x_: zml.Tensor, residual_: zml.Tensor, post_mix_: zml.Tensor, res_mix_: zml.Tensor) zml.Tensor {
    const x = x_.withPartialTags(.{ .s, .d }).convert(.f32);
    const residual = residual_.withPartialTags(.{ .s, .hc, .d }).convert(.f32);
    const post_mix = post_mix_.withPartialTags(.{ .s, .hc }).convert(.f32);
    const res_mix = res_mix_.withPartialTags(.{ .s, .hc_in, .hc_out }).convert(.f32);

    const residual_in = residual.rename(.{ .hc = .hc_in });
    const mixed_residual = res_mix.transpose(.{ .s, .hc_out, .hc_in })
        .dot(residual_in, .hc_in)
        .rename(.{ .hc_out = .hc });
    const mixed_x = x.insertAxes(.d, .{.hc})
        .broad(mixed_residual.shape())
        .mul(post_mix.insertAxes(.hc, .{.d}).broad(mixed_residual.shape()));
    return mixed_x.add(mixed_residual).convert(residual_.dtype());
}

pub const Attention = struct {
    wq_a: zml.nn.Linear,
    wkv: zml.nn.Linear,
    q_norm: RmsNorm,
    kv_norm: RmsNorm,
    wq_b: zml.nn.Linear,
    wo_a: zml.Tensor,
    wo_b: zml.nn.Linear,
    num_heads: u32,
    head_dim: u32,
    rope_head_dim: u32,
    o_groups: u32,
    o_lora_rank: u32,
    rope_opts: zml.nn.RopeOpts,

    pub fn init(store: zml.io.TensorStore.View, config: Config) Attention {
        var rope_scaling = config.rope_scaling;
        rope_scaling.setRopeTheta(config.rope_theta);
        return .{
            .wq_a = linear(store, "wq_a", .{ .dout = .replicated, .d = .replicated }),
            .wkv = linear(store, "wkv", .{ .dout = .replicated, .d = .replicated }),
            .q_norm = .init(store.withPrefix("q_norm"), config.rms_norm_eps),
            .kv_norm = .init(store.withPrefix("kv_norm"), config.rms_norm_eps),
            .wq_b = linear(store, "wq_b", .{ .dout = .model, .d = .replicated }),
            .wo_a = store.withPrefix("wo_a").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
            .wo_b = linear(store, "wo_b", .{ .dout = .replicated, .d = .model }),
            .num_heads = config.num_attention_heads,
            .head_dim = config.head_dim,
            .rope_head_dim = config.qk_rope_head_dim,
            .o_groups = config.o_groups,
            .o_lora_rank = config.o_lora_rank,
            .rope_opts = .{
                .layout = .sequential,
                .scaling = rope_scaling,
            },
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Attention)) void {
        unloadLinear(&self.wq_a);
        unloadLinear(&self.wkv);
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.kv_norm);
        unloadLinear(&self.wq_b);
        self.wo_a.deinit();
        unloadLinear(&self.wo_b);
    }

    pub fn forward(self: Attention, x_: zml.Tensor, positions_: zml.Tensor) zml.Tensor {
        const x = x_.withPartialTags(.{ .s, .d }).withPartitioning(.{ .d = .replicated });
        const positions = positions_.withPartialTags(.{.s});
        var qr = self.wq_a.forward(x.convert(self.wq_a.weight.dtype())).rename(.{ .dout = .d });
        qr = self.q_norm.forward(qr).convert(self.wq_b.weight.dtype());

        var q = self.wq_b.forward(qr)
            .splitAxis(.dout, .{ .h = self.num_heads, .hd = self.head_dim })
            .withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });

        var kv = self.kv_norm.forward(self.wkv.forward(x.convert(self.wkv.weight.dtype())).rename(.{ .dout = .d }));
        kv = kv.rename(.{ .d = .hd });
        var k = kv.insertAxes(.hd, .{.h}).broad(q.shape()).withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        var v = k;

        q = applyTrailingRope(q, positions, self.rope_head_dim, self.rope_opts);
        k = applyTrailingRope(k, positions, self.rope_head_dim, self.rope_opts);

        const attn = zml.nn.sdpa(
            q.rename(.{ .s = .q }).convert(.f32),
            k.rename(.{ .s = .k }).convert(.f32),
            v.rename(.{ .s = .k }).convert(.f32),
            .{},
        )
            .rename(.{ .q = .s })
            .withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });

        const projected_attn = applyTrailingInverseRope(attn, positions, self.rope_head_dim, self.rope_opts);
        const heads_per_group = @divExact(self.num_heads, self.o_groups);
        const grouped = projected_attn
            .splitAxis(.h, .{ .g = self.o_groups, .gh = heads_per_group })
            .merge(.{ .gd = .{ .gh, .hd } });
        const wo_a = self.wo_a.withTags(.{ .dout, .d })
            .reshape(.{ .g = self.o_groups, .orank = self.o_lora_rank, .gd = grouped.dim(.gd) })
            .convert(projected_attn.dtype());
        const compressed = grouped.dot(wo_a, .gd)
            .merge(.{ .d = .{ .g, .orank } })
            .convert(self.wo_b.weight.dtype());
        return self.wo_b.forward(compressed).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
    }
};

pub const Moe = struct {
    gate: Router,
    experts: []Expert,
    shared_experts: ?Mlp,
    routed_scaling_factor: f32,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, global_layer_index: u32) !Moe {
        _ = global_layer_index;
        if (!std.mem.eql(u8, config.scoring_func, "sqrtsoftplus")) return error.UnsupportedDeepSeekRouter;

        const experts = try allocator.alloc(Expert, config.n_routed_experts);
        errdefer allocator.free(experts);
        for (experts, 0..) |*expert, i| {
            expert.* = .init(store.withPrefix("experts").withLayer(i));
        }

        return .{
            .gate = .init(store.withPrefix("gate"), config),
            .experts = experts,
            .shared_experts = if ((config.n_shared_experts orelse 0) > 0) .init(store.withPrefix("shared_experts")) else null,
            .routed_scaling_factor = config.routed_scaling_factor,
        };
    }

    pub fn deinit(self: Moe, allocator: std.mem.Allocator) void {
        allocator.free(self.experts);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Moe), allocator: std.mem.Allocator) void {
        Router.unloadBuffers(&self.gate);
        for (self.experts) |*expert| Expert.unloadBuffers(expert);
        allocator.free(self.experts);
        if (self.shared_experts) |*shared| Mlp.unloadBuffers(shared);
    }

    pub fn forward(self: Moe, x_: zml.Tensor, input_ids: zml.Tensor, moe_metadata: zml.moe.Metadata, moe_parameters: zml.moe.Parameters) zml.Tensor {
        _ = input_ids;
        const x = x_.withPartialTags(.{ .s, .d }).insertAxes(.s, .{.b});
        const routing_scores, const topk_ids = self.gate.forward(x);

        var gate_up_parts: [512]zml.Tensor = undefined;
        var down_parts: [512]zml.Tensor = undefined;
        stdx.debug.assert(self.experts.len <= gate_up_parts.len, "DSpark example supports up to {} routed experts, got {}", .{ gate_up_parts.len, self.experts.len });
        for (self.experts, 0..) |expert, i| {
            gate_up_parts[i] = expert.gateUp();
            down_parts[i] = expert.down();
        }

        const gate_up = stackMany(gate_up_parts[0..self.experts.len], 0, .expert)
            .withPartitioning(.{ .expert = .experts, .out = .replicated, .in = .replicated });
        const down = stackMany(down_parts[0..self.experts.len], 0, .expert)
            .withPartitioning(.{ .expert = .experts, .out = .replicated, .mid = .replicated });

        var out = zml.moe.forwardMoe(
            x,
            topk_ids,
            routing_scores,
            gate_up,
            null,
            null,
            down,
            null,
            null,
            moe_metadata,
            moe_parameters,
        ) catch |err| stdx.debug.panic("moe backend failed: {}", .{err});

        if (self.shared_experts) |shared| {
            out = out.add(shared.forward(x).rename(.{ .dout = .d }));
        }
        return out.squeeze(.b);
    }
};

pub const Router = struct {
    gate: zml.nn.Linear,
    bias: ?zml.Tensor,
    num_experts_per_tok: u32,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config) Router {
        return .{
            .gate = .init(
                store.createTensor("weight", .{ .expert, .d }, .{ .expert = .replicated, .d = .replicated }),
                null,
                .d,
            ),
            .bias = store.maybeCreateTensor("bias", .{.expert}, .replicated),
            .num_experts_per_tok = config.num_experts_per_tok,
            .norm_topk_prob = config.norm_topk_prob,
            .routed_scaling_factor = config.routed_scaling_factor,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Router)) void {
        self.gate.weight.deinit();
        if (self.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: Router, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const logits = self.gate.forward(x).convert(.f32);
        const scores = softplus(logits).sqrt();
        const scores_for_choice = if (self.bias) |bias| scores.add(bias.convert(.f32).broad(scores.shape())) else scores;
        const topk = scores_for_choice.topK(.{ .top_expert = .expert }, self.num_experts_per_tok, .{});
        const topk_ids = topk.indices.convert(.i32);
        var topk_weights = scores.gather(.{ .expert = topk.indices }, .{});
        if (self.norm_topk_prob) {
            topk_weights = topk_weights.div(topk_weights.sum(.top_expert).broad(topk_weights.shape()).addConstant(1e-20));
        }
        return .{ topk_weights.scale(self.routed_scaling_factor), topk_ids };
    }
};

pub const Expert = struct {
    w1: zml.nn.Linear,
    w2: zml.nn.Linear,
    w3: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Expert {
        return .{
            .w1 = linear(store, "w1", .{ .dout = .replicated, .d = .replicated }),
            .w2 = linear(store, "w2", .{ .dout = .replicated, .d = .replicated }),
            .w3 = linear(store, "w3", .{ .dout = .replicated, .d = .replicated }),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Expert)) void {
        unloadLinear(&self.w1);
        unloadLinear(&self.w2);
        unloadLinear(&self.w3);
    }

    pub fn gateUp(self: Expert) zml.Tensor {
        return zml.Tensor.concatenate(&.{
            self.w1.weight.withTags(.{ .out, .in }),
            self.w3.weight.withTags(.{ .out, .in }),
        }, .out);
    }

    pub fn down(self: Expert) zml.Tensor {
        return self.w2.weight.withTags(.{ .out, .mid });
    }
};

pub const Mlp = struct {
    w1: zml.nn.Linear,
    w2: zml.nn.Linear,
    w3: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .w1 = linear(store, "w1", .{ .dout = .model, .d = .replicated }),
            .w2 = linear(store, "w2", .{ .dout = .replicated, .d = .model }),
            .w3 = linear(store, "w3", .{ .dout = .model, .d = .replicated }),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        unloadLinear(&self.w1);
        unloadLinear(&self.w2);
        unloadLinear(&self.w3);
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        return self.w2.forward(
            self.w1.forward(x.convert(self.w1.weight.dtype()))
                .silu()
                .mul(self.w3.forward(x.convert(self.w3.weight.dtype())))
                .rename(.{ .dout = .d }),
        );
    }
};

pub const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{
            .weight = store.createTensor("weight", .{.d}, .{ .d = .replicated }),
            .eps = eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        return zml.nn.rmsNorm(x, .d, self.eps)
            .mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }
};

fn linear(store: zml.io.TensorStore.View, comptime prefix: []const u8, partitioning: anytype) zml.nn.Linear {
    return .init(
        store.withPrefix(prefix).createTensor("weight", .{ .dout, .d }, partitioning),
        null,
        .d,
    );
}

fn qwenLinear(store: zml.io.TensorStore.View, partitioning: anytype, bias_partitioning: anytype) zml.nn.Linear {
    return .init(
        store.createTensor("weight", .{ .dout, .d }, partitioning),
        store.maybeCreateTensor("bias", .{.dout}, bias_partitioning),
        .d,
    );
}

fn unloadLinear(linear_: anytype) void {
    linear_.weight.deinit();
    if (linear_.bias) |*bias| bias.deinit();
}

fn scalarAt(tensor: zml.Tensor, axis: anytype, index: i64) zml.Tensor {
    return tensor.slice1d(axis, .single(index)).squeeze(axis);
}

fn softplus(x: zml.Tensor) zml.Tensor {
    return x.exp().addConstant(1).log();
}

fn applyLeadingRope(x: zml.Tensor, positions: zml.Tensor, rope_dim: u32, rope_opts: zml.nn.RopeOpts) zml.Tensor {
    if (rope_dim == x.dim(.hd)) {
        return zml.nn.rope(x, positions, rope_opts);
    }
    const x_rot = x.slice1d(.hd, .{ .start = 0, .end = rope_dim });
    const x_pass = x.slice1d(.hd, .{ .start = rope_dim, .end = x.dim(.hd) });
    return zml.Tensor.concatenate(&.{ zml.nn.rope(x_rot, positions, rope_opts), x_pass }, .hd);
}

fn applyTrailingRope(x: zml.Tensor, positions: zml.Tensor, rope_dim: u32, rope_opts: zml.nn.RopeOpts) zml.Tensor {
    if (rope_dim == x.dim(.hd)) {
        return zml.nn.rope(x, positions, rope_opts);
    }
    const pass_dim = x.dim(.hd) - rope_dim;
    const x_pass = x.slice1d(.hd, .{ .start = 0, .end = pass_dim });
    const x_rot = x.slice1d(.hd, .{ .start = pass_dim, .end = x.dim(.hd) });
    return zml.Tensor.concatenate(&.{ x_pass, zml.nn.rope(x_rot, positions, rope_opts) }, .hd);
}

fn applyTrailingInverseRope(x: zml.Tensor, positions: zml.Tensor, rope_dim: u32, rope_opts: zml.nn.RopeOpts) zml.Tensor {
    return applyTrailingRope(x, positions.convert(.f32).negate(), rope_dim, rope_opts);
}

fn stackMany(tensors: []const zml.Tensor, axis_: anytype, tag: anytype) zml.Tensor {
    stdx.debug.assert(tensors.len > 0, "stackMany requires at least one tensor", .{});
    stdx.debug.assert(tensors.len <= 1024, "stackMany supports up to 1024 tensors, got {}", .{tensors.len});

    const shape0 = tensors[0].shape();
    const res_shape = shape0.insertTag(axis_, 1, zml.Shape.toTag(tag));
    const axis = if (@TypeOf(axis_) == @EnumLiteral() and axis_ == .last)
        shape0.rank()
    else
        shape0.axis(axis_);

    var groups: [32]zml.Tensor = undefined;
    var group_count: usize = 0;
    var i: usize = 0;
    while (i < tensors.len) {
        var chunk: [32]zml.Tensor = undefined;
        const chunk_len = @min(@as(usize, 32), tensors.len - i);
        for (0..chunk_len) |j| {
            chunk[j] = tensors[i + j].reshape(res_shape);
        }
        groups[group_count] = zml.Tensor.concatenate(chunk[0..chunk_len], axis);
        group_count += 1;
        i += chunk_len;
    }
    return zml.Tensor.concatenate(groups[0..group_count], axis);
}
