const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;
const common = @import("../common.zig");

pub const KvCache = common.KvCache;

pub const Config = struct {
    model_type: []const u8 = "",
    architectures: []const []const u8 = &.{},
    vocab_size: u32,
    bos_token_id: u32,
    eos_token_id: stdx.json.Union(union(enum) {
        int: u32,
        ints: []u32,
    }),
    pad_token_id: ?u32 = null,
    head_dim: ?u32 = null,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    attention_bias: bool = false,
    hidden_act: []const u8 = "silu",
    rope_theta: f32 = 1_000_000,
    rope_parameters: RopeParameters = .{},
    max_position_embeddings: u32,
    rms_norm_eps: f32,
    hf_rope_impl: bool = true,
    tie_word_embeddings: bool = true,
    rope_scaling: ?zml.nn.RopeOpts.Scaling = null,

    pub fn ropeTheta(self: Config) f32 {
        return self.rope_parameters.rope_theta orelse self.rope_theta;
    }
};

pub const RopeParameters = struct {
    rope_theta: ?f32 = null,
    partial_rotary_factor: f32 = 1.0,
    rope_type: []const u8 = "default",
};

pub const Buffers = zml.Bufferized(Model);

pub const SamplingConfig = struct {
    temperature: f32 = 0.0,
};

pub fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !zml.tokenizer.Tokenizer {
    const file = try dir.openFile(io, "tokenizer.json", .{});
    defer file.close(io);

    var reader = file.reader(io, &.{});
    const bytes = try reader.interface.readAlloc(allocator, try file.length(io));
    defer allocator.free(bytes);

    return try .fromBytes(allocator, bytes);
}

pub fn tokenizePrompt(
    allocator: std.mem.Allocator,
    tokenizer: *zml.tokenizer.Tokenizer,
    config: Config,
    prompt: []const u8,
) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    _ = config;
    const im_start = tokenizer.tokenId("<|im_start|>") orelse return error.NoSuchToken;
    const im_end = tokenizer.tokenId("<|im_end|>") orelse return error.NoSuchToken;
    const newline = tokenizer.tokenId("\\n") orelse return error.NoSuchToken;

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, 32);
    try tokens.append(allocator, im_start);
    const user_tokens = try encoder.encodeAlloc(allocator, "user\n");
    defer allocator.free(user_tokens);
    try tokens.appendSlice(allocator, user_tokens);
    const prompt_tokens = try encoder.encodeAlloc(allocator, prompt);
    defer allocator.free(prompt_tokens);
    try tokens.appendSlice(allocator, prompt_tokens);
    try tokens.appendSlice(allocator, &.{ im_end, newline, im_start });
    const assistant_tokens = try encoder.encodeAlloc(allocator, "assistant\n");
    defer allocator.free(assistant_tokens);
    try tokens.appendSlice(allocator, assistant_tokens);

    return tokens.toOwnedSlice(allocator);
}

pub fn isEosToken(config: *const Config, token_id: u32) bool {
    return switch (config.eos_token_id.value) {
        .int => |eos| token_id == eos,
        .ints => |eos_list| for (eos_list) |eos| {
            if (token_id == eos) break true;
        } else false,
    };
}

/// Qwen3 architecture, using huggingface transformers naming.
/// Dimensions of activations: {.b, .s, .d}
pub const Model = struct {
    lm_head: ?zml.nn.Linear,
    model: Qwen3,

    pub fn init(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: Config,
    ) !Model {
        const lm_head: ?zml.nn.Linear = if (store.withPrefix("lm_head").maybeCreateTensor(
            "weight",
            .{ .dout, .d },
            .{ .dout = .model, .d = .replicated },
        )) |weight|
            .init(weight, null, .d)
        else
            null;

        return .{
            .lm_head = lm_head,
            .model = try .init(allocator, store.withPrefix("model"), config),
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
    }

    pub fn initKvCache(self: Model, config: Config, cache_seq_len: u32) KvCache {
        return .init(.init(.{
            .layer = config.num_hidden_layers,
            .k = cache_seq_len,
            .h = config.num_key_value_heads,
            .hd = config.head_dim orelse config.hidden_size / config.num_attention_heads,
        }, self.model.embed_tokens.weight.dtype()));
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
            .dma_chunk_size = 128 * zml.MiB,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model), allocator: std.mem.Allocator) void {
        if (self.lm_head) |*lm_head| lm_head.weight.deinit();
        Qwen3.unloadBuffers(&self.model, allocator);
    }

    pub fn logitsForward(self: Model, out_: zml.Tensor) zml.Tensor {
        const out = out_.withPartialTags(.{ .s, .d });

        var logits = blk: {
            if (self.lm_head) |lm_head| {
                break :blk lm_head.forward(out).rename(.{ .dout = .voc });
            } else {
                break :blk self.model.embed_tokens.weight.withTags(.{ .voc, .d }).dot(out, .d);
            }
        };

        if (logits.shape().hasTag(.voc) == null)
            logits = logits.rename(.{ .d = .voc });

        return logits;
    }

    fn sampleTargetTokens(self: Model, out_: zml.Tensor, sampling: SamplingConfig, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        const logits = self.logitsForward(out_);
        return sampleLogits(logits, sampling, rng);
    }

    fn sampleLogits(logits_: zml.Tensor, sampling: SamplingConfig, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        const logits = logits_.withPartialTags(.{.voc});
        const topk: u32 = if (sampling.temperature < 0.00001) 1 else @intCast(logits.dim(.voc));
        const temperature: f32 = if (sampling.temperature < 0.00001) 1.0 else sampling.temperature;
        const tokens, const updated_rng = zml.nn.sampleTokens(logits, .{ .topk = topk, .temperature = temperature }, rng);
        return .{ tokens.convert(.u32), updated_rng };
    }

    fn sampleLastTargetToken(self: Model, out_: zml.Tensor, sampling: SamplingConfig, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        const out = out_.withPartialTags(.{ .s, .d });
        const last = out.slice1d(.s, .single(out.dim(.s) - 1)).reshape(.{ .s = 1, .d = out.dim(.d) });
        return self.sampleTargetTokens(last, sampling, rng);
    }

    pub fn prefillForward(
        self: Model,
        tokens_: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
        target_layer_ids: []const u32,
        sampling: SamplingConfig,
        rng: zml.Tensor.Rng,
    ) struct { zml.Tensor, zml.Tensor, KvCache, zml.Tensor.Rng } {
        const target_hidden, const out, const updated_kv_cache = self.model.forward(
            tokens_.withPartialTags(.{.s}),
            token_index,
            kv_cache,
            attention_metadata,
            attention_parameters,
            target_layer_ids,
        );

        const sampled_last, const updated_rng = self.sampleLastTargetToken(out, sampling, rng);
        return .{ target_hidden, sampled_last, updated_kv_cache, updated_rng };
    }

    pub fn verifyForward(
        self: Model,
        tokens_: zml.Tensor,
        draft_logits_: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
        target_layer_ids: []const u32,
        sampling: SamplingConfig,
        rng: zml.Tensor.Rng,
    ) struct { zml.Tensor, zml.Tensor, zml.Tensor, KvCache, zml.Tensor.Rng } {
        const target_hidden, const out, const updated_kv_cache = self.model.forward(
            tokens_.withPartialTags(.{.s}),
            token_index,
            kv_cache,
            attention_metadata,
            attention_parameters,
            target_layer_ids,
        );

        const target_logits = self.logitsForward(out);
        const valid_draft_tokens, const correction_token, const updated_rng = verifyDraftTokens(
            tokens_,
            draft_logits_,
            target_logits,
            sampling,
            rng,
        );
        return .{ target_hidden, valid_draft_tokens, correction_token, updated_kv_cache, updated_rng };
    }

    pub fn embedForward(self: Model, tokens_: zml.Tensor) zml.Tensor {
        return self.model.embed_tokens.forward(tokens_.withPartialTags(.{.s})).withPartialTags(.{.d});
    }
};

fn verifyDraftTokens(
    draft_tokens_: zml.Tensor,
    draft_logits_: zml.Tensor,
    target_logits_: zml.Tensor,
    sampling: SamplingConfig,
    rng: zml.Tensor.Rng,
) struct { zml.Tensor, zml.Tensor, zml.Tensor.Rng } {
    const draft_tokens = draft_tokens_.withPartialTags(.{.s});
    const draft_logits = draft_logits_.withPartialTags(.{ .s, .voc });
    const target_logits = target_logits_.withPartialTags(.{ .s, .voc });
    const draft_len = draft_tokens.dim(.s);
    const verify_len = draft_len - 1;

    if (sampling.temperature < 0.00001) {
        const sampled_tokens, const updated_rng = Model.sampleLogits(target_logits, sampling, rng);
        const proposals = draft_tokens.slice1d(.s, .{ .start = 1, .end = draft_len });
        const posterior = sampled_tokens.slice1d(.s, .{ .start = 0, .end = verify_len });
        const matches = proposals.cmp(.EQ, posterior);
        const candidate_idx = zml.Tensor.iota(.init(.{ .s = verify_len }, .u32), .s).convert(.u32);
        const sentinel = zml.Tensor.scalar(@as(u32, @intCast(verify_len)), .u32).broad(candidate_idx.shape());
        const valid_draft_tokens = matches.select(sentinel, candidate_idx).min(.s).squeeze(.s);
        const correction_token = sampled_tokens.gather(.{ .s = valid_draft_tokens }, .{}).convert(.u32);
        return .{ valid_draft_tokens, correction_token, updated_rng };
    }

    const proposals = draft_tokens.slice1d(.s, .{ .start = 1, .end = draft_len });
    const draft_proposal_logits = draft_logits.slice1d(.s, .{ .start = 1, .end = draft_len });
    const target_proposal_logits = target_logits.slice1d(.s, .{ .start = 0, .end = verify_len });
    const draft_probs = tokenProbs(draft_proposal_logits, sampling);
    const target_probs = tokenProbs(target_proposal_logits, sampling);

    const p = target_probs.gather(.{ .voc = proposals }, .{});
    const q = draft_probs.gather(.{ .voc = proposals }, .{});
    const epsilon = zml.Tensor.scalar(std.math.floatEps(f32), .f32).broad(q.shape());
    const acceptance = p.div(q.maximum(epsilon));

    const accept_rng, const u = rng.uniform(acceptance.shape().withDtype(.f32), .{ .min = std.math.floatEps(f32), .max = 1 });
    const accepted = acceptance.cmp(.GT, u);
    const candidate_idx = zml.Tensor.iota(.init(.{ .s = verify_len }, .u32), .s).convert(.u32);
    const sentinel = zml.Tensor.scalar(@as(u32, @intCast(verify_len)), .u32).broad(candidate_idx.shape());
    const valid_draft_tokens = accepted.select(sentinel, candidate_idx).min(.s).squeeze(.s);

    const residual_probs = target_probs.sub(draft_probs).relu();
    const residual_idx = valid_draft_tokens.minimum(zml.Tensor.scalar(@as(u32, @intCast(verify_len - 1)), .u32));
    const residual = residual_probs.gather(.{ .s = residual_idx }, .{});
    const residual_logits = residual.log();

    const no_rejection = valid_draft_tokens.cmp(.EQ, zml.Tensor.scalar(@as(u32, @intCast(verify_len)), .u32));
    const target_correction_logits = target_logits.gather(.{ .s = valid_draft_tokens }, .{}).convert(.f32).scale(1 / sampling.temperature);
    const correction_logits = no_rejection.broad(target_correction_logits.shape()).select(target_correction_logits, residual_logits);
    const correction_token, const updated_rng = Model.sampleLogits(correction_logits, .{ .temperature = 1.0 }, accept_rng);
    return .{ valid_draft_tokens, correction_token.convert(.u32), updated_rng };
}

fn tokenProbs(logits_: zml.Tensor, sampling: SamplingConfig) zml.Tensor {
    const logits = logits_.withPartialTags(.{.voc}).convert(.f32);
    return logits.scale(1 / sampling.temperature).softmax(.voc);
}

pub const Qwen3 = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    norm: RmsNorm,
    layers: []TransformerLayer,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Qwen3 {
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = try .init(store.withPrefix("layers").withLayer(i), config);
        }

        return .{
            .embed_tokens = .{ .weight = store.createTensor(
                "embed_tokens.weight",
                .{ .voc, .d },
                .{ .voc = .replicated, .d = .model },
            ) },
            .norm = .{
                .weight = store.withPrefix("norm").createTensor("weight", .{.d}, .{ .d = .replicated }),
                .eps = config.rms_norm_eps,
            },
            .layers = layers,
        };
    }

    pub fn deinit(self: Qwen3, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Qwen3), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        RmsNorm.unloadBuffers(&self.norm);
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
    }

    pub fn forward(
        self: Qwen3,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
        target_layer_ids: []const u32,
    ) struct { zml.Tensor, zml.Tensor, KvCache } {
        const embeds = embedForward(self.embed_tokens, tokens);
        var hidden = embeds;
        var updated_kv_cache = kv_cache;

        var selected: [32]zml.Tensor = undefined;
        var selected_len: usize = 0;
        for (self.layers, 0..) |layer, i| {
            hidden, updated_kv_cache = layer.forward(
                hidden,
                token_index,
                updated_kv_cache.atLayer(i),
                attention_metadata,
                attention_parameters,
            );

            for (target_layer_ids) |target_layer_id| {
                if (target_layer_id == i) {
                    selected[selected_len] = hidden;
                    selected_len += 1;
                    break;
                }
            }
        }

        const target_hidden = zml.Tensor.concatenate(selected[0..selected_len], .d);
        const out = self.norm.forward(hidden);
        return .{ target_hidden, out, updated_kv_cache.reuseBuffer(kv_cache) };
    }
};

fn embedForward(embed_tokens_: zml.nn.TokenEmbedding, tokens_: zml.Tensor) zml.Tensor {
    return embed_tokens_.forward(tokens_).withPartialTags(.{.d});
}

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: SelfAttn,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,

    pub fn init(store: zml.io.TensorStore.View, config: Config) !TransformerLayer {
        return .{
            .input_layernorm = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps),
            .self_attn = try .init(store.withPrefix("self_attn"), config),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps),
            .mlp = .init(store.withPrefix("mlp")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        SelfAttn.unloadBuffers(&self.self_attn);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
        Mlp.unloadBuffers(&self.mlp);
    }

    pub fn forward(
        self: TransformerLayer,
        x0: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { zml.Tensor, KvCache } {
        // Self Attention
        //log.debug("TransformerLayer({f}) -> {f}", .{ x0, self.input_layernorm.forward(x0) });
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {f}", .{x0});

        // Keep the residual stream replicated to avoid repeated gathers before q/k/v.
        const x0_replicated = x0.withPartitioning(.{ .d = .replicated });
        const x0_normalized = self.input_layernorm.forward(x0_replicated);
        const delta0, const updated_kv_cache = self.self_attn.forward(
            x0_normalized,
            token_index,
            kv_cache,
            attention_metadata,
            attention_parameters,
        );

        // Fully Connected
        const x1 = x0_replicated.add(delta0).withPartitioning(.{ .d = .replicated });
        const x1_normalized = self.post_attention_layernorm.forward(x1);
        const x2 = self.mlp.forward(x1_normalized)
            .rename(.{ .dout = .d })
            .withPartitioning(.{ .d = .replicated })
            .add(x1)
            .withPartitioning(.{ .d = .replicated });

        return .{ x2.reuseBuffer(x0), updated_kv_cache };
    }
};

const RmsNorm = struct {
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

    /// L2 normalization of input tensor along `.d` axis.
    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const normalized = zml.nn.rmsNorm(x, .d, self.eps);
        return normalized.mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }
};

const Mlp = struct {
    up_proj: zml.nn.Linear, // (dim -> hidden_dim)
    gate_proj: zml.nn.Linear, // (dim -> hidden_dim)
    down_proj: zml.nn.Linear, // (hidden_dim -> dim)

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .up_proj = .init(store.createTensor("up_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .gate_proj = .init(store.createTensor("gate_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .down_proj = .init(store.createTensor("down_proj.weight", .{ .dout, .d }, .{ .d = .model }), null, .d),
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

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const proj = self.up_proj.forward(x);
        var output = self.gate_proj.forward(x);
        output = output.silu().mul(proj).rename(.{ .dout = .d });
        return self.down_proj.forward(output);
    }
};

pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,

    q_norm: RmsNorm,
    k_norm: RmsNorm,

    o_proj: zml.nn.Linear,
    num_heads: i64 = undefined,
    num_kv_heads: i64 = 0,
    rope_opts: zml.nn.RopeOpts = undefined,

    fn initProj(
        store: zml.io.TensorStore.View,
        partitioning: anytype,
        bias_partitioning: anytype,
        with_bias: bool,
    ) zml.nn.Linear {
        return .init(
            store.createTensor("weight", .{ .dout, .d }, partitioning),
            if (with_bias) store.createTensor("bias", .{.dout}, bias_partitioning) else null,
            .d,
        );
    }

    pub fn init(store: zml.io.TensorStore.View, config: Config) !SelfAttn {
        var rope_scaling = config.rope_scaling orelse .{ .default = .{} };
        rope_scaling.setRopeTheta(config.ropeTheta());
        return .{
            .q_proj = initProj(store.withPrefix("q_proj"), .{ .dout = .model }, .{ .dout = .model }, config.attention_bias),
            .k_proj = initProj(store.withPrefix("k_proj"), .{ .dout = .model }, .{ .dout = .model }, config.attention_bias),
            .v_proj = initProj(store.withPrefix("v_proj"), .{ .dout = .model }, .{ .dout = .model }, config.attention_bias),
            .o_proj = initProj(store.withPrefix("o_proj"), .{ .d = .model }, .{ .dout = .replicated }, config.attention_bias),
            .q_norm = .init(store.withPrefix("q_norm"), config.rms_norm_eps),
            .k_norm = .init(store.withPrefix("k_norm"), config.rms_norm_eps),
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
            .rope_opts = .{
                .layout = if (config.hf_rope_impl) .sequential else .interleaved,
                .scaling = rope_scaling,
            },
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

    /// Self zml.attention.attention.
    ///   - If token_index is set, x is assumed to be the representation of one new token,
    /// and kv_cache will be read for the previous tokens.
    ///   - If token_index is not set, x is assumed to be the representation of all tokens
    /// since the beginning of the sequence, and kv_cache won't be read.
    /// In both case, kv_cache will be updated with the computed key and value.
    /// x: {.b, .s, .d } -> .{.b, .s, .d}
    pub fn forward(
        self: SelfAttn,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { zml.Tensor, KvCache } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;

        // Make hidden state replicated once and reuse it across q/k/v projections.
        // This avoids paying gather-style collectives independently for each projection.
        const x_qkv = x.withPartitioning(.{ .d = .replicated });

        var q = self.q_proj.forward(x_qkv).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = self.k_proj.forward(x_qkv).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        var v = self.v_proj.forward(x_qkv).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        q = q.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        k = k.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });

        // In self-attention, .s axis is used both for keys and queries.
        const pos_index = b: {
            const temp = zml.Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const dtype = q.dtype();
        const new_kv_cache = kv_cache.update(k, v, token_index);
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);
        k = k.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });

        const attn_output = zml.attention.attention.attention(
            q,
            k,
            v,
            token_index,
            attention_metadata,
            attention_parameters,
        ).withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated });

        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const delta = self.o_proj.forward(attn)
            .rename(.{ .dout = .d })
            .withPartitioning(.{ .d = .replicated });
        return .{ delta, new_kv_cache };
    }
};
