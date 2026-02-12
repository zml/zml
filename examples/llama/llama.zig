const std = @import("std");
const testing = std.testing;

const zml = @import("zml");
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;
const stdx = zml.stdx;

const log = std.log.scoped(.llama);

pub const TransferCtx = struct {
    allocator: std.mem.Allocator,
    pool: *zml.mem.DynamicBufferPool,
    transferred_bytes: *usize,
    progress: *std.Progress.Node,
};

/// Llama architecture, using huggingface transformers naming.
/// Dimensions of activations: {.b, .s, .d}
pub const LlamaLM = struct {
    pub const Config = struct {
        bos_token_id: u32,
        eos_token_id: stdx.json.Union(union(enum) {
            int: u32,
            ints: []u32,
        }),
        head_dim: ?u32 = null,
        hidden_size: u32,
        num_hidden_layers: u32,
        num_attention_heads: u32,
        num_key_value_heads: u32,
        rope_theta: f32,
        max_position_embeddings: u32,
        rms_norm_eps: f32,
        hf_rope_impl: bool = true,
        tie_word_embeddings: bool = false,
        rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = .{} },
    };

    pub const Options = struct {
        sampling_strategy: ?zml.nn.SamplingStrategy,
        max_seq_len: u32,
    };

    lm_head: ?zml.nn.Linear,
    model: Llama,

    // Options controlling generation
    gen_opts: zml.nn.SamplingStrategy = .{},
    config: Config,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, options: Options) !LlamaLM {
        const lm_head: ?zml.nn.Linear = if (store.withPrefix("lm_head").maybeCreateTensorWithTags("weight", .{ .dout, .d })) |weight|
            .init(weight, null, .d)
        else
            null;

        return .{
            .lm_head = lm_head,
            .model = try .init(allocator, store.withPrefix("model"), config),
            .gen_opts = options.sampling_strategy orelse .{},
            .config = config,
        };
    }

    pub fn deinit(self: LlamaLM, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
    }

    pub fn load(
        self: *const LlamaLM,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(LlamaLM) {
        progress.increaseEstimatedTotalItems(store.view().count());
        var timer: std.time.Timer = try .start();
        var total_bytes: usize = 0;
        defer {
            const took = timer.read();
            log.info("Loaded weights [{Bi:.2}, {D}, {Bi:.2}/s]", .{
                total_bytes,
                took,
                total_bytes / took * std.time.ns_per_s,
            });
        }
        return zml.io.load(LlamaLM, self, allocator, io, platform, .{
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = progress,
            .store = store,
            .parallelism = 16,
            .total_bytes = &total_bytes,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(LlamaLM), allocator: std.mem.Allocator) void {
        if (self.lm_head) |*lm_head| lm_head.weight.deinit();
        Llama.unloadBuffers(&self.model, allocator);
    }

    /// Predicts the token at `token_index` position.
    /// Returns:
    ///  - updated `tokens`,
    ///  - updated KV cache
    ///  - a Rng state to allow for probabilistic generation
    pub fn forward(
        self: LlamaLM,
        tokens_: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        rng: Tensor.Rng,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { Tensor, KvCache, Tensor.Rng } {
        stdx.debug.assert(tokens_.dtype() == .u32 and tokens_.rank() >= 1 and token_index.dtype() == .u32 and token_index.rank() <= 1, "Can't run Llama ! Expected >=1d tokens and 0d token_index, got: {f} and {f}", .{ tokens_, token_index });
        const tokens = tokens_.withPartialTags(.{.s});
        const out, const updated_kv_cache = self.model.forward(tokens, token_index, kv_cache, attention_metadata, attention_parameters);
        const new_tokens, const new_rng = self.sampleTokens(self.lm_head, out, rng, self.gen_opts);
        return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), updated_kv_cache, new_rng };
    }

    pub fn sampleTokens(
        self: LlamaLM,
        lm_head_: ?zml.nn.Linear,
        out_: Tensor,
        rng: Tensor.Rng,
        opts: zml.nn.SamplingStrategy,
    ) struct { Tensor, Tensor.Rng } {
        const out = out_.withPartialTags(.{ .s, .d });

        var logits = blk: {
            if (lm_head_) |lm_head| {
                break :blk lm_head.forward(out).rename(.{ .dout = .d });
            } else {
                break :blk self.model.embed_tokens.weight.withTags(.{ .voc, .d }).dot(out, .d);
            }
        };

        if (logits.shape().hasTag(.voc) == null)
            logits = logits.rename(.{ .d = .voc });

        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, opts, rng);
        return .{ next_tokens, new_rng };
    }
};

pub const Llama = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    norm: RmsNorm,
    layers: []TransformerLayer,

    max_seq_len: u32 = 0,
    num_heads: u32 = 32,
    num_kv_heads: u32 = 32,
    rope_opts: zml.nn.RopeOpts = .{
        .layout = .interleaved,
        .freq_base = 10_000,
    },

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: LlamaLM.Config) !Llama {
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = try .init(store.withPrefix("layers").withLayer(i), config);
        }

        return .{
            .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight") },
            .norm = .{ .weight = store.withPrefix("norm").createTensorWithTags("weight", .{.d}), .eps = config.rms_norm_eps },
            .layers = layers,
        };
    }

    pub fn deinit(self: Llama, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn loadBuffers(
        self: *const Llama,
        bufferize_ctx: zml.io.BufferizeContext(TransferCtx),
        group: *stdx.Io.LimitedGroup,
        store: zml.io.TensorStore.View,
        cb: zml.io.CallbackTensorBufferTransfer(TransferCtx),
    ) !zml.Bufferized(Llama) {
        const bufferized_layers = try bufferize_ctx.allocator.alloc(zml.Bufferized(TransformerLayer), self.layers.len);

        var bufferized: zml.Bufferized(Llama) = .{ .embed_tokens = undefined, .norm = undefined, .layers = bufferized_layers };

        const transfers = try zml.io.bufferize(TransferCtx, bufferize_ctx, &self.embed_tokens, &bufferized.embed_tokens, store.withPrefix("embed_tokens"));
        for (transfers) |t| {
            try group.concurrent(bufferize_ctx.io, cb, .{t});
        }

        const norm_transfers = try zml.io.bufferize(TransferCtx, bufferize_ctx, &self.norm, &bufferized.norm, store.withPrefix("norm"));
        for (norm_transfers) |t| {
            try group.concurrent(bufferize_ctx.io, cb, .{t});
        }

        for (self.layers, bufferized_layers, 0..) |layer, *buf_layer, i| {
            const layer_transfers = try zml.io.bufferize(TransferCtx, bufferize_ctx, &layer, buf_layer, store.withPrefix("layers").withLayer(i));
            for (layer_transfers) |t| {
                try group.concurrent(bufferize_ctx.io, cb, .{t});
            }
        }

        return bufferized;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Llama), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        RmsNorm.unloadBuffers(&self.norm);
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
    }

    /// Forward one token, using KV cache for previous tokens.
    /// Returns result and updated KV cache.
    pub fn forward(
        self: Llama,
        tokens: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { Tensor, KvCache } {
        const embeds = embed(self.embed_tokens, tokens);
        var hidden = embeds;

        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden, updated_kv_cache = layer.forward(hidden, token_index, updated_kv_cache.atLayer(i), attention_metadata, attention_parameters);
        }
        const output = self.norm.forward(hidden);

        return .{ output, updated_kv_cache.reuseBuffer(kv_cache) };
    }

    pub fn embed(embed_tokens_: zml.nn.TokenEmbedding, tokens_: Tensor) Tensor {
        return embed_tokens_.forward(tokens_).withPartialTags(.{.d});
    }
};

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: SelfAttn,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,

    pub fn init(store: zml.io.TensorStore.View, config: LlamaLM.Config) !TransformerLayer {
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
        x0: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { Tensor, KvCache } {
        // Self Attention
        //log.debug("TransformerLayer({f}) -> {f}", .{ x0, self.input_layernorm.forward(x0) });
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {f}", .{x0});

        const x0_normalized = self.input_layernorm.forward(x0);
        const delta0, const updated_kv_cache = self.self_attn.forward(x0_normalized, token_index, kv_cache, attention_metadata, attention_parameters);
        const x1 = x0.add(delta0);

        // Fully Connected
        const x1_normalized = self.post_attention_layernorm.forward(x1);
        const x2 = self.mlp.forward(x1_normalized).add(x1).rename(.{ .dout = .d });

        return .{ x2.reuseBuffer(x0), updated_kv_cache };
    }
};

const RmsNorm = struct {
    weight: Tensor,
    eps: f32 = 1e-5,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{ .weight = store.createTensorWithTags("weight", .{.d}), .eps = eps };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    /// L2 normalization of input tensor along `.d` axis.
    pub fn forward(self: RmsNorm, input: Tensor) Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const normalized = zml.nn.rmsNorm(x, .d, self.eps);
        return normalized.mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }
};

const Mlp = struct {
    up_proj: zml.nn.Linear, // (dim -> hidden_dim)
    gate_proj: zml.nn.Linear, // (dim -> hidden_dim)
    down_proj: zml.nn.Linear, // (hidden_dim -> dim)

    fn initLinear(store: zml.io.TensorStore.View) zml.nn.Linear {
        return .init(store.createTensorWithTags("weight", .{ .dout, .d }), store.maybeCreateTensorWithTags("bias", .{.dout}), .d);
    }

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .up_proj = initLinear(store.withPrefix("up_proj")),
            .gate_proj = initLinear(store.withPrefix("gate_proj")),
            .down_proj = initLinear(store.withPrefix("down_proj")),
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

    q_norm: ?RmsNorm,
    k_norm: ?RmsNorm,

    o_proj: zml.nn.Linear,
    num_heads: i64 = undefined,
    num_kv_heads: i64 = 0,
    rope_opts: zml.nn.RopeOpts = undefined,

    fn initProj(store: zml.io.TensorStore.View) zml.nn.Linear {
        return .init(store.createTensorWithTags("weight", .{ .dout, .d }), store.maybeCreateTensorWithTags("bias", .{.dout}), .d);
    }

    pub fn init(store: zml.io.TensorStore.View, config: LlamaLM.Config) !SelfAttn {
        return .{
            .q_proj = initProj(store.withPrefix("q_proj")),
            .k_proj = initProj(store.withPrefix("k_proj")),
            .v_proj = initProj(store.withPrefix("v_proj")),
            .o_proj = initProj(store.withPrefix("o_proj")),
            // TODO(Corentin): fix that
            .q_norm = null,
            .k_norm = null,
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
            .rope_opts = .{
                .layout = if (config.hf_rope_impl) .sequential else .interleaved,
                .freq_base = config.rope_theta,
                .scaling = config.rope_scaling,
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

        if (self.q_norm) |*q_norm| RmsNorm.unloadBuffers(q_norm);
        if (self.k_norm) |*k_norm| RmsNorm.unloadBuffers(k_norm);
    }

    /// Self Attention.
    ///   - If token_index is set, x is assumed to be the representation of one new token,
    /// and kv_cache will be read for the previous tokens.
    ///   - If token_index is not set, x is assumed to be the representation of all tokens
    /// since the beginning of the sequence, and kv_cache won't be read.
    /// In both case, kv_cache will be updated with the computed key and value.
    /// x: {.b, .s, .d } -> .{.b, .s, .d}
    pub fn forward(
        self: SelfAttn,
        x: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { Tensor, KvCache } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;
        var q = self.q_proj.forward(x).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = self.k_proj.forward(x).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        var v = self.v_proj.forward(x).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });

        // In self-attention, .s axis is used both for keys and queries.
        const pos_index = b: {
            const temp = Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };

        if (self.q_norm) |norm| q = norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        if (self.k_norm) |norm| k = norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const dtype = q.dtype();
        const new_kv_cache = kv_cache.update(k, v, token_index);
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);

        const attn_output = zml.attention.attention.attention(q, k, v, token_index, attention_metadata, attention_parameters);

        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return .{ self.o_proj.forward(attn), new_kv_cache };
    }
};

pub const KvCache = struct {
    k: Tensor,
    v: Tensor,
    layer_index: Tensor,

    pub fn init(kv_shape: zml.Shape) KvCache {
        return .{
            .k = .fromShape(kv_shape),
            .v = .fromShape(kv_shape),
            .layer_index = .init(.{}, .u32),
        };
    }

    pub fn initShape(kv_shape: zml.Shape) ShapeOf(KvCache) {
        return .{
            .k = kv_shape,
            .v = kv_shape,
            .layer_index = zml.Shape.init(.{}, .u32),
        };
    }

    pub fn initBuffer(self: KvCache, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(KvCache) {
        return .{
            .k = try zml.Buffer.uninitialized(io, platform, self.k.shape(), .{}),
            .v = try zml.Buffer.uninitialized(io, platform, self.v.shape(), .{}),
            .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(KvCache)) void {
        self.k.deinit();
        self.v.deinit();
        self.layer_index.deinit();
    }

    pub fn keys(self: KvCache) Tensor {
        return self.k.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn values(self: KvCache) Tensor {
        return self.v.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn update(self: KvCache, new_k: Tensor, new_v: Tensor, token_index: ?Tensor) KvCache {
        const k_shape = self.k.shape().drop(.layer);
        var layer = self.layer_index;
        layer = if (token_index) |idx| layer.broad(idx.shape()) else layer;

        return if (token_index) |idx| .{
            .k = self.k.scatterSlices(
                .{ .layer = layer, .k = idx },
                new_k.convert(self.k.dtype()).transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(self.k),
            .v = self.v.scatterSlices(
                .{ .layer = layer, .k = idx },
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

    pub fn atLayer(self: KvCache, layer_index: usize) KvCache {
        return .{
            .k = self.k,
            .v = self.v,
            .layer_index = Tensor.scalar(layer_index, .u32),
        };
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{
            .k = self.k.reuseBuffer(other.k),
            .v = self.v.reuseBuffer(other.v),
            .layer_index = self.layer_index.reuseBuffer(other.layer_index),
        };
    }
};
