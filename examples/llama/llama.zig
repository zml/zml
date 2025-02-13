const std = @import("std");
const stdx = @import("stdx");
const zml = @import("zml");

const testing = std.testing;
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;

const gguf = zml.io.gguf;
const expectClose = zml.testing.expectClose;
const log = std.log.scoped(.llama);

/// Llama architecture, using huggingface transformers naming.
/// Dimensions of activations: {.b, .s, .d}
pub const LlamaLM = struct {
    pub const Config = struct {
        bos_token_id: u32,
        eos_token_id: stdx.json.Union(union(enum) {
            int: u32,
            ints: []u32,
        }),
        num_hidden_layers: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        rope_theta: f32,
        max_position_embeddings: usize,
        rms_norm_eps: f32,
        hf_rope_impl: bool = true,
    };

    pub const Options = struct {
        sampling_strategy: ?zml.nn.SamplingStrategy,
        max_seq_len: usize,
    };

    lm_head: ?zml.nn.Linear,
    model: Llama,

    // Options controlling generation
    gen_opts: zml.nn.SamplingStrategy = .{},
    config: Config,

    pub fn init(self: *LlamaLM, config: Config, options: Options) void {
        self.config = config;
        self.gen_opts = options.sampling_strategy orelse .{};
        self.model.max_seq_len = @intCast(options.max_seq_len);
        self.model.num_heads = @intCast(config.num_attention_heads);
        self.model.num_kv_heads = @intCast(config.num_key_value_heads);
        self.model.rope_opts = .{
            .impl = if (config.hf_rope_impl) .sequential else .interleaved,
            .freq_base = config.rope_theta,
        };
        for (self.model.layers) |*layer| {
            layer.self_attn.num_heads = self.model.num_heads;
            layer.self_attn.num_kv_heads = self.model.num_kv_heads;
            layer.self_attn.rope_opts = self.model.rope_opts;
            layer.input_layernorm.eps = config.rms_norm_eps;
            layer.post_attention_layernorm.eps = config.rms_norm_eps;
            layer.mlp.up_proj.weight = layer.mlp.up_proj.weight.withSharding(.{0});
            layer.mlp.gate_proj.weight = layer.mlp.gate_proj.weight.withSharding(.{0});
            layer.mlp.down_proj.weight = layer.mlp.down_proj.weight.withSharding(.{1});

            layer.self_attn.q_proj.weight = layer.self_attn.q_proj.weight.withSharding(.{0});
            layer.self_attn.k_proj.weight = layer.self_attn.k_proj.weight.withSharding(.{0});
            layer.self_attn.v_proj.weight = layer.self_attn.v_proj.weight.withSharding(.{0});
            layer.self_attn.o_proj.weight = layer.self_attn.o_proj.weight.withSharding(.{1});
        }

        // TODO(Corentin): Fix lm_head sharding when top-k sampling is enabled.
        // It currently crashes/compilation fails
        if (self.gen_opts.topk == 1 and self.lm_head != null) {
            self.lm_head.?.weight = self.lm_head.?.weight.withSharding(.{0});
        }
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
    ) struct { Tensor, KvCache, Tensor.Rng } {
        stdx.debug.assert(tokens_.dtype() == .u32 and tokens_.rank() >= 1 and token_index.dtype() == .u32 and token_index.rank() <= 1, "Can't run Llama ! Expected >=1d tokens and 0d token_index, got: {} and {}", .{ tokens_, token_index });
        const tokens = tokens_.withPartialTags(.{.s});
        const out, const updated_kv_cache = zml.call(self.model, .forward, .{ tokens, token_index, kv_cache });
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
                break :blk zml.call(lm_head, .forward, .{out});
            } else {
                break :blk self.model.embed_tokens.weight.withTags(.{ .voc, .d }).dot(out, .{.d});
            }
        };

        if (logits.shape().hasTag(.voc) == null)
            logits = logits.rename(.{ .d = .voc });

        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, opts, rng);
        return .{ next_tokens, new_rng };
    }

    pub fn increment(_: u8, token_index: Tensor) Tensor {
        return token_index.addConstant(1).reuseBuffer(token_index);
    }
};

pub const Llama = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    norm: RmsNorm,
    layers: []TransformerLayer,

    max_seq_len: u32 = 0,
    num_heads: i64 = 32,
    num_kv_heads: i64 = 32,
    rope_opts: zml.nn.RopeOpts = .{
        .impl = .interleaved,
        .freq_base = 10_000,
    },

    const Shape = struct {
        s: u32,
        layer: u16,
        hd: u16,
        nh: u16,
        nkvh: u16,
        dtype: zml.DataType,
    };

    pub fn shape(self: Llama) Shape {
        const key_dim = self.layers[0].self_attn.k_proj.weight.dim(0);
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;

        return .{
            .s = self.max_seq_len,
            .layer = @intCast(self.layers.len),
            .hd = @intCast(@divExact(key_dim, num_kv_heads)),
            .nh = @intCast(self.num_heads),
            .nkvh = @intCast(num_kv_heads),
            .dtype = self.embed_tokens.weight.dtype(),
        };
    }

    /// Forward one token, using KV cache for previous tokens.
    /// Returns result and updated KV cache.
    pub fn forward(self: Llama, tokens: Tensor, token_index: Tensor, kv_cache: KvCache) struct { Tensor, KvCache } {
        const embeds = embed(self.embed_tokens, tokens);
        var hidden = embeds;

        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden, updated_kv_cache = zml.call(layer, .forward, .{ hidden, token_index, updated_kv_cache.atLayer(i) });
        }
        const output = zml.call(self.norm, .forward, .{hidden});

        return .{ output, updated_kv_cache.reuseBuffer(kv_cache) };
    }

    pub fn embed(embed_tokens_: zml.nn.TokenEmbedding, tokens_: Tensor) Tensor {
        return zml.call(embed_tokens_, .forward, .{tokens_}).withPartialTags(.{.d});
    }

    fn initKvCache(self: Llama, embed_shape: zml.Shape) KvCache {
        const dims = self.shape();
        var kv_shape = embed_shape.insert(0, .{ .layer = dims.layer }).rename(.{ .s = .k }).splitAxes(.{ .d = .{ .h = dims.nkvh, .hd = dims.hd } });
        const perm = kv_shape.contiguousPerm(.{ .k, .h, .hd });
        kv_shape = kv_shape.transpose(perm.constSlice());
        return KvCache.init(kv_shape);
    }
};

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: SelfAttn,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,

    pub fn forward(
        self: TransformerLayer,
        x0: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
    ) struct { Tensor, KvCache } {
        // Self Attention
        //log.debug("TransformerLayer({}) -> {}", .{ x0, self.input_layernorm.forward(x0) });
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {}", .{x0});

        const x0_normalized = zml.call(self.input_layernorm, .forward, .{x0});
        const delta0, const updated_kv_cache = zml.call(self.self_attn, .forward, .{ x0_normalized, token_index, kv_cache });
        const x1 = x0.add(delta0);

        // Fully Connected
        const x1_normalized = zml.call(self.post_attention_layernorm, .forward, .{x1});
        const x2 = zml.call(self.mlp, .forward, .{x1_normalized}).add(x1);

        return .{ x2.reuseBuffer(x0), updated_kv_cache };
    }
};

const RmsNorm = struct {
    weight: Tensor,
    eps: f32 = 1e-5,

    /// L2 normalization of input tensor along `.d` axis.
    pub fn forward(self: RmsNorm, input: Tensor) Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        // upcast to improve precision
        const xf32 = x.convert(.f32);
        const mean = xf32.mul(xf32).mean(.d);
        const rsqrt = Tensor.rsqrt(mean.addConstant(self.eps)).convert(x.dtype());
        const normalized = x.mul(rsqrt.broad(x.shape()));

        return normalized.mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }
};

const Mlp = struct {
    up_proj: zml.nn.Linear, // (dim -> hidden_dim)
    gate_proj: zml.nn.Linear, // (dim -> hidden_dim)
    down_proj: zml.nn.Linear, // (hidden_dim -> dim)

    pub fn forward(self: Mlp, x: Tensor) Tensor {
        const proj = zml.call(self.up_proj, .forward, .{x});
        var output = zml.call(self.gate_proj, .forward, .{x});
        output = output.silu().mul(proj);
        return zml.call(self.down_proj, .forward, .{output});
    }
};

pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,

    o_proj: zml.nn.Linear,
    num_heads: i64 = undefined,
    num_kv_heads: i64 = 0,
    rope_opts: zml.nn.RopeOpts = undefined,

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
    ) struct { Tensor, KvCache } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;
        var q = zml.call(self.q_proj, .forward, .{x}).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto }).withSharding(.{.h});
        var k = zml.call(self.k_proj, .forward, .{x}).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto }).withSharding(.{.h});
        var v = zml.call(self.v_proj, .forward, .{x}).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto }).withSharding(.{.h});

        // Generate the attention mask.
        const seq_len = kv_cache.k.dim(.k);
        var attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, x.dtype(), null);

        // Note: in Pytorch it would be very inefficient to generate the full attn_mask,
        // then slice into it, but XLA is able to optimize this correctly.
        attn_mask = attn_mask.gatherSlices(zml.Shape.init(.{ .q = x.dim(.s) }, attn_mask.dtype()), token_index.reshape(.{ .coord = 1 }), .{});

        // In self-attention, .s axis is used both for keys and queries.
        const pos_index = b: {
            const temp = Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };

        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const dtype = q.dtype();
        const new_kv_cache = kv_cache.update(k, v, token_index);
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);

        const attn_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .allow_cudnn = true });
        // const attn_output = zml.nn.sdpaMemEfficient(q, k, v, .{ .attn_mask = attn_mask }, .{ .q_chunk_size = 4096, .k_chunk_size = 1024 });
        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return .{ zml.call(self.o_proj, .forward, .{attn}), new_kv_cache };
    }

    fn initKvCache(key_shape: zml.Shape) KvCache {
        // When we call initKvCache, we haven't renamed .s to .k yet.
        var kv_shape = key_shape.insert(0, .{ .layer = 1 }).rename(.{ .s = .k });
        const perm = kv_shape.contiguousPerm(.{ .h, .k, .hd });
        kv_shape = kv_shape.transpose(perm.constSlice());
        var res = KvCache.init(kv_shape);
        res.layer_index = Tensor.scalar(0, .u32);
        return res;
    }
};

pub const KvCache = struct {
    k: Tensor,
    v: Tensor,
    layer_index: Tensor,

    pub fn init(kv_shape: zml.Shape) KvCache {
        // The KV-cache is initialized with ones to detect reads of uninitialized memory.
        return .{
            .k = Tensor.constant(kv_shape, kv_shape.dtype().one()).withSharding(.{.h}),
            .v = Tensor.constant(kv_shape, kv_shape.dtype().one()).withSharding(.{.h}),
            .layer_index = Tensor.scalar(-1, .u32),
        };
    }

    pub fn initShape(kv_shape: zml.Shape) ShapeOf(KvCache) {
        return .{
            .k = kv_shape,
            .v = kv_shape,
            .layer_index = zml.Shape.init(.{}, .u32),
        };
    }

    pub fn initBuffer(kv_shape: zml.Shape, platform: zml.Platform) !zml.Bufferized(KvCache) {
        return .{
            .k = try zml.Buffer.constant(platform, kv_shape, 1),
            .v = try zml.Buffer.constant(platform, kv_shape, 1),
            .layer_index = try zml.Buffer.constant(platform, zml.Shape.init(.{}, .u32), 0),
        };
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
