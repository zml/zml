const std = @import("std");
const testing = std.testing;

const zml = @import("zml");
const meta = zml.meta;
const flags = @import("tigerbeetle/flags");

const log = std.log.scoped(.llama);
const gguf = zml.io.gguf;
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;
const expectClose = zml.testing.expectClose;

pub const LlamaOptions = struct {
    gen_opts: zml.nn.SamplingStrategy,
    max_seq_len: u32,
    num_heads: i64,
    num_kv_heads: i64,
    rms_norm_eps: f32,
    rope_opts: zml.nn.RopeOpts,
};

/// Llama architecture, using huggingface transformers naming.
/// Dimensions of activations: {.b, .s, .d}
pub const LlamaLM = struct {
    lm_head: zml.nn.Linear,
    model: Llama,

    // Options controlling generation
    gen_opts: zml.nn.SamplingStrategy = .{},

    pub fn init(self: *LlamaLM, options: LlamaOptions) void {
        self.gen_opts = options.gen_opts;
        self.model.max_seq_len = options.max_seq_len;
        self.model.num_heads = options.num_heads;
        self.model.num_kv_heads = options.num_kv_heads;
        self.model.rope_opts = options.rope_opts;
        for (self.model.layers) |*layer| {
            layer.self_attn.num_heads = options.num_heads;
            layer.self_attn.num_kv_heads = options.num_kv_heads;
            layer.self_attn.rope_opts = options.rope_opts;
            layer.input_layernorm.eps = options.rms_norm_eps;
            layer.post_attention_layernorm.eps = options.rms_norm_eps;
        }
    }

    /// Predicts the token at `token_index` position.
    /// Returns:
    ///  - updated `tokens`,
    ///  - `token_idx` + 1,
    ///  - updated KV cache
    ///  - a Rng state to allow for probabilistic generation
    pub fn forward(
        self: LlamaLM,
        tokens_: Tensor,
        token_index: Tensor,
        kv_cache: ?KvCache,
        rng: Tensor.Rng,
    ) struct { Tensor, Tensor, KvCache, Tensor.Rng } {
        meta.assert(tokens_.dtype() == .i32 and tokens_.rank() >= 1 and token_index.dtype() == .i32 and token_index.rank() == 0, "Can't run Llama ! Expected >=1d tokens and 0d token_index, got: {} and {}", .{ tokens_, token_index });

        var tokens = tokens_.withPartialTags(.{.s});
        const out, const updated_kv_cache = zml.call(self.model, .forward, .{ tokens, if (kv_cache == null) null else token_index, kv_cache });
        tokens, const new_rng = updateTokens(self.lm_head, tokens, token_index, out, rng, self.gen_opts);
        return .{ tokens, increment(0, token_index), updated_kv_cache, new_rng };
    }

    pub fn updateTokens(
        lm_head: zml.nn.Linear,
        tokens_: Tensor,
        token_index: Tensor,
        out_: Tensor,
        rng: Tensor.Rng,
        opts: zml.nn.SamplingStrategy,
    ) struct { Tensor, Tensor.Rng } {
        const tokens = tokens_.withPartialTags(.{.s});
        const out = out_.withPartialTags(.{ .s, .d });

        const next_token_pred = out.dynamicSlice(.{ .s = .{ .start = token_index, .len = 1 } });
        var logits = zml.call(lm_head, .forward, .{next_token_pred});
        if (logits.shape().hasTag(.voc) == null)
            logits = logits.rename(.{ .d = .voc });

        const next_token, const new_rng = zml.nn.sampleTokens(logits, opts, rng);
        const next_token_index = token_index.addConstant(1);
        const new_tokens = tokens.dynamicUpdateSlice(.{ .s = next_token_index }, next_token);

        return .{ new_tokens.reuseBuffer(tokens_), new_rng };
    }

    pub fn increment(_: u8, token_index: Tensor) Tensor {
        return token_index.addConstant(1).reuseBuffer(token_index);
    }

    /// Run the generation entirely within pjrt.
    pub fn generate(self: LlamaLM, tokens: Tensor, token_index: Tensor, rng: Tensor.Rng) Tensor {
        // Generate the first token using the prompt and generate the KV-cache initial values.
        const prefill = zml.call(self, .forward, .{ tokens, token_index, null, rng });

        const Gen = struct {
            /// Same as LlamaLM.forward but without optional in the signature
            pub fn forward(lm: LlamaLM, t_ids: Tensor, t_idx: Tensor, kv_cache_: KvCache, inner_rng: Tensor.Rng) struct { Tensor, Tensor, KvCache, Tensor.Rng } {
                var kv_cache = kv_cache_;
                kv_cache.k = kv_cache.k.withPartialTags(.{ .layer, .h, .k, .hd });
                kv_cache.v = kv_cache.v.withPartialTags(.{ .layer, .h, .k, .hd });
                return zml.call(lm, .forward, .{ t_ids._ctx, t_ids, t_idx, kv_cache, inner_rng });
            }
            // / Stops when we generated `max_seq_len` tokens.
            pub fn shouldContinue(lm: LlamaLM, t_ids: Tensor, t_idx: Tensor, kv_cache: KvCache, inner_rng: Tensor.Rng) Tensor {
                _ = kv_cache;
                _ = inner_rng;
                std.debug.assert(t_ids.dim(1) == lm.model.max_seq_len);
                return t_idx.cmp(.LT, Tensor.scalar(t_ids._ctx, lm.model.max_seq_len, t_idx.dtype()));
            }
        };
        // Generate remaining tokens using the KV-cache, return tokens.
        return zml.ops.while_(Gen.shouldContinue, Gen.forward, self, prefill)[0];
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
    pub fn forward(self: Llama, tokens: Tensor, token_index: ?Tensor, kv_cache: ?KvCache) struct { Tensor, KvCache } {
        const embeds = embed(self.embed_tokens, tokens, token_index);

        var hidden = embeds;
        const kv_cache0 = kv_cache orelse self.initKvCache(embeds.shape());

        var updated_kv_cache = kv_cache0;
        for (self.layers, 0..) |layer, i| {
            hidden, updated_kv_cache = zml.call(layer, .forward, .{ hidden, token_index, updated_kv_cache.atLayer(i) });
            hidden = hidden.withPartialTags(.{ .s, .d });
        }
        // TODO: tags seem to be lost by `callFunc`.
        const output = zml.call(self.norm, .forward, .{hidden.withPartialTags(.{ .s, .d })});

        return .{ output, updated_kv_cache.reuseBuffer(kv_cache0) };
    }

    pub fn embed(embed_tokens_: zml.nn.TokenEmbedding, tokens_: Tensor, token_index: ?Tensor) Tensor {
        const tokens = if (token_index) |idx|
            tokens_.dynamicSlice1d(-1, 1, idx)
        else
            tokens_;
        return zml.call(embed_tokens_, .forward, .{tokens}).withPartialTags(.{ .s, .d });
    }

    fn initKvCache(self: Llama, embed_shape: zml.Shape) KvCache {
        const dims = self.shape();
        var kv_shape = embed_shape.insert(0, .{ .layer = dims.layer }).rename(.{ .s = .k }).splitAxes(.{ .d = .{ .h = dims.nkvh, .hd = dims.hd } });
        const perm = kv_shape.contiguousPerm(.{ .h, .k, .hd });
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
        token_index: ?Tensor,
        kv_cache: ?KvCache,
    ) struct { Tensor, KvCache } {
        // Self Attention
        //log.debug("TransformerLayer({}) -> {}", .{ x0, self.input_layernorm.forward(x0) });
        meta.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {}", .{x0});

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
        token_index: ?Tensor,
        kv_cache_: ?KvCache,
    ) struct { Tensor, KvCache } {
        // log.debug("x.shape: {}", .{x.shape()});
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;
        var q = zml.call(self.q_proj, .forward, .{x}).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = zml.call(self.k_proj, .forward, .{x}).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        var v = zml.call(self.v_proj, .forward, .{x}).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        // Generate the attention mask.
        const kv_cache = kv_cache_ orelse initKvCache(k.shape());
        const seq_len = kv_cache.k.dim(.k);
        var attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, x.dtype(), null);
        var cos, var sin = zml.nn.ropeCosSin(.{ .s = seq_len, .hd = k.dim(.hd) }, x.dtype(), self.rope_opts);
        if (token_index) |idx| {
            // Note: in Pytorch it would be very inefficient to generate the full ropeCosSin and attn_mask matrices, then slice into it,
            // but XLA is able to optimize this correctly.
            attn_mask = attn_mask.dynamicSlice(.{ .q = .{ .start = idx, .len = 1 } });
            cos = cos.dynamicSlice(.{ .s = .{ .start = idx, .len = 1 } });
            sin = sin.dynamicSlice(.{ .s = .{ .start = idx, .len = 1 } });
        }

        // In self-attention, .s axis is used both for keys and queries.
        q = zml.nn.rope(q, .{ cos, sin }, self.rope_opts);
        k = zml.nn.rope(k, .{ cos, sin }, self.rope_opts);
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const new_kv_cache = kv_cache.update(k, v, token_index orelse Tensor.scalar(0, .i32));
        if (token_index) |_| {
            std.debug.assert(q.dim(.q) == 1);
            k = new_kv_cache.keys();
            v = new_kv_cache.values();
        }

        const attn_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .allow_cudnn = false });
        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return .{ zml.call(self.o_proj, .forward, .{attn}), new_kv_cache };
    }

    fn initKvCache(key_shape: zml.Shape) KvCache {
        // When we call initKvCache, we haven't renamed .s to .k yet.
        var kv_shape = key_shape.insert(0, .{ .layer = 1 }).rename(.{ .s = .k });
        const perm = kv_shape.contiguousPerm(.{ .h, .k, .hd });
        kv_shape = kv_shape.transpose(perm.constSlice());
        var res = KvCache.init(kv_shape);
        res.layer_index = Tensor.scalar(0, .i32);
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
            .k = Tensor.constant(kv_shape, kv_shape.dtype().one()),
            .v = Tensor.constant(kv_shape, kv_shape.dtype().one()),
            .layer_index = Tensor.scalar(-1, .i32),
        };
    }

    pub fn initShape(kv_shape: zml.Shape) ShapeOf(KvCache) {
        return .{
            .k = kv_shape,
            .v = kv_shape,
            .layer_index = zml.Shape.init(.{}, .i32),
        };
    }

    pub fn keys(self: KvCache) Tensor {
        return self.k.dynamicSlice(.{ .layer = .{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn values(self: KvCache) Tensor {
        return self.v.dynamicSlice(.{ .layer = .{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn update(self: KvCache, new_k: Tensor, new_v: Tensor, token_index: Tensor) KvCache {
        return .{
            .k = self.k.dynamicUpdateSlice(
                .{ .layer = self.layer_index, .k = token_index },
                // transpose to match kv-cache layout
                new_k.contiguous(.{ .h, .k, .hd }),
            ).reuseBuffer(self.k),
            .v = self.v.dynamicUpdateSlice(
                .{ .layer = self.layer_index, .k = token_index },
                // transpose to match kv-cache layout
                new_v.contiguous(.{ .h, .k, .hd }),
            ).reuseBuffer(self.v),
            .layer_index = self.layer_index,
        };
    }

    pub fn atLayer(self: KvCache, layer_index: usize) KvCache {
        return .{
            .k = self.k,
            .v = self.v,
            .layer_index = Tensor.scalar(layer_index, .i32),
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
