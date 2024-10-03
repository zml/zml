const std = @import("std");
const testing = std.testing;

const zml = @import("zml");
const meta = zml.meta;

const log = std.log.scoped(.sdxl);
const gguf = zml.io.gguf;
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;
const expectClose = zml.testing.expectClose;

pub const LlamaOptions = struct {
    num_heads: i64,
    num_kv_heads: i64,
    rms_norm_eps: f32,
    rope_opts: zml.nn.RopeOpts,
};

/// ClipTextTransformer architecture, using huggingface transformers naming.
/// Dimensions of activations: {.b, .s, .d}
pub const LlamaLM = struct {
    lm_head: zml.nn.Linear,
    model: ClipTextTransformer,

    pub fn init(self: *LlamaLM, options: LlamaOptions) void {
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
        rng: Tensor.Rng,
    ) struct { Tensor, Tensor, Tensor.Rng } {
        meta.assert(tokens_.dtype() == .i32 and tokens_.rank() >= 1 and token_index.dtype() == .i32 and token_index.rank() == 0, "Can't run ClipTextTransformer ! Expected >=1d tokens and 0d token_index, got: {} and {}", .{ tokens_, token_index });

        var tokens = tokens_.withPartialTags(.{.s});
        const out = zml.call(self.model, .forward, .{ tokens, null });
        tokens, const new_rng = updateTokens(self.lm_head, tokens, token_index, out, rng, self.gen_opts);
        return .{ tokens, increment(0, token_index), new_rng };
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

        const next_token_pred = out.gatherValues(.s, token_index, .{});
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
};

pub const ClipTextTransformer = struct {
    embeddings: struct {
        token_embedding: zml.nn.TokenEmbedding,
        position_embedding: struct { weight: zml.Tensor },
    },
    encoder: struct { layers: []TransformerLayer },
    final_layer_norm: zml.nn.LayerNorm,

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

    pub fn shape(self: ClipTextTransformer) Shape {
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
    pub fn forward(self: ClipTextTransformer, tokens: Tensor, token_index: ?Tensor) Tensor {
        const embeds = embed(self.embed_tokens, tokens, token_index);

        var hidden = embeds;
        for (self.layers, 0..) |layer, i| {
            _ = i; // autofix
            hidden = zml.call(layer, .forward, .{
                hidden,
                token_index,
            });
            hidden = hidden.withPartialTags(.{ .s, .d });
        }
        // TODO: tags seem to be lost by `callFunc`.
        const output = zml.call(self.norm, .forward, .{hidden.withPartialTags(.{ .s, .d })});

        return output;
    }

    pub fn embed(embed_tokens_: zml.nn.TokenEmbedding, tokens_: Tensor, token_index: ?Tensor) Tensor {
        const tokens = if (token_index) |idx|
            tokens_.dynamicSlice1d(-1, 1, idx)
        else
            tokens_;
        return zml.call(embed_tokens_, .forward, .{tokens}).withPartialTags(.{ .s, .d });
    }
};

pub const TransformerLayer = struct {
    layer_norm1: zml.nn.LayerNorm,
    self_attn: SelfAttn,
    layer_norm2: zml.nn.LayerNorm,
    mlp: ClipMlp,

    pub fn forward(
        self: TransformerLayer,
        x0: Tensor,
        token_index: ?Tensor,
    ) Tensor {
        // Self Attention
        //log.debug("TransformerLayer({}) -> {}", .{ x0, self.input_layernorm.forward(x0) });
        meta.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {}", .{x0});

        const x0_normalized = zml.call(self.input_layernorm, .forward, .{x0});
        const delta0 = zml.call(self.self_attn, .forward, .{ x0_normalized, token_index });
        const x1 = x0.add(delta0);

        // Fully Connected
        const x1_normalized = zml.call(self.post_attention_layernorm, .forward, .{x1});
        const x2 = zml.call(self.mlp, .forward, .{x1_normalized}).add(x1);

        return x2.reuseBuffer(x0);
    }
};

const ClipMlp = struct {
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,
    activation: zml.nn.Activation = .gelu,

    pub fn forward(self: ClipMlp, x: Tensor) Tensor {
        var y = x;
        y = self.fc1.forward(y);
        y = self.activation.forward(y);
        y = self.fc2.forward(y);
        return y;
    }
};

pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,

    out_proj: zml.nn.Linear,
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
    pub fn forward(self: SelfAttn, x: Tensor, token_index: ?Tensor) Tensor {
        // log.debug("x.shape: {}", .{x.shape()});
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;
        var q = zml.call(self.q_proj, .forward, .{x}).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = zml.call(self.k_proj, .forward, .{x}).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        var v = zml.call(self.v_proj, .forward, .{x}).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        // Generate the attention mask.
        const seq_len = k.dim(.k);
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

        const attn_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .allow_cudnn = false });
        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return .{zml.call(self.o_proj, .forward, .{attn})};
    }
};
