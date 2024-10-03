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

pub const ClipTextConfig = struct {
    num_heads: u32,
    max_position_embeddings: u32,
    hidden_act: zml.nn.Activation = .gelu,
    layer_norm_eps: f32 = 1e-5,
};

pub const ClipTextTransformer = struct {
    embeddings: ClipTextEmbeddings,
    encoder: struct { layers: []TransformerLayer },
    final_layer_norm: zml.nn.LayerNorm,

    config: ClipTextConfig = undefined,

    pub fn init(self: *ClipTextTransformer, config: ClipTextConfig) void {
        self.config = config;
        for (self.encoder.layers) |*layer| {
            layer.self_attn.num_heads = config.num_heads;
            layer.layer_norm1.eps = config.layer_norm_eps;
            layer.layer_norm2.eps = config.layer_norm_eps;
            layer.mlp.activation = config.hidden_act;
        }
        self.embeddings.max_position_embeddings = config.max_position_embeddings;
    }

    /// Forward one token, using KV cache for previous tokens.
    /// Returns result and updated KV cache.
    pub fn forward(self: ClipTextTransformer, tokens: Tensor) Tensor {
        const embeds = zml.call(self.embeddings, .forward, .{tokens}).withPartialTags(.{ .s, .d });
        var hidden = embeds;
        for (self.encoder.layers) |layer| {
            hidden = zml.call(layer, .forward, .{hidden});
            // TODO: tags seem to be lost by `callFunc`.
            hidden = hidden.withPartialTags(.{ .s, .d });
        }
        return zml.call(self.final_layer_norm, .forward, .{hidden});
    }
};

pub const TransformerLayer = struct {
    layer_norm1: zml.nn.LayerNorm,
    self_attn: SelfAttn,
    layer_norm2: zml.nn.LayerNorm,
    mlp: ClipMlp,

    pub fn forward(
        self: TransformerLayer,
        input: Tensor,
    ) Tensor {
        // Self Attention
        const x0 = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{ .b, .s, .d });

        const x0_normalized = zml.call(self.layer_norm1, .forward, .{x0});
        const delta0 = zml.call(self.self_attn, .forward, .{x0_normalized});
        const x1 = x0.add(delta0);

        // Fully Connected
        const x1_normalized = zml.call(self.layer_norm2, .forward, .{x1});
        const x2 = zml.call(self.mlp, .forward, .{x1_normalized}).add(x1);

        return x2.reuseBuffer(x0);
    }
};

const ClipTextEmbeddings = struct {
    token_embedding: zml.nn.TokenEmbedding,
    position_embedding: zml.nn.TokenEmbedding,
    max_position_embeddings: u32,

    pub fn forward(self: ClipTextEmbeddings, token_ids: Tensor) Tensor {
        const position_ids = Tensor.arange(.{ .end = self.max_position_embeddings }, .i32).broadcastLeft(token_ids.shape());

        return self.token_embedding.forward(token_ids).add(self.position_embedding.forward(position_ids));
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
    num_heads: u32,

    /// Self Attention.
    ///   - If token_index is set, x is assumed to be the representation of one new token,
    /// and kv_cache will be read for the previous tokens.
    ///   - If token_index is not set, x is assumed to be the representation of all tokens
    /// since the beginning of the sequence, and kv_cache won't be read.
    /// In both case, kv_cache will be updated with the computed key and value.
    /// x: {.b, .s, .d } -> .{.b, .s, .d}
    pub fn forward(self: SelfAttn, x: Tensor) Tensor {
        // log.debug("x.shape: {}", .{x.shape()});
        const nh = self.num_heads;
        var q = zml.call(self.q_proj, .forward, .{x}).splitAxis(-1, .{ .h = nh, .hd = .auto });
        var k = zml.call(self.k_proj, .forward, .{x}).splitAxis(-1, .{ .h = nh, .hd = .auto });
        var v = zml.call(self.v_proj, .forward, .{x}).splitAxis(-1, .{ .h = nh, .hd = .auto });
        // Generate the attention mask.
        const seq_len = k.dim(.s);
        const attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, x.dtype(), null);

        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        log.info("q={}, k={}, v={}", .{ q, k, v });
        const attn_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .allow_cudnn = false });
        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return zml.call(self.out_proj, .forward, .{attn});
    }
};
