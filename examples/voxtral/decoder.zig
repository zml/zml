const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;

const cfg = @import("config.zig");
const Config = cfg.Config;

const enc = @import("encoder.zig");
const SwiGluFfn = enc.SwiGluFfn;

const common = @import("common.zig");
const rmsNorm = common.rmsNorm;

/// Adapter: encoder→decoder projection.
/// Reshapes encoder output [s, d] → [s/dsf, d*dsf] (concatenate `downsample_factor` consecutive timesteps),
/// then projects through a 2-layer MLP: Linear(d*dsf→dim) → GELU → Linear(dim→dim).
pub const Adapter = struct {
    proj0: zml.nn.Linear,
    proj1: zml.nn.Linear,
    downsample_factor: u32,

    pub fn init(store: zml.io.TensorStore.View, config: Config) Adapter {
        const proj_store = store.withPrefix("mm_streams_embeddings.embedding_module.audio_language_projection");

        return .{
            .proj0 = common.linear(proj_store.withLayer(0)),
            .proj1 = common.linear(proj_store.withLayer(2)),
            .downsample_factor = config.downsample_factor(),
        };
    }

    pub fn load(self: *const Adapter, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, progress: *std.Progress.Node) !zml.Bufferized(Adapter) {
        return common.loadModel(Adapter, self, allocator, io, platform, store, progress, "adapter");
    }

    pub fn unload(self: *zml.Bufferized(Adapter)) void {
        common.deinitBufferized(self);
    }

    /// encoder_out: [s, d] → [s/dsf, dim]
    pub fn forward(self: Adapter, encoder_out: Tensor) Tensor {
        var h = encoder_out;
        const dsf = self.downsample_factor;

        // Reshape [s, d] → [s/dsf, d*dsf] by concatenating consecutive timesteps.
        h = h.splitAxis(.s, .{ .s = .auto, .group = dsf })
            .merge(.{ .d = .{ .group, .d } });

        return self.proj1.forward(self.proj0.forward(h).gelu().rename(.{ .dout = .d })).rename(.{ .dout = .d });
    }
};

/// Top-level decoder: runs all transformer layers, returns hidden states + updated KV cache.
pub const Decoder = struct {
    /// Token embeddings: used for both input embedding and output logit computation.
    tok_embeddings: Tensor,
    layers: []DecoderLayer,
    norm: Tensor,
    norm_eps: f32,
    config: Config,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) Decoder {
        const layers = allocator.alloc(DecoderLayer, config.n_layers) catch unreachable;

        for (layers, 0..) |*layer, i| {
            layer.* = DecoderLayer.init(store.withPrefix("layers").withLayer(i), config);
        }

        return .{
            .tok_embeddings = store.withPrefix("mm_streams_embeddings.embedding_module.tok_embeddings").createTensorWithTags("weight", .{ .voc, .d }),
            .layers = layers,
            .norm = store.withPrefix("norm").createTensorWithTags("weight", .{.d}),
            .norm_eps = config.norm_eps,
            .config = config,
        };
    }

    pub fn deinit(self: Decoder, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn load(self: *const Decoder, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, progress: *std.Progress.Node) !zml.Bufferized(Decoder) {
        return common.loadModel(Decoder, self, allocator, io, platform, store, progress, "decoder");
    }

    pub fn unload(self: *zml.Bufferized(Decoder), allocator: std.mem.Allocator) void {
        common.deinitBufferized(self);
        allocator.free(self.layers);
    }

    /// Run all decoder layers on input embeddings, sample next tokens and return updated KV cache + RNG.
    /// text_tokens: [s] u32, audio_embed: [s, d], token_index: scalar, kv_cache: KvCache, t_cond: [d], rng: Rng
    /// Returns: (tokens [s] u32, KvCache, Rng)
    pub fn forward(self: Decoder, text_tokens: Tensor, audio_embed: Tensor, token_index: Tensor, kv_cache: KvCache, t_cond: Tensor, rng: Tensor.Rng, attention_metadata: zml.attention.Metadata, attention_parameters: zml.attention.Parameters) struct { Tensor, KvCache, Tensor.Rng } {
        const text_embeds = self.tok_embeddings.gather(.{ .voc = text_tokens }, .{});
        const input_embeds = text_embeds.add(audio_embed);

        const t = t_cond.convert(input_embeds.dtype());

        var h = input_embeds;
        var cache = kv_cache;
        const attn_config = self.config.attentionConfig();

        for (self.layers, 0..) |layer, i| {
            const layer_cache = cache.atLayer(i);
            const result = layer.forward(h, token_index, layer_cache, t, attn_config, attention_metadata, attention_parameters);
            h = result[0];
            const updated_layer_cache = result[1];

            cache = updated_layer_cache.reuseBuffer(layer_cache);
        }

        h = rmsNorm(h, self.norm, self.norm_eps);

        // Compute logits in f32 for precision
        const logits = h.dot(self.tok_embeddings, .d).convert(.f32);
        const output_tokens, const new_rng = zml.nn.sampleTokens(logits, .{ .temperature = 0 }, rng);

        return .{ output_tokens.convert(.u32), cache, new_rng };
    }

    /// Like forward, but also returns the incremented token index (on-device).
    pub fn forwardStep(self: Decoder, text_tokens: Tensor, audio_embed: Tensor, token_index: Tensor, kv_cache: KvCache, t_cond: Tensor, rng: Tensor.Rng, attention_metadata: zml.attention.Metadata, attention_parameters: zml.attention.Parameters) struct { Tensor, KvCache, Tensor.Rng, Tensor } {
        const output_tokens, const cache, const new_rng = self.forward(text_tokens, audio_embed, token_index, kv_cache, t_cond, rng, attention_metadata, attention_parameters);
        const next_index = token_index.addConstant(1).reuseBuffer(token_index);
        return .{ output_tokens, cache, new_rng, next_index };
    }
};

/// AdaRmsNorm: per-layer time conditioning.
/// Computes ada_scale = up(gelu(down(t_cond))), then returns h * (1 + ada_scale).
pub const AdaRmsNorm = struct {
    down: zml.nn.Linear,
    up: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) AdaRmsNorm {
        const ada_store = store.withPrefix("ada_rms_norm_t_cond");
        return .{
            .down = common.linear(ada_store.withLayer(0)),
            .up = common.linear(ada_store.withLayer(2)),
        };
    }

    pub fn unload(self: *zml.Bufferized(AdaRmsNorm)) void {
        common.deinitBufferized(self);
    }

    /// h: [s, d], t_cond: [d] → [s, d]
    pub fn forward(self: AdaRmsNorm, h: Tensor, t_cond: Tensor) Tensor {
        const ada_scale = self.up.forward(self.down.forward(t_cond).gelu().rename(.{ .dout = .d })).rename(.{ .dout = .d });
        const one_plus_scale = ada_scale.addConstant(1.0).broad(h.shape());

        return h.mul(one_plus_scale);
    }
};

pub const KvCache = common.KvCache;

pub const SelfAttention = common.SelfAttention(false);

/// Single decoder transformer layer with AdaRmsNorm time conditioning.
pub const DecoderLayer = struct {
    attention_norm: Tensor,
    attention: SelfAttention,
    ffn_norm: Tensor,
    feed_forward: SwiGluFfn,
    ada_norm: AdaRmsNorm,
    norm_eps: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config) DecoderLayer {
        return .{
            .attention_norm = store.withPrefix("attention_norm").createTensorWithTags("weight", .{.d}),
            .attention = SelfAttention.init(store.withPrefix("attention")),
            .ffn_norm = store.withPrefix("ffn_norm").createTensorWithTags("weight", .{.d}),
            .feed_forward = SwiGluFfn.init(store.withPrefix("feed_forward")),
            .ada_norm = AdaRmsNorm.init(store),
            .norm_eps = config.norm_eps,
        };
    }

    pub fn unload(self: *zml.Bufferized(DecoderLayer)) void {
        common.deinitBufferized(self);
    }

    /// h: [s, d], token_index: scalar, kv_cache: KvCache, t_cond: [d] → ([s, d], KvCache)
    pub fn forward(self: DecoderLayer, h: Tensor, token_index: Tensor, kv_cache: KvCache, t_cond: Tensor, attn_config: cfg.AttentionConfig, attention_metadata: zml.attention.Metadata, attention_parameters: zml.attention.Parameters) struct { Tensor, KvCache } {
        // Pre-attention norm → attention (with KV cache) → residual
        const attn_out, const updated_kv_cache = self.attention.forward(
            rmsNorm(h, self.attention_norm, self.norm_eps),
            token_index,
            kv_cache,
            attn_config,
            attention_metadata,
            attention_parameters,
        );

        var out = h.add(attn_out);

        // Pre-FFN norm → ada conditioning → FFN → residual
        const normed = rmsNorm(out, self.ffn_norm, self.norm_eps);
        const conditioned = self.ada_norm.forward(normed, t_cond);
        const ffn_out = self.feed_forward.forward(conditioned);

        out = out.add(ffn_out);

        return .{ out, updated_kv_cache };
    }
};
