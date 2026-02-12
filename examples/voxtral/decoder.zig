const std = @import("std");
const log = std.log;

const zml = @import("zml");
const Tensor = zml.Tensor;
const Shape = zml.Shape;

const cfg = @import("config.zig");
const Config = cfg.Config;

const enc = @import("encoder.zig");
const Linear = enc.Linear;
const RmsNorm = enc.RmsNorm;
const SwiGluFfn = enc.SwiGluFfn;

/// Adapter: encoder→decoder projection.
/// Reshapes encoder output [s, d=1280] → [s/4, d=5120] (concatenate 4 timesteps),
/// then projects through a 2-layer MLP: Linear(5120→3072) → GELU → Linear(3072→3072).
pub const Adapter = struct {
    proj0: Linear,
    proj1: Linear,

    pub fn init(store: zml.io.TensorStore.View) Adapter {
        const proj_store = store.withPrefix("mm_streams_embeddings.embedding_module.audio_language_projection");
        return .{
            .proj0 = Linear.init(proj_store.withLayer(0)),
            .proj1 = Linear.init(proj_store.withLayer(2)),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Adapter)) void {
        Linear.unloadBuffers(&self.proj0);
        Linear.unloadBuffers(&self.proj1);
    }

    /// encoder_out: [s, d=1280] → [s/4, d=3072]
    pub fn forward(self: Adapter, encoder_out: Tensor) Tensor {
        // Reshape [s, d=1280] → [s/4, d=5120] by concatenating 4 consecutive timesteps.
        // Split s into (s_new, group=4), then merge (group, d) → d_new.
        const h = encoder_out
            .splitAxis(.s, .{ .s = .auto, .group = 4 })
            .merge(.{ .d = .{ .group, .d } });

        return self.proj1.forward(self.proj0.forward(h).gelu());
    }
};

/// AdaRmsNorm: per-layer time conditioning.
/// Computes ada_scale = up(gelu(down(t_cond))), then returns h * (1 + ada_scale).
pub const AdaRmsNorm = struct {
    down: Linear,
    up: Linear,

    pub fn init(store: zml.io.TensorStore.View) AdaRmsNorm {
        const ada_store = store.withPrefix("ada_rms_norm_t_cond");
        return .{
            .down = Linear.init(ada_store.withLayer(0)),
            .up = Linear.init(ada_store.withLayer(2)),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AdaRmsNorm)) void {
        Linear.unloadBuffers(&self.down);
        Linear.unloadBuffers(&self.up);
    }

    /// Apply adaptive conditioning: h * (1 + ada_scale).
    /// h: [s, d], t_cond: [d] → [s, d]
    pub fn forward(self: AdaRmsNorm, h: Tensor, t_cond: Tensor) Tensor {
        const ada_scale = self.up.forward(self.down.forward(t_cond).gelu());
        const one_plus_scale = ada_scale.broad(h.shape()).add(Tensor.scalar(1.0, h.dtype()).broad(h.shape()));
        return h.mul(one_plus_scale);
    }
};

/// KV cache for all decoder layers.
/// Stores K/V tensors with shape {layer, k=max_seq_len, h, hd}.
pub const KvCache = struct {
    k: Tensor,
    v: Tensor,
    layer_index: Tensor,

    pub fn init(kv_shape: Shape) KvCache {
        return .{
            .k = .fromShape(kv_shape),
            .v = .fromShape(kv_shape),
            .layer_index = .init(.{}, .u32),
        };
    }

    pub fn initShape(kv_shape: Shape) zml.ShapeOf(KvCache) {
        return .{
            .k = kv_shape,
            .v = kv_shape,
            .layer_index = Shape.init(.{}, .u32),
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

/// Decoder self-attention with KV cache.
pub const SelfAttention = struct {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,

    pub fn init(store: zml.io.TensorStore.View) SelfAttention {
        return .{
            .wq = Linear.init(store.withPrefix("wq")),
            .wk = Linear.init(store.withPrefix("wk")),
            .wv = Linear.init(store.withPrefix("wv")),
            .wo = Linear.init(store.withPrefix("wo")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SelfAttention)) void {
        Linear.unloadBuffers(&self.wq);
        Linear.unloadBuffers(&self.wk);
        Linear.unloadBuffers(&self.wv);
        Linear.unloadBuffers(&self.wo);
    }

    /// x: [s, d], token_index: scalar, kv_cache: KvCache → ([s, d], KvCache)
    pub fn forward(self: SelfAttention, x: Tensor, token_index: Tensor, kv_cache: KvCache, config: Config) struct { Tensor, KvCache } {
        const dtype = x.dtype();

        // Q, K, V projections
        var q = self.wq.forward(x);
        var k = self.wk.forward(x);
        var v = self.wv.forward(x);

        // Split into heads: [s, d] → [s, h, hd]
        q = q.splitAxis(.d, .{ .h = config.n_heads, .hd = config.head_dim });
        k = k.splitAxis(.d, .{ .h = config.n_kv_heads, .hd = config.head_dim });
        v = v.splitAxis(.d, .{ .h = config.n_kv_heads, .hd = config.head_dim });

        // Position indices for RoPE
        const pos_index = b: {
            const temp = Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype())
                .withTags(.{.s}).broad(Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };

        // RoPE (interleaved)
        const rope_opts: zml.nn.RopeOpts = .{
            .layout = .interleaved,
            .freq_base = config.rope_theta,
        };
        q = zml.nn.rope(q, pos_index, rope_opts);
        k = zml.nn.rope(k, pos_index, rope_opts);

        // Rename for attention: .s → .q/.k
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        // Update KV cache and retrieve cached K/V
        const new_kv_cache = kv_cache.update(k, v, token_index);
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);

        // Causal attention with sliding window
        const full_seq_len = k.dim(.k);
        var attn_mask = zml.nn.causalAttnMask(.{ .q = full_seq_len, .k = full_seq_len }, dtype, config.sliding_window);
        attn_mask = attn_mask.gatherSlices(
            Shape.init(.{ .q = q.dim(.q) }, attn_mask.dtype()),
            token_index.reshape(.{ .coord = 1 }),
            .{},
        );
        const attn_out = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask });

        // Merge heads: [q, h, hd] → [s, d]
        const merged = attn_out.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });

        return .{ self.wo.forward(merged), new_kv_cache };
    }
};

/// Single decoder transformer layer with AdaRmsNorm time conditioning.
pub const DecoderLayer = struct {
    attention_norm: RmsNorm,
    attention: SelfAttention,
    ffn_norm: RmsNorm,
    feed_forward: SwiGluFfn,
    ada_norm: AdaRmsNorm,

    pub fn init(store: zml.io.TensorStore.View, config: Config) DecoderLayer {
        return .{
            .attention_norm = RmsNorm.init(store.withPrefix("attention_norm"), config.norm_eps),
            .attention = SelfAttention.init(store.withPrefix("attention")),
            .ffn_norm = RmsNorm.init(store.withPrefix("ffn_norm"), config.norm_eps),
            .feed_forward = SwiGluFfn.init(store.withPrefix("feed_forward")),
            .ada_norm = AdaRmsNorm.init(store),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(DecoderLayer)) void {
        RmsNorm.unloadBuffers(&self.attention_norm);
        SelfAttention.unloadBuffers(&self.attention);
        RmsNorm.unloadBuffers(&self.ffn_norm);
        SwiGluFfn.unloadBuffers(&self.feed_forward);
        AdaRmsNorm.unloadBuffers(&self.ada_norm);
    }

    /// h: [s, d], token_index: scalar, kv_cache: KvCache, t_cond: [d] → ([s, d], KvCache)
    pub fn forward(self: DecoderLayer, h: Tensor, token_index: Tensor, kv_cache: KvCache, t_cond: Tensor, config: Config) struct { Tensor, KvCache } {
        // Pre-attention norm → attention (with KV cache) → residual
        const attn_out, const updated_kv_cache = self.attention.forward(
            self.attention_norm.forward(h),
            token_index,
            kv_cache,
            config,
        );
        var out = h.add(attn_out);

        // Pre-FFN norm → ada conditioning → FFN → residual
        const normed = self.ffn_norm.forward(out);
        const conditioned = self.ada_norm.forward(normed, t_cond);
        const ffn_out = self.feed_forward.forward(conditioned);
        out = out.add(ffn_out);

        return .{ out, updated_kv_cache };
    }
};

/// Top-level decoder: runs all transformer layers, returns hidden states + updated KV cache.
pub const Decoder = struct {
    tok_embeddings: Tensor,
    layers: []DecoderLayer,
    norm: RmsNorm,
    config: Config,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) Decoder {
        const layers = allocator.alloc(DecoderLayer, config.n_layers) catch unreachable;
        for (layers, 0..) |*layer, i| {
            layer.* = DecoderLayer.init(store.withPrefix("layers").withLayer(i), config);
        }

        return .{
            .tok_embeddings = store.withPrefix("mm_streams_embeddings.embedding_module.tok_embeddings").createTensorWithTags("weight", .{ .vocab, .d }),
            .layers = layers,
            .norm = RmsNorm.init(store.withPrefix("norm"), config.norm_eps),
            .config = config,
        };
    }

    pub fn deinit(self: Decoder, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Decoder), allocator: std.mem.Allocator) void {
        self.tok_embeddings.deinit();
        for (self.layers) |*layer| {
            DecoderLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
    }

    /// Run all decoder layers on input embeddings.
    /// input_embeds: [s, d], token_index: scalar, kv_cache: KvCache, t_cond: [d]
    /// Returns: ([s, d], KvCache)
    pub fn forward(self: Decoder, input_embeds: Tensor, token_index: Tensor, kv_cache: KvCache, t_cond: Tensor) struct { Tensor, KvCache } {
        var h = input_embeds;
        var cache = kv_cache;

        for (self.layers, 0..) |layer, i| {
            const layer_cache = cache.atLayer(i);
            const result = layer.forward(h, token_index, layer_cache, t_cond, self.config);
            h = result[0];
            const updated_layer_cache = result[1];
            cache = updated_layer_cache.reuseBuffer(layer_cache);
        }

        h = self.norm.forward(h);
        return .{ h, cache };
    }

    /// Compute output logits via tied embeddings: hidden.dot(tok_embeddings, .d)
    /// hidden: [s, d] → [s, vocab]
    pub fn logits(self: Decoder, hidden: Tensor) Tensor {
        return hidden.dot(self.tok_embeddings.convert(hidden.dtype()), .d);
    }
};
