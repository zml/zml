const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;
const Shape = zml.Shape;

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
        common.deinitLinear(&self.proj0);
        common.deinitLinear(&self.proj1);
    }

    /// encoder_out: [s, d] → [s/dsf, dim]
    pub fn forward(self: Adapter, encoder_out: Tensor) Tensor {
        var h = encoder_out;
        const dsf = self.downsample_factor;

        // Pad encoder output to next multiple of downsample_factor
        const seq_len: usize = @intCast(h.dim(.s));
        const remainder = seq_len % dsf;
        if (remainder > 0) {
            h = h.pad(0, .{ .s = Tensor.Pad{ .high = @as(i64, @intCast(dsf - remainder)) } });
        }

        // Reshape [s, d] → [s/dsf, d*dsf] by concatenating consecutive timesteps.
        h = h.splitAxis(.s, .{ .s = .auto, .group = dsf })
            .merge(.{ .d = .{ .group, .d } });

        return self.proj1.forward(self.proj0.forward(h).gelu().rename(.{ .dout = .d })).rename(.{ .dout = .d });
    }
};

pub const Embedder = struct {
    tok_embeddings: Tensor,

    pub fn init(store: zml.io.TensorStore.View) Embedder {
        return .{
            .tok_embeddings = store.withPrefix("mm_streams_embeddings.embedding_module.tok_embeddings").createTensorWithTags("weight", .{ .vocab, .d }),
        };
    }

    pub fn forward(self: Embedder, full_adapter_out: Tensor, text_tokens: Tensor, pos: Tensor) Tensor {
        const audio_embeds = full_adapter_out.dynamicSlice(.{
            .s = Tensor.DynSlice{ .start = pos, .len = @intCast(text_tokens.dim(.s)) },
        });

        const text_embeds = self.tok_embeddings.gather(.{ .vocab = text_tokens }, .{});

        return text_embeds.add(audio_embeds);
    }

    pub fn load(self: *const Embedder, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, progress: *std.Progress.Node) !zml.Bufferized(Embedder) {
        return common.loadModel(Embedder, self, allocator, io, platform, store, progress, "embedder");
    }

    pub fn unload(self: *zml.Bufferized(Embedder)) void {
        self.tok_embeddings.deinit();
    }
};

/// Top-level decoder: runs all transformer layers, returns hidden states + updated KV cache.
pub const Decoder = struct {
    /// Same weights as Embedder.tok_embeddings; duplicated here for logit computation.
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
            .tok_embeddings = store.withPrefix("mm_streams_embeddings.embedding_module.tok_embeddings").createTensorWithTags("weight", .{ .vocab, .d }),
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
        self.tok_embeddings.deinit();

        for (self.layers) |*layer| {
            DecoderLayer.unload(layer);
        }

        allocator.free(self.layers);
        self.norm.deinit();
    }

    /// Run all decoder layers on input embeddings, compute logits and return argmax tokens.
    /// input_embeds: [s, d], token_index: scalar, kv_cache: KvCache, t_cond: [d]
    /// Returns: (tokens [s] u32, KvCache)
    pub fn forward(self: Decoder, input_embeds: Tensor, token_index: Tensor, kv_cache: KvCache, t_cond: Tensor) struct { Tensor, KvCache } {
        const t = t_cond.convert(input_embeds.dtype());
        var h = input_embeds;
        var cache = kv_cache;

        for (self.layers, 0..) |layer, i| {
            const layer_cache = cache.atLayer(i);
            const result = layer.forward(h, token_index, layer_cache, t, self.config);
            h = result[0];
            const updated_layer_cache = result[1];

            cache = updated_layer_cache.reuseBuffer(layer_cache);
        }

        h = rmsNorm(h, self.norm, self.norm_eps);

        // Compute logits in f32 for precision (matching Python reference)
        const output_logits = h.convert(.f32).dot(self.tok_embeddings.convert(.f32), .d);
        const output_tokens = output_logits.argMax(.vocab).indices.convert(.u32);

        return .{ output_tokens, cache };
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
        common.deinitLinear(&self.down);
        common.deinitLinear(&self.up);
    }

    /// h: [s, d], t_cond: [d] → [s, d]
    pub fn forward(self: AdaRmsNorm, h: Tensor, t_cond: Tensor) Tensor {
        const ada_scale = self.up.forward(self.down.forward(t_cond).gelu().rename(.{ .dout = .d })).rename(.{ .dout = .d });
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
        return .{
            .k = scatterCache(self.k, new_k, self.layer_index, token_index),
            .v = scatterCache(self.v, new_v, self.layer_index, token_index),
            .layer_index = self.layer_index,
        };
    }

    fn scatterCache(cache: Tensor, new: Tensor, layer_index: Tensor, token_index: ?Tensor) Tensor {
        const k_shape = cache.shape().drop(.layer);
        const converted = new.convert(cache.dtype()).transpose(k_shape);
        const scatter_opts: Tensor.ScatterOpts = .{ .indices_are_sorted = true, .update_fn = Tensor.ScatterOpts.override };

        return if (token_index) |idx|
            cache.scatterSlices(.{ .layer = layer_index.broad(idx.shape()), .k = idx }, converted, scatter_opts).reuseBuffer(cache)
        else
            cache.scatterSlices(.{ .layer = layer_index }, converted, scatter_opts).reuseBuffer(cache);
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
    wq: zml.nn.Linear,
    wk: zml.nn.Linear,
    wv: zml.nn.Linear,
    wo: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) SelfAttention {
        return .{
            .wq = common.linear(store.withPrefix("wq")),
            .wk = common.linear(store.withPrefix("wk")),
            .wv = common.linear(store.withPrefix("wv")),
            .wo = common.linear(store.withPrefix("wo")),
        };
    }

    pub fn unload(self: *zml.Bufferized(SelfAttention)) void {
        common.deinitLinear(&self.wq);
        common.deinitLinear(&self.wk);
        common.deinitLinear(&self.wv);
        common.deinitLinear(&self.wo);
    }

    /// x: [s, d], token_index: scalar, kv_cache: KvCache → ([s, d], KvCache)
    pub fn forward(self: SelfAttention, x: Tensor, token_index: Tensor, kv_cache: KvCache, config: Config) struct { Tensor, KvCache } {
        const dtype = x.dtype();

        var q = self.wq.forward(x);
        var k = self.wk.forward(x);
        var v = self.wv.forward(x);

        // Split into heads: [s, dout] → [s, h, hd]
        q = q.splitAxis(.dout, .{ .h = config.n_heads, .hd = config.head_dim });
        k = k.splitAxis(.dout, .{ .h = config.n_kv_heads, .hd = config.head_dim });
        v = v.splitAxis(.dout, .{ .h = config.n_kv_heads, .hd = config.head_dim });

        // Position indices for RoPE
        const pos_index = b: {
            const temp = Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype())
                .withTags(.{.s}).broad(Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };

        const rope_opts: zml.nn.RopeOpts = .{
            .layout = .interleaved,
            .freq_base = config.rope_theta,
        };

        q = zml.nn.rope(q, pos_index, rope_opts);
        k = zml.nn.rope(k, pos_index, rope_opts);

        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        // Update KV cache and retrieve cached K/V
        const new_kv_cache = kv_cache.update(k, v, pos_index.rename(.{ .s = .k }));
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

        // Merge heads and output projection: [q, h, hd] → [s, d]
        const merged = attn_out.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return .{ self.wo.forward(merged).rename(.{ .dout = .d }), new_kv_cache };
    }
};

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
        self.attention_norm.deinit();
        SelfAttention.unload(&self.attention);

        self.ffn_norm.deinit();
        SwiGluFfn.unload(&self.feed_forward);

        AdaRmsNorm.unload(&self.ada_norm);
    }

    /// h: [s, d], token_index: scalar, kv_cache: KvCache, t_cond: [d] → ([s, d], KvCache)
    pub fn forward(self: DecoderLayer, h: Tensor, token_index: Tensor, kv_cache: KvCache, t_cond: Tensor, config: Config) struct { Tensor, KvCache } {
        // Pre-attention norm → attention (with KV cache) → residual
        const attn_out, const updated_kv_cache = self.attention.forward(
            rmsNorm(h, self.attention_norm, self.norm_eps),
            token_index,
            kv_cache,
            config,
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
