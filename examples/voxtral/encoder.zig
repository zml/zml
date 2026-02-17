const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;

const cfg = @import("config.zig");
const Config = cfg.Config;

const common = @import("common.zig");
const rmsNorm = common.rmsNorm;
const KvCache = common.KvCache;

pub const Encoder = struct {
    conv0: CausalConv1d,
    conv1: CausalConv1d,
    layers: []TransformerLayer,
    norm: Tensor,
    norm_eps: f32,
    config: Config,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) Encoder {
        const encoder_store = store.withPrefix("mm_streams_embeddings.embedding_module.whisper_encoder");
        const conv_store = encoder_store.withPrefix("conv_layers");
        const transformer_store = encoder_store.withPrefix("transformer");

        const enc = config.encoder();
        const layers = allocator.alloc(TransformerLayer, enc.n_layers) catch unreachable;
        for (layers, 0..) |*layer, i| {
            layer.* = TransformerLayer.init(transformer_store.withPrefix("layers").withLayer(i), config);
        }

        return .{
            .conv0 = CausalConv1d.init(conv_store.withLayer(0).withPrefix("conv"), 1),
            .conv1 = CausalConv1d.init(conv_store.withLayer(1).withPrefix("conv"), 2),
            .layers = layers,
            .norm = transformer_store.withPrefix("norm").createTensorWithTags("weight", .{.d}),
            .norm_eps = enc.norm_eps,
            .config = config,
        };
    }

    pub fn deinit(self: Encoder, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn load(self: *const Encoder, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, progress: *std.Progress.Node) !zml.Bufferized(Encoder) {
        return common.loadModel(Encoder, self, allocator, io, platform, store, progress, "encoder");
    }

    pub fn unload(self: *zml.Bufferized(Encoder), allocator: std.mem.Allocator) void {
        CausalConv1d.unload(&self.conv0);
        CausalConv1d.unload(&self.conv1);

        for (self.layers) |*layer| {
            TransformerLayer.unload(layer);
        }

        allocator.free(self.layers);
        self.norm.deinit();
    }

    /// Streaming conv stem: processes a mel chunk with history prepended.
    /// Input: mel_chunk [channels=128, time=mel_history+mel_per_step]
    /// Output: [s=dsf, d=1280] — only the new frames (history-derived outputs discarded)
    pub fn convStemStep(self: Encoder, mel_chunk: Tensor) Tensor {
        const full_output = self.convStem(mel_chunk);
        // full_output: [s=(12+1)/2=6, d=1280] when chunk is 12 frames
        // Skip first 2 (history-derived), keep last dsf=4
        const dsf: u32 = self.config.downsample_factor();
        const total: u32 = @intCast(full_output.dim(.s));
        return full_output.slice1d(.s, .{ .start = total - dsf });
    }

    /// Run the convolutional stem: conv0 + gelu + conv1 + gelu + reshape.
    /// mel: [channels=128, time=frames] → [s=(frames+1)/2, d=1280]
    pub fn convStem(self: Encoder, mel: Tensor) Tensor {
        const dtype = self.conv0.weight.dtype();
        var h = mel.convert(dtype).insertAxes(.channels, .{.batch});

        h = self.conv0.forward(h).gelu();
        h = self.conv1.forward(h).gelu();

        // [batch=1, channels=1280, time'] -> [s, d=1280]
        h = h.squeeze(.batch);
        return h.transpose(.{ .time, .channels }).rename(.{ .time = .s, .channels = .d });
    }

    /// Process a chunk of conv stem output through all transformer layers with KV cache.
    /// x: [s=dsf, d=1280], token_index: scalar, kv_cache: KvCache
    /// Returns: ([s=dsf, d=1280], KvCache)
    pub fn transformer(self: Encoder, x: Tensor, token_index: Tensor, kv_cache: KvCache, attention_metadata: zml.attention.Metadata, attention_parameters: zml.attention.Parameters) struct { Tensor, KvCache } {
        var h = x;
        var cache = kv_cache;

        for (self.layers, 0..) |layer, i| {
            const layer_cache = cache.atLayer(i);
            const result = layer.forward(h, token_index, layer_cache, self.config, attention_metadata, attention_parameters);
            h = result[0];
            const updated_layer_cache = result[1];
            cache = updated_layer_cache.reuseBuffer(layer_cache);
        }

        return .{ rmsNorm(h, self.norm, self.norm_eps), cache };
    }
};

/// Expects input layout [batch, channels, time] and kernel [cout, cin, k].
pub const CausalConv1d = struct {
    weight: Tensor,
    bias: Tensor,
    stride: i64,

    pub fn init(store: zml.io.TensorStore.View, stride: i64) CausalConv1d {
        return .{
            .weight = store.createTensorWithTags("weight", .{ .cout, .cin, .k }),
            .bias = store.createTensorWithTags("bias", .{.channels}),
            .stride = stride,
        };
    }

    pub fn unload(self: *zml.Bufferized(CausalConv1d)) void {
        self.weight.deinit();
        self.bias.deinit();
    }

    pub fn forward(self: CausalConv1d, input: Tensor) Tensor {
        const kernel_size: i64 = @intCast(self.weight.dim(.k));
        const stride = self.stride;
        const input_len: i64 = @intCast(input.dim(.time));

        // Causal padding: padding_total goes on the left.
        // Right padding is computed to align the output length with stride.
        const padding_left = kernel_size - stride;
        const numerator = input_len - kernel_size + padding_left;
        const n_frames = @divTrunc(numerator + stride - 1, stride) + 1;
        const target_length = (n_frames - 1) * stride + kernel_size - padding_left;
        const padding_right = target_length - input_len;

        const dtype = input.dtype();
        std.debug.assert(dtype == self.weight.dtype());

        var h = input.conv1d(self.weight, .{
            .window_strides = stride,
            .padding = &.{ padding_left, padding_right },
        });

        // Bias: [channels] broadcasts to [batch, channels, time]
        h = h.add(self.bias.broad(h.shape()));
        return h;
    }
};

pub const TransformerLayer = struct {
    attention_norm: Tensor,
    attention: SelfAttention,
    ffn_norm: Tensor,
    feed_forward: SwiGluFfn,
    norm_eps: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config) TransformerLayer {
        const enc = config.encoder();
        return .{
            .attention_norm = store.withPrefix("attention_norm").createTensorWithTags("weight", .{.d}),
            .attention = SelfAttention.init(store.withPrefix("attention")),
            .ffn_norm = store.withPrefix("ffn_norm").createTensorWithTags("weight", .{.d}),
            .feed_forward = SwiGluFfn.init(store.withPrefix("feed_forward")),
            .norm_eps = enc.norm_eps,
        };
    }

    pub fn unload(self: *zml.Bufferized(TransformerLayer)) void {
        self.attention_norm.deinit();
        SelfAttention.unload(&self.attention);

        self.ffn_norm.deinit();
        SwiGluFfn.unload(&self.feed_forward);
    }

    /// h: [s, d], token_index: scalar, kv_cache: KvCache → ([s, d], KvCache)
    pub fn forward(self: TransformerLayer, h: Tensor, token_index: Tensor, kv_cache: KvCache, config: Config, attention_metadata: zml.attention.Metadata, attention_parameters: zml.attention.Parameters) struct { Tensor, KvCache } {
        const attn_out, const updated_kv_cache = self.attention.forward(
            rmsNorm(h, self.attention_norm, self.norm_eps),
            token_index,
            kv_cache,
            config,
            attention_metadata,
            attention_parameters,
        );
        var out = h.add(attn_out);

        const ffn_out = self.feed_forward.forward(rmsNorm(out, self.ffn_norm, self.norm_eps));
        out = out.add(ffn_out);

        return .{ out, updated_kv_cache };
    }
};

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
    pub fn forward(self: SelfAttention, x: Tensor, token_index: Tensor, kv_cache: KvCache, config: Config, attention_metadata: zml.attention.Metadata, attention_parameters: zml.attention.Parameters) struct { Tensor, KvCache } {
        const enc = config.encoder();
        const dtype = x.dtype();

        var q = self.wq.forward(x);
        var k = self.wk.forward(x);
        var v = self.wv.forward(x);

        q = q.splitAxis(.dout, .{ .h = enc.n_heads, .hd = enc.head_dim });
        k = k.splitAxis(.dout, .{ .h = enc.n_kv_heads, .hd = enc.head_dim });
        v = v.splitAxis(.dout, .{ .h = enc.n_kv_heads, .hd = enc.head_dim });

        // Position indices for RoPE: token_index + arange(s)
        const pos_index = b: {
            const temp = Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype())
                .withTags(.{.s}).broad(zml.Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };

        const rope_opts: zml.nn.RopeOpts = .{
            .layout = .interleaved,
            .freq_base = enc.rope_theta,
        };

        q = zml.nn.rope(q, pos_index, rope_opts);
        k = zml.nn.rope(k, pos_index, rope_opts);

        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        // Update KV cache with modular indexing for sliding window circular buffer
        const cache_size = Tensor.scalar(@as(u32, @intCast(kv_cache.k.dim(.k))), token_index.dtype());
        const cache_pos = pos_index.remainder(cache_size.broad(pos_index.shape()));
        const new_kv_cache = kv_cache.update(k, v, cache_pos.rename(.{ .s = .k }));
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);

        // EXPERIMENTAL
        // Reorder K/V from circular buffer to temporal order for correct causal masking.
        // When the cache wraps, physical order != temporal order, which breaks FA2's
        // position-based causal mask for multi-token (q_len > 1) attention.
        {
            const pos_end = token_index.addConstant(x.dim(.s));
            const is_full = pos_end.cmp(.GE, cache_size);
            const rotation_start = pos_end.remainder(cache_size);
            const safe_start = is_full.select(rotation_start, Tensor.scalar(@as(u32, 0), token_index.dtype()));
            const reorder_arange = Tensor.arange(.{ .end = kv_cache.k.dim(.k) }, token_index.dtype()).withTags(.{.kk});
            const reorder_idx = reorder_arange.add(safe_start.broad(reorder_arange.shape()))
                .remainder(cache_size.broad(reorder_arange.shape()));
            k = k.gather(.{ .k = reorder_idx }, .{}).rename(.{ .kk = .k });
            v = v.gather(.{ .k = reorder_idx }, .{}).rename(.{ .kk = .k });
        }

        // Cap token_index so seqused_k doesn't exceed cache bounds
        const max_token_index = Tensor.scalar(@as(u32, @intCast(kv_cache.k.dim(.k) - x.dim(.s))), token_index.dtype());
        const attn_token_index = token_index.minimum(max_token_index);
        const attn_out = zml.attention.attention(q, k, v, attn_token_index, attention_metadata, attention_parameters, .{
            .sliding_window = @intCast(enc.sliding_window),
        });

        const merged = attn_out.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return .{ self.wo.forward(merged).rename(.{ .dout = .d }), new_kv_cache };
    }
};

pub const SwiGluFfn = struct {
    w1: zml.nn.Linear,
    w2: zml.nn.Linear,
    w3: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) SwiGluFfn {
        return .{
            .w1 = common.linear(store.withPrefix("w1")),
            .w2 = common.linear(store.withPrefix("w2")),
            .w3 = common.linear(store.withPrefix("w3")),
        };
    }

    pub fn unload(self: *zml.Bufferized(SwiGluFfn)) void {
        common.deinitLinear(&self.w1);
        common.deinitLinear(&self.w2);
        common.deinitLinear(&self.w3);
    }

    /// x: [s, d] -> [s, d]
    pub fn forward(self: SwiGluFfn, x: Tensor) Tensor {
        const gate = self.w1.forward(x).silu();
        const up = self.w3.forward(x);

        return self.w2.forward(gate.mul(up).rename(.{ .dout = .d })).rename(.{ .dout = .d });
    }
};
