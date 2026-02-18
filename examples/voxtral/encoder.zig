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
        common.deinitBufferized(self);
        allocator.free(self.layers);
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
        const attn_config = self.config.encoder().attentionConfig();

        for (self.layers, 0..) |layer, i| {
            const layer_cache = cache.atLayer(i);
            const result = layer.forward(h, token_index, layer_cache, attn_config, attention_metadata, attention_parameters);
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
        common.deinitBufferized(self);
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
        common.deinitBufferized(self);
    }

    /// h: [s, d], token_index: scalar, kv_cache: KvCache → ([s, d], KvCache)
    pub fn forward(self: TransformerLayer, h: Tensor, token_index: Tensor, kv_cache: KvCache, attn_config: cfg.AttentionConfig, attention_metadata: zml.attention.Metadata, attention_parameters: zml.attention.Parameters) struct { Tensor, KvCache } {
        const attn_out, const updated_kv_cache = self.attention.forward(
            rmsNorm(h, self.attention_norm, self.norm_eps),
            token_index,
            kv_cache,
            attn_config,
            attention_metadata,
            attention_parameters,
        );
        var out = h.add(attn_out);

        const ffn_out = self.feed_forward.forward(rmsNorm(out, self.ffn_norm, self.norm_eps));
        out = out.add(ffn_out);

        return .{ out, updated_kv_cache };
    }
};

pub const SelfAttention = common.SelfAttention(true);

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
        common.deinitBufferized(self);
    }

    /// x: [s, d] -> [s, d]
    pub fn forward(self: SwiGluFfn, x: Tensor) Tensor {
        const gate = self.w1.forward(x).silu();
        const up = self.w3.forward(x);

        return self.w2.forward(gate.mul(up).rename(.{ .dout = .d })).rename(.{ .dout = .d });
    }
};
