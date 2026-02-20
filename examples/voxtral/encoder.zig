const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;

const cfg = @import("config.zig");
const Config = cfg.Config;

const common = @import("common.zig");
const rmsNorm = common.rmsNorm;
const KvCache = common.KvCache;

pub const Encoder = struct {
    conv0_weight: Tensor,
    conv0_bias: Tensor,
    conv1_weight: Tensor,
    conv1_bias: Tensor,
    layers: []TransformerLayer,
    norm: Tensor,
    norm_eps: f32,
    config: Config,

    pub const ConvState = struct {
        conv1: Tensor, // [batch=1, channels=128, time=2]
        conv2: Tensor, // [batch=1, channels=enc_dim, time=2]

        pub fn initBuffer(self: ConvState, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(ConvState) {
            return .{
                .conv1 = try .uninitialized(io, platform, self.conv1.shape(), .{}),
                .conv2 = try .uninitialized(io, platform, self.conv2.shape(), .{}),
            };
        }

        pub fn deinitBuffer(self: *zml.Bufferized(ConvState)) void {
            self.conv1.deinit();
            self.conv2.deinit();
        }
    };

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) Encoder {
        const encoder_store = store.withPrefix("mm_streams_embeddings.embedding_module.whisper_encoder");
        const conv_store = encoder_store.withPrefix("conv_layers");
        const transformer_store = encoder_store.withPrefix("transformer");

        const enc = config.encoder();
        const layers = allocator.alloc(TransformerLayer, enc.n_layers) catch unreachable;
        for (layers, 0..) |*layer, i| {
            layer.* = TransformerLayer.init(transformer_store.withPrefix("layers").withLayer(i), config);
        }

        const conv0_store = conv_store.withLayer(0).withPrefix("conv");
        const conv1_store = conv_store.withLayer(1).withPrefix("conv");

        return .{
            .conv0_weight = conv0_store.createTensorWithTags("weight", .{ .cout, .cin, .k }),
            .conv0_bias = conv0_store.createTensorWithTags("bias", .{.channels}),
            .conv1_weight = conv1_store.createTensorWithTags("weight", .{ .cout, .cin, .k }),
            .conv1_bias = conv1_store.createTensorWithTags("bias", .{.channels}),
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

    /// Prefill conv stem: runs conv on all prefill mel frames and extracts
    /// conv states (last kernel_size-1=2 frames of each intermediate) for subsequent streaming.
    /// Uses padding={2,0} to match reference zero-initialized state (kernel_size-1=2).
    /// mel: [channels=128, time=frames] → ([s=(frames+1)/2, d=enc_dim], ConvState)
    pub fn convStemPrefill(self: Encoder, mel: Tensor) struct { Tensor, ConvState } {
        const dtype = self.conv0_weight.dtype();
        const h = mel.convert(dtype).insertAxes(.channels, .{.batch});

        // Conv0 (stride=1): pad kernel_size-1=2 zeros on left (matches zero-initialized state)
        var conv0_out = h.conv1d(self.conv0_weight, .{ .window_strides = 1, .padding = &.{ 2, 0 } });
        conv0_out = conv0_out.add(self.conv0_bias.broad(conv0_out.shape())).gelu();

        // Conv1 (stride=2): pad kernel_size-1=2 zeros on left (matches zero-initialized state)
        var conv1_out = conv0_out.conv1d(self.conv1_weight, .{ .window_strides = 2, .padding = &.{ 2, 0 } });
        conv1_out = conv1_out.add(self.conv1_bias.broad(conv1_out.shape())).gelu();

        // Output: [s=(frames+1)/2, d=enc_dim]
        const output = conv1_out.squeeze(.batch)
            .transpose(.{ .time, .channels })
            .rename(.{ .time = .s, .channels = .d });

        // Extract conv states from tails (last kernel_size-1=2 frames)
        const h_time: u32 = @intCast(h.dim(.time));
        const conv0_time: u32 = @intCast(conv0_out.dim(.time));

        return .{ output, .{
            .conv1 = h.slice1d(.time, .{ .start = h_time - 2 }),
            .conv2 = conv0_out.slice1d(.time, .{ .start = conv0_time - 2 }),
        } };
    }

    /// Streaming conv stem: processes mel_per_step mel frames with explicit conv states.
    /// Concatenates state (2 frames of left context) with new input, runs conv1d (no padding),
    /// then saves the tail as new state. Matches the reference ExecuTorch implementation.
    /// mel_chunk: [channels=128, time=mel_per_step], conv_state: ConvState
    /// Returns: ([s=dsf, d=enc_dim], new ConvState)
    pub fn convStemStep(self: Encoder, mel_chunk: Tensor, conv_state: ConvState) struct { Tensor, ConvState } {
        const dtype = self.conv0_weight.dtype();
        const mel_with_batch = mel_chunk.convert(dtype).insertAxes(.channels, .{.batch});

        // Conv0 (stride=1): concat state with new mel, conv1d with no padding
        const conv0_input = Tensor.concatenate(&.{ conv_state.conv1, mel_with_batch }, .time);
        var conv0_out = conv0_input.conv1d(self.conv0_weight, .{ .window_strides = 1 });
        conv0_out = conv0_out.add(self.conv0_bias.broad(conv0_out.shape())).gelu();
        const mel_time: u32 = @intCast(mel_with_batch.dim(.time));
        const new_conv1_state = mel_with_batch.slice1d(.time, .{ .start = mel_time - 2 });

        // Conv1 (stride=2): concat state with conv0 output, conv1d with no padding
        const conv1_input = Tensor.concatenate(&.{ conv_state.conv2, conv0_out }, .time);
        var conv1_out = conv1_input.conv1d(self.conv1_weight, .{ .window_strides = 2 });
        conv1_out = conv1_out.add(self.conv1_bias.broad(conv1_out.shape())).gelu();
        const conv0_out_time: u32 = @intCast(conv0_out.dim(.time));
        const new_conv2_state = conv0_out.slice1d(.time, .{ .start = conv0_out_time - 2 });

        // Reshape to [s=dsf, d=enc_dim]
        const output = conv1_out.squeeze(.batch)
            .transpose(.{ .time, .channels })
            .rename(.{ .time = .s, .channels = .d });

        return .{ output, .{
            .conv1 = new_conv1_state,
            .conv2 = new_conv2_state,
        } };
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

    /// Like transformer, but also returns the incremented token index (on-device).
    pub fn transformerStep(self: Encoder, x: Tensor, token_index: Tensor, kv_cache: KvCache, attention_metadata: zml.attention.Metadata, attention_parameters: zml.attention.Parameters) struct { Tensor, KvCache, Tensor } {
        const output, const cache = self.transformer(x, token_index, kv_cache, attention_metadata, attention_parameters);
        const next_index = token_index.addConstant(self.config.downsample_factor()).reuseBuffer(token_index);
        return .{ output, cache, next_index };
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

pub const SwiGluFfn = common.SwiGluFfn;
