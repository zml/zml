const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;
const attention = zml.attention.attention;

const common = @import("../common.zig");
const inference = @import("inference.zig");

const log = std.log.scoped(.lfm);

pub const Config = struct {
    architectures: []const []const u8,
    block_auto_adjust_ff_dim: bool,
    block_dim: u32,
    block_ff_dim: u32,
    block_ffn_dim_multiplier: f32,
    block_mlp_init_scale: f32,
    block_multiple_of: u32,
    block_norm_eps: f32,
    block_out_init_scale: f32,
    block_use_swiglu: bool,
    block_use_xavier_init: bool,
    bos_token_id: u32,
    conv_L_cache: u32,
    conv_bias: bool,
    conv_dim: u32,
    conv_use_xavier_init: bool,
    dtype: []const u8,
    eos_token_id: u32,
    hidden_size: u32,
    initializer_range: f32,
    intermediate_size: u32,
    layer_types: []const []const u8,
    max_position_embeddings: u32,
    model_type: []const u8,
    norm_eps: f32,
    num_attention_heads: u32,
    num_heads: u32,
    num_hidden_layers: u32,
    num_key_value_heads: u32,
    pad_token_id: u32,
    rope_theta: f32,
    tie_embedding: bool,
    transformers_version: []const u8,
    use_cache: bool,
    use_pos_enc: bool,
    vocab_size: u32,
};

pub const LoadedModel = struct {
    inner: Model,
    parsed_config: std.json.Parsed(Config),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
        generation: common.GenerationOptions,
    ) !LoadedModel {
        const parsed_config = try common.parseConfig(Config, allocator, io, repo);
        errdefer parsed_config.deinit();

        return .{
            .inner = .init(allocator, store, parsed_config.value, generation),
            .parsed_config = parsed_config,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        self.parsed_config.deinit();
    }

    pub fn loadBuffers(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: common.Shardings,
    ) !Buffers {
        const now: std.Io.Timestamp = .now(io, .awake);
        var total_bytes: usize = 0;

        defer {
            const took = now.untilNow(io, .awake);
            const bytes_per_sec: u64 = @intFromFloat(
                @as(f64, @floatFromInt(total_bytes)) /
                    (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s),
            );
            log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });
        }

        const all_shardings = shardings.all();
        return zml.io.load(Model, &self.inner, allocator, io, platform, store, .{
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = progress,
            .shardings = &all_shardings,
            .parallelism = 16,
            .total_bytes = &total_bytes,
        });
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        _ = self;
        Model.unloadBuffers(buffers, allocator);
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.attention.Backend,
        shardings: common.Shardings,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !inference.CompiledModel {
        const params = inference.CompilationParameters.init(self.inner, self.parsed_config.value, @intCast(seqlen), backend, false, shardings);
        return inference.CompiledModel.init(allocator, io, @constCast(platform), self, self.inner, params, progress);
    }
};

pub const Buffers = zml.Bufferized(Model);

pub const Model = struct {
    embed_tokens: TokenEmbedding,
    lm_head: LmHead,
    layers: []DecoderLayer,
    num_attention_layers: usize,
    num_conv_layers: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        root_store: zml.io.TensorStore.View,
        config: Config,
        generation: common.GenerationOptions,
    ) Model {
        const store = root_store.withPrefix("model");
        stdx.debug.assert(config.layer_types.len == config.num_hidden_layers, "Expected layer_types len {} to match num_hidden_layers {}", .{ config.layer_types.len, config.num_hidden_layers });

        const layers = allocator.alloc(DecoderLayer, config.num_hidden_layers) catch unreachable;
        var num_attention_layers: usize = 0;
        var num_conv_layers: usize = 0;
        for (layers, 0..) |*layer, i| {
            const layer_store = store.withPrefix("layers").withLayer(i);
            const kind = DecoderLayer.parseOperatorKind(config.layer_types[i]);
            switch (kind) {
                .conv => num_conv_layers += 1,
                .full_attention => num_attention_layers += 1,
            }
            layer.* = DecoderLayer.init(config, layer_store, kind);
        }

        return .{
            .embed_tokens = .init(store.withPrefix("embed_tokens")),
            .lm_head = LmHead.init(store, config, generation.sampling_strategy),
            .layers = layers,
            .num_attention_layers = num_attention_layers,
            .num_conv_layers = num_conv_layers,
        };
    }

    pub fn loadBuffers(
        self: *const Model,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: common.Shardings,
    ) !zml.Bufferized(Model) {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);
        var total_bytes: usize = 0;
        defer {
            const took = now.untilNow(io, .awake);
            const took_ns: usize = @max(1, @as(usize, @intCast(took.toNanoseconds())));
            log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{
                total_bytes,
                took,
                total_bytes * std.time.ns_per_s / took_ns,
            });
        }
        const all_shardings = shardings.all();
        return zml.io.load(Model, self, allocator, io, platform, store, .{
            .dma_chunks = 8,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = progress,
            .parallelism = 16,
            .total_bytes = &total_bytes,
            .shardings = &all_shardings,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        LmHead.unloadBuffers(&self.lm_head);
        for (self.layers) |*layer| {
            DecoderLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
    }

    pub fn forward(
        self: Model,
        tokens: zml.Tensor,
        tokens_position_offset: zml.Tensor,
        actual_seq_len: zml.Tensor,
        rng: zml.Tensor.Rng,
        cache_: Cache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
        conv_parameters: ConvParameters,
    ) struct { zml.Tensor, Cache, zml.Tensor.Rng } {
        stdx.debug.assert(tokens.shape().hasTags(.{ .batch, .seq }), "Tokens should have tags {{.batch, .seq}}, got {f}", .{tokens.shape()});

        const embeds = self.embed_tokens.forward(tokens);

        var hidden = embeds;
        var cache = cache_;
        // Incremented by the individual layers.
        var conv_cache_index = zml.Tensor.scalar(@as(u32, 0), .u32);
        var kv_cache_index = zml.Tensor.scalar(@as(u32, 0), .u32);
        for (self.layers) |layer| {
            hidden, cache, conv_cache_index, kv_cache_index = layer.forward(
                hidden,
                tokens_position_offset,
                actual_seq_len,
                cache,
                conv_cache_index,
                kv_cache_index,
                attention_metadata,
                attention_parameters,
                conv_parameters,
            );
        }
        const new_tokens, const new_rng = self.lm_head.forward(hidden, self.embed_tokens, tokens, rng);
        return .{ new_tokens, cache.reuseBuffer(cache_), new_rng };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }
};

pub const TokenEmbedding = struct {
    weight: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) TokenEmbedding {
        return .{ .weight = store.createTensor("weight", .{ .voc, .d }, null) };
    }

    pub fn forward(self: TokenEmbedding, tokens: zml.Tensor) zml.Tensor {
        stdx.debug.assert(tokens.dtype().isInteger(), "TokenEmbedding expects an integer input, received: {f}", .{tokens});
        stdx.debug.assert(self.weight.rank() == 2, "TokenEmbedding expects it's weight zml.Tensor to be a 2D matrix, got {f}", .{self.weight});
        return self.weight.gather(.{ .voc = tokens }, .{});
    }

    pub fn unembed(self: TokenEmbedding, embeds: zml.Tensor) zml.Tensor {
        stdx.debug.assert(embeds.shape().hasTags(.{.d}), "TokenEmbedding expects the input embeds to have a .d tag, got {f}", .{embeds.shape()});
        return self.weight.dot(embeds, .d);
    }
};

pub const LmHead = struct {
    embedding_norm: RmsNorm,
    sampling_strategy: zml.nn.SamplingStrategy,

    pub fn init(store: zml.io.TensorStore.View, config: Config, sampling_strategy: zml.nn.SamplingStrategy) LmHead {
        return .{
            .embedding_norm = RmsNorm.init(store.withPrefix("embedding_norm"), config.norm_eps, .d),
            .sampling_strategy = sampling_strategy,
        };
    }

    pub fn forward(self: LmHead, hidden: zml.Tensor, embed_tokens: TokenEmbedding, tokens: zml.Tensor, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        const logits = embed_tokens.unembed(self.embedding_norm.forward(hidden));
        const new_tokens, const new_rng = zml.nn.sampleTokens(logits, self.sampling_strategy, rng);
        return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), new_rng };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(LmHead)) void {
        RmsNorm.unloadBuffers(&self.embedding_norm);
    }
};

pub const OperatorKind = enum { conv, full_attention };
const Operator = union(enum) { conv: ShortConv, self_attn: Attention };

pub const DecoderLayer = struct {
    operator: Operator,
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
    feed_forward: Mlp,

    pub fn parseOperatorKind(layer_type: []const u8) OperatorKind {
        return std.meta.stringToEnum(OperatorKind, layer_type) orelse {
            stdx.debug.assert(false, "Unsupported layer type {s}", .{layer_type});
            unreachable;
        };
    }

    pub fn init(config: Config, store: zml.io.TensorStore.View, kind: OperatorKind) DecoderLayer {
        const operator_norm = RmsNorm.init(store.withPrefix("operator_norm"), config.norm_eps, .d);
        const ffn_norm = RmsNorm.init(store.withPrefix("ffn_norm"), config.norm_eps, .d);
        const feed_forward = Mlp.init(store.withPrefix("feed_forward"));
        const operator: Operator = switch (kind) {
            .conv => .{ .conv = ShortConv.init(config, store.withPrefix("conv")) },
            .full_attention => .{ .self_attn = Attention.init(config, store.withPrefix("self_attn")) },
        };

        return .{
            .operator = operator,
            .operator_norm = operator_norm,
            .ffn_norm = ffn_norm,
            .feed_forward = feed_forward,
        };
    }

    pub fn forward(
        self: DecoderLayer,
        input: zml.Tensor,
        tokens_position_offset: zml.Tensor,
        actual_seq_len: zml.Tensor,
        cache_: Cache,
        conv_cache_index_: zml.Tensor,
        kv_cache_index_: zml.Tensor,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
        conv_parameters: ConvParameters,
    ) struct { zml.Tensor, Cache, zml.Tensor, zml.Tensor } {
        var cache = cache_;
        var conv_cache_index = conv_cache_index_;
        var kv_cache_index = kv_cache_index_;
        const residual = switch (self.operator) {
            .conv => |operator| b: {
                const residual, const updated_conv_cache = operator.forward(
                    self.operator_norm.forward(input),
                    tokens_position_offset,
                    actual_seq_len,
                    cache.conv,
                    conv_cache_index,
                    conv_parameters,
                );
                cache.conv = updated_conv_cache;
                conv_cache_index = conv_cache_index.add(zml.Tensor.scalar(@as(u32, 1), .u32));
                break :b residual;
            },
            .self_attn => |operator| b: {
                const residual, const updated_kv_cache = operator.forward(
                    self.operator_norm.forward(input),
                    tokens_position_offset,
                    cache.kv,
                    kv_cache_index,
                    attention_metadata,
                    attention_parameters,
                );
                cache.kv = updated_kv_cache;
                kv_cache_index = kv_cache_index.add(zml.Tensor.scalar(@as(u32, 1), .u32));
                break :b residual;
            },
        };

        const x = input.add(residual);

        return .{
            x.add(self.feed_forward.forward(self.ffn_norm.forward(x))).reuseBuffer(input),
            cache.reuseBuffer(cache_),
            conv_cache_index.reuseBuffer(conv_cache_index_),
            kv_cache_index.reuseBuffer(kv_cache_index_),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(DecoderLayer)) void {
        switch (self.operator) {
            .conv => |*operator| ShortConv.unloadBuffers(operator),
            .self_attn => |*operator| Attention.unloadBuffers(operator),
        }
        RmsNorm.unloadBuffers(&self.operator_norm);
        RmsNorm.unloadBuffers(&self.ffn_norm);
        Mlp.unloadBuffers(&self.feed_forward);
    }
};

pub const ConvParameters = struct { is_prefill: bool };

pub const ShortConv = struct {
    in_proj: Linear,
    out_proj: Linear,
    kernel: zml.Tensor,
    config: Config,

    pub fn init(config: Config, store: zml.io.TensorStore.View) ShortConv {
        stdx.debug.assert(!config.conv_bias, "conv_bias is not supported.", .{});
        return .{
            .in_proj = initLinear(store.withPrefix("in_proj"), .d),
            .out_proj = initLinear(store.withPrefix("out_proj"), .d),
            .kernel = store.createTensor("conv.weight", .{ .out, .in, .kernel_size }, null),
            .config = config,
        };
    }

    pub fn forward(self: ShortConv, input: zml.Tensor, tokens_position_offset: zml.Tensor, actual_seq_len: zml.Tensor, cache_: ConvCache, cache_index: zml.Tensor, parameters: ConvParameters) struct { zml.Tensor, ConvCache } {
        var cache = cache_;
        const BCx = self.in_proj.forward(input);

        const B, const C, const x = BCx.chunkExact(.d, 3);
        const Bx = B.mul(x);

        const conv_out = if (parameters.is_prefill) b: {
            const actual_seq_len_i32 = actual_seq_len.convert(.i32);
            const start = actual_seq_len_i32.sub(zml.Tensor.scalar(@as(i32, @intCast(self.config.conv_L_cache)), .i32)).maximum(zml.Tensor.scalar(@as(i32, 0), .i32));
            const cache_seq_indices = lkp: {
                const n = actual_seq_len_i32.sub(start);
                const left_pad = zml.Tensor.scalar(@as(i32, @intCast(self.config.conv_L_cache)), .i32).sub(n);
                const sh = tokens_position_offset.shape().insert(.last, .{ .seq = self.config.conv_L_cache });
                break :lkp zml.Tensor.iota(sh, .seq).convert(.u32).add(left_pad.convert(.u32).broad(sh)).broad(sh);
            };
            const scatter_data = Bx.dynamicSlice(.{ .seq = @as(zml.Tensor.DynSlice, .{ .start = start.convert(.u32), .len = @intCast(self.config.conv_L_cache) }) });
            cache.state = cache.state.scatterSlices(.{ .layer = cache_index, .seq = cache_seq_indices }, scatter_data, .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(cache.state);
            const padding = self.config.conv_L_cache - 1;
            const conv_out_ = zml.Tensor.conv1d(Bx, self.kernel, .{
                .padding = &.{ padding, 0 },
                .input_batch_dimension = Bx.axis(.batch),
                .input_feature_dimension = B.axis(.d),
                .input_spatial_dimensions = Bx.axis(.seq),
                .kernel_output_feature_dimension = self.kernel.axis(.out),
                .kernel_input_feature_dimension = self.kernel.axis(.in),
                .kernel_spatial_dimensions = self.kernel.axis(.kernel_size),
                .output_batch_dimension = Bx.axis(.batch),
                .output_feature_dimension = Bx.axis(.d),
                .output_spatial_dimensions = Bx.axis(.seq),
                .feature_group_count = Bx.dim(.d),
            });
            break :b conv_out_.slice1d(.seq, .{ .end = Bx.dim(.seq) });
        } else b: {
            cache.state = cache.state.rollRight1d(.seq, -1).scatterSlices(.{ .layer = cache_index, .seq = tokens_position_offset.convert(.u32).clamp(.scalar(@as(u32, 0), .u32), .scalar(self.config.conv_L_cache - 1, .u32)) }, Bx, .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(cache.state);
            const kernel = self.kernel.squeeze(.in).rename(.{ .out = .d, .kernel_size = .seq });
            const sh = kernel.shape().insert(.last, .{ .batch = tokens_position_offset.dim(.batch) });
            break :b cache.state.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = cache_index, .len = 1 } }).squeeze(.layer).mul(kernel.broad(sh).transpose(.{ .batch, .seq, .d })).sum(.seq);
        };

        const y = C.mul(conv_out);

        const output = self.out_proj.forward(y);
        return .{ output.reuseBuffer(input), cache.reuseBuffer(cache_) };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(ShortConv)) void {
        Linear.unloadBuffers(&self.in_proj);
        Linear.unloadBuffers(&self.out_proj);
        self.kernel.deinit();
    }
};

pub const Attention = struct {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    q_layernorm: RmsNorm,
    k_layernorm: RmsNorm,
    head_dim: usize,
    num_key_value_groups: usize,
    rope_opts: zml.nn.RopeOpts,

    pub fn init(config: Config, store: zml.io.TensorStore.View) Attention {
        const head_dim = config.hidden_size / config.num_attention_heads;
        const num_key_value_groups = config.num_attention_heads / config.num_key_value_heads;
        return .{
            .q_proj = initLinear(store.withPrefix("q_proj"), .d),
            .k_proj = initLinear(store.withPrefix("k_proj"), .d),
            .v_proj = initLinear(store.withPrefix("v_proj"), .d),
            .out_proj = initLinear(store.withPrefix("out_proj"), .d),
            .q_layernorm = RmsNorm.init(store.withPrefix("q_layernorm"), config.norm_eps, .hd),
            .k_layernorm = RmsNorm.init(store.withPrefix("k_layernorm"), config.norm_eps, .hd),
            .head_dim = head_dim,
            .num_key_value_groups = num_key_value_groups,
            .rope_opts = .{ .layout = .sequential, .scaling = .{ .default = .{ .rope_theta = config.rope_theta } } },
        };
    }

    pub fn forward(
        self: Attention,
        x: zml.Tensor,
        tokens_position_offset: zml.Tensor,
        kv_cache: KvCache,
        cache_index: zml.Tensor,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { zml.Tensor, KvCache } {
        var q = self.q_proj.forward(x).splitAxis(-1, .{ .h = .auto, .hd = self.head_dim });
        var k = self.k_proj.forward(x).splitAxis(-1, .{ .h = .auto, .hd = self.head_dim });
        var v = self.v_proj.forward(x).splitAxis(-1, .{ .h = .auto, .hd = self.head_dim });

        q = self.q_layernorm.forward(q);
        k = self.k_layernorm.forward(k);

        const token_positions = b: {
            const sh = tokens_position_offset.shape().insert(.last, .{ .seq = x.dim(.seq) });
            break :b zml.Tensor.iota(sh, .seq).convert(.u32).add(tokens_position_offset.broad(sh));
        };

        q = zml.nn.rope(q, token_positions, self.rope_opts);
        k = zml.nn.rope(k, token_positions, self.rope_opts);

        q = q.rename(.{ .seq = .q });
        k = k.rename(.{ .seq = .k });
        v = v.rename(.{ .seq = .k });

        const new_kv_cache = kv_cache.update(k, v, tokens_position_offset, cache_index);
        k = new_kv_cache.keys(cache_index);
        v = new_kv_cache.values(cache_index);

        stdx.debug.assert(q.dim(.batch) == 1, "LFM attention currently expects batch size 1 for flash attention backend, got {}", .{q.dim(.batch)});
        const attn = switch (attention_parameters) {
            .vanilla => zml.attention.attention.attention(q, k, v, tokens_position_offset, attention_metadata, attention_parameters).merge(.{ .d = .{ .h, .hd } }),
            .neuron => zml.attention.attention.attention(q.squeeze(.batch), k.squeeze(.batch), v.squeeze(.batch), tokens_position_offset.squeeze(.batch), attention_metadata, attention_parameters).merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .seq }).insertAxes(.seq, .{.batch}),
            .cuda_fa2, .cuda_fa3 => zml.attention.attention.attention(q.squeeze(.batch), k.squeeze(.batch), v.squeeze(.batch), tokens_position_offset.squeeze(.batch), attention_metadata, attention_parameters).merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .seq }).insertAxes(.seq, .{.batch}),
        };

        return .{ self.out_proj.forward(attn).reuseBuffer(x), new_kv_cache.reuseBuffer(kv_cache) };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Attention)) void {
        Linear.unloadBuffers(&self.q_proj);
        Linear.unloadBuffers(&self.k_proj);
        Linear.unloadBuffers(&self.v_proj);
        Linear.unloadBuffers(&self.out_proj);
        RmsNorm.unloadBuffers(&self.q_layernorm);
        RmsNorm.unloadBuffers(&self.k_layernorm);
    }
};

pub const Cache = struct {
    conv: ConvCache,
    kv: KvCache,

    pub fn initBuffers(self: Cache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, sharding: zml.sharding.Sharding) !zml.Bufferized(Cache) {
        return .{ .conv = try self.conv.initBuffers(allocator, io, platform, sharding), .kv = try self.kv.initBuffers(allocator, io, platform, sharding) };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Cache)) void {
        ConvCache.unloadBuffers(&self.conv);
        KvCache.unloadBuffers(&self.kv);
    }

    pub fn reuseBuffer(self: Cache, other: Cache) Cache {
        return .{ .conv = self.conv.reuseBuffer(other.conv), .kv = self.kv.reuseBuffer(other.kv) };
    }
};

pub const ConvCache = struct {
    state: zml.Tensor,

    pub fn init(shape: zml.Shape) ConvCache {
        return .{ .state = .fromShape(shape) };
    }

    pub fn initBuffers(self: ConvCache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, sharding: zml.sharding.Sharding) !zml.Bufferized(ConvCache) {
        const sh = self.state.shape();
        const host = try allocator.alloc(u8, sh.byteSize());
        defer allocator.free(host);
        @memset(host, 0);
        return .{ .state = try zml.Buffer.fromBytes(io, platform, sh, sharding, host) };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(ConvCache)) void {
        self.state.deinit();
    }

    pub fn reuseBuffer(self: ConvCache, other: ConvCache) ConvCache {
        return .{ .state = self.state.reuseBuffer(other.state) };
    }
};

pub const KvCache = struct {
    k: zml.Tensor,
    v: zml.Tensor,

    pub fn init(kv_shape: zml.Shape) KvCache {
        return .{ .k = .fromShape(kv_shape), .v = .fromShape(kv_shape) };
    }

    pub fn initBuffers(self: KvCache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, sharding: zml.sharding.Sharding) !zml.Bufferized(KvCache) {
        const sh = self.k.shape();
        const host = try allocator.alloc(u8, sh.byteSize());
        defer allocator.free(host);
        @memset(host, 0);

        // Decode kernels receive the full static cache shape. Vanilla attention
        // masks future positions semantically, but custom kernels may still load
        // those lanes while applying the mask, so unwritten slots must be stable.
        return .{ .k = try zml.Buffer.fromBytes(io, platform, sh, sharding, host), .v = try zml.Buffer.fromBytes(io, platform, self.v.shape(), sharding, host) };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(KvCache)) void {
        self.k.deinit();
        self.v.deinit();
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{ .k = self.k.reuseBuffer(other.k), .v = self.v.reuseBuffer(other.v) };
    }

    pub fn keys(self: KvCache, cache_index: zml.Tensor) zml.Tensor {
        return self.k.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = cache_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn values(self: KvCache, cache_index: zml.Tensor) zml.Tensor {
        return self.v.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = cache_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn update(self: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_position: zml.Tensor, cache_index: zml.Tensor) KvCache {
        const k_shape = self.k.shape().drop(.layer);
        const layer = cache_index.broad(token_position.shape());
        return .{
            .k = self.k.scatterSlices(.{ .layer = layer, .k = token_position }, new_k.transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.k),
            .v = self.v.scatterSlices(.{ .layer = layer, .k = token_position }, new_v.transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.v),
        };
    }
};

fn initLinear(store: zml.io.TensorStore.View, tag: anytype) Linear {
    return .init(store.createTensor("weight", .{ .out, tag }, null), null, tag);
}

pub const Linear = struct {
    weight: zml.Tensor,
    bias: ?zml.Tensor = null,
    in_tag: zml.Shape.Tag,
    out_tag: zml.Shape.Tag,

    pub fn init(weight: zml.Tensor, bias: ?zml.Tensor, tag: anytype) Linear {
        stdx.debug.guard(weight.shape().hasTag(tag) != null, @src());
        const axis = weight.shape().axis(tag);
        const out_tag = weight.shape().tag(1 - axis);
        return .{ .weight = weight, .bias = bias, .in_tag = zml.Shape.toTag(tag), .out_tag = out_tag };
    }

    pub fn unloadBuffers(linear: *zml.Bufferized(Linear)) void {
        linear.weight.deinit();
        if (linear.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: Linear, x: zml.Tensor) zml.Tensor {
        var y = x.dot(self.weight, self.in_tag).renameTag(self.out_tag, self.in_tag);
        return if (self.bias) |bias| y.add(bias.broad(y.shape())).reuseBuffer(x) else y;
    }
};

const Mlp = struct {
    w1: Linear,
    w2: Linear,
    w3: Linear,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{ .w1 = initLinear(store.withPrefix("w1"), .d), .w2 = initLinear(store.withPrefix("w2"), .d), .w3 = initLinear(store.withPrefix("w3"), .d) };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        Linear.unloadBuffers(&self.w1);
        Linear.unloadBuffers(&self.w2);
        Linear.unloadBuffers(&self.w3);
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        return self.w2.forward(self.w1.forward(x).silu().mul(self.w3.forward(x))).reuseBuffer(x);
    }
};

const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32,
    tag: zml.Shape.Tag,

    pub fn init(store: zml.io.TensorStore.View, eps: f32, tag: anytype) RmsNorm {
        return .{ .weight = store.createTensor("weight", .{tag}, null), .eps = eps, .tag = zml.Shape.toTag(tag) };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const normalized = zml.nn.rmsNorm(input, self.tag, self.eps);
        return normalized.mul(self.weight.broad(input.shape())).reuseBuffer(input);
    }
};
