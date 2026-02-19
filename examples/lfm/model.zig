const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const attention = zml.attention.attention;

const cfg = @import("config.zig");
pub const Config = cfg.Config;

const log = std.log.scoped(.lfm);

pub const Model = struct {
    embed_tokens: TokenEmbedding,
    lm_head: LmHead,
    layers: []DecoderLayer,
    num_attention_layers: usize,
    num_conv_layers: usize,

    pub fn init(allocator: std.mem.Allocator, root_store: zml.io.TensorStore.View, config: Config) Model {
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
            .lm_head = LmHead.init(store, config),
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
    ) !zml.Bufferized(Model) {
        progress.increaseEstimatedTotalItems(store.view().count());

        const now: std.Io.Timestamp = .now(io, .awake);
        var total_bytes: usize = 0;

        defer {
            const took = now.untilNow(io, .awake);
            const took_ns: usize = @max(1, @as(usize, @intCast(took.toNanoseconds())));
            log.info("Loaded weights [{Bi:.2}, {D}, {Bi:.2}/s]", .{
                total_bytes,
                stdx.fmt.fmtDuration(took),
                total_bytes * std.time.ns_per_s / took_ns,
            });
        }

        return zml.io.load(Model, self, allocator, io, platform, .{
            .dma_chunks = 8,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = progress,
            .store = store,
            .parallelism = 16,
            .total_bytes = &total_bytes,
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
        //
        self: Model,
        tokens: Tensor,
        tokens_position_offset: Tensor,
        actual_seq_len: Tensor,
        rng: Tensor.Rng,
        cache_: Cache,
        attention_metadata: attention.Metadata,
        attention_parameters: attention.Parameters,
        conv_parameters: ConvParameters,
    ) struct { Tensor, Cache, Tensor.Rng } {
        stdx.debug.assert(tokens.shape().hasTags(.{ .batch, .seq }), "Tokens should have tags {{.batch, .seq}}, got {f}", .{tokens.shape()});

        const embeds = self.embed_tokens.forward(tokens);

        var hidden = embeds;
        var cache = cache_;
        var conv_cache_index = Tensor.scalar(@as(u32, 0), .u32);
        var kv_cache_index = Tensor.scalar(@as(u32, 0), .u32);
        for (self.layers) |layer| {
            hidden, cache, conv_cache_index, kv_cache_index = layer.forward(hidden, tokens_position_offset, actual_seq_len, cache, conv_cache_index, kv_cache_index, attention_metadata, attention_parameters, conv_parameters);
        }

        const new_tokens, const new_rng = self.lm_head.forward(hidden, self.embed_tokens, tokens, rng);
        return .{ new_tokens, cache.reuseBuffer(cache_), new_rng };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }
};

pub const TokenEmbedding = struct {
    weight: Tensor,

    pub fn init(store: zml.io.TensorStore.View) TokenEmbedding {
        return .{ .weight = store.createTensor("weight").withTags(.{ .voc, .d }) };
    }

    pub fn forward(self: TokenEmbedding, tokens: Tensor) Tensor {
        stdx.debug.assert(tokens.dtype().isInteger(), "TokenEmbedding expects an integer input, received: {f}", .{tokens});
        stdx.debug.assert(self.weight.rank() == 2, "TokenEmbedding expects it's weight Tensor to be a 2D matrix, got {f}", .{self.weight});
        return self.weight.gather(.{ .voc = tokens }, .{});
    }

    pub fn unembed(self: TokenEmbedding, embeds: Tensor) Tensor {
        stdx.debug.assert(embeds.shape().hasTags(.{.d}), "TokenEmbedding expects the input embeds to have a .d tag, got {f}", .{embeds.shape()});
        return self.weight.dot(embeds, .d);
    }
};

pub const LmHead = struct {
    embedding_norm: RmsNorm,

    pub fn init(store: zml.io.TensorStore.View, config: Config) LmHead {
        return .{
            .embedding_norm = RmsNorm.init(store.withPrefix("embedding_norm"), config.norm_eps, .d),
        };
    }

    pub fn forward(self: LmHead, hidden: Tensor, embed_tokens: TokenEmbedding, tokens: Tensor, rng: Tensor.Rng) struct { Tensor, Tensor.Rng } {
        const logits = embed_tokens.unembed(self.embedding_norm.forward(hidden));
        const new_tokens, const new_rng = zml.nn.sampleTokens(logits, .{}, rng);
        return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), new_rng };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(LmHead)) void {
        RmsNorm.unloadBuffers(&self.embedding_norm);
    }
};

pub const OperatorKind = enum { conv, full_attention };
const Operator = union(enum) {
    conv: ShortConv,
    self_attn: Attention,
};

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
        input: Tensor,
        tokens_position_offset: Tensor,
        actual_seq_len: Tensor,
        cache_: Cache,
        conv_cache_index_: Tensor,
        kv_cache_index_: Tensor,
        attention_metadata: attention.Metadata,
        attention_parameters: attention.Parameters,
        conv_parameters: ConvParameters,
    ) struct { Tensor, Cache, Tensor, Tensor } {
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
                conv_cache_index = conv_cache_index.add(Tensor.scalar(@as(u32, 1), .u32));
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
                kv_cache_index = kv_cache_index.add(Tensor.scalar(@as(u32, 1), .u32));
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
        return ShortConv{ .in_proj = initLinear(store.withPrefix("in_proj"), .d), .out_proj = initLinear(store.withPrefix("out_proj"), .d), .kernel = store.createTensorWithTags("conv.weight", .{ .out, .in, .kernel_size }), .config = config };
    }

    pub fn forward(
        self: ShortConv,
        input: Tensor,
        tokens_position_offset: Tensor,
        actual_seq_len: Tensor,
        cache_: ConvCache,
        cache_index: Tensor,
        parameters: ConvParameters,
    ) struct { Tensor, ConvCache } {
        var cache = cache_;
        // Allows us to have a consistent API for the generic DecoderLayer
        const BCx = self.in_proj.forward(input);

        const B, const C, const x = BCx.chunkExact(.d, 3);
        const Bx = B.mul(x);

        const conv_out = if (parameters.is_prefill) b: {
            const actual_seq_len_i32 = actual_seq_len.convert(.i32);
            const start = actual_seq_len_i32.sub(Tensor.scalar(@as(i32, @intCast(self.config.conv_L_cache)), .i32)).maximum(Tensor.scalar(@as(i32, 0), .i32));
            const cache_seq_indices = lkp: {
                const n = actual_seq_len_i32.sub(start);
                const left_pad = Tensor.scalar(@as(i32, @intCast(self.config.conv_L_cache)), .i32).sub(n);
                const sh = tokens_position_offset.shape().insert(.last, .{ .seq = self.config.conv_L_cache });
                break :lkp Tensor.iota(sh, .seq).convert(.u32).add(left_pad.convert(.u32).broad(sh)).broad(sh);
            };
            const scatter_data = Bx.dynamicSlice(.{ .seq = @as(Tensor.DynSlice, .{ .start = start.convert(.u32), .len = @intCast(self.config.conv_L_cache) }) });
            cache.state = cache.state.scatterSlices(
                .{ .layer = cache_index, .seq = cache_seq_indices },
                scatter_data,
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(cache.state);
            const padding = self.config.conv_L_cache - 1;
            const conv_out = Tensor.conv1d(Bx, self.kernel, .{
                .padding = &.{ padding, 0 },
                .input_batch_dimension = Bx.axis(.batch),
                .input_feature_dimension = B.axis(.d),
                .input_spatial_dimensions = B.axis(.seq),
                .kernel_output_feature_dimension = self.kernel.axis(.out),
                .kernel_input_feature_dimension = self.kernel.axis(.in),
                .kernel_spatial_dimensions = self.kernel.axis(.kernel_size),
                .output_batch_dimension = Bx.axis(.batch),
                .output_feature_dimension = Bx.axis(.d),
                .output_spatial_dimensions = Bx.axis(.seq),
                .feature_group_count = Bx.dim(.d),
            });
            break :b conv_out.slice1d(.seq, .{ .end = Bx.dim(.seq) });
        } else b: {
            cache.state = cache.state.rollRight1d(.seq, -1)
                .scatterSlices(
                    .{ .layer = cache_index, .seq = tokens_position_offset.convert(.u32).clamp(.scalar(@as(u32, @intCast(0)), .u32), .scalar(self.config.conv_L_cache - 1, .u32)) },
                    Bx,
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(cache.state);
            const kernel = self.kernel.squeeze(.in).rename(.{ .out = .d, .kernel_size = .seq });
            const sh = kernel.shape().insert(.last, .{ .batch = tokens_position_offset.dim(.batch) });

            break :b cache.state.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = cache_index, .len = 1 } }).squeeze(.layer)
                .mul(kernel.broad(sh).transpose(.{ .batch, .seq, .d }))
                .sum(.seq);
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
            //
            .q_proj = initLinear(store.withPrefix("q_proj"), .d),
            .k_proj = initLinear(store.withPrefix("k_proj"), .d),
            .v_proj = initLinear(store.withPrefix("v_proj"), .d),
            .out_proj = initLinear(store.withPrefix("out_proj"), .d),
            .q_layernorm = RmsNorm.init(store.withPrefix("q_layernorm"), config.norm_eps, .hd),
            .k_layernorm = RmsNorm.init(store.withPrefix("k_layernorm"), config.norm_eps, .hd),
            .head_dim = head_dim,
            .num_key_value_groups = num_key_value_groups,
            .rope_opts = .{
                .layout = .sequential,
                .freq_base = config.rope_theta,
                .scaling = .{ .default = .{} },
            },
        };
    }

    pub fn forward(
        self: Attention,
        x: Tensor,
        tokens_position_offset: Tensor,
        kv_cache: KvCache,
        cache_index: Tensor,
        attention_metadata: attention.Metadata,
        attention_parameters: attention.Parameters,
    ) struct { Tensor, KvCache } {
        var q = self.q_proj.forward(x).splitAxis(-1, .{ .h = .auto, .hd = self.head_dim });
        var k = self.k_proj.forward(x).splitAxis(-1, .{ .h = .auto, .hd = self.head_dim });
        var v = self.v_proj.forward(x).splitAxis(-1, .{ .h = .auto, .hd = self.head_dim });

        q = self.q_layernorm.forward(q);
        k = self.k_layernorm.forward(k);

        const token_positions = b: {
            const sh = tokens_position_offset.shape().insert(.last, .{ .seq = x.dim(.seq) });
            break :b Tensor.iota(sh, .seq).convert(.u32).add(tokens_position_offset.broad(sh));
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
            .vanilla => attention.attention(
                q,
                k,
                v,
                tokens_position_offset,
                attention_metadata,
                attention_parameters,
            ).merge(.{ .d = .{ .h, .hd } }),
            // flash attention doesn't support batch dimension at all in practice.
            else => attention.attention(
                q.squeeze(.batch),
                k.squeeze(.batch),
                v.squeeze(.batch),
                tokens_position_offset.squeeze(.batch),
                attention_metadata,
                attention_parameters,
            ).merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .seq }).insertAxes(.seq, .{.batch}),
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

    pub fn initBuffers(self: Cache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Cache) {
        return .{
            .conv = try self.conv.initBuffers(allocator, io, platform),
            .kv = try self.kv.initBuffers(io, platform),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Cache)) void {
        ConvCache.unloadBuffers(&self.conv);
        KvCache.unloadBuffers(&self.kv);
    }

    pub fn reuseBuffer(self: Cache, other: Cache) Cache {
        return .{
            .conv = self.conv.reuseBuffer(other.conv),
            .kv = self.kv.reuseBuffer(other.kv),
        };
    }
};

pub const ConvCache = struct {
    state: Tensor,

    pub fn init(shape: zml.Shape) ConvCache {
        return .{
            .state = .fromShape(shape),
        };
    }

    pub fn initBuffers(self: ConvCache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(ConvCache) {
        const sh = self.state.shape();
        const host = try allocator.alloc(u8, sh.byteSize());
        defer allocator.free(host);
        @memset(host, 0);

        return .{ .state = try zml.Buffer.fromBytes(io, platform, sh, host) };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(ConvCache)) void {
        self.state.deinit();
    }

    pub fn reuseBuffer(self: ConvCache, other: ConvCache) ConvCache {
        return .{
            .state = self.state.reuseBuffer(other.state),
        };
    }
};

pub const KvCache = struct {
    k: Tensor,
    v: Tensor,

    pub fn init(kv_shape: zml.Shape) KvCache {
        return .{
            .k = .fromShape(kv_shape),
            .v = .fromShape(kv_shape),
        };
    }

    pub fn initBuffers(self: KvCache, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(KvCache) {
        return .{
            .k = try zml.Buffer.uninitialized(io, platform, self.k.shape(), .{}),
            .v = try zml.Buffer.uninitialized(io, platform, self.v.shape(), .{}),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(KvCache)) void {
        self.k.deinit();
        self.v.deinit();
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{
            .k = self.k.reuseBuffer(other.k),
            .v = self.v.reuseBuffer(other.v),
        };
    }

    pub fn keys(self: KvCache, cache_index: Tensor) Tensor {
        return self.k.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = cache_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn values(self: KvCache, cache_index: Tensor) Tensor {
        return self.v.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = cache_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn update(self: KvCache, new_k: Tensor, new_v: Tensor, token_position: Tensor, cache_index: Tensor) KvCache {
        const k_shape = self.k.shape().drop(.layer);
        const layer = cache_index.broad(token_position.shape());

        return .{
            .k = self.k.scatterSlices(
                .{ .layer = layer, .k = token_position },
                new_k.transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(self.k),
            .v = self.v.scatterSlices(
                .{ .layer = layer, .k = token_position },
                new_v.transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(self.v),
        };
    }
};

fn initLinear(store: zml.io.TensorStore.View, tag: anytype) Linear {
    return .init(store.createTensorWithTags("weight", .{ .out, tag }), null, tag);
}

pub const Linear = struct {
    weight: Tensor,
    bias: ?Tensor = null,
    in_tag: Shape.Tag,
    out_tag: Shape.Tag,

    /// Assumes that the weights are tagged `.in` and `.out`.
    pub fn init(weight: Tensor, bias: ?Tensor, tag: anytype) Linear {
        stdx.debug.guard(weight.shape().hasTag(tag) != null, @src());
        const axis = weight.shape().axis(tag);
        const out_tag = weight.shape().tag(1 - axis);
        return .{
            .weight = weight,
            .bias = bias,
            .in_tag = zml.Shape.toTag(tag),
            .out_tag = out_tag,
        };
    }

    pub fn unloadBuffers(linear: *zml.Bufferized(Linear)) void {
        linear.weight.deinit();
        if (linear.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: Linear, x: Tensor) Tensor {
        var y = x.dot(self.weight, self.in_tag).renameTag(self.out_tag, self.in_tag);

        return if (self.bias) |bias| y.add(bias.broad(y.shape())).reuseBuffer(x) else y;
    }
};

const Mlp = struct {
    w1: Linear,
    w2: Linear,
    w3: Linear,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .w1 = initLinear(store.withPrefix("w1"), .d),
            .w2 = initLinear(store.withPrefix("w2"), .d),
            .w3 = initLinear(store.withPrefix("w3"), .d),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        Linear.unloadBuffers(&self.w1);
        Linear.unloadBuffers(&self.w2);
        Linear.unloadBuffers(&self.w3);
    }

    pub fn forward(self: Mlp, x: Tensor) Tensor {
        return self.w2.forward(self.w1.forward(x).silu().mul(self.w3.forward(x))).reuseBuffer(x);
    }
};

const RmsNorm = struct {
    weight: Tensor,
    eps: f32,
    tag: Shape.Tag,

    pub fn init(store: zml.io.TensorStore.View, eps: f32, tag: anytype) RmsNorm {
        return .{ .weight = store.createTensorWithTags("weight", .{tag}), .eps = eps, .tag = zml.Shape.toTag(tag) };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    /// L2 normalization of input tensor along `.d` axis.
    pub fn forward(self: RmsNorm, input: Tensor) Tensor {
        const normalized = zml.nn.rmsNorm(input, self.tag, self.eps);
        return normalized.mul(self.weight.broad(input.shape())).reuseBuffer(input);
    }
};
