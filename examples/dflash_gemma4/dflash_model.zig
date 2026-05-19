const std = @import("std");

const zml = @import("zml");

pub const KvCache = struct {
    kv: zml.Tensor,
    layer_index: ?u32,

    pub const Buffer = zml.Bufferized(KvCache);

    pub fn init(kv_shape: zml.Shape) KvCache {
        const sharded_shape = kv_shape.insert(.k, .{ .kv = 2 }).withPartitioning(.{ .h = .model });
        return .{
            .kv = .fromShape(sharded_shape),
            .layer_index = null,
        };
    }

    pub fn deinitBuffer(kv: *Buffer) void {
        kv.kv.deinit();
    }

    pub fn replaceBuffers(dst: *Buffer, src: *Buffer) void {
        replaceBuffer(&dst.kv, &src.kv);
    }

    pub fn initZeroBuffer(
        kv: KvCache,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        sharding: zml.Sharding,
    ) !Buffer {
        return .{
            .kv = try zeroBuffer(allocator, io, platform, kv.kv.shape(), sharding),
        };
    }

    pub fn keys(kv: KvCache) zml.Tensor {
        return kv.kv
            .slice1d(.layer, .single(kv.layer_index orelse @panic("forgot to call atLayer")))
            .slice1d(.kv, .single(0));
    }

    pub fn values(kv: KvCache) zml.Tensor {
        return kv.kv
            .slice1d(.layer, .single(kv.layer_index orelse @panic("forgot to call atLayer")))
            .slice1d(.kv, .single(1));
    }

    pub fn update(kv: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: zml.Tensor) KvCache {
        const update_shape = kv.kv.shape().drop(.layer);
        const layer: zml.Tensor = .scalar(kv.layer_index orelse @panic("forgot to call atLayer"), .u32);
        const new_kv = zml.Tensor.stack(&.{
            new_k.convert(kv.kv.dtype()),
            new_v.convert(kv.kv.dtype()),
        }, .k, .kv);

        return .{
            .kv = kv.kv.scatterSlices(.{ .layer = layer, .k = token_index }, new_kv.transpose(update_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.kv),
            .layer_index = kv.layer_index,
        };
    }

    pub fn atLayer(kv: KvCache, layer_index: usize) KvCache {
        return .{
            .kv = kv.kv,
            .layer_index = @intCast(layer_index),
        };
    }

    pub fn reuseBuffer(kv: KvCache, other: KvCache) KvCache {
        return .{
            .kv = kv.kv.reuseBuffer(other.kv),
            .layer_index = null,
        };
    }
};

pub const Config = struct {
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: ?u32 = null,
    attention_bias: bool = false,
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = .{} },
    block_size: u32,
    sliding_window: ?u32 = null,
    use_sliding_window: bool = false,
    layer_types: []const LayerType = &.{},
    dflash_config: DFlashConfig,
};

pub const LayerType = enum {
    full_attention,
    sliding_attention,
};

pub const DFlashConfig = struct {
    target_layer_ids: []const u32,
    mask_token_id: ?u32 = null,
};

pub const Buffers = zml.Bufferized(Model);

pub const Model = struct {
    layers: []DecoderLayer,
    norm: RmsNorm,
    fc: zml.nn.Linear,
    hidden_norm: RmsNorm,
    target_layer_ids: []u32,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Model {
        const layers = try allocator.alloc(DecoderLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);

        if (config.layer_types.len != 0 and config.layer_types.len != config.num_hidden_layers) return error.InvalidDFlashLayerTypes;
        for (layers, 0..) |*layer, i| {
            const layer_type = if (config.layer_types.len == 0) .full_attention else config.layer_types[i];
            layer.* = try .init(store.withPrefix("layers").withLayer(i), config, layer_type);
        }

        const target_layer_ids = try allocator.alloc(u32, config.dflash_config.target_layer_ids.len);
        @memcpy(target_layer_ids, config.dflash_config.target_layer_ids);
        errdefer allocator.free(target_layer_ids);

        return .{
            .layers = layers,
            .norm = .init(store.withPrefix("norm"), config.rms_norm_eps),
            .fc = .init(
                store.createTensor("fc.weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                null,
                .d,
            ),
            .hidden_norm = .init(store.withPrefix("hidden_norm"), config.rms_norm_eps),
            .target_layer_ids = target_layer_ids,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
        allocator.free(self.target_layer_ids);
    }

    pub fn unloadBuffers(self: *Buffers, allocator: std.mem.Allocator) void {
        for (self.layers) |*layer| DecoderLayer.unloadBuffers(layer);
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
        self.fc.weight.deinit();
        if (self.fc.bias) |*bias| bias.deinit();
        RmsNorm.unloadBuffers(&self.hidden_norm);
    }

    pub fn initKvCache(self: Model, config: Config, cache_seq_len: u32) KvCache {
        _ = self;
        return .init(.init(.{
            .layer = config.num_hidden_layers,
            .k = cache_seq_len,
            .h = config.num_key_value_heads,
            .hd = config.head_dim orelse config.hidden_size / config.num_attention_heads,
        }, .f32));
    }

    pub fn loadBuffers(
        self: *const Model,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
    ) !Buffers {
        return zml.io.load(Model, self, allocator, io, platform, store, .{
            .parallelism = 16,
            .shardings = shardings,
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
        });
    }

    pub fn forward(
        self: Model,
        target_hidden_: zml.Tensor,
        noise_embedding_: zml.Tensor,
        position_ids_: zml.Tensor,
        kv_cache: KvCache,
        cache_index: zml.Tensor,
        active_context_len: zml.Tensor,
    ) struct { zml.Tensor, KvCache } {
        var hidden = noise_embedding_.withPartialTags(.{ .s, .d }).convert(self.norm.weight.dtype());
        const target_hidden = self.hidden_norm.forward(
            self.fc.forward(target_hidden_.withPartialTags(.{ .s, .d }).convert(self.fc.weight.dtype()))
                .rename(.{ .dout = .d }),
        );
        const position_ids = position_ids_.withPartialTags(.{.s});
        var updated_kv_cache = kv_cache;

        for (self.layers, 0..) |layer, i| {
            hidden, updated_kv_cache = layer.forward(
                hidden,
                target_hidden,
                position_ids,
                updated_kv_cache.atLayer(i),
                cache_index,
                active_context_len,
            );
        }

        return .{ self.norm.forward(hidden), updated_kv_cache.reuseBuffer(kv_cache) };
    }
};

pub const DecoderLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: DFlashAttention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,

    pub fn init(store: zml.io.TensorStore.View, config: Config, layer_type: LayerType) !DecoderLayer {
        return .{
            .input_layernorm = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps),
            .self_attn = try .init(store.withPrefix("self_attn"), config, layer_type),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps),
            .mlp = .init(store.withPrefix("mlp")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(DecoderLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        DFlashAttention.unloadBuffers(&self.self_attn);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
        Mlp.unloadBuffers(&self.mlp);
    }

    pub fn forward(
        self: DecoderLayer,
        hidden_states: zml.Tensor,
        target_hidden: zml.Tensor,
        position_ids: zml.Tensor,
        kv_cache: KvCache,
        cache_index: zml.Tensor,
        active_context_len: zml.Tensor,
    ) struct { zml.Tensor, KvCache } {
        const residual = hidden_states.withPartitioning(.{ .d = .replicated });
        const attn_out, const updated_kv_cache = self.self_attn.forward(
            self.input_layernorm.forward(residual),
            target_hidden,
            position_ids,
            kv_cache,
            cache_index,
            active_context_len,
        );

        const post_attn = residual.add(attn_out).withPartitioning(.{ .d = .replicated });
        const mlp_out = self.mlp.forward(self.post_attention_layernorm.forward(post_attn))
            .rename(.{ .dout = .d })
            .withPartitioning(.{ .d = .replicated });

        return .{ post_attn.add(mlp_out).withPartitioning(.{ .d = .replicated }), updated_kv_cache };
    }
};

pub const DFlashAttention = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: i64,
    num_kv_heads: i64,
    rope_opts: zml.nn.RopeOpts,
    sliding_window: ?u32,

    pub fn init(store: zml.io.TensorStore.View, config: Config, layer_type: LayerType) !DFlashAttention {
        var rope_scaling = config.rope_scaling;
        rope_scaling.setRopeTheta(config.rope_theta);
        const sliding_window = switch (layer_type) {
            .full_attention => null,
            .sliding_attention => config.sliding_window orelse return error.MissingDFlashSlidingWindow,
        };

        return .{
            .q_proj = linear(store, "q_proj", config.attention_bias, .{ .dout = .model, .d = .replicated }),
            .k_proj = linear(store, "k_proj", config.attention_bias, .{ .dout = .model, .d = .replicated }),
            .v_proj = linear(store, "v_proj", config.attention_bias, .{ .dout = .model, .d = .replicated }),
            .o_proj = linear(store, "o_proj", config.attention_bias, .{ .d = .model }),
            .q_norm = .init(store.withPrefix("q_norm"), config.rms_norm_eps),
            .k_norm = .init(store.withPrefix("k_norm"), config.rms_norm_eps),
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
            .rope_opts = .{
                .layout = .sequential,
                .scaling = rope_scaling,
            },
            .sliding_window = sliding_window,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(DFlashAttention)) void {
        unloadLinear(&self.q_proj);
        unloadLinear(&self.k_proj);
        unloadLinear(&self.v_proj);
        unloadLinear(&self.o_proj);
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    pub fn forward(
        self: DFlashAttention,
        hidden_states_: zml.Tensor,
        target_hidden_: zml.Tensor,
        position_ids_: zml.Tensor,
        kv_cache: KvCache,
        cache_index: zml.Tensor,
        active_context_len: zml.Tensor,
    ) struct { zml.Tensor, KvCache } {
        const hidden_states = hidden_states_.withPartialTags(.{ .s, .d }).withPartitioning(.{ .d = .replicated });
        const target_hidden = target_hidden_.withPartialTags(.{ .s, .d }).withPartitioning(.{ .d = .replicated });
        const position_ids = position_ids_.withPartialTags(.{.s});

        var q = self.q_proj.forward(hidden_states)
            .splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });

        var new_k = zml.Tensor.concatenate(&.{
            self.k_proj.forward(target_hidden),
            self.k_proj.forward(hidden_states),
        }, .s).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = .auto });

        var new_v = zml.Tensor.concatenate(&.{
            linearForwardF32(self.v_proj, target_hidden),
            linearForwardF32(self.v_proj, hidden_states),
        }, .s).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = .auto });

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        new_k = self.k_norm.forward(new_k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        const q_pos = position_ids.slice1d(.s, .{
            .start = position_ids.dim(.s) - hidden_states.dim(.s),
            .end = position_ids.dim(.s),
        });
        q = zml.nn.rope(q, q_pos, self.rope_opts).rename(.{ .s = .q });
        new_k = zml.nn.rope(new_k, position_ids, self.rope_opts).rename(.{ .s = .k });
        new_v = new_v.rename(.{ .s = .k });

        const updated_kv_cache = updateBucketed(kv_cache, new_k, new_v, cache_index, active_context_len, target_hidden.dim(.s));
        var k = updated_kv_cache.keys().convert(q.dtype());
        var v = updated_kv_cache.values().convert(.f32);
        k = k.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });

        const attn_shape = zml.Shape.init(.{ .q = q.dim(.q), .k = k.dim(.k) }, .bool);
        const k_idx = zml.Tensor.iota(attn_shape, .k).convert(cache_index.dtype());
        const active_context_end = cache_index.add(active_context_len.convert(cache_index.dtype()));
        const valid_k = active_context_end.add(zml.Tensor.scalar(hidden_states.dim(.s), cache_index.dtype()));
        var valid_mask = k_idx.cmp(.LT, valid_k.broad(k_idx.shape()));
        if (self.sliding_window) |window| {
            const q_idx = zml.Tensor.iota(attn_shape, .q).convert(cache_index.dtype());
            const q_abs = q_idx.add(active_context_end.broad(q_idx.shape()));
            valid_mask = valid_mask.logical(.AND, q_abs.cmp(.LT, k_idx.addConstant(window)));
        }
        const zeros = zml.Tensor.scalar(@as(f32, 0.0), .f32).broad(attn_shape.withDtype(.f32));
        const minus_inf = zml.Tensor.scalar(-std.math.inf(f32), .f32).broad(attn_shape.withDtype(.f32));
        const attn_mask = valid_mask.select(zeros, minus_inf).convert(q.dtype());

        const attn = zml.nn.sdpa(q.convert(.f32), k.convert(.f32), v, .{ .attn_mask = attn_mask.convert(.f32) })
            .withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated })
            .merge(.{ .d = .{ .h, .hd } })
            .rename(.{ .q = .s })
            .convert(self.o_proj.weight.dtype());

        return .{
            self.o_proj.forward(attn)
                .rename(.{ .dout = .d })
                .withPartitioning(.{ .d = .replicated }),
            updated_kv_cache,
        };
    }
};

pub const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{
            .weight = store.createTensor("weight", .{.d}, .{ .d = .replicated }),
            .eps = eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        return zml.nn.rmsNorm(x, .d, self.eps)
            .mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }
};

pub const Mlp = struct {
    gate_proj: zml.nn.Linear,
    up_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .gate_proj = .init(store.createTensor("gate_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .up_proj = .init(store.createTensor("up_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .down_proj = .init(store.createTensor("down_proj.weight", .{ .dout, .d }, .{ .d = .model }), null, .d),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        unloadLinear(&self.gate_proj);
        unloadLinear(&self.up_proj);
        unloadLinear(&self.down_proj);
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        return self.down_proj.forward(
            self.gate_proj.forward(x)
                .silu()
                .mul(self.up_proj.forward(x))
                .rename(.{ .dout = .d }),
        );
    }
};

fn linear(
    store: zml.io.TensorStore.View,
    comptime prefix: []const u8,
    with_bias: bool,
    partitioning: anytype,
) zml.nn.Linear {
    const prefixed = store.withPrefix(prefix);
    return .init(
        prefixed.createTensor("weight", .{ .dout, .d }, partitioning),
        if (with_bias) prefixed.createTensor("bias", .{.dout}, .{ .dout = .replicated }) else null,
        .d,
    );
}

fn unloadLinear(linear_: anytype) void {
    if (linear_.bias) |*bias| bias.deinit();
    linear_.weight.deinit();
}

fn linearForwardF32(linear_: zml.nn.Linear, x: zml.Tensor) zml.Tensor {
    var y = x.convert(.f32).dot(linear_.weight.convert(.f32), linear_.tag);
    return if (linear_.bias) |bias| y.add(bias.convert(.f32).broad(y.shape())) else y;
}

fn updateBucketed(kv: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: zml.Tensor, active_context_len: zml.Tensor, context_len: i64) KvCache {
    const merged_k, const merged_v = mergeBucketed(kv, new_k, new_v, token_index, active_context_len, context_len);
    return kv.update(merged_k, merged_v, token_index);
}

fn mergeBucketed(kv: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: zml.Tensor, active_context_len: zml.Tensor, context_len: i64) struct { zml.Tensor, zml.Tensor } {
    const proposal_len = new_k.dim(.k) - context_len;
    const merged_len = context_len + proposal_len;
    const context_start = zml.Tensor.scalar(@as(u32, 0), .u32);
    const proposal_start = active_context_len.convert(.u32);

    const context_k = new_k.slice1d(.k, .{ .start = 0, .end = context_len });
    const context_v = new_v.slice1d(.k, .{ .start = 0, .end = context_len });
    const proposal_k = new_k.slice1d(.k, .{ .start = context_len, .end = new_k.dim(.k) });
    const proposal_v = new_v.slice1d(.k, .{ .start = context_len, .end = new_v.dim(.k) });

    var merged_k = kv.keys().dynamicSlice(.{ .k = zml.Tensor.DynSlice{ .start = token_index, .len = merged_len } }).convert(new_k.dtype());
    var merged_v = kv.values().dynamicSlice(.{ .k = zml.Tensor.DynSlice{ .start = token_index, .len = merged_len } }).convert(new_v.dtype());
    merged_k = merged_k.dynamicUpdateSlice(.{ .k = context_start }, context_k);
    merged_v = merged_v.dynamicUpdateSlice(.{ .k = context_start }, context_v);
    merged_k = merged_k.dynamicUpdateSlice(.{ .k = proposal_start }, proposal_k);
    merged_v = merged_v.dynamicUpdateSlice(.{ .k = proposal_start }, proposal_v);
    return .{ merged_k, merged_v };
}

fn zeroBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    sharding: zml.Sharding,
) !zml.Buffer {
    const bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(bytes);
    @memset(bytes, 0);
    return zml.Buffer.fromSlice(io, platform, zml.Slice.init(shape, bytes), sharding);
}

fn replaceBuffer(dst: *zml.Buffer, src: *zml.Buffer) void {
    if (!sameBufferHandle(dst.*, src.*)) {
        dst.deinit();
    }
    dst.* = src.*;
}

fn sameBufferHandle(a: zml.Buffer, b: zml.Buffer) bool {
    if (a._shards.len != b._shards.len) return false;
    for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
        if (a_shard != b_shard) return false;
    }
    return true;
}
