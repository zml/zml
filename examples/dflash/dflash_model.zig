const std = @import("std");

const zml = @import("zml");

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
    dflash_config: DFlashConfig,
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
    block_size: u32,
    mask_token_id: ?u32,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Model {
        const layers = try allocator.alloc(DecoderLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = try .init(store.withPrefix("layers").withLayer(i), config);
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
            .block_size = config.block_size,
            .mask_token_id = config.dflash_config.mask_token_id,
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

    /// Forward pass for DFlashDraftModel.
    /// `target_hidden` is the concatenated target feature from the selected target layers.
    /// `noise_embedding` is the target model embedding of the current noisy block.
    pub fn forward(
        self: Model,
        target_hidden_: zml.Tensor,
        noise_embedding_: zml.Tensor,
        position_ids_: zml.Tensor,
    ) zml.Tensor {
        var hidden = noise_embedding_.withPartialTags(.{ .s, .d }).convert(self.norm.weight.dtype());
        const target_hidden = self.hidden_norm.forward(
            self.fc.forward(target_hidden_.withPartialTags(.{ .s, .d }).convert(self.fc.weight.dtype()))
                .rename(.{ .dout = .d }),
        );
        const position_ids = position_ids_.withPartialTags(.{.s});

        for (self.layers) |layer| {
            hidden = layer.forward(hidden, target_hidden, position_ids);
        }

        return self.norm.forward(hidden);
    }

    pub fn forwardF32(
        self: Model,
        target_hidden: zml.Tensor,
        noise_embedding: zml.Tensor,
        position_ids: zml.Tensor,
    ) zml.Tensor {
        return self.forward(target_hidden, noise_embedding, position_ids).convert(.f32);
    }

    pub fn forwardCached(
        self: Model,
        target_hidden_: zml.Tensor,
        noise_embedding_: zml.Tensor,
        position_ids_: zml.Tensor,
        kv_cache: KvCache,
        cache_index: zml.Tensor,
    ) struct { zml.Tensor, KvCache } {
        var hidden = noise_embedding_.withPartialTags(.{ .s, .d }).convert(self.norm.weight.dtype());
        const target_hidden = self.hidden_norm.forward(
            self.fc.forward(target_hidden_.withPartialTags(.{ .s, .d }).convert(self.fc.weight.dtype()))
                .rename(.{ .dout = .d }),
        );
        const position_ids = position_ids_.withPartialTags(.{.s});
        var updated_kv_cache = kv_cache;

        for (self.layers, 0..) |layer, i| {
            hidden, updated_kv_cache = layer.forwardCached(
                hidden,
                target_hidden,
                position_ids,
                updated_kv_cache.atLayer(i),
                cache_index,
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

    pub fn init(store: zml.io.TensorStore.View, config: Config) !DecoderLayer {
        return .{
            .input_layernorm = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps),
            .self_attn = try .init(store.withPrefix("self_attn"), config),
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

    pub fn forward(self: DecoderLayer, hidden_states: zml.Tensor, target_hidden: zml.Tensor, position_ids: zml.Tensor) zml.Tensor {
        const residual = hidden_states.withPartitioning(.{ .d = .replicated });
        const attn_out = self.self_attn.forward(
            self.input_layernorm.forward(residual),
            target_hidden,
            position_ids,
        );

        const post_attn = residual.add(attn_out).withPartitioning(.{ .d = .replicated });
        const mlp_out = self.mlp.forward(self.post_attention_layernorm.forward(post_attn))
            .rename(.{ .dout = .d })
            .withPartitioning(.{ .d = .replicated });

        return post_attn.add(mlp_out).withPartitioning(.{ .d = .replicated });
    }

    pub fn forwardCached(
        self: DecoderLayer,
        hidden_states: zml.Tensor,
        target_hidden: zml.Tensor,
        position_ids: zml.Tensor,
        kv_cache: KvCache,
        cache_index: zml.Tensor,
    ) struct { zml.Tensor, KvCache } {
        const residual = hidden_states.withPartitioning(.{ .d = .replicated });
        const attn_out, const updated_kv_cache = self.self_attn.forwardCached(
            self.input_layernorm.forward(residual),
            target_hidden,
            position_ids,
            kv_cache,
            cache_index,
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

    pub fn init(store: zml.io.TensorStore.View, config: Config) !DFlashAttention {
        var rope_scaling = config.rope_scaling;
        rope_scaling.setRopeTheta(config.rope_theta);

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
    ) zml.Tensor {
        const hidden_states = hidden_states_.withPartialTags(.{ .s, .d }).withPartitioning(.{ .d = .replicated });
        const target_hidden = target_hidden_.withPartialTags(.{ .s, .d }).withPartitioning(.{ .d = .replicated });
        const position_ids = position_ids_.withPartialTags(.{.s});

        var q = self.q_proj.forward(hidden_states)
            .splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });

        var k = zml.Tensor.concatenate(&.{
            self.k_proj.forward(target_hidden),
            self.k_proj.forward(hidden_states),
        }, .s).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = .auto });

        var v = zml.Tensor.concatenate(&.{
            linearForwardF32(self.v_proj, target_hidden),
            linearForwardF32(self.v_proj, hidden_states),
        }, .s).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = .auto });

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        const q_pos = if (position_ids.dim(.s) == hidden_states.dim(.s))
            position_ids
        else
            position_ids.slice1d(.s, .{
                .start = position_ids.dim(.s) - hidden_states.dim(.s),
                .end = position_ids.dim(.s),
            });

        q = zml.nn.rope(q, q_pos, self.rope_opts).rename(.{ .s = .q });
        k = zml.nn.rope(k, position_ids, self.rope_opts).rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const attn = zml.nn.sdpa(q.convert(.f32), k.convert(.f32), v.convert(.f32), .{}) // no attention mask => full non-causal attention
            .withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated })
            .merge(.{ .d = .{ .h, .hd } })
            .rename(.{ .q = .s })
            .convert(self.o_proj.weight.dtype());

        return self.o_proj.forward(attn)
            .rename(.{ .dout = .d })
            .withPartitioning(.{ .d = .replicated });
    }

    pub fn forwardCached(
        self: DFlashAttention,
        hidden_states_: zml.Tensor,
        target_hidden_: zml.Tensor,
        position_ids_: zml.Tensor,
        kv_cache: KvCache,
        cache_index: zml.Tensor,
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

        const updated_kv_cache = kv_cache.update(new_k, new_v, cache_index);
        var k = updated_kv_cache.keys().convert(q.dtype());
        var v = updated_kv_cache.values().convert(.f32);
        k = k.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });

        const valid_k = cache_index.add(zml.Tensor.scalar(position_ids.dim(.s), cache_index.dtype()));
        const attn_shape = zml.Shape.init(.{ .q = q.dim(.q), .k = k.dim(.k) }, .bool);
        const k_idx = zml.Tensor.iota(attn_shape, .k).convert(cache_index.dtype());
        const valid_mask = k_idx.cmp(.LT, valid_k.broad(k_idx.shape()));
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

pub const KvCache = struct {
    k: zml.Tensor,
    v: zml.Tensor,
    layer_index: ?u32,

    pub const Buffer = zml.Bufferized(KvCache);

    pub fn init(kv_shape: zml.Shape) KvCache {
        const sharded_shape = kv_shape.withPartitioning(.{ .h = .model });
        return .{
            .k = .fromShape(sharded_shape),
            .v = .fromShape(sharded_shape),
            .layer_index = null,
        };
    }

    pub fn initBuffer(kv: KvCache, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !Buffer {
        return .{
            .k = try zml.Buffer.uninitialized(io, platform, kv.k.shape(), sharding, .{}),
            .v = try zml.Buffer.uninitialized(io, platform, kv.v.shape(), sharding, .{}),
        };
    }

    pub fn deinitBuffer(kv: *Buffer) void {
        kv.k.deinit();
        kv.v.deinit();
    }

    pub fn keys(kv: KvCache) zml.Tensor {
        return kv.k.slice1d(.layer, .single(kv.layer_index orelse @panic("forgot to call atLayer")));
    }

    pub fn values(kv: KvCache) zml.Tensor {
        return kv.v.slice1d(.layer, .single(kv.layer_index orelse @panic("forgot to call atLayer")));
    }

    pub fn update(kv: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: zml.Tensor) KvCache {
        const k_shape = kv.k.shape().drop(.layer);
        const layer: zml.Tensor = .scalar(kv.layer_index orelse @panic("forgot to call atLayer"), .u32);
        return .{
            .k = kv.k.scatterSlices(.{ .layer = layer, .k = token_index }, new_k.convert(kv.k.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.k),
            .v = kv.v.scatterSlices(.{ .layer = layer, .k = token_index }, new_v.convert(kv.v.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.v),
            .layer_index = kv.layer_index,
        };
    }

    pub fn atLayer(kv: KvCache, layer_index: usize) KvCache {
        return .{
            .k = kv.k,
            .v = kv.v,
            .layer_index = @intCast(layer_index),
        };
    }

    pub fn reuseBuffer(kv: KvCache, other: KvCache) KvCache {
        return .{
            .k = kv.k.reuseBuffer(other.k),
            .v = kv.v.reuseBuffer(other.v),
            .layer_index = null,
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
