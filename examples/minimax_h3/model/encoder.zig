const std = @import("std");

const zml = @import("zml");

const config = @import("../core/config.zig");
const weights = @import("../core/weights.zig");

const log = std.log.scoped(.minimax_h3_encoder);

pub const Config = config.EncoderConfig;

const rmsNorm = weights.rmsNorm;

fn linear(store: zml.io.TensorStore.View, weight_name: []const u8, partitions: anytype) zml.nn.Linear {
    return weights.linear(store, weight_name, null, partitions, .replicated);
}

const Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .up_proj = linear(store, "up_proj.weight", .{ .dout = .model }),
            .gate_proj = linear(store, "gate_proj.weight", .{ .dout = .model }),
            .down_proj = linear(store, "down_proj.weight", .{ .d = .model }),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        zml.nn.Linear.unloadBuffers(&self.up_proj);
        zml.nn.Linear.unloadBuffers(&self.gate_proj);
        zml.nn.Linear.unloadBuffers(&self.down_proj);
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const proj = self.up_proj.forward(x);
        var output = self.gate_proj.forward(x);
        output = output.silu().mul(proj).rename(.{ .dout = .d });
        return self.down_proj.forward(output);
    }
};

const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,
    q_norm: zml.nn.RmsNorm,
    k_norm: zml.nn.RmsNorm,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    attn_backend: zml.attention.Backend = .vanilla,

    pub fn init(store: zml.io.TensorStore.View, cfg: Config) SelfAttn {
        return .{
            .q_proj = linear(store, "q_proj.weight", .{ .dout = .model }),
            .k_proj = linear(store, "k_proj.weight", .{ .dout = .model }),
            .v_proj = linear(store, "v_proj.weight", .{ .dout = .model }),
            .o_proj = linear(store, "o_proj.weight", .{ .d = .model }),
            .q_norm = rmsNorm(store.withPrefix("q_norm"), .{.hd}, cfg.rms_norm_eps),
            .k_norm = rmsNorm(store.withPrefix("k_norm"), .{.hd}, cfg.rms_norm_eps),
            .num_heads = cfg.num_attention_heads,
            .num_kv_heads = cfg.num_key_value_heads,
            .head_dim = cfg.head_dim,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SelfAttn)) void {
        zml.nn.Linear.unloadBuffers(&self.q_proj);
        zml.nn.Linear.unloadBuffers(&self.k_proj);
        zml.nn.Linear.unloadBuffers(&self.v_proj);
        zml.nn.Linear.unloadBuffers(&self.o_proj);
        zml.nn.RmsNorm.unloadBuffers(&self.q_norm);
        zml.nn.RmsNorm.unloadBuffers(&self.k_norm);
    }

    pub fn forward(self: SelfAttn, x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const x_qkv = x.withPartitioning(.{ .d = .replicated });
        var q = self.q_proj.forward(x_qkv).splitAxis(-1, .{ .h = self.num_heads, .hd = self.head_dim }).withPartitioning(.{ .h = .model });
        var k = self.k_proj.forward(x_qkv).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = self.head_dim }).withPartitioning(.{ .h = .model });
        const v = self.v_proj.forward(x_qkv).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = self.head_dim }).withPartitioning(.{ .h = .model });

        q = self.q_norm.forward(q);
        k = self.k_norm.forward(k);
        q = zml.nn.applyRotary(q, cos, sin);
        k = zml.nn.applyRotary(k, cos, sin);

        const q_s = q.rename(.{ .s = .q });
        const k_s = k.rename(.{ .s = .k });
        const v_s = v.rename(.{ .s = .k });
        const attn = zml.attention.dense(q_s, k_s, v_s, self.attn_backend, .{ .is_causal = true })
            .rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } });
        return self.o_proj.forward(attn).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
    }
};

pub const TransformerLayer = struct {
    input_layernorm: zml.nn.RmsNorm,
    self_attn: SelfAttn,
    post_attention_layernorm: zml.nn.RmsNorm,
    mlp: Mlp,

    pub const Input = struct {
        layer: TransformerLayer,
        hidden: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
        visual_delta: zml.Tensor,
    };

    pub const Output = struct {
        hidden: zml.Tensor,
    };

    pub fn init(store: zml.io.TensorStore.View, cfg: Config) TransformerLayer {
        return .{
            .input_layernorm = rmsNorm(store.withPrefix("input_layernorm"), .{.d}, cfg.rms_norm_eps),
            .self_attn = .init(store.withPrefix("self_attn"), cfg),
            .post_attention_layernorm = rmsNorm(store.withPrefix("post_attention_layernorm"), .{.d}, cfg.rms_norm_eps),
            .mlp = .init(store.withPrefix("mlp")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        zml.nn.RmsNorm.unloadBuffers(&self.input_layernorm);
        SelfAttn.unloadBuffers(&self.self_attn);
        zml.nn.RmsNorm.unloadBuffers(&self.post_attention_layernorm);
        Mlp.unloadBuffers(&self.mlp);
    }

    pub fn forward(input: Input) Output {
        const self = input.layer;
        const residual = input.hidden.withPartitioning(.{ .d = .replicated });
        const attn = self.self_attn.forward(self.input_layernorm.forward(residual), input.cos, input.sin);
        const x1 = residual.add(attn).withPartitioning(.{ .d = .replicated });
        const mlp = self.mlp.forward(self.post_attention_layernorm.forward(x1)).rename(.{ .dout = .d });
        const hidden = x1.add(mlp).add(input.visual_delta.convert(x1.dtype())).withPartitioning(.{ .d = .replicated });
        return .{ .hidden = hidden.reuseBuffer(input.hidden) };
    }
};

pub const EmbedTokens = struct {
    embed_tokens: zml.nn.TokenEmbedding,

    pub const Input = struct {
        embedding: EmbedTokens,
        tokens: zml.Tensor,
    };

    pub const Output = struct {
        hidden: zml.Tensor,
    };

    pub fn unloadBuffers(self: *zml.Bufferized(EmbedTokens)) void {
        zml.nn.TokenEmbedding.unloadBuffers(&self.embed_tokens);
    }

    pub fn forward(input: Input) Output {
        const tokens = input.tokens.withPartialTags(.{.s});
        return .{ .hidden = input.embedding.embed_tokens.forward(tokens)
            .withPartialTags(.{.d})
            .withPartitioning(.{ .d = .replicated }) };
    }
};

pub const Model = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    cfg: Config,

    pub fn init(allocator: std.mem.Allocator, store_: zml.io.TensorStore.View, cfg: Config) !Model {
        const store = rootView(store_);
        const used: usize = @intCast(cfg.used_hidden_layers);
        const layers = try allocator.alloc(TransformerLayer, used);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| {
            layer.* = .init(store.withPrefix("layers").withLayer(i), cfg);
        }
        return .{
            .embed_tokens = embedTokens(store),
            .layers = layers,
            .cfg = cfg,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn applyBackend(self: *Model, kind: zml.attention.Backend) void {
        for (self.layers) |*layer| layer.self_attn.attn_backend = kind;
    }
};

fn embedTokens(store: zml.io.TensorStore.View) zml.nn.TokenEmbedding {
    return .{ .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, .{ .voc = .replicated, .d = .model }) };
}

fn rootView(store: zml.io.TensorStore.View) zml.io.TensorStore.View {
    return store.withPrefix("model.language_model");
}

pub const LoadedModel = struct {
    inner: Model,
    cfg: Config,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View) !LoadedModel {
        const cfg = try config.loadEncoderConfig(allocator, io, repo);
        log.info("encoder: {d} layers hidden={d} heads={d}", .{
            cfg.used_hidden_layers,
            cfg.hidden_size,
            cfg.num_attention_heads,
        });
        return .{
            .inner = try .init(allocator, store, cfg),
            .cfg = cfg,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
    }

    pub fn loadEmbed(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(EmbedTokens) {
        const part = EmbedTokens{ .embed_tokens = self.inner.embed_tokens };
        return weights.load(allocator, io, platform, store, shardings, EmbedTokens, &part, progress, null);
    }

    pub fn loadLayer(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        index: usize,
        progress: *std.Progress.Node,
        loader: ?*zml.io.Loader,
    ) !zml.Bufferized(TransformerLayer) {
        return weights.load(allocator, io, platform, store, shardings, TransformerLayer, &self.inner.layers[index], progress, loader);
    }
};
