const std = @import("std");

const zml = @import("zml");

const config_mod = @import("config.zig");
const weights = @import("weights.zig");

const log = std.log.scoped(.minimax_h3_encoder);

pub const Config = config_mod.EncoderConfig;

const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{
            .weight = store.createTensor("weight", .{.d}, .replicated),
            .eps = eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const x = input.withPartialTags(.{.d});
        const normalized = zml.nn.rmsNorm(x, .d, self.eps);
        return normalized.mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }
};

const Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .up_proj = .init(store.createTensor("up_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .gate_proj = .init(store.createTensor("gate_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .down_proj = .init(store.createTensor("down_proj.weight", .{ .dout, .d }, .{ .d = .model }), null, .d),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        self.up_proj.weight.deinit();
        self.gate_proj.weight.deinit();
        self.down_proj.weight.deinit();
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
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    rope_opts: zml.nn.RopeOpts,

    pub fn init(store: zml.io.TensorStore.View, cfg: Config) SelfAttn {
        return .{
            .q_proj = .init(store.createTensor("q_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .k_proj = .init(store.createTensor("k_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .v_proj = .init(store.createTensor("v_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .o_proj = .init(store.createTensor("o_proj.weight", .{ .dout, .d }, .{ .d = .model }), null, .d),
            .q_norm = .init(store.withPrefix("q_norm"), cfg.rms_norm_eps),
            .k_norm = .init(store.withPrefix("k_norm"), cfg.rms_norm_eps),
            .num_heads = cfg.num_attention_heads,
            .num_kv_heads = cfg.num_key_value_heads,
            .head_dim = cfg.head_dim,
            .rope_opts = .{
                .layout = .real_im_pass,
                .scaling = .{ .default = .{ .rope_theta = cfg.rope_theta } },
            },
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SelfAttn)) void {
        self.q_proj.weight.deinit();
        self.k_proj.weight.deinit();
        self.v_proj.weight.deinit();
        self.o_proj.weight.deinit();
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    pub fn forward(self: SelfAttn, x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const x_qkv = x.withPartitioning(.{ .d = .replicated });
        var q = self.q_proj.forward(x_qkv).splitAxis(-1, .{ .h = self.num_heads, .hd = self.head_dim }).withPartitioning(.{ .h = .model });
        var k = self.k_proj.forward(x_qkv).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = self.head_dim }).withPartitioning(.{ .h = .model });
        var v = self.v_proj.forward(x_qkv).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = self.head_dim }).withPartitioning(.{ .h = .model });

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        q = applyRotary(q, cos, sin);
        k = applyRotary(k, cos, sin);

        // Official Qwen3-VL text attention is causal (`is_causal = True`).
        const mask = zml.nn.causalAttnMask(.{ .q = q.dim(.s), .k = k.dim(.s) }, q.dtype(), null);
        const attn = zml.nn.sdpa(
            q.rename(.{ .s = .q }),
            k.rename(.{ .s = .k }),
            v.rename(.{ .s = .k }),
            .{ .attn_mask = mask },
        ).rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } });
        return self.o_proj.forward(attn).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
    }
};

fn applyRotary(x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
    const half = @divExact(x.dim(.hd), 2);
    const x1 = x.slice1d(.hd, .{ .start = 0, .end = half });
    const x2 = x.slice1d(.hd, .{ .start = half, .end = x.dim(.hd) });
    const rotated = zml.Tensor.concatenate(&.{ x2.negate(), x1 }, .hd);
    return x.mul(cos.broad(x.shape())).add(rotated.mul(sin.broad(x.shape())));
}

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: SelfAttn,
    post_attention_layernorm: RmsNorm,
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
            .input_layernorm = .init(store.withPrefix("input_layernorm"), cfg.rms_norm_eps),
            .self_attn = .init(store.withPrefix("self_attn"), cfg),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), cfg.rms_norm_eps),
            .mlp = .init(store.withPrefix("mlp")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        SelfAttn.unloadBuffers(&self.self_attn);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
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
        self.embed_tokens.weight.deinit();
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
            .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, .{ .voc = .replicated, .d = .model }) },
            .layers = layers,
            .cfg = cfg,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

};

fn rootView(store: zml.io.TensorStore.View) zml.io.TensorStore.View {
    if (store.hasKey("embed_tokens.weight")) return store;
    if (store.hasKey("model.language_model.embed_tokens.weight")) return store.withPrefix("model.language_model");
    if (store.hasKey("language_model.embed_tokens.weight")) return store.withPrefix("language_model");
    if (store.hasKey("model.embed_tokens.weight")) return store.withPrefix("model");
    return store;
}

pub const LoadedModel = struct {
    inner: Model,
    parsed_config: ?std.json.Parsed(config_mod.EncoderFileConfig),
    cfg: Config,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View) !LoadedModel {
        const parsed: ?std.json.Parsed(config_mod.EncoderFileConfig) = config_mod.parseJson(config_mod.EncoderFileConfig, allocator, io, repo, "config.json") catch null;
        const cfg = if (parsed) |p| p.value.resolve() else Config{};
        log.info("encoder: {d} layers hidden={d} heads={d}", .{
            cfg.used_hidden_layers,
            cfg.hidden_size,
            cfg.num_attention_heads,
        });
        return .{
            .inner = try .init(allocator, store, cfg),
            .parsed_config = parsed,
            .cfg = cfg,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        if (self.parsed_config) |*parsed| parsed.deinit();
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
