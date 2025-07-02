// examples/llama/llama.zig
const std = @import("std");
const testing = std.testing;

const stdx = @import("stdx");
const zml = @import("zml");
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;

const log = std.log.scoped(.llama);

/// Llama architecture, using huggingface transformers naming.
/// Dimensions of activations: {.b, .s, .d}
pub const LlamaLM = struct {
    pub const Config = struct {
        bos_token_id: u32,
        eos_token_id: stdx.json.Union(union(enum) {
            int: u32,
            ints: []u32,
        }),
        num_hidden_layers: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        rope_theta: f32,
        max_position_embeddings: usize,
        rms_norm_eps: f32,
        hf_rope_impl: bool = true,
        rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = {} },
    };

    pub const Options = struct {
        sampling_strategy: ?zml.nn.SamplingStrategy,
        max_seq_len: usize,
    };

    lm_head: ?zml.nn.Linear,
    model: Llama,

    // Runtime configuration, not part of the model's weights/arch
    gen_opts: zml.nn.SamplingStrategy = .{},
    config: Config,

    pub fn init(self: *LlamaLM, compute_mesh: zml.Mesh, vocab_mesh: zml.Mesh, config: Config, options: Options) void {
        self.config = config;
        self.gen_opts = options.sampling_strategy orelse .{};

        self.model.init(config, options.max_seq_len, compute_mesh, vocab_mesh);

        // Shard the final output layer across all devices
        if (self.lm_head) |*lm_head| {
            lm_head.weight = lm_head.weight
                .withTags(.{ .d, .vocab })
                .withMesh(vocab_mesh)
                .withSharding(.{ .vocab = .model });
        }
    }

    pub fn forward(
        self: LlamaLM,
        tokens_: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        rng: Tensor.Rng,
    ) struct { Tensor, KvCache, Tensor.Rng } {
        stdx.debug.assert(tokens_.dtype() == .u32 and tokens_.rank() >= 1 and token_index.dtype() == .u32 and token_index.rank() <= 1, "Can't run Llama ! Expected >=1d tokens and 0d token_index, got: {} and {}", .{ tokens_, token_index });
        const tokens = tokens_.withPartialTags(.{.s});

        zml.pushMesh(self.model.layers[0].input_layernorm.weight.mesh());
        const out, const updated_kv_cache = zml.call(self.model, .forward, .{ tokens, token_index, kv_cache });
        zml.popMesh();

        const new_tokens, const new_rng = self.sampleTokens(self.lm_head, out, rng, self.gen_opts);
        return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), updated_kv_cache, new_rng };
    }

    pub fn sampleTokens(
        self: LlamaLM,
        lm_head_: ?zml.nn.Linear,
        out_: Tensor,
        rng: Tensor.Rng,
        opts: zml.nn.SamplingStrategy,
    ) struct { Tensor, Tensor.Rng } {
        const vocab_mesh = if (lm_head_) |lm_head| lm_head.weight.mesh() else self.model.embed_tokens.weight.mesh();
        zml.pushMesh(vocab_mesh);
        defer zml.popMesh();

        const out = out_.withPartialTags(.{ .s, .d });

        var logits = blk: {
            if (lm_head_) |lm_head| {
                break :blk zml.call(lm_head, .forward, .{out});
            } else {
                break :blk self.model.embed_tokens.weight.withTags(.{ .voc, .d }).dot(out, .{.d});
            }
        };

        if (logits.shape().hasTag(.voc) == null)
            logits = logits.rename(.{ .d = .voc });

        logits = zml.ops.allGather(logits, .model, vocab_mesh);

        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, opts, rng);
        return .{ next_tokens, new_rng };
    }
};

pub const Llama = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    norm: RmsNorm,
    layers: []TransformerLayer,

    max_seq_len: u32 = 0,
    num_heads: i64 = 32,
    num_kv_heads: i64 = 32,
    rope_opts: zml.nn.RopeOpts = .{},

    const Shape = struct {
        s: u32,
        layer: u16,
        hd: u16,
        nh: u16,
        nkvh: u16,
        dtype: zml.DataType,
    };

    pub fn shape(self: Llama) Shape {
        const d_model = self.embed_tokens.weight.dim(1);
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;
        const num_q_heads = self.num_heads;
        const head_dim = @divExact(d_model, num_q_heads);

        return .{
            .s = self.max_seq_len,
            .layer = @intCast(self.layers.len),
            .hd = @intCast(head_dim),
            .nh = @intCast(self.num_heads),
            .nkvh = @intCast(num_kv_heads),
            .dtype = self.embed_tokens.weight.dtype(),
        };
    }

    pub fn init(self: *Llama, config: LlamaLM.Config, max_seq_len: usize, compute_mesh: zml.Mesh, vocab_mesh: zml.Mesh) void {
        self.num_heads = @intCast(config.num_attention_heads);
        self.num_kv_heads = @intCast(config.num_key_value_heads);
        self.max_seq_len = @intCast(max_seq_len);
        self.rope_opts = .{
            .layout = if (config.hf_rope_impl) .sequential else .interleaved,
            .freq_base = config.rope_theta,
            .scaling = config.rope_scaling,
        };

        self.norm.init(config, compute_mesh);

        self.embed_tokens.weight = self.embed_tokens.weight
            .withTags(.{ .vocab, .d })
            .withMesh(vocab_mesh)
            .withSharding(.{ .vocab = .replicated, .d = .replicated });

        for (self.layers) |*layer| {
            layer.init(config, compute_mesh);
        }
    }

    pub fn forward(self: Llama, tokens: Tensor, token_index: Tensor, kv_cache: KvCache) struct { Tensor, KvCache } {
        var hidden = embed(self.embed_tokens, tokens);
        var updated_kv_cache = kv_cache;

        for (self.layers, 0..) |layer, i| {
            hidden, updated_kv_cache = zml.call(layer, .forward, .{ hidden, token_index, updated_kv_cache.atLayer(i) });
        }
        const output = zml.call(self.norm, .forward, .{hidden});

        return .{ output, updated_kv_cache.reuseBuffer(kv_cache) };
    }

    pub fn embed(embed_tokens_: zml.nn.TokenEmbedding, tokens_: Tensor) Tensor {
        const embeddings = zml.call(embed_tokens_, .forward, .{tokens_});
        return embeddings.withPartialTags(.{.d});
    }
};

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: SelfAttn,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,

    pub fn init(self: *TransformerLayer, config: LlamaLM.Config, mesh: zml.Mesh) void {
        self.input_layernorm.init(config, mesh);
        self.self_attn.init(config, mesh);
        self.post_attention_layernorm.init(config, mesh);
        self.mlp.init(mesh);
    }

    pub fn forward(
        self: TransformerLayer,
        x0_: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
    ) struct { Tensor, KvCache } {
        const x0 = x0_;
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {}", .{x0});

        const x0_normalized = zml.call(self.input_layernorm, .forward, .{x0});
        const delta0, const updated_kv_cache = zml.call(self.self_attn, .forward, .{ x0_normalized, token_index, kv_cache });
        const x1 = x0.add(delta0);

        const x1_normalized = zml.call(self.post_attention_layernorm, .forward, .{x1});
        const delta1 = zml.call(self.mlp, .forward, .{x1_normalized});
        const x2 = x1.add(delta1);

        return .{ x2.reuseBuffer(x0), updated_kv_cache };
    }
};

pub const RmsNorm = struct {
    weight: Tensor,
    eps: f32 = 1e-5,

    pub fn init(self: *RmsNorm, config: LlamaLM.Config, mesh: zml.Mesh) void {
        self.eps = config.rms_norm_eps;
        self.weight = self.weight.withTags(.{.d}).withMesh(mesh).withSharding(.{ .d = .replicated });
    }

    pub fn forward(self: RmsNorm, input: Tensor) Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const normalized = zml.nn.rmsNorm(x, .d, self.eps);
        return normalized.mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }
};

pub const Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(self: *Mlp, mesh: zml.Mesh) void {
        // Column-Parallel: Shard on the output/column dimension
        self.gate_proj.weight = self.gate_proj.weight
            .withTags(.{ .hidden, .d })
            .withMesh(mesh)
            .withSharding(.{ .hidden = .model });

        self.up_proj.weight = self.up_proj.weight
            .withTags(.{ .hidden, .d })
            .withMesh(mesh)
            .withSharding(.{ .hidden = .model });

        // Row-Parallel: Shard on the INPUT/column dimension
        self.down_proj.weight = self.down_proj.weight
            .withTags(.{ .d, .hidden })
            .withMesh(mesh)
            .withSharding(.{ .hidden = .model });
    }

    pub fn forward(self: Mlp, x: Tensor) Tensor {
        const gate_out = zml.call(self.gate_proj, .forward, .{x});
        const up_out = zml.call(self.up_proj, .forward, .{x});
        const output = gate_out.silu().mul(up_out);
        return zml.call(self.down_proj, .forward, .{output});
    }
};

pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,
    num_heads: i64 = undefined,
    num_kv_heads: i64 = 0,
    rope_opts: zml.nn.RopeOpts = undefined,

    pub fn init(self: *SelfAttn, config: LlamaLM.Config, mesh: zml.Mesh) void {
        self.num_heads = @intCast(config.num_attention_heads);
        self.num_kv_heads = @intCast(config.num_key_value_heads);
        self.rope_opts = .{
            .layout = if (config.hf_rope_impl) .sequential else .interleaved,
            .freq_base = config.rope_theta,
            .scaling = config.rope_scaling,
        };

        self.q_proj.weight = self.q_proj.weight.withTags(.{ .q_hidden, .d }).withMesh(mesh).withSharding(.{ .q_hidden = .model });
        self.k_proj.weight = self.k_proj.weight.withTags(.{ .kv_hidden, .d }).withMesh(mesh).withSharding(.{ .kv_hidden = .model });
        self.v_proj.weight = self.v_proj.weight.withTags(.{ .kv_hidden, .d }).withMesh(mesh).withSharding(.{ .kv_hidden = .model });
        self.o_proj.weight = self.o_proj.weight.withTags(.{ .d, .o_hidden }).withMesh(mesh).withSharding(.{ .o_hidden = .model });
    }

    pub fn forward(
        self: SelfAttn,
        x: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
    ) struct { Tensor, KvCache } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;

        const q_proj_out = zml.call(self.q_proj, .forward, .{x});
        const k_proj_out = zml.call(self.k_proj, .forward, .{x});
        const v_proj_out = zml.call(self.v_proj, .forward, .{x});

        var q = q_proj_out.withPartialTags(.{.q_hidden}).splitAxis(.q_hidden, .{ .h = self.num_heads, .hd = .auto }).withSharding(.{ .h = .model });
        var k = k_proj_out.withPartialTags(.{.kv_hidden}).splitAxis(.kv_hidden, .{ .h = num_kv_heads, .hd = .auto }).withSharding(.{ .h = .model });
        var v = v_proj_out.withPartialTags(.{.kv_hidden}).splitAxis(.kv_hidden, .{ .h = num_kv_heads, .hd = .auto }).withSharding(.{ .h = .model });

        const pos_index = blk: {
            const temp = Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :blk temp.add(token_index.broad(temp.shape()));
        };
        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);

        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const new_kv_cache = kv_cache.update(k, v, token_index);

        const k_cached = new_kv_cache.keys();
        const v_cached = new_kv_cache.values();

        const attn_mask = if (x.dim(.s) > 1) blk: {
            var mask = zml.nn.causalAttnMask(.{ .q = k_cached.dim(.k), .k = k_cached.dim(.k) }, x.dtype(), null);
            break :blk mask.gatherSlices(.{ .q = x.dim(.s) }, token_index.reshape(.{ .coord = 1 }), .{});
        } else null;

        const attn_output = zml.nn.sdpa(q, k_cached, v_cached, .{ .attn_mask = attn_mask, .allow_cudnn = true });

        const attn = attn_output.merge(.{ .o_hidden = .{ .h, .hd } }).rename(.{ .q = .s });

        const partial_output = zml.call(self.o_proj, .forward, .{attn});

        return .{ partial_output, new_kv_cache };
    }
};

pub const KvCache = struct {
    k: Tensor,
    v: Tensor,
    layer_index: Tensor,

    pub fn initShape(kv_shape: zml.Shape) ShapeOf(KvCache) {
        const sharded_shape = kv_shape.withPartitioning(.{ .h = .model });
        return .{
            .k = sharded_shape,
            .v = sharded_shape,
            .layer_index = zml.Shape.init(.{}, .u32),
        };
    }

    pub fn initBuffer(kv_shape: zml.Shape, mesh: zml.Mesh, platform: zml.Platform) !zml.Bufferized(KvCache) {
        const sharding = zml.Sharding.init(mesh, kv_shape.withPartitioning(.{ .h = .model }));

        return .{
            .k = try zml.Buffer.uninitialized(platform, sharding, .{}),
            .v = try zml.Buffer.uninitialized(platform, sharding, .{}),
            .layer_index = try zml.Buffer.constant(platform, .init(mesh, zml.Shape.init(.{}, .u32)), 0),
        };
    }

    pub fn keys(self: KvCache) Tensor {
        return self.k.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn values(self: KvCache) Tensor {
        return self.v.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn update(self: KvCache, new_k: Tensor, new_v: Tensor, token_index: ?Tensor) KvCache {
        const k_shape = self.k.shape().drop(.layer);
        var layer = self.layer_index;
        layer = if (token_index) |idx| layer.broad(idx.shape()) else layer;

        return if (token_index) |idx| .{
            .k = self.k.scatterSlices(
                .{ .layer = layer, .k = idx },
                new_k.convert(self.k.dtype()).transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(self.k),
            .v = self.v.scatterSlices(
                .{ .layer = layer, .k = idx },
                new_v.convert(self.v.dtype()).transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(self.v),
            .layer_index = self.layer_index,
        } else .{
            .k = self.k.scatterSlices(
                .{ .layer = layer },
                new_k.convert(self.k.dtype()).transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(self.k),
            .v = self.v.scatterSlices(
                .{ .layer = layer },
                new_v.convert(self.v.dtype()).transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(self.v),
            .layer_index = self.layer_index,
        };
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
