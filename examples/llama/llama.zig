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

    // Options controlling generation
    gen_opts: zml.nn.SamplingStrategy = .{},
    config: Config,

    pub fn init(self: *LlamaLM, mesh: zml.Mesh, config: Config, options: Options) void {
        self.config = config;
        self.gen_opts = options.sampling_strategy orelse .{};

        const vocab_mesh = mesh.flatten(.all_devices);

        self.model.init(config, options.max_seq_len, mesh, vocab_mesh);

        // TODO(Corentin): Fix lm_head sharding when top-k sampling is enabled.
        // It currently crashes/compilation fails
        if (self.gen_opts.topk == 1 and self.lm_head != null) {
            self.lm_head.?.weight = self.lm_head.?.weight.withTags(.{ .d, .vocab }).withMesh(vocab_mesh).withSharding(.{ .vocab = .all_devices });
        }
    }

    /// Predicts the token at `token_index` position.
    /// Returns:
    ///  - updated `tokens`,
    ///  - updated KV cache
    ///  - a Rng state to allow for probabilistic generation
    pub fn forward(
        self: LlamaLM,
        tokens_: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        rng: Tensor.Rng,
    ) struct { Tensor, KvCache, Tensor.Rng } {
        stdx.debug.assert(tokens_.dtype() == .u32 and tokens_.rank() >= 1 and token_index.dtype() == .u32 and token_index.rank() <= 1, "Can't run Llama ! Expected >=1d tokens and 0d token_index, got: {} and {}", .{ tokens_, token_index });
        const tokens = tokens_.withPartialTags(.{.s});
        const out, const updated_kv_cache = self.model.forward(tokens, token_index, kv_cache);
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
        const out = out_.withPartialTags(.{ .s, .d });

        var logits = blk: {
            if (lm_head_) |lm_head| {
                break :blk lm_head.forward(out);
            } else {
                break :blk self.model.embed_tokens.weight.withTags(.{ .voc, .d }).dot(out, .{.d});
            }
        };

        if (logits.shape().hasTag(.voc) == null)
            logits = logits.rename(.{ .d = .voc });

        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, opts, rng);
        return .{ next_tokens, new_rng };
    }

    pub fn increment(_: u8, token_index: Tensor) Tensor {
        return token_index.addConstant(1).reuseBuffer(token_index);
    }
};

pub const Llama = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    norm: RmsNorm,
    layers: []TransformerLayer,

    max_seq_len: u32 = 0,
    num_heads: i64 = 32,
    num_kv_heads: i64 = 32,
    rope_opts: zml.nn.RopeOpts = .{
        .layout = .interleaved,
        .freq_base = 10_000,
    },

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
            .withSharding(.{ .vocab = .all_devices });

        for (self.layers) |*layer| {
            layer.init(config, compute_mesh);
        }
    }

    /// Forward one token, using KV cache for previous tokens.
    /// Returns result and updated KV cache.
    pub fn forward(self: Llama, tokens: Tensor, token_index: Tensor, kv_cache: KvCache) struct { Tensor, KvCache } {
        log.info(">>>>>>>>>>>>>>> Llama.forward(tokens={}, token_index={}, kv_cache={} vocab_mesh={any})", .{ tokens, token_index, kv_cache, self.embed_tokens.weight._mesh });
        const vocab_mesh = self.embed_tokens.weight._mesh.?;
        zml.pushMesh(vocab_mesh);
        const embeds = embed(self.embed_tokens, tokens);
        zml.popMesh();

        zml.pushMesh(self.layers[0].self_attn.q_proj.weight._mesh.?);

        var hidden = embeds.withSharding(.{ .s = .replicated, .d = .model });

        if (embeds.dim(.s) > 1) {
            hidden = hidden.withSharding(.{ .s = .data, .d = .model });
        }

        var updated_kv_cache = kv_cache;

        for (self.layers, 0..) |layer, i| {
            hidden, updated_kv_cache = layer.forward(hidden, token_index, updated_kv_cache.atLayer(i));
        }
        const output = self.norm.forward(hidden);

        zml.popMesh();

        return .{ output, updated_kv_cache.reuseBuffer(kv_cache) };
    }

    pub fn embed(embed_tokens_: zml.nn.TokenEmbedding, tokens_: Tensor) Tensor {
        return embed_tokens_.forward(tokens_).withPartialTags(.{.d});
    }

    fn initKvCache(self: Llama, embed_shape: zml.Shape) KvCache {
        const dims = self.shape();
        var kv_shape = embed_shape.insert(0, .{ .layer = dims.layer }).rename(.{ .s = .k }).splitAxes(.{ .d = .{ .h = dims.nkvh, .hd = dims.hd } });
        const perm = kv_shape.contiguousPerm(.{ .k, .h, .hd });
        kv_shape = kv_shape.transpose(perm.constSlice());
        return KvCache.init(kv_shape);
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
        // const x0 = x0_.withTags(.{ .s, .d }).withSharding(.{ .s = .data, .d = .model });
        const x0 = x0_;
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {}", .{x0});

        // Self Attention
        const x0_normalized = self.input_layernorm.forward(x0);
        const delta0, const updated_kv_cache = self.self_attn.forward(
            x0_normalized,
            token_index,
            kv_cache,
        );
        log.warn("TransformerLayer.forward: delta0={}", .{delta0});
        const x1 = x0.add(delta0);

        // Fully Connected
        const x1_normalized = self.post_attention_layernorm.forward(x1);
        // FIX: Add the MLP output to the Attention output (x1).
        const delta1 = self.mlp.forward(x1_normalized);
        const x2 = x1.add(delta1);

        return .{ x2.reuseBuffer(x0), updated_kv_cache };
    }
};

const RmsNorm = struct {
    weight: Tensor,
    eps: f32 = 1e-5,

    pub fn init(self: *RmsNorm, config: LlamaLM.Config, mesh: zml.Mesh) void {
        self.eps = config.rms_norm_eps;
        self.weight = self.weight.withTags(.{.d}).withMesh(mesh);

        if (mesh.topology.hasTag(.model) != null) {
            self.weight = self.weight.withSharding(.{ .d = .model });
        }
    }

    /// L2 normalization of input tensor along `.d` axis.
    pub fn forward(self: RmsNorm, input: Tensor) Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        log.warn("RmsNorm.forward({})", .{x});
        const normalized = zml.nn.rmsNorm(x, .d, self.eps);
        return normalized.mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }
};

const Mlp = struct {
    up_proj: zml.nn.Linear, // (dim -> hidden_dim)
    gate_proj: zml.nn.Linear, // (dim -> hidden_dim)
    down_proj: zml.nn.Linear, // (hidden_dim -> dim)

    pub fn init(self: *Mlp, mesh: zml.Mesh) void {
        const has_model_axis = mesh.topology.hasTag(.model) != null;

        self.gate_proj.weight = self.gate_proj.weight.withTags(.{ .d, .hidden }).withMesh(mesh);
        self.up_proj.weight = self.up_proj.weight.withTags(.{ .d, .hidden }).withMesh(mesh);
        self.down_proj.weight = self.down_proj.weight.withTags(.{ .hidden, .d }).withMesh(mesh);

        if (has_model_axis) {
            self.gate_proj.weight = self.gate_proj.weight.withSharding(.{ .hidden = .model });
            self.up_proj.weight = self.up_proj.weight.withSharding(.{ .hidden = .model });
            self.down_proj.weight = self.down_proj.weight.withSharding(.{ .d = .model });
        }
    }

    pub fn forward(self: Mlp, x: Tensor) Tensor {
        const proj = self.up_proj.forward(x);
        var output: zml.Tensor = self.gate_proj.forward(x);
        output = output.silu().mul(proj);
        output = self.down_proj.forward(output);

        return output;
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

        const has_model_axis = mesh.topology.hasTag(.model) != null;

        self.q_proj.weight = self.q_proj.weight.withTags(.{ .d, .q_hidden }).withMesh(mesh);
        self.k_proj.weight = self.k_proj.weight.withTags(.{ .d, .kv_hidden }).withMesh(mesh);
        self.v_proj.weight = self.v_proj.weight.withTags(.{ .d, .kv_hidden }).withMesh(mesh);
        self.o_proj.weight = self.o_proj.weight.withTags(.{ .o_hidden, .d }).withMesh(mesh);

        if (has_model_axis) {
            self.q_proj.weight = self.q_proj.weight.withSharding(.{ .q_hidden = .model });
            self.k_proj.weight = self.k_proj.weight.withSharding(.{ .kv_hidden = .model });
            self.v_proj.weight = self.v_proj.weight.withSharding(.{ .kv_hidden = .model });
            self.o_proj.weight = self.o_proj.weight.withSharding(.{ .o_hidden = .model });
        }
    }

    /// Self Attention.
    ///   - If token_index is set, x is assumed to be the representation of one new token,
    /// and kv_cache will be read for the previous tokens.
    ///   - If token_index is not set, x is assumed to be the representation of all tokens
    /// since the beginning of the sequence, and kv_cache won't be read.
    /// In both case, kv_cache will be updated with the computed key and value.
    /// x: {.b, .s, .d } -> .{.b, .s, .d}
    pub fn forward(
        self: SelfAttn,
        x: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
    ) struct { Tensor, KvCache } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;

        const q_proj_out = self.q_proj.forward(x);
        const k_proj_out = self.k_proj.forward(x);
        const v_proj_out = self.v_proj.forward(x);
        log.warn("SelfAttn.forward: q_proj_out={}, k_proj_out={}, v_proj_out={}", .{ q_proj_out, k_proj_out, v_proj_out });
        var q = q_proj_out.splitAxis(.d, .{ .h = self.num_heads, .hd = .auto });
        var k = k_proj_out.splitAxis(.d, .{ .h = num_kv_heads, .hd = .auto });
        var v = v_proj_out.splitAxis(.d, .{ .h = num_kv_heads, .hd = .auto });

        // Generate the attention mask.
        const seq_len = kv_cache.k.dim(.k);
        var attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, x.dtype(), null);
        attn_mask = attn_mask.gatherSlices(.{ .q = x.dim(.s) }, token_index.reshape(.{ .coord = 1 }), .{});

        const pos_index = blk: {
            const temp = Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :blk temp.add(token_index.broad(temp.shape()));
        };

        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const dtype = q.dtype();
        const new_kv_cache = kv_cache.update(k, v, token_index);
        var k_cached = new_kv_cache.keys().convert(dtype);
        var v_cached = new_kv_cache.values().convert(dtype);

        if (self.num_kv_heads > 0 and self.num_heads > self.num_kv_heads) {
            const num_groups = @divExact(self.num_heads, self.num_kv_heads);
            k_cached = k_cached.repeat1d(.h, @intCast(num_groups));
            v_cached = v_cached.repeat1d(.h, @intCast(num_groups));
        }

        const attn_output = zml.nn.sdpa(q, k_cached, v_cached, .{ .attn_mask = attn_mask, .allow_cudnn = true });

        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });

        var output = self.o_proj.forward(attn).rename(.{ .o_hidden = .d });
        log.warn("SelfAttn.forward: attn_output={}, output={}", .{ attn_output, output });
        if (x.dim(.s) > 1) {
            // Prefill phase: Shard the sequence dimension.
            output = output.withSharding(.{ .s = .data, .d = .model });
        } else {
            // Generation phase (s=1): Replicate the sequence dimension.
            output = output.withSharding(.{ .s = .replicated, .d = .model });
        }

        return .{ output, new_kv_cache };
    }

    fn initKvCache(key_shape: zml.Shape) KvCache {
        // When we call initKvCache, we haven't renamed .s to .k yet.
        var kv_shape = key_shape.insert(0, .{ .layer = 1 }).rename(.{ .s = .k });
        const perm = kv_shape.contiguousPerm(.{ .h, .k, .hd });
        kv_shape = kv_shape.transpose(perm.constSlice());
        var res = KvCache.init(kv_shape);
        res.layer_index = Tensor.scalar(0, .u32);
        return res;
    }
};

pub const KvCache = struct {
    k: Tensor,
    v: Tensor,
    layer_index: Tensor,

    pub fn init(kv_shape: zml.Shape, mesh: zml.Mesh) KvCache {
        const sharded_shape = kv_shape.withPartitioning(.{ .k = .data, .h = .model });

        return .{
            .k = Tensor.constant(sharded_shape, kv_shape.dtype().one()).withMesh(mesh),
            .v = Tensor.constant(sharded_shape, kv_shape.dtype().one()).withMesh(mesh),
            .layer_index = Tensor.scalar(-1, .u32),
        };
    }

    pub fn initShape(
        kv_shape: zml.Shape,
    ) ShapeOf(KvCache) {
        const sharded_shape = kv_shape.withPartitioning(.{ .k = .data, .h = .model });

        return .{
            .k = sharded_shape,
            .v = sharded_shape,
            .layer_index = zml.Shape.init(.{}, .u32),
        };
    }

    pub fn initBuffer(allocator: std.mem.Allocator, kv_shape: zml.Shape, mesh: zml.Mesh, platform: zml.Platform) !zml.Bufferized(KvCache) {
        const sharding = zml.Sharding.init(mesh, kv_shape.withPartitioning(.{ .k = .data, .h = .model }));
        const slice = try zml.slice.fill(f32, allocator, kv_shape, 1.0);
        defer allocator.free(slice);
        return .{
            .k = try zml.Buffer.from(platform, sharding, slice, .{ .wait = true }),
            .v = try zml.Buffer.from(platform, sharding, slice, .{ .wait = true }),
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
        var layer = self.layer_index;
        layer = if (token_index) |idx| layer.broad(idx.shape()) else layer;

        return if (token_index) |idx| .{
            .k = self.k.scatterSlices(
                .{ .layer = layer, .k = idx },
                new_k.convert(self.k.dtype()),
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(self.k),
            .v = self.v.scatterSlices(
                .{ .layer = layer, .k = idx },
                new_v.convert(self.v.dtype()),
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(self.v),
            .layer_index = self.layer_index,
        } else .{
            .k = self.k.scatterSlices(
                .{ .layer = layer },
                new_k.convert(self.k.dtype()),
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(self.k),
            .v = self.v.scatterSlices(
                .{ .layer = layer },
                new_v.convert(self.v.dtype()),
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
