// examples/llama/llama.zig
const std = @import("std");
const testing = std.testing;

const stdx = @import("stdx");
const zml = @import("zml");
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;

const log = std.log.scoped(.llama);

pub const Linear = struct {
    weight: Tensor,
    bias: ?Tensor = null,

    pub fn forward(self: Linear, x: Tensor) Tensor {
        var y = x.dotGeneral(self.weight.convert(x.dtype()), &.{.{ -1, -1 }}, &.{});

        if (y.shape().tag(-1) == zml.Shape.TagUnknown) {
            y._shape._tags.set(y.rank() - 1, x.shape().tag(-1));
        }

        const weight_sharding = self.weight.shape()._partitioning;
        const out_dim_idx = y.rank() - 1;

        if (weight_sharding.get(out_dim_idx) != .replicated and weight_sharding.get(out_dim_idx) != .unknown) {
            const mesh = self.weight._mesh orelse zml.currentMesh();
            const mesh_axis = weight_sharding.get(out_dim_idx).toTag();
            y = zml.ops.allReduce(y, mesh_axis, mesh);
        } else if (weight_sharding.get(0) != .replicated and weight_sharding.get(0) != .unknown) {
            var new_sharding = zml.Shape.PartitionArray.init(0) catch unreachable;
            new_sharding.appendNTimesAssumeCapacity(.replicated, y.rank());
            new_sharding.set(out_dim_idx, weight_sharding.get(0));
            y._shape._partitioning = new_sharding;
        }

        if (self.bias) |bias| {
            return y.add(bias.broadcastLeft(y.shape()));
        } else {
            return y;
        }
    }
};

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
                .withTags(.{ .d, .voc })
                .withMesh(vocab_mesh)
                .withSharding(.{ .voc = .model });
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
        const final_tokens = new_tokens.convert(tokens.dtype());
        return .{ final_tokens, updated_kv_cache, new_rng };
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

        var logits_sharded = blk: {
            if (lm_head_) |lm_head| {
                break :blk zml.call(lm_head, .forward, .{out});
            } else {
                break :blk self.model.embed_tokens.weight.withTags(.{ .voc, .d }).dot(out, .{.d});
            }
        };

        if (logits_sharded.shape().hasTag(.voc) == null)
            logits_sharded = logits_sharded.rename(.{ .d = .voc });

        const logits = zml.ops.allGather(logits_sharded, .voc, vocab_mesh);

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

        const replicated_sharding = .{ .d = .replicated };
        self.norm.init(config, compute_mesh, replicated_sharding);

        self.embed_tokens.weight = self.embed_tokens.weight
            .withTags(.{ .voc, .d })
            .withMesh(vocab_mesh)
            .withSharding(.{ .d = .replicated, .voc = .model });

        for (self.layers) |*layer| {
            layer.init(config, compute_mesh, replicated_sharding);
        }
    }

    pub fn forward(self: Llama, tokens: Tensor, token_index: Tensor, kv_cache: KvCache) struct { Tensor, KvCache } {
        var hidden = embed(self.embed_tokens, tokens);
        var updated_kv_cache = kv_cache;

        for (self.layers, 0..) |layer, i| {
            hidden, updated_kv_cache = zml.call(layer, .forward, .{ hidden, token_index, updated_kv_cache.atLayer(i) });
        }

        const output = zml.call(self.norm, .forward, .{hidden});

        return .{ output, updated_kv_cache };
    }

    pub fn embed(embed_tokens_: zml.nn.TokenEmbedding, tokens_: Tensor) Tensor {
        const embeddings = zml.call(embed_tokens_, .forward, .{tokens_});
        var tagged_embeddings = embeddings.withTags(.{ .s, .d });

        if (tokens_.dim(.s) > 1) {
            return tagged_embeddings.withSharding(.{ .s = .s });
        } else {
            return tagged_embeddings.replicated();
        }
    }
};

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: SelfAttn,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,

    pub fn init(self: *TransformerLayer, config: LlamaLM.Config, mesh: zml.Mesh, sharding: anytype) void {
        self.input_layernorm.init(config, mesh, sharding);
        self.self_attn.init(config, mesh, sharding);
        self.post_attention_layernorm.init(config, mesh, sharding);
        self.mlp.init(mesh, sharding);
    }

    pub fn forward(
        self: TransformerLayer,
        x0: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
    ) struct { Tensor, KvCache } {
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {}", .{x0});

        const x0_normalized = zml.call(self.input_layernorm, .forward, .{x0});

        const sharded_delta0, const updated_kv_cache = zml.call(self.self_attn, .forward, .{ x0_normalized, token_index, kv_cache });

        const x1 = x0.add(sharded_delta0);

        const x1_normalized = zml.call(self.post_attention_layernorm, .forward, .{x1});
        const sharded_delta1 = zml.call(self.mlp, .forward, .{x1_normalized});

        const x2 = x1.add(sharded_delta1);

        return .{ x2.reuseBuffer(x0), updated_kv_cache };
    }
};

pub const RmsNorm = struct {
    weight: Tensor,
    eps: f32 = 1e-5,

    pub fn init(self: *RmsNorm, config: LlamaLM.Config, mesh: zml.Mesh, sharding: anytype) void {
        self.eps = config.rms_norm_eps;
        self.weight = self.weight.withTags(.{.d}).withMesh(mesh).withSharding(sharding);
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

    pub fn init(self: *Mlp, mesh: zml.Mesh, sharding: anytype) void {
        self.gate_proj.weight = self.gate_proj.weight.withTags(.{ .hidden, .d }).withMesh(mesh).withSharding(sharding);
        self.up_proj.weight = self.up_proj.weight.withTags(.{ .hidden, .d }).withMesh(mesh).withSharding(sharding);
        self.down_proj.weight = self.down_proj.weight.withTags(.{ .d, .hidden }).withMesh(mesh).withSharding(sharding);
    }

    pub fn forward(self: Mlp, x: Tensor) Tensor {
        const gate_out = zml.call(self.gate_proj, .forward, .{x});
        const up_out = zml.call(self.up_proj, .forward, .{x});
        const output = gate_out.silu().mul(up_out);
        const down_proj_partial = zml.call(self.down_proj, .forward, .{output});

        const down_proj_final = if (x.dim(.s) > 1)
            down_proj_partial.withSharding(.{ .s = .s })
        else
            down_proj_partial;

        return down_proj_final;
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

    pub fn init(self: *SelfAttn, config: LlamaLM.Config, mesh: zml.Mesh, sharding: anytype) void {
        self.num_heads = @intCast(config.num_attention_heads);
        self.num_kv_heads = @intCast(config.num_key_value_heads);
        self.rope_opts = .{
            .layout = if (config.hf_rope_impl) .sequential else .interleaved,
            .freq_base = config.rope_theta,
            .scaling = config.rope_scaling,
        };

        self.q_proj.weight = self.q_proj.weight.withTags(.{ .q_hidden, .d }).withMesh(mesh).withSharding(sharding);
        self.k_proj.weight = self.k_proj.weight.withTags(.{ .kv_hidden, .d }).withMesh(mesh).withSharding(sharding);
        self.v_proj.weight = self.v_proj.weight.withTags(.{ .kv_hidden, .d }).withMesh(mesh).withSharding(sharding);
        self.o_proj.weight = self.o_proj.weight.withTags(.{ .d, .o_hidden }).withMesh(mesh).withSharding(sharding);
    }

    pub fn forward(
        self: SelfAttn,
        x: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
    ) struct { Tensor, KvCache } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;
        const mesh = self.q_proj.weight.mesh();

        const q_proj_out = zml.call(self.q_proj, .forward, .{x});
        const k_proj_out = zml.call(self.k_proj, .forward, .{x});
        const v_proj_out = zml.call(self.v_proj, .forward, .{x});

        var q = q_proj_out.withPartialTags(.{.q_hidden}).splitAxis(.q_hidden, .{ .h = self.num_heads, .hd = .auto });
        var k = k_proj_out.withPartialTags(.{.kv_hidden}).splitAxis(.kv_hidden, .{ .h = num_kv_heads, .hd = .auto });
        var v = v_proj_out.withPartialTags(.{.kv_hidden}).splitAxis(.kv_hidden, .{ .h = num_kv_heads, .hd = .auto });

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

        const k_cached = zml.ops.allGather(new_kv_cache.keys(), .k, mesh);
        const v_cached = zml.ops.allGather(new_kv_cache.values(), .k, mesh);

        const attn_mask = if (x.dim(.s) > 1) blk: {
            var mask = zml.nn.causalAttnMask(.{ .q = k_cached.dim(.k), .k = k_cached.dim(.k) }, x.dtype(), null);
            break :blk mask.gatherSlices(.{ .q = x.dim(.s) }, token_index.reshape(.{ .coord = 1 }), .{});
        } else null;

        const attn_output = zml.nn.sdpa(q, k_cached, v_cached, .{ .attn_mask = attn_mask, .allow_cudnn = true });
        const attn = attn_output.merge(.{ .o_hidden = .{ .h, .hd } }).rename(.{ .q = .s });

        const sharded_output = zml.call(self.o_proj, .forward, .{attn});

        const output_final = if (x.dim(.s) > 1)
            sharded_output.withSharding(.{ .s = .s })
        else
            sharded_output;

        return .{ output_final, new_kv_cache };
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
        const kv_sharding = zml.Sharding.init(mesh, kv_shape.withPartitioning(.{ .h = .model }));
        const layer_index_sharding = zml.Sharding.init(mesh, zml.Shape.init(.{}, .u32));

        return .{
            .k = try zml.Buffer.uninitialized(platform, kv_sharding, .{}),
            .v = try zml.Buffer.uninitialized(platform, kv_sharding, .{}),
            .layer_index = try zml.Buffer.constant(platform, layer_index_sharding, 0),
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
};
