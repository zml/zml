///! GptOss architecture, using huggingface transformers naming.
///! Dimensions of activations: {.b, .s, .d}
const std = @import("std");

const stdx = @import("stdx");
const zml = @import("zml");

const GptOss = @This();
const log = std.log.scoped(.GptOss);

pub const Config = struct {
    bos_token_id: u32 = 199998,
    eos_token_id: stdx.json.Union(union(enum) {
        int: u32,
        ints: []const u32,
    }),
    head_dim: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    experts_per_token: u32,
    rope_theta: f32,
    max_position_embeddings: u32,
    rms_norm_eps: f32,
    sliding_window: u32,
    hf_rope_impl: bool = true,
    rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = {} },
};

pub const Options = struct {
    sampling_strategy: zml.nn.SamplingStrategy,
    max_seq_len: u32,
    max_prompt_len: u32,
    tokens_per_expert_ratio: f32,
};

pub const Mode = union(enum) {
    /// In prefill mode we pass the actual len of the prompt
    prefill: zml.Tensor,
    /// In gen mode we pass the position of the next token
    gen: zml.Tensor,
};

lm_head: ?zml.nn.Linear,
model: Model,

config: Config,
options: Options,

pub fn init(allocator: std.mem.Allocator, store: zml.aio.BufferStore, config: Config, options: Options) !GptOss {
    var self: GptOss = .{
        .config = config,
        .options = options,
        .model = .{
            .max_seq_len = @intCast(options.max_seq_len),
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
            .rope_opts = .{
                .layout = if (config.hf_rope_impl) .sequential else .interleaved,
                .freq_base = config.rope_theta,
                .scaling = config.rope_scaling,
            },

            .embed_tokens = .{
                .weight = store.getTensor("model.embed_tokens.weight").withSharding(.{1}),
            },
            .layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers),
            .norm = .{
                .weight = store.getTensor("model.norm.weight"),
                .eps = config.rms_norm_eps,
            },
        },
        .lm_head = .{ .weight = store.getTensor("lm_head.weight").withSharding(.{0}) },
    };

    var prefix: zml.aio.PrefixBuilder = try .initCapacity(allocator, 1024);
    try prefix.push(stdx.noalloc, "model.layers");
    for (self.model.layers, 0..) |*layer, i| {
        try prefix.pushDigit(stdx.noalloc, i);
        defer prefix.pop();

        var self_attn: SelfAttn = .{
            .sinks = store.getTensor(prefix.concat("self_attn.sinks")),
            .q_proj = try zml.aio.populateModelWithPrefix(zml.nn.Linear, allocator, store, prefix.concat("self_attn.q_proj")),
            .k_proj = try zml.aio.populateModelWithPrefix(zml.nn.Linear, allocator, store, prefix.concat("self_attn.k_proj")),
            .v_proj = try zml.aio.populateModelWithPrefix(zml.nn.Linear, allocator, store, prefix.concat("self_attn.v_proj")),
            .o_proj = try zml.aio.populateModelWithPrefix(zml.nn.Linear, allocator, store, prefix.concat("self_attn.o_proj")),

            .sliding_window = if (i % 2 == 0) config.sliding_window else null,
            .num_heads = self.model.num_heads,
            .num_kv_heads = self.model.num_kv_heads,
            .rope_opts = self.model.rope_opts,
        };

        self_attn.q_proj.weight = self_attn.q_proj.weight.withSharding(.{0});
        self_attn.k_proj.weight = self_attn.k_proj.weight.withSharding(.{0});
        self_attn.v_proj.weight = self_attn.v_proj.weight.withSharding(.{0});
        self_attn.o_proj.weight = self_attn.o_proj.weight.withSharding(.{1});

        const on_disk_moe = try zml.aio.populateModelWithPrefix(MoE.OnDisk, allocator, store, prefix.concat("mlp"));
        var moe = on_disk_moe.rewrite(config.experts_per_token, options);
        {
            moe.experts.gate_up_proj.blocks = moe.experts.gate_up_proj.blocks.withSharding(.{.expert});
            moe.experts.down_proj.blocks = moe.experts.down_proj.blocks.withSharding(.{.expert});
        }

        layer.* = .{
            .input_layernorm = .{
                .weight = store.getTensor(prefix.concat("input_layernorm.weight")),
                .eps = config.rms_norm_eps,
            },
            .post_attention_layernorm = .{
                .weight = store.getTensor(prefix.concat("post_attention_layernorm.weight")),
                .eps = config.rms_norm_eps,
            },
            .self_attn = self_attn,
            .mlp = moe,
        };
    }

    // TODO(Corentin): Fix lm_head sharding when top-k sampling is enabled.
    // It currently crashes/compilation fails
    if (self.options.sampling_strategy.topk == 1 and self.lm_head != null) {
        self.lm_head.?.weight = self.lm_head.?.weight.withSharding(.{0});
    }

    return self;
}

/// Predicts the token at `token_index` position.
/// Returns:
///  - updated `tokens`,
///  - updated KV cache
///  - a Rng state to allow for probabilistic generation
pub fn forward(
    self: GptOss,
    tokens_: zml.Tensor,
    mode: Mode,
    kv_cache: KvCache,
    rng: zml.Tensor.Rng,
) struct { zml.Tensor, KvCache, zml.Tensor.Rng } {
    const tokens = tokens_.withPartialTags(.{.s});

    // token index is the position in the kv cache where to write results.
    const token_index: zml.Tensor = switch (mode) {
        .gen => |token_index| token_index,
        .prefill => .scalar(0, .u32),
    };
    var out, const updated_kv_cache = zml.call(self.model, .forward, .{ tokens, token_index, kv_cache });

    switch (mode) {
        // In prefill we only pass the last token to the lm head.
        .prefill => |prompt_len| out = out.gather(.{ .s = prompt_len.convert(.i32).addConstant(-1) }, .{ .indices_are_sorted = true }),
        .gen => {},
    }

    var new_token, const new_rng = self.sampleTokens(self.lm_head, out, rng, self.options.sampling_strategy);
    new_token = new_token.convert(.u32);
    new_token = switch (mode) {
        .gen => new_token.reuseBuffer(tokens),
        .prefill => new_token.appendAxes(.{.s}),
    };
    return .{ new_token, updated_kv_cache, new_rng };
}

fn sampleTokens(
    self: GptOss,
    lm_head_: ?zml.nn.Linear,
    out_: zml.Tensor,
    rng: zml.Tensor.Rng,
    opts: zml.nn.SamplingStrategy,
) struct { zml.Tensor, zml.Tensor.Rng } {
    const out = out_.withPartialTags(.{.d});

    var logits = blk: {
        if (lm_head_) |lm_head| {
            break :blk zml.call(lm_head, .forward, .{out});
        } else {
            break :blk self.model.embed_tokens.weight.withTags(.{ .voc, .d }).dot(out, .{.d});
        }
    };

    if (logits.shape().hasTag(.voc) == null)
        logits = logits.rename(.{ .d = .voc });

    const next_tokens, const new_rng = zml.nn.sampleTokens(logits, opts, rng);
    return .{ next_tokens, new_rng };
}

pub fn loadBuffers(self: GptOss, allocator: std.mem.Allocator, store: zml.aio.BufferStore, platform: zml.Platform) !zml.Bufferized(GptOss) {
    var prefix: zml.aio.PrefixBuilder = try .initCapacity(allocator, 256);
    defer prefix.deinit(allocator);

    const noalloc = stdx.noalloc;
    const loaded: zml.Bufferized(GptOss) = .{
        .model = .{
            .embed_tokens = try store.loadModelById(zml.nn.TokenEmbedding, noalloc, self.model.embed_tokens, platform),
            .layers = try allocator.alloc(zml.Bufferized(TransformerLayer), self.model.layers.len),
            .norm = try store.loadModelById(RmsNorm, noalloc, self.model.norm, platform),
        },
        .lm_head = try store.loadModelById(?zml.nn.Linear, noalloc, self.lm_head, platform),
    };

    prefix.push(noalloc, "model.layers") catch unreachable;
    for (loaded.model.layers, self.model.layers, 0..) |*d_layer, layer, layer_id| {
        const ckpt = prefix.checkpoint();
        defer prefix.restore(ckpt);

        prefix.pushDigit(noalloc, layer_id) catch unreachable;
        d_layer.* = .{
            .input_layernorm = try store.loadModelById(RmsNorm, noalloc, layer.input_layernorm, platform),
            .self_attn = try store.loadModelById(SelfAttn, noalloc, layer.self_attn, platform),
            .post_attention_layernorm = try store.loadModelById(RmsNorm, noalloc, layer.post_attention_layernorm, platform),
            .mlp = try store.loadModelById(MoE, noalloc, layer.mlp, platform),
        };
    }

    return loaded;
}

pub const Model = struct {
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

    /// Forward one token, using KV cache for previous tokens.
    /// Returns result and updated KV cache.
    pub fn forward(self: Model, tokens: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache) struct { zml.Tensor, KvCache } {
        const embeds = embed(self.embed_tokens, tokens);
        var hidden = embeds;

        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden, updated_kv_cache = zml.call(layer, .forward, .{ hidden, token_index, updated_kv_cache.atLayer(i) });
        }
        const output = zml.call(self.norm, .forward, .{hidden});

        return .{ output, updated_kv_cache.reuseBuffer(kv_cache) };
    }

    pub fn embed(embed_tokens_: zml.nn.TokenEmbedding, tokens_: zml.Tensor) zml.Tensor {
        return zml.call(embed_tokens_, .forward, .{tokens_}).withPartialTags(.{.d});
    }
};

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: SelfAttn,
    post_attention_layernorm: RmsNorm,
    mlp: MoE,

    pub fn forward(
        self: TransformerLayer,
        x0: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
    ) struct { zml.Tensor, KvCache } {
        // Self Attention
        //log.debug("TransformerLayer({}) -> {}", .{ x0, self.input_layernorm.forward(x0) });
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {f}", .{x0});

        const x0_normalized = zml.call(self.input_layernorm, .forward, .{x0});
        const delta0, const updated_kv_cache = zml.call(self.self_attn, .forward, .{ x0_normalized, token_index, kv_cache });
        const x1 = x0.add(delta0);

        // Fully Connected
        const x1_normalized = zml.call(self.post_attention_layernorm, .forward, .{x1});
        const x2 = zml.call(self.mlp, .forward, .{x1_normalized}).add(x1);

        return .{ x2.reuseBuffer(x0), updated_kv_cache };
    }
};

const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32 = 1e-6,

    /// L2 normalization of input tensor along `.d` axis.
    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        // Note: contrary to Llama here the full layer is done in .f32, not just the variance computation.
        const normalized = zml.nn.rmsNorm(x.convert(.f32), .d, self.eps);
        return normalized.mul(self.weight.convert(.f32).withTags(.{.d}).broad(x.shape())).convert(input.dtype());
    }
};

const MoE = struct {
    experts: Mlp,
    router: zml.nn.Linear,
    moe_opts: MoeOpts,

    pub fn forward(self: MoE, input: zml.Tensor) zml.Tensor {
        log.warn("compiling moe with {f}", .{input});
        // Note: GptOss applies softmax on the routing score.
        // We delay the softmax to mixtureOfExperts where the actual routing is done.
        // This allow to do re-routing without introducing nans.
        const gating = self.router.forward(input);
        return mixtureOfExperts(Mlp, self.experts, input, gating, self.moe_opts);
    }

    pub const OnDisk = struct {
        router: zml.nn.Linear,
        experts: struct {
            down_proj_bias: zml.Tensor,
            down_proj_blocks: zml.Tensor,
            down_proj_scales: zml.Tensor,
            gate_up_proj_bias: zml.Tensor,
            gate_up_proj_blocks: zml.Tensor,
            gate_up_proj_scales: zml.Tensor,
        },

        pub fn rewrite(on_disk: OnDisk, experts_per_token: u32, options: Options) MoE {
            const e = on_disk.experts;
            return .{
                .experts = .{
                    .gate_up_proj = .{
                        // We need to bitcast the scale cause safetensors doesn't encode f8 types correctly
                        .scale = e.gate_up_proj_scales.withTags(.{ .expert, .out, .d }),
                        // We don't bitcast here because PJRT doesn't handle packed host buffers
                        .blocks = e.gate_up_proj_blocks.withTags(.{ .expert, .out, .d, .d_block }),
                        .blocks_dtype = .f4e2m1,
                        .bias = e.gate_up_proj_bias.withTags(.{ .expert, .d }),
                    },
                    .down_proj = .{
                        .blocks = e.down_proj_blocks.withTags(.{ .expert, .out, .d, .d_block }),
                        .blocks_dtype = .f4e2m1,
                        .scale = e.down_proj_scales.withTags(.{ .expert, .out, .d }),
                        .bias = e.down_proj_bias.withTags(.{ .expert, .d }),
                    },
                },

                .router = .{
                    .weight = on_disk.router.weight.withTags(.{ .expert, .d }),
                    .bias = on_disk.router.bias.?.withTags(.{.expert}),
                },

                .moe_opts = .{
                    .experts_per_token = experts_per_token,
                    .tokens_per_expert_ratio = options.tokens_per_expert_ratio,
                    .normalization = .softmax,
                },
            };
        }
    };
};

pub const Mlp = struct {
    gate_up_proj: BlockScaledLinear, // {.out = intermediate_size * 2, .d = hidden_size / block_size, .d_block = block_size }
    down_proj: BlockScaledLinear, // {.out = hidden_size * 2, .d = intermediate_size / block_size, .d_block = block_size }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const dt = x.dtype();
        var gate, var up = zml.nn.splitRealImg(self.gate_up_proj.forward(x), .interleaved);
        gate = .minimum(gate, .scalar(7, dt));
        up = .clamp(up, .scalar(-7, dt), .scalar(7, dt));

        const out = gate.quickGelu().mul(up.addConstant(1));
        return zml.call(self.down_proj, .forward, .{out});
    }

    pub fn format(self: Mlp, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("Mlp(gate_up_proj=.{f}, down_proj=.{f})", .{ self.gate_up_proj, self.down_proj });
    }
};

pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    sinks: zml.Tensor,

    o_proj: zml.nn.Linear,

    sliding_window: ?u32,
    num_heads: i64,
    num_kv_heads: i64,
    rope_opts: zml.nn.RopeOpts,

    /// Self Attention.
    ///   - If token_index is set, x is assumed to be the representation of one new token,
    /// and kv_cache will be read for the previous tokens.
    ///   - If token_index is not set, x is assumed to be the representation of all tokens
    /// since the beginning of the sequence, and kv_cache won't be read.
    /// In both case, kv_cache will be updated with the computed key and value.
    /// x: {.b, .s, .d } -> .{.b, .s, .d}
    pub fn forward(
        self: SelfAttn,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
    ) struct { zml.Tensor, KvCache } {
        const num_kv_heads = self.num_kv_heads;
        var q = zml.call(self.q_proj, .forward, .{x}).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto }).withSharding(.{.h});
        var k = zml.call(self.k_proj, .forward, .{x}).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto }).withSharding(.{.h});
        var v = zml.call(self.v_proj, .forward, .{x}).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto }).withSharding(.{.h});

        // Generate the attention mask.
        const seq_len = kv_cache.k.dim(.k);
        var attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, x.dtype(), self.sliding_window);

        // Note: in Pytorch it would be very inefficient to generate the full attn_mask,
        // then slice into it, but XLA is able to optimize this correctly.
        attn_mask = attn_mask.gatherSlices(zml.Shape.init(.{ .q = x.dim(.s) }, attn_mask.dtype()), token_index.reshape(.{ .coord = 1 }), .{});

        // In self-attention, .s axis is used both for keys and queries.
        const pos_index = b: {
            const temp = zml.Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };

        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const dtype = q.dtype();
        const new_kv_cache = kv_cache.update(k, v, token_index);
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);

        // TODO ringbuffer kv cache.

        const softmax_bias = self.sinks.withTags(.{.h});
        const attn_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .allow_cudnn = true, .softmax_bias = softmax_bias });
        // const attn_output = zml.nn.sdpaMemEfficient(q, k, v, .{ .attn_mask = attn_mask }, .{ .q_chunk_size = 4096, .k_chunk_size = 1024 });
        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return .{ zml.call(self.o_proj, .forward, .{attn}), new_kv_cache };
    }
};

pub const KvCache = struct {
    k: zml.Tensor,
    v: zml.Tensor,
    layer_index: zml.Tensor,

    pub fn init(kv_shape: zml.Shape) KvCache {
        // The KV-cache is initialized with ones to detect reads of uninitialized memory.
        return .{
            .k = .constant(kv_shape, kv_shape.dtype().one()).withSharding(.{.h}),
            .v = .constant(kv_shape, kv_shape.dtype().one()).withSharding(.{.h}),
            .layer_index = .scalar(-1, .u32),
        };
    }

    pub fn initShape(kv_shape: zml.Shape) zml.ShapeOf(KvCache) {
        return .{
            .k = kv_shape,
            .v = kv_shape,
            .layer_index = zml.Shape.init(.{}, .u32),
        };
    }

    pub fn initBuffer(kv_shape: zml.Shape, platform: zml.Platform) !zml.Bufferized(KvCache) {
        return .{
            .k = try zml.Buffer.uninitialized(platform, kv_shape, .{}),
            .v = try zml.Buffer.uninitialized(platform, kv_shape, .{}),
            .layer_index = try zml.Buffer.uninitialized(platform, .scalar(.u32), .{}),
        };
    }

    pub fn keys(self: KvCache) zml.Tensor {
        return self.k.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn values(self: KvCache) zml.Tensor {
        return self.v.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn update(self: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: ?zml.Tensor) KvCache {
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
            .layer_index = .scalar(layer_index, .u32),
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

pub const BlockScaledLinear = struct {
    blocks: zml.Tensor,
    scale: zml.Tensor,
    bias: ?zml.Tensor = null,
    blocks_dtype: zml.DataType,

    pub fn forward(self: BlockScaledLinear, x: zml.Tensor) zml.Tensor {
        const ctx = x.getContext();
        const res_shape = x.shape().setDim(-1, self.blocks.dim(-3));

        // Bitcast to our actual type. This allows to load weights in a packed layout.
        const blocks_0 = self.blocks.bitCast(self.blocks_dtype);
        const blocks = blocks_0.merge(.{ .d_block = .{ .d_block, .bitcast } });

        const scale = self.scale.bitCast(.f8e8m0);

        // log.warn("BlockScaledLinear({}): {f} -> {f}", .{ self, x, res_shape });
        const y = switch (ctx._platform.target) {
            else => y: {
                var dequantized_weight: zml.Tensor = .mul(
                    blocks.convert(x.dtype()),
                    scale.convert(x.dtype()).appendAxes(.{.d_block}),
                );
                var y = x.dot(dequantized_weight.merge(.{ .d = .{ .d, .d_block } }), .{.d});
                // std.log.warn("output shape: {f}", .{y});
                std.debug.assert(y.shape().eql(res_shape));
                y._shape = res_shape;
                break :y y;
            },
        };
        return if (self.bias) |bias| y.add(bias.broad(y.shape())) else y;
    }

    pub fn format(self: BlockScaledLinear, writer: *std.Io.Writer) !void {
        try writer.print("BlockScaledLinear(blocks={f}, scale={f}, bias={?f}, dt={t})", .{ self.blocks, self.scale, self.bias, self.blocks_dtype });
    }
};

const MoeOpts = struct {
    experts_per_token: u32,
    tokens_per_expert_ratio: ?f32 = 0.0,
    normalization: Normalization,

    pub const Normalization = enum { linear, softmax };
};

/// We have three algorithms,
/// * one for single-stream inference (naive),
/// * one for small batch sized with exact precision that sends all tokens to all experts.
///   this isn't too costly as long as the batch size is small and the experts are IO bound.
/// * one for big batch size that assign a fixed compute budget per expert and
/// experts chose the tokens they want to handle. This introduces noise since it's possible
/// a token doesn't get their requested expert.
///   The parameter `tokens_per_expert_ratio` control how much compute budget is granted:
///   expert_budget = ratio * (num_tokens * experts_per_token / num_experts).
///   Bigger values of ratio will ensure it's rare a token doesn't get it's top 2 tokens.
///
/// The preferred algorithm is the batched one,
/// it is selected as soon there is enough tokens to guarantee that experts will be active most of the time.
///
/// - input: .{ .s, .d } per-entry vector
/// - gating: .{ .s, .expert } per-entry expert-affinity
/// - experts: .{ .expert, .d_out, .d } expert layer (need to have a .forward method).
/// -> output: .{ .s, .d_out }
pub fn mixtureOfExperts(Expert: type, experts: Expert, input: zml.Tensor, gating: zml.Tensor, opts: MoeOpts) zml.Tensor {
    log.warn("mixtureOfExperts({s}, {f}, {f}, {})", .{ @typeName(Expert), input, gating, opts });
    const num_tokens: u32 = @intCast(input.dim(.s));
    const num_experts = gating.dim(.expert);
    stdx.debug.assert(opts.experts_per_token > 0, "mixtureOfExperts expects opts.experts_per_token > 0, got {}", .{opts});

    if (num_tokens == 1) {
        return moePerTokenRouting(Expert, experts, input, gating, opts);
    }

    const tokens_per_expert: u32 = if (opts.tokens_per_expert_ratio) |ratio| tpe: {
        const compute_budget = ratio * @as(f32, @floatFromInt(num_tokens * opts.experts_per_token));
        var tpe: u32 = @intFromFloat(stdx.math.divFloat(f32, compute_budget, num_experts));
        // Round to next multiple of 8 to avoid weird shapes.
        if (tpe % 8 != 0) tpe += 8 - (tpe % 8);
        break :tpe tpe;
    } else num_tokens;

    if (3 * tokens_per_expert <= 2 * num_tokens) {
        const routing, const tokens_ids_per_expert = dispatchTokens(gating, .{
            .tokens_per_expert = tokens_per_expert,
            .experts_per_token = opts.experts_per_token,
            .normalization = opts.normalization,
        });
        const scores_per_expert = routing.transpose(.{ .expert, .s }).gather(.{ .s = tokens_ids_per_expert }, .{});
        const input_per_expert = input.gather(.{ .s = tokens_ids_per_expert }, .{});
        var output_per_expert = experts.forward(input_per_expert);
        output_per_expert = output_per_expert.mul(scores_per_expert.convert(output_per_expert.dtype()).broad(output_per_expert.shape()));

        // Reverse engineer the normal output shape that one expert would have produced for all tokens.
        // If this fall short, we could use the "sliced_expert" strategy and call forward ourselves.
        const output_shape = output_per_expert.shape().drop(.expert).rename(.{ .top_token = .s }).setDim(.s, num_tokens);
        const output = zml.Tensor.scatterSlices(
            .constant(output_shape, output_shape.dtype().zero()),
            .{ .s = tokens_ids_per_expert },
            output_per_expert,
            .{ .update_fn = zml.Tensor.ScatterOpts.increment },
        );

        log.warn("mixtureOfExperts({s}, {f}, {f}) -> fixed budget impl tpe: {d}, tokens: {d}", .{ @typeName(Expert), input, gating, tokens_per_expert, num_tokens });
        return output;
    } else {
        return mixtureOfExpertsAllToAll(Expert, experts, input, gating, opts);
    }
}

/// Few tokens: most experts are unused, experts have at most one token.
/// Select active experts and compute with that.
pub fn moePerTokenRouting(Expert: type, experts: Expert, input: zml.Tensor, gating: zml.Tensor, opts: MoeOpts) zml.Tensor {
    const num_tokens: u32 = @intCast(input.dim(.s));
    stdx.debug.assert(num_tokens < 32, "Trying to unroll a lot of tokens !", .{});
    const per_token_outputs = input.getContext().allocator().alloc(zml.Tensor, num_tokens) catch @panic("OOM");

    const routing = gating.topK(.{ .top_expert = .expert }, opts.experts_per_token, .{});
    const per_token_score = switch (opts.normalization) {
        .linear => routing.values.div(routing.values.sum(.top_expert)),
        .softmax => routing.values.softmax(.top_expert),
    };

    for (per_token_outputs, 0..num_tokens) |*output, tok_id| {
        for (0..opts.experts_per_token) |expert_rank| {
            const expert_id = routing.indices.choose(.{ .s = tok_id, .top_expert = expert_rank }).asScalar();
            const expert_score = per_token_score.choose(.{ .s = tok_id, .top_expert = expert_rank }).asScalar();

            var sliced_expert: Expert = undefined;
            zml.meta.mapAlloc(struct {
                pub fn cb(expert_id_: zml.Tensor, expert_weight: zml.Tensor) zml.Tensor {
                    return expert_weight.gather(.{ .expert = expert_id_ }, .{});
                }
            }.cb, stdx.noalloc, expert_id, experts, &sliced_expert) catch unreachable;

            // TODO how does this work when the two experts are on different gpus?
            // does the compute overlap ?
            var expert_output = sliced_expert.forward(input.choose(.{ .s = tok_id }));
            expert_output = .mul(
                expert_output,
                expert_score.convert(input.dtype()).broad(expert_output.shape()),
            );
            output.* = if (expert_rank > 0) output.add(expert_output) else expert_output;
        }
    }

    log.warn("mixtureOfExperts({s}, {f}, {f}) -> single-stream impl", .{ @typeName(Expert), input, gating });
    return .stack(per_token_outputs, 0, .s);
}

/// Send all tokens to all experts, and apply gating.
pub fn mixtureOfExpertsAllToAll(Expert: type, experts: Expert, input: zml.Tensor, gating: zml.Tensor, opts: MoeOpts) zml.Tensor {
    log.warn("mixtureOfExperts({s}, {f}, {f}) -> all to all impl", .{ @typeName(Expert), input, gating });
    const num_experts = gating.dim(.expert);
    const hard_gating = hardGating(gating, opts).print();
    // TODO: `input.insertAxes(0, .{.expert}).repeat1d(.expert, num_experts)` is too verbose for just broadcasting along a new axis`
    const output_per_expert = experts.forward(input.insertAxes(0, .{.expert}).repeat1d(.expert, @intCast(num_experts)));
    return output_per_expert.dot(hard_gating.convert(input.dtype()), .expert);
}

/// Given `(token, expert) -> scores`,
/// keeps only the top-k expert per token, and normalize the scores accordingly.
/// Non selected experts will have a 0 score.
pub fn hardGating(gating: zml.Tensor, opts: MoeOpts) zml.Tensor {
    const routing = gating.topK(.{ .top_expert = .expert }, opts.experts_per_token, .{});

    const per_token_score = switch (opts.normalization) {
        .linear => routing.values.div(routing.values.sum(.top_expert)),
        .softmax => routing.values.softmax(.top_expert),
    };

    return zml.Tensor.scatterSlices(
        .zeroes(gating.shape()),
        .{ .expert = routing.indices },
        per_token_score,
        .{ .indices_are_unique = true },
    );
}

/// Lot of tokens, each experts chose their tokens.
/// It means that some tokens may have only one expert assigned.
/// Each token will get assigned to at least one expert IIF the input gating is sums up to 1 (typically softmax output).
/// Returns the actual `(token, expert) -> scores` used.
pub fn dispatchTokens(
    gating: zml.Tensor,
    opts: struct {
        tokens_per_expert: u32,
        experts_per_token: u32,
        normalization: MoeOpts.Normalization,
    },
) [2]zml.Tensor {
    const num_experts = gating.dim(.expert);

    const token_pref = gating.argsort(.expert, .{ .descending = true });
    var expert_rank: zml.Tensor = .scatterSlices(
        .zeroes(gating.shape().withDtype(.i32)),
        .{ .expert = token_pref },
        .addConstant(.iota(gating.shape(), .expert), 1),
        .{ .indices_are_unique = true },
    );
    // The pow(expert_rank) here means that we strongly favor top 1 over top 2 and top 2 over top 3.
    // expert_routing: (expert, top_token) -> token
    const expert_routing = gating.pow(expert_rank.convert(gating.dtype())).topK(.{ .top_token = .s }, opts.tokens_per_expert, .{});
    const scores_per_expert = gating.gather(.{ .s = expert_routing.indices }, .{});

    // Update the gating coefficient to account for the expert routing.
    // Each (token, expert) which can't be computed within the given budget is left to 0.
    const gating_v2: zml.Tensor = .scatterSlices(
        .zeroes(gating.shape()),
        .{ .s = expert_routing.indices },
        scores_per_expert,
        .{ .indices_are_unique = true, .update_fn = zml.Tensor.ScatterOpts.override },
    );
    // Now set to zero the scores (token, expert) for tokens that have been assigned more than experts_per_token.
    const lowest_experts = gating_v2.topK(.{ .top_expert = .expert }, @intCast(num_experts - opts.experts_per_token), .{ .descending = false });
    var gating_v3: zml.Tensor = .scatterSlices(
        gating_v2,
        .{ .expert = lowest_experts.indices },
        .zeroes(lowest_experts.values.shape()),
        .{ .indices_are_unique = true, .update_fn = zml.Tensor.ScatterOpts.override },
    );
    // Then normalize so the sum of experts scores for one token sums up to 1.
    gating_v3 = switch (opts.normalization) {
        .linear => gating_v3.div(gating_v3.sum(.expert)),
        .softmax => gating_v3.softmax(.expert),
    };
    const tokens_ids_per_expert = expert_routing.indices.transpose(.{ .expert, .top_token });

    return .{ gating_v3, tokens_ids_per_expert };
}

test mixtureOfExperts {
    const async = @import("async");

    const Expert = struct {
        weight: zml.Tensor,

        pub fn forward(self: @This(), x: zml.Tensor) zml.Tensor {
            // I'm always dumb founded by the weird output layout of `dot`.
            return self.weight.dot(x, .{.d}).rename(.{ .d_out = .d }).contiguous(.{.d});
        }
    };

    const SimpleMoE = struct {
        router: zml.nn.Linear,
        experts: Expert,

        pub fn forward(self: @This(), x: zml.Tensor) zml.Tensor {
            const routing = self.router.forward(x).convert(.f32).softmax(.expert);
            return mixtureOfExperts(Expert, self.experts, x, routing, .{ .experts_per_token = 2, .tokens_per_expert_ratio = 2 });
        }
    };

    const platform = zml.testing.env();
    const store = zml.aio.detectFormatAndOpen(std.testing.allocator, "zml/test_data/moe.activations.pt") catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer store.deinit();

    var model_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer model_arena.deinit();
    var moe_model = try zml.aio.populateModel(SimpleMoE, model_arena.allocator(), store);
    moe_model.router.weight = moe_model.router.weight.withTags(.{ .expert, .d });
    moe_model.experts.weight = moe_model.experts.weight.withTags(.{ .expert, .d_out, .d });

    var compilation_prefill = try async.async(zml.compileModel, .{
        std.testing.allocator,
        SimpleMoE.forward,
        moe_model,
        .{zml.Shape.init(.{ .s = 24, .d = 64 }, .bf16)},
        platform,
    });
    var compilation_gen = try async.async(zml.compileModel, .{
        std.testing.allocator,
        SimpleMoE.forward,
        moe_model,
        .{zml.Shape.init(.{ .s = 1, .d = 64 }, .bf16)},
        platform,
    });

    var moe_weights = try zml.aio.loadModelBuffers(MoE, moe_model, store, std.testing.allocator, platform);
    defer zml.aio.unloadBuffers(&moe_weights);

    {
        const moe_prefill = (try compilation_prefill.awaitt()).prepare(moe_weights);
        defer moe_prefill.deinit();

        const x = try store.get("in").?.toDevice(platform);
        const actual = moe_prefill.call(.{x});

        try zml.testing.expectClose(store.get("out").?, actual, 5e-2);
    }

    {
        const moe_gen = (try compilation_gen.awaitt()).prepare(moe_weights);
        defer moe_gen.deinit();

        const x = try store.get("in").?.slice1d(0, .{ .start = 12, .end = 13 }).toDevice(platform);
        const actual = moe_gen.call(.{x});

        try zml.testing.expectClose(store.get("out").?.slice1d(0, .{ .start = 12, .end = 13 }), actual, 5e-2);
    }
}
