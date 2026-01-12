const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;
const stdx = zml.stdx;

const log = std.log.scoped(.gpt_oss);

pub const GptOss = struct {
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

    // Restored Mode to distinguish behavior
    pub const Mode = union(enum) {
        /// In prefill mode, we pass the length of the prompt (to select the last token output).
        /// The KV cache is updated linearly from 0.
        prefill: zml.Tensor,
        /// In gen mode, we pass the specific position of the current token.
        gen: zml.Tensor,
    };

    lm_head: ?zml.nn.Linear,
    model: Model,
    config: Config,
    options: Options,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, options: Options) !GptOss {
        const lm_head: ?zml.nn.Linear = if (store.withPrefix("lm_head").maybeCreateTensorWithTags("weight", .{ .voc, .d })) |weight|
            .init(weight, null, .d)
        else
            null;

        return .{
            .lm_head = lm_head,
            .model = try Model.init(allocator, store.withPrefix("model"), config, options),
            .config = config,
            .options = options,
        };
    }

    pub fn deinit(self: GptOss, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
    }

    pub fn loadBuffers(self: *const GptOss, allocator: std.mem.Allocator, io: std.Io, store: zml.io.TensorStore.View, platform: zml.Platform) !zml.Bufferized(GptOss) {
        const lm_head = if (self.lm_head) |lm_head| try zml.io.loadBuffersFromId(allocator, io, lm_head, store.withPrefix("lm_head"), platform) else null;
        const model = try self.model.loadBuffers(allocator, io, store.withPrefix("model"), platform);
        return .{
            .lm_head = lm_head,
            .model = model,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(GptOss), allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }

    pub fn forward(
        self: GptOss,
        tokens_: zml.Tensor,
        mode: Mode,
        kv_cache: KvCache,
        rng: Tensor.Rng,
    ) struct { Tensor, KvCache, Tensor.Rng } {
        log.info(" tokens: {f}", .{tokens_.shape()});
        switch (mode) {
            .prefill => {
                log.info("mode prefill {f}", .{mode.prefill.shape()});
            },
            .gen => {
                log.info("mode gen {f}", .{mode.gen.shape()});
            },
        }

        const tokens = tokens_.withPartialTags(.{.s});

        // Model forward returns the sequence of hidden states
        var out, const updated_kv_cache = self.model.forward(tokens, mode, kv_cache);

        log.info("out: {f}", .{out.shape()});

        // Logic to select which token to predict
        switch (mode) {
            // In prefill, we only care about predicting the token AFTER the prompt.
            // So we select the last hidden state: out[prompt_len - 1]
            .prefill => |prompt_len| {
                const last_idx = prompt_len.convert(.i32).sub(zml.Tensor.scalar(1, .i32));
                out = out.gather(.{ .s = last_idx }, .{ .indices_are_sorted = true });
            },
            // In generation, we are processing 1 token, so we take the whole result.
            .gen => {},
        }

        const new_tokens, const new_rng = self.sampleTokens(self.lm_head, out, rng, self.options.sampling_strategy);

        // If we are in prefill, the output shape is {1} (the next token), but the input `tokens` was {seq_len}.
        // We cannot reuseBuffer(tokens) in prefill because sizes differ.
        const res = switch (mode) {
            .gen => new_tokens.convert(tokens.dtype()).reuseBuffer(tokens),
            .prefill => new_tokens.convert(tokens.dtype()).appendAxes(.{.s}),
        };

        log.info("Generated tokens: {f}", .{res.shape()});

        return .{ res, updated_kv_cache, new_rng };
    }

    pub fn sampleTokens(
        self: GptOss,
        lm_head_: ?zml.nn.Linear,
        out_: Tensor,
        rng: Tensor.Rng,
        opts: zml.nn.SamplingStrategy,
    ) struct { Tensor, Tensor.Rng } {
        const out = out_.withPartialTags(.{.d});

        var logits = blk: {
            if (lm_head_) |lm_head| {
                break :blk lm_head.forward(out);
            } else {
                break :blk self.model.embed_tokens.weight.withTags(.{ .voc, .d }).dot(out, .d);
            }
        };

        if (logits.shape().hasTag(.voc) == null)
            logits = logits.rename(.{ .d = .voc });

        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, opts, rng);
        return .{ next_tokens, new_rng };
    }
};

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

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: GptOss.Config, options: GptOss.Options) !Model {
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = try .init(store.withPrefix("layers").withLayer(i), config, options);
        }

        return .{
            .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight") },
            .norm = .{ .weight = store.withPrefix("norm").createTensorWithTags("weight", .{.d}), .eps = config.rms_norm_eps },
            .layers = layers,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn loadBuffers(self: *const Model, allocator: std.mem.Allocator, io: std.Io, store: zml.io.TensorStore.View, platform: zml.Platform) !zml.Bufferized(Model) {
        const layers = try allocator.alloc(zml.Bufferized(TransformerLayer), self.layers.len);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = try zml.io.loadBuffersFromId(allocator, io, self.layers[i], store.withLayer(i), platform);
        }

        return .{
            .embed_tokens = try zml.io.loadBuffersFromId(allocator, io, self.embed_tokens, store.withPrefix("embed_tokens"), platform),
            .layers = layers,
            .norm = try zml.io.loadBuffersFromId(allocator, io, self.norm, store.withPrefix("norm"), platform),
        };
    }

    pub fn forward(self: Model, tokens: Tensor, mode: GptOss.Mode, kv_cache: KvCache) struct { Tensor, KvCache } {
        const embeds = embed(self.embed_tokens, tokens);
        var hidden = embeds;

        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden, updated_kv_cache = layer.forward(hidden, mode, updated_kv_cache.atLayer(i));
        }
        const output = self.norm.forward(hidden);

        return .{ output, updated_kv_cache.reuseBuffer(kv_cache) };
    }

    pub fn embed(embed_tokens_: zml.nn.TokenEmbedding, tokens_: Tensor) Tensor {
        return embed_tokens_.forward(tokens_).withPartialTags(.{.d});
    }
};

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: SelfAttn,
    post_attention_layernorm: RmsNorm,
    mlp: MoE,

    pub fn init(store: zml.io.TensorStore.View, config: GptOss.Config, options: GptOss.Options) !TransformerLayer {
        return .{
            .input_layernorm = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps),
            .self_attn = try .init(store.withPrefix("self_attn"), config),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps),
            .mlp = .init(store.withPrefix("mlp"), config, options),
        };
    }

    pub fn forward(
        self: TransformerLayer,
        x0: Tensor,
        mode: GptOss.Mode,
        kv_cache: KvCache,
    ) struct { Tensor, KvCache } {
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {f}", .{x0});

        const x0_normalized = self.input_layernorm.forward(x0);
        const delta0, const updated_kv_cache = self.self_attn.forward(x0_normalized, mode, kv_cache);
        const x1 = x0.add(delta0);

        const x1_normalized = self.post_attention_layernorm.forward(x1);
        const x2 = self.mlp.forward(x1_normalized).add(x1);

        return .{ x2.reuseBuffer(x0), updated_kv_cache };
    }
};

pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    sinks: Tensor,
    o_proj: zml.nn.Linear,

    q_norm: ?RmsNorm,
    k_norm: ?RmsNorm,

    num_heads: i64 = undefined,
    num_kv_heads: i64 = 0,
    rope_opts: zml.nn.RopeOpts = undefined,
    sliding_window: ?u32 = null,

    fn initProj(store: zml.io.TensorStore.View) zml.nn.Linear {
        return .init(store.createTensorWithTags("weight", .{ .dout, .d }), store.maybeCreateTensorWithTags("bias", .{.dout}), .d);
    }

    pub fn init(store: zml.io.TensorStore.View, config: GptOss.Config) !SelfAttn {
        return .{
            .q_proj = initProj(store.withPrefix("q_proj")),
            .k_proj = initProj(store.withPrefix("k_proj")),
            .v_proj = initProj(store.withPrefix("v_proj")),
            .o_proj = initProj(store.withPrefix("o_proj")),
            .sinks = store.createTensorWithTags("sinks", .{.h}),
            .q_norm = null,
            .k_norm = null,
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
            .sliding_window = if (config.sliding_window > 0) config.sliding_window else null,
            .rope_opts = .{
                .layout = if (config.hf_rope_impl) .sequential else .interleaved,
                .freq_base = config.rope_theta,
                .scaling = config.rope_scaling,
            },
        };
    }

    pub fn forward(
        self: SelfAttn,
        x: Tensor,
        mode: GptOss.Mode,
        kv_cache: KvCache,
    ) struct { Tensor, KvCache } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;
        var q = self.q_proj.forward(x).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = self.k_proj.forward(x).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        var v = self.v_proj.forward(x).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });

        // Generate the attention mask.
        const seq_len = kv_cache.k.dim(.k);
        var attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, x.dtype(), self.sliding_window);

        // Calculate Rope Offset and Mask Slicing
        const rope_offset = switch (mode) {
            .prefill => zml.Tensor.scalar(0, .u32),
            .gen => |idx| idx,
        };

        // Slice the mask based on the current position
        attn_mask = attn_mask.gatherSlices(zml.Shape.init(.{ .q = x.dim(.s) }, attn_mask.dtype()), rope_offset.reshape(.{ .coord = 1 }), .{});

        const pos_index = b: {
            const temp = Tensor.arange(.{ .end = x.dim(.s) }, rope_offset.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = x.dim(.s) }, rope_offset.dtype()));
            break :b temp.add(rope_offset.broad(temp.shape()));
        };

        if (self.q_norm) |norm| q = norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        if (self.k_norm) |norm| k = norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const dtype = q.dtype();

        // Determine KV Update Index:
        // Prefill: null (auto-detect linear write)
        // Gen: rope_offset (scalar index)
        const kv_idx = switch (mode) {
            .prefill => null,
            .gen => |idx| idx.rename(.{ .s = .k }),
        };

        const new_kv_cache = kv_cache.update(k, v, kv_idx);

        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);

        const attn_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .softmax_bias = self.sinks, .allow_cudnn = true });
        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return .{ self.o_proj.forward(attn), new_kv_cache };
    }
};

const RmsNorm = struct {
    weight: Tensor,
    eps: f32 = 1e-5,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{ .weight = store.createTensorWithTags("weight", .{.d}), .eps = eps };
    }

    pub fn forward(self: RmsNorm, input: Tensor) Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const normalized = zml.nn.rmsNorm(x.convert(.f32), .d, self.eps);
        return normalized.mul(self.weight.convert(.f32).withTags(.{.d}).broad(x.shape())).convert(input.dtype());
    }
};

const MoE = struct {
    experts: Mlp,
    router: zml.nn.Linear,
    moe_opts: MoeOpts,

    pub fn init(store: zml.io.TensorStore.View, config: GptOss.Config, options: GptOss.Options) MoE {
        const moe_on_disk: OnDisk = .{ .router = zml.nn.Linear.init(store.createTensorWithTags("router.weight", .{ .exp, .d }), store.createTensorWithTags("router.bias", .{.exp}), .d), .experts = .{
            .down_proj_bias = store.createTensorWithTags("experts.down_proj_bias", .{ .exp, .d }),
            .down_proj_blocks = store.createTensorWithTags("experts.down_proj_blocks", .{ .exp, .out, .d, .d_blocks }),
            .down_proj_scales = store.createTensorWithTags("experts.down_proj_scales", .{ .exp, .out, .d }),
            .gate_up_proj_bias = store.createTensorWithTags("experts.gate_up_proj_bias", .{ .exp, .d }),
            .gate_up_proj_blocks = store.createTensorWithTags("experts.gate_up_proj_blocks", .{ .exp, .out, .d, .d_blocks }),
            .gate_up_proj_scales = store.createTensorWithTags("experts.gate_up_proj_scales", .{ .exp, .out, .d }),
        } };
        return OnDisk.rewrite(moe_on_disk, config.experts_per_token, options);
    }

    pub fn forward(self: MoE, input: Tensor) Tensor {
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

        pub fn rewrite(on_disk: OnDisk, experts_per_token: u32, options: GptOss.Options) MoE {
            const e = on_disk.experts;
            return .{
                .experts = .{
                    .gate_up_proj = .{
                        .scale = e.gate_up_proj_scales.withTags(.{ .expert, .out, .d }),
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
                .router = zml.nn.Linear.init(on_disk.router.weight.withTags(.{ .expert, .d }), on_disk.router.bias.?.withTags(.{.expert}), .d),
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
    gate_up_proj: BlockScaledLinear,
    down_proj: BlockScaledLinear,

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const dt = x.dtype();
        var gate, var up = zml.nn.splitRealImg(self.gate_up_proj.forward(x), .interleaved);
        gate = .minimum(gate, .scalar(7, dt));
        up = .clamp(up, .scalar(-7, dt), .scalar(7, dt));
        const out = gate.quickGelu().mul(up.addConstant(1));
        return self.down_proj.forward(out);
    }
};

pub const BlockScaledLinear = struct {
    blocks: zml.Tensor,
    scale: zml.Tensor,
    bias: ?zml.Tensor = null,
    blocks_dtype: zml.DataType,

    pub fn forward(self: BlockScaledLinear, x: zml.Tensor) zml.Tensor {
        const res_shape = x.shape().setDim(-1, self.blocks.dim(-3));
        const blocks_0 = self.blocks.bitCast(self.blocks_dtype);
        const blocks = blocks_0.merge(.{ .d_block = .{ .d_block, .bitcast } });
        const scale = self.scale.bitCast(.f8e8m0);

        const y = y: {
            var dequantized_weight: zml.Tensor = .mul(
                blocks.convert(x.dtype()),
                scale.convert(x.dtype()).appendAxes(.{.d_block}),
            );
            var y = x.dot(dequantized_weight.merge(.{ .d = .{ .d, .d_block } }), .d);
            y._shape = res_shape;
            break :y y;
        };
        return if (self.bias) |bias| y.add(bias.broad(y.shape())) else y;
    }
};

const MoeOpts = struct {
    experts_per_token: u32,
    tokens_per_expert_ratio: ?f32 = 0.0,
    normalization: Normalization,
    pub const Normalization = enum { linear, softmax };
};

pub fn mixtureOfExperts(Expert: type, experts: Expert, input: zml.Tensor, gating: zml.Tensor, opts: MoeOpts) zml.Tensor {
    const num_tokens: u32 = @intCast(input.dim(.s));
    const num_experts = gating.dim(.expert);

    if (num_tokens == 1) {
        return moePerTokenRouting(Expert, experts, input, gating, opts);
    }

    const tokens_per_expert: u32 = if (opts.tokens_per_expert_ratio) |ratio| tpe: {
        const compute_budget = ratio * @as(f32, @floatFromInt(num_tokens * opts.experts_per_token));
        var tpe: u32 = @intFromFloat(stdx.math.divFloat(f32, compute_budget, num_experts));
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

        const output_shape = output_per_expert.shape().drop(.expert).rename(.{ .top_token = .s }).setDim(.s, num_tokens);
        const output = zml.Tensor.scatterSlices(
            .zeroes(output_shape),
            .{ .s = tokens_ids_per_expert },
            output_per_expert,
            .{ .update_fn = zml.Tensor.ScatterOpts.increment },
        );
        return output;
    } else {
        return mixtureOfExpertsAllToAll(Expert, experts, input, gating, opts);
    }
}

pub fn moePerTokenRouting(Expert: type, experts: Expert, input: zml.Tensor, gating: zml.Tensor, opts: MoeOpts) zml.Tensor {
    const num_tokens: u32 = @intCast(input.dim(.s));
    const per_token_outputs = zml.module.CompilationContext.current().allocator.alloc(Tensor, num_tokens) catch @panic("OOM");

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

            var expert_output = sliced_expert.forward(input.choose(.{ .s = tok_id }));
            expert_output = .mul(
                expert_output,
                expert_score.convert(input.dtype()).broad(expert_output.shape()),
            );
            output.* = if (expert_rank > 0) output.add(expert_output) else expert_output;
        }
    }
    return .stack(per_token_outputs, 0, .s);
}

pub fn mixtureOfExpertsAllToAll(Expert: type, experts: Expert, input: zml.Tensor, gating: zml.Tensor, opts: MoeOpts) zml.Tensor {
    const num_experts = gating.dim(.expert);
    const hard_gating = hardGating(gating, opts).print();
    const output_per_expert = experts.forward(input.insertAxes(0, .{.expert}).repeat1d(.expert, @intCast(num_experts)));
    return output_per_expert.dot(hard_gating.convert(input.dtype()), .expert);
}

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
    const expert_routing = gating.pow(expert_rank.convert(gating.dtype())).topK(.{ .top_token = .s }, opts.tokens_per_expert, .{});
    const scores_per_expert = gating.gather(.{ .s = expert_routing.indices }, .{});
    const gating_v2: zml.Tensor = .scatterSlices(
        .zeroes(gating.shape()),
        .{ .s = expert_routing.indices },
        scores_per_expert,
        .{ .indices_are_unique = true, .update_fn = zml.Tensor.ScatterOpts.override },
    );
    const lowest_experts = gating_v2.topK(.{ .top_expert = .expert }, @intCast(num_experts - opts.experts_per_token), .{ .descending = false });
    var gating_v3: zml.Tensor = .scatterSlices(
        gating_v2,
        .{ .expert = lowest_experts.indices },
        .zeroes(lowest_experts.values.shape()),
        .{ .indices_are_unique = true, .update_fn = zml.Tensor.ScatterOpts.override },
    );
    gating_v3 = switch (opts.normalization) {
        .linear => gating_v3.div(gating_v3.sum(.expert)),
        .softmax => gating_v3.softmax(.expert),
    };
    const tokens_ids_per_expert = expert_routing.indices.transpose(.{ .expert, .top_token });
    return .{ gating_v3, tokens_ids_per_expert };
}

pub const KvCache = struct {
    k: Tensor,
    v: Tensor,
    layer_index: Tensor,

    pub fn init(kv_shape: zml.Shape) KvCache {
        return .{
            .k = .fromShape(kv_shape),
            .v = .fromShape(kv_shape),
            .layer_index = .init(.{}, .u32),
        };
    }

    // pub fn initShape(kv_shape: zml.Shape) ShapeOf(KvCache) {
    //     return .{
    //         .k = kv_shape,
    //         .v = kv_shape,
    //         .layer_index = zml.Shape.init(.{}, .u32),
    //     };
    // }

    pub fn initBuffer(self: KvCache, io: std.Io, platform: zml.Platform) !zml.Bufferized(KvCache) {
        return .{
            .k = try zml.Buffer.uninitialized(io, platform, self.k.shape(), .{}),
            .v = try zml.Buffer.uninitialized(io, platform, self.v.shape(), .{}),
            .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(KvCache)) void {
        self.k.deinit();
        self.v.deinit();
        self.layer_index.deinit();
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

        // Reverted to robust logic: if token_index is null, we auto-arange from 0.
        // This handles prefill (linear write) vs gen (scatter write).
        const idx = if (token_index) |idx| idx else zml.Tensor.arange(.{ .end = new_k.dim(.k) }, .u32).withTags(.{.k});

        return .{
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
