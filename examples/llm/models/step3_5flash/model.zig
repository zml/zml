const std = @import("std");
const zml = @import("zml");
const stdx = zml.stdx;
const common = @import("../common.zig");

pub const Config = struct {
    architectures: []const []const u8 = &.{},
    model_type: []const u8,

    auto_map: ?AutoMap = null,

    rope_scaling: RopeScaling,
    yarn_only_types: []const []const u8 = &.{},

    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    max_seq_len: u32,
    vocab_size: u32,

    torch_dtype: []const u8 = "bfloat16",

    use_qk_norm: bool = false,

    moe_layers_enum: []const u8 = "",
    num_attention_heads: u32,
    num_attention_groups: u32,
    head_dim: u32,

    use_moe: bool = false,
    moe_num_experts: u32 = 0,
    moe_top_k: u32 = 0,
    moe_intermediate_size: u32 = 0,
    share_expert_dim: u32 = 0,
    moe_layer_offset: u32 = 0,
    moe_every_n_layer: u32 = 1,
    norm_expert_weight: bool = false,
    moe_router_activation: []const u8 = "sigmoid",
    moe_router_scaling_factor: f32 = 1.0,

    att_impl_type: []const u8 = "GQA",
    tie_word_embeddings: bool = false,

    rope_theta: []const f32,

    use_head_wise_attn_gate: bool = false,
    sliding_window: u32 = 0,

    use_moe_router_bias: bool = false,
    need_fp32_gate: bool = false,
    sink: bool = false,

    layer_types: []const []const u8 = &.{},
    use_rope_layers: []const u32 = &.{},

    num_nextn_predict_layers: u32 = 0,
    partial_rotary_factors: []const f32 = &.{},

    attention_other_setting: ?AttentionOtherSetting = null,

    swiglu_limits: []const f32 = &.{},
    swiglu_limits_shared: []const f32 = &.{},

    zero_centered: bool = false,
    max_position_embeddings: u32,

    pub const AutoMap = struct {
        AutoConfig: []const u8,
        AutoModelForCausalLM: []const u8,
    };

    pub const RopeScaling = struct {
        rope_type: []const u8,
        factor: f32,
        original_max_position_embeddings: u32,
        low_freq_factor: f32,
        high_freq_factor: f32,
    };

    pub const AttentionOtherSetting = struct {
        attention_type: []const u8,
        num_attention_heads: u32,
        num_attention_groups: u32,
        head_dim: u32,
        true_head_dim: u32,
    };

    pub fn numKeyValueHeads(self: Config) u32 {
        return self.num_attention_groups;
    }
};

// Options
pub const Options = struct {
    sampling_strategy: ?zml.nn.SamplingStrategy,
    max_seq_len: u32,
};

// LayerType

// Rope
// - parameters

// There are some partitioning functions re: KV Cache

// TextRotaryEmbedding

<<<<<<< HEAD
// Moe
=======
// Router
pub const Router = struct {
    gate: zml.nn.Linear, // no bias inside the Linear; router_bias is applied post-sigmoid
    router_bias: zml.Tensor,
    num_experts_per_tok: u32,
    routed_scaling_factor: f32,

    // k = num_experts_per_tok
    // `store` is the view at the parent `...moe` prefix; HF layout is:
    //   <prefix>.gate.weight
    //   <prefix>.router_bias   (optional)
    pub fn init(
        store: zml.io.TensorStore.View,
        num_experts_per_tok: u32,
        routed_scaling_factor: f32,
    ) Router {
        return .{
            .gate = .init(
                store.createTensor("gate.weight", .{ .expert, .d }, .{ .expert = .replicated, .d = .replicated }),
                null,
                .d,
            ),
            .router_bias = store.createTensor("router_bias", .{.expert}, .{ .expert = .replicated }),
            .num_experts_per_tok = num_experts_per_tok,
            .routed_scaling_factor = routed_scaling_factor,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Router)) void {
        self.gate.weight.deinit();
        self.router_bias.deinit();
    }

    /// Returns (topk weights, topk indices) with shapes {.b, .s, .topk}.
    /// Matches `Step3p5MoEMLP.router_bias_func`:
    ///   logits = x @ W^T
    ///   probs  = sigmoid(logits)        (in fp32)
    ///   biased = probs + router_bias    (only used to choose indices)
    ///   ids    = topk(biased)
    ///   wts    = gather(probs, ids)     (from UNBIASED probs)
    ///   if renormalize: wts /= sum(wts) + 1e-20
    pub fn forward(self: Router, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        // Match HF reference: always run gate matmul + sigmoid in fp32.
        const tagged = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });
        const input = tagged.convert(.f32);

        const gate_f32: zml.nn.Linear = .init(
            self.gate.weight.convert(.f32),
            null,
            .d,
        );

        // calculate raw logits
        const logits = gate_f32.forward(input);

        // calculate scores of how strongly a token matches an expert
        const expert_scores = logits.sigmoid();

        // add bias to influence how experts are distributed
        const expert_scores_with_bias = expert_scores.add(self.router_bias.convert(.f32).broad(expert_scores.shape()));

        // with N=288 total experts, pick top_k (k=8)
        const topk = expert_scores_with_bias.topK(.{ .topk = .expert }, self.num_experts_per_tok, .{});
        const topk_ids = topk.indices.convert(.i32);

        // Gather from UNBIASED probs — bias is only used to pick indices.
        const router_scores = expert_scores.gather(.{ .expert = topk_ids }, .{});

        // Renormalize so the topk weights sum to 1. Scaling by
        // routed_scaling_factor happens in the caller (Moe.forward), matching
        // the HF reference where `route()` returns unscaled weights.
        const denom = router_scores.sum(.topk).addConstant(1e-20);
        const normalized = router_scores.div(denom);

        return .{ normalized, topk_ids };
    }
};

// Moe
pub const Moe = struct {
    up_proj: zml.Tensor,
    gate_proj: zml.Tensor,
    down_proj: zml.Tensor,
    router: Router,

    pub fn init(store: zml.io.TensorStore.View) !Moe {
        // init the up, gate, down tensors

        const up_proj_tensor = store.createTensor(
            "up_proj.weight",
            .{ .expert, .dout, .d },
            .{ .expert = .replicated, .dout = .replicated, .d = .replicated },
        );

        const gate_proj_tensor = store.createTensor(
            "gate_proj.weight",
            .{ .expert, .dout, .d },
            .{ .expert = .replicated, .dout = .replicated, .d = .replicated },
        );

        const down_proj_tensor = store.createTensor(
            "down_proj.weight",
            .{ .expert, .dout, .d },
            .{ .expert = .replicated, .dout = .replicated, .d = .replicated },
        );

        return .{
            // swiglu limit temporarily hardcoded to 0
            // .shared_expert = .init(store.parent().withPrefix("share_expert"), 0),
            .up_proj = up_proj_tensor,
            .gate_proj = gate_proj_tensor,
            .down_proj = down_proj_tensor,
            .router = .init(store, 8, 3.0),
        };
    }

    pub fn deinit(self: *zml.Bufferized(Moe)) void {
        self.up_proj.deinit();
        self.gate_proj.deinit();
        self.down_proj.deinit();
    }

    pub fn forward(self: Moe, x: zml.Tensor) zml.Tensor {
        const input = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });

        // collect topk weights and indices (renormalized, unscaled)
        const routing_scores, const topk_ids = self.router.forward(input);

        // apply routed_scaling_factor here, matching reference's
        // `routing_weights = routing_weights * self.routed_scaling_factor`
        const scaled = routing_scores.scale(self.router.routed_scaling_factor);

        // Triton MoE backend expects the top-k axis tagged as `.top_expert`.
        const topk_ids_tensor = topk_ids.rename(.{ .topk = .top_expert });
        const scaled_tensor = scaled.rename(.{ .topk = .top_expert });

        // concat the gate and up
        const gate_up_proj = zml.Tensor.concatenate(&.{ self.gate_proj, self.up_proj }, .dout).rename(.{ .dout = .out, .d = .in });

        // we must rename down proj as well
        const down_proj = self.down_proj.rename(.{ .dout = .out, .d = .in });

        // hardcoded zml.moe.metadata, zml.moe.parameters
        const moe_metadata = zml.moe.Metadata.init(.{ .triton = .{} });
        const moe_parameters = zml.moe.Parameters.init(.{ .triton = .{ .num_experts_per_tok = self.router.num_experts_per_tok, .activation = .silu } });

        // get all expert outputs as tensor via fused triton kernel instead of Python loop
        // NOTE: swiglu limit not considered. may have to edit
        const moe_output = zml.moe.forwardMoe(
            input,
            topk_ids_tensor,
            scaled_tensor,
            gate_up_proj,
            null,
            null,
            down_proj,
            null,
            null,
            moe_metadata,
            moe_parameters,
        ) catch |err| stdx.debug.panic("moe backend failed: {}", .{err});

        return moe_output;
    }
};
// hidden size
// intermediate size
// router bias
// routed scaling factor

// gating
>>>>>>> c856b0f1 (examples/llm: clean up tests and model annotations)

// LoadedModel
pub const LoadedModel = struct {
    inner: Model,
    parsed_config: std.json.Parsed(Config),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
        generation: common.GenerationOptions,
    ) LoadedModel {
        _ = store; // autofix
        const parsed_config = try common.parseConfig(Config, allocator, io, repo);
        errdefer parsed_config.deinit();

        const options: Options = .{
            .sampling_strategy = generation.sampling_strategy,
            .max_seq_len = parsed_config.value.max_position_embeddings,
        };
        _ = options; // autofix

        return .{
            .inner = try .init(),
            .parsed_config = parsed_config,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        self.parsed_config.deinit();
    }

    pub fn loadBuffers(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: common.Shardings,
    ) !Buffers {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);
        var total_bytes: usize = 0;
        defer {
            const took = now.untilNow(io, .awake);
            const bytes_per_sec: u64 = @intFromFloat(
                @as(f64, @floatFromInt(total_bytes)) /
                    (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s),
            );
            std.log.scoped(.step3p5flash).info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });
        }

        return zml.io.load(Model, &self.inner, allocator, io, platform, store, .{
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = progress,
            .shardings = &shardings.all(),
            .parallelism = 16,
            .total_bytes = &total_bytes,
        });
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        _ = self;
        if (buffers.lm_head) |*lm_head| lm_head.weight.deinit();
        Step3p5Flash.unloadBuffers(&buffers.replicated, allocator);
    }

    // pub fn compile(
    //     self: *const LoadedModel,
    //     allocator: std.mem.Allocator,
    //     io: std.Io,
    //     platform: *const zml.Platform,
    //     backend: zml.attention.Backend,
    //     shardings: common.Shardings,
    //     seqlen: usize,
    //     progress: *std.Progress.Node,
    // ) !inference.CompiledModel {
    //     const params = inference.CompilationParameters.init(self.inner, self.parsed_config.value, @intCast(seqlen), backend, shardings);
    //     return inference.CompiledModel.init(allocator, io, platform, self, self.inner, params, progress);
    // }
};

// Buffers
pub const Buffers = zml.Bufferized(Model);

const Model = struct {
    lm_head: ?zml.nn.Linear,
    // model: Step3p5Flash,

    gen_opts: zml.nn.SamplingStrategy = .{},
    config: Config,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, options: Options) !Model {
        const lm_head: ?zml.nn.Linear = if (store.withPrefix("lm_head").maybeCreateTensor(
            "weight",
            .{ .dout, .d },
            .{ .dout = .replicated, .d = .replicated },
        )) |weight|
            .init(weight, null, .d)
        else
            null;

        return .{
            .lm_head = lm_head,
            .replicated = try .init(allocator, store.withPrefix("model"), config),
            .gen_opts = options.sampling_strategy orelse .{},
            .config = config,
        };
    }
};

// Step3p5Flash
const Step3p5Flash = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    norm: RmsNorm,
    layers: []TransformerLayer,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Step3p5Flash {
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = try .init(store.withPrefix("layers").withLayer(i), config);
        }

        return .{
            .embed_tokens = .{ .weight = store.createTensor(
                "embed_tokens.weight",
                .{ .voc, .d },
                .{ .voc = .replicated, .d = .replicated },
            ) },
            .norm = .{
                .weight = store.withPrefix("norm").createTensor("weight", .{.d}, .{ .d = .replicated }),
                .eps = config.rms_norm_eps,
            },
            .layers = layers,
        };
    }
};

// TransformerLayer
pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    // self_attn: SelfAttn,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,

    pub fn init(store: zml.io.TensorStore.View, config: Config) !TransformerLayer {
        return .{
            .input_layernorm = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps),
            .self_attn = try .init(store.withPrefix("self_attn"), config),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm")),
            .mlp = .init(store.withPrefix("mlp")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        // SelfAttn.unloadBuffers(&self.self_attn);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
        Mlp.unloadBuffers(&self.mlp);
    }

    pub fn forward(
        self: TransformerLayer,
        x0: zml.Tensor,
        token_index: zml.Tensor,
        // kv_cache: KvCache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { zml.Tensor } {
        // ) struct { zml.Tensor, KvCache } {
        // Self Attention
        //log.debug("TransformerLayer({f}) -> {f}", .{ x0, self.input_layernorm.forward(x0) });
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {f}", .{x0});

        // Keep the residual stream replicated to avoid repeated gathers before q/k/v.
        const x0_replicated = x0.withPartitioning(.{ .d = .replicated });
        const x0_normalized = self.input_layernorm.forward(x0_replicated);
        const delta0, const updated_kv_cache = self.self_attn.forward(
            x0_normalized,
            token_index,
            // kv_cache,
            attention_metadata,
            attention_parameters,
        );

        // Fully Connected
        const x1 = x0_replicated.add(delta0).withPartitioning(.{ .d = .replicated });
        const x1_normalized = self.post_attention_layernorm.forward(x1);
        const x2 = self.mlp.forward(x1_normalized)
            .rename(.{ .dout = .d })
            .withPartitioning(.{ .d = .replicated })
            .add(x1)
            .withPartitioning(.{ .d = .replicated });

        return .{ x2.reuseBuffer(x0), updated_kv_cache };
    }
};

// RmsNorm
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

    /// L2 normalization of input tensor along .d axis
    /// Step 3.5 Flash uses offset-style RMSNorm: the effective scale is `1 + weight`,
    /// not just `weight`.
    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const normalized = zml.nn.rmsNorm(x, .d, self.eps);
        const scale = self.weight.convert(.f32).addConstant(1).convert(x.dtype()).withTags(.{.d});
        std.log.warn("normalized={f} scale={f}", .{ normalized.shape(), scale.shape() });
        return normalized.mul(scale.broad(normalized.shape()));
    }
};

pub const Mlp = struct {
    up_proj: zml.nn.Linear, // (dim -> hidden_dim)
    gate_proj: zml.nn.Linear, // (dim -> hidden_dim)
    down_proj: zml.nn.Linear, // (hidden_dim -> dim)
    limit: ?i32,

    pub fn init(store: zml.io.TensorStore.View, swiglu_limit: ?i32) Mlp {
        return .{
            .up_proj = .init(
                store.createTensor("up_proj.weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }),
                null,
                .d,
            ),
            .gate_proj = .init(
                store.createTensor("gate_proj.weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }),
                null,
                .d,
            ),
            .down_proj = .init(
                store.createTensor("down_proj.weight", .{ .d, .dout }, .{ .d = .replicated, .dout = .replicated }),
                null,
                .dout,
            ),
            .limit = swiglu_limit,
        };
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        // Add tags to input before providing to our layer.
        const input = x.withTags(.{ .b, .s, .d });

        var up_proj = self.up_proj.forward(input);
        var gate = self.gate_proj.forward(input);
        gate = gate.silu();

        // Step 3.5 Flash clamps gate projection asymmetrically
        if (self.limit) |limit| {
            if (limit != 0) {
                const lim_f = @as(f32, @floatFromInt(limit));
                const max_t = zml.Tensor.scalar(lim_f, gate.dtype());
                const min_t = zml.Tensor.scalar(-lim_f, gate.dtype());

                // Step 3.5 Flash has asymmetric clamping of gate projection
                gate = gate.minimum(max_t);
                up_proj = up_proj.clamp(min_t, max_t);
            }
        }
        return self.down_proj.forward(gate.mul(up_proj));
    }
};

// SwAttn

// SelfAttn

// KvCache
