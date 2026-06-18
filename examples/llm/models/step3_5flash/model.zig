const std = @import("std");
const zml = @import("zml");
const stdx = zml.stdx;
const common = @import("../common.zig");
const inference = @import("inference.zig");

const log = std.log.scoped(.step3_5flash);

pub const Config = struct {
    architectures: []const []const u8 = &.{},
    model_type: []const u8,
    bos_token_id: ?u32 = null,
    eos_token_id: ?stdx.json.Union(union(enum) {
        int: u32,
        ints: []u32,
    }) = null,
    im_start_token_id: ?u32 = null,
    im_end_token_id: ?u32 = null,

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

    moe_layers_enum: MoeLayersEnum = .{},
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

    layer_types: []const AttnType = &.{},
    use_rope_layers: []const u32 = &.{},

    num_nextn_predict_layers: u32 = 0,
    partial_rotary_factors: []const f32 = &.{},

    attention_other_setting: ?AttentionOtherSetting = null,

    swiglu_limits: []const f32 = &.{},
    swiglu_limits_shared: []const f32 = &.{},

    zero_centered: bool = false,
    max_position_embeddings: u32 = 0,

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

    /// `moe_layers_enum` arrives either as a JSON array of u32 or as a comma-separated string.
    pub const MoeLayersEnum = struct {
        layers: []const u32 = &.{},

        pub fn jsonParse(
            allocator: std.mem.Allocator,
            source: anytype,
            options: std.json.ParseOptions,
        ) !MoeLayersEnum {
            if (try source.peekNextTokenType() == .array_begin) {
                return .{ .layers = try std.json.innerParse([]const u32, allocator, source, options) };
            }

            const s = switch (try source.nextAlloc(allocator, options.allocate.?)) {
                inline .string, .allocated_string => |v| v,
                else => return error.UnexpectedToken,
            };

            var list: std.ArrayList(u32) = .empty;
            errdefer list.deinit(allocator);
            var it = std.mem.tokenizeScalar(u8, s, ',');
            while (it.next()) |part| {
                const n = std.fmt.parseInt(u32, std.mem.trim(u8, part, " \t"), 10) catch return error.UnexpectedToken;
                try list.append(allocator, n);
            }
            return .{ .layers = try list.toOwnedSlice(allocator) };
        }
    };

    pub fn numKeyValueHeads(self: Config) u32 {
        return self.num_attention_groups;
    }

    pub fn numMainLayers(self: Config) u32 {
        return self.num_hidden_layers - self.num_nextn_predict_layers;
    }
};

pub const AttnType = enum {
    full_attention,
    sliding_attention,
};

pub const FfnType = enum {
    moe,
    mlp,
};

/// sharding - when .model mesh size exceeds the number of KV heads, replicate KV heads (by repeating along `.h`) such that each model-parallel shard owns at least one head.
fn kvHeadsAreReplicated(axis_size: i64, model_partitions: i64) struct { bool, ?u32 } {
    const is_replicated = axis_size > 0 and axis_size < model_partitions and @rem(model_partitions, axis_size) == 0;
    const repeat_factor = if (is_replicated) @as(u32, @intCast(@divExact(model_partitions, axis_size))) else null;
    return .{ is_replicated, repeat_factor };
}

fn partitionProjectedKv(tensor: zml.Tensor, replicate_kv_heads: bool, repeat_factor: ?u32) zml.Tensor {
    return if (replicate_kv_heads)
        tensor.repeat1d(.h, @as(u63, repeat_factor.?)).withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated })
    else
        tensor.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
}

fn partitionCachedKv(tensor: zml.Tensor, replicate_kv_heads: bool, repeat_factor: ?u32) zml.Tensor {
    return if (replicate_kv_heads)
        tensor.rename(.{ .s = .k }).repeat1d(.h, @as(u63, repeat_factor.?)).withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated })
    else
        tensor.rename(.{ .s = .k }).withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });
}

pub fn partitionKvCacheShape(kv_shape: zml.Shape, kv_heads: i64, model_partitions: i64) zml.Shape {
    const is_replicated, const repeat_factor = kvHeadsAreReplicated(kv_heads, model_partitions);

    if (is_replicated) {
        const repeated_h = kv_shape.dim(.h) * @as(i64, @intCast(repeat_factor.?));
        return kv_shape
            .setDim(.h, repeated_h)
            .withPartitioning(.{ .h = .model });
    }

    return kv_shape.withPartitioning(.{ .h = .model });
}

pub const Options = Model.GenOptions;

pub const Model = struct {
    pub const GenOptions = struct {
        sampling_strategy: zml.nn.SamplingStrategy = .{},
        max_seq_len: u32 = default_config.max_position_embeddings,
    };

    text_model: TextModel,
    lm_head: zml.nn.Linear,
    config: Config,
    gen_options: GenOptions,

    pub fn init(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: Config,
        gen_options: GenOptions,
        model_partitions: i64,
    ) !Model {
        return .{
            .text_model = try .init(allocator, store.withPrefix("model"), model_partitions),
            .lm_head = .init(
                store.createTensor("lm_head.weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                null,
                .d,
            ),
            .config = config,
            .gen_options = gen_options,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        self.text_model.deinit(allocator);
    }

    pub fn load(
        self: *const Model,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(Model) {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);
        var total_bytes: usize = 0;
        defer {
            const took = now.untilNow(io, .awake);
            const bytes_per_sec: u64 = @intFromFloat(
                @as(f64, @floatFromInt(total_bytes)) /
                    (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s),
            );
            log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });
        }
        return zml.io.load(Model, self, allocator, io, platform, store, .{
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = progress,
            .shardings = shardings,
            .parallelism = 16,
            .total_bytes = &total_bytes,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model), allocator: std.mem.Allocator) void {
        TextModel.unloadBuffers(&self.text_model, allocator);
        self.lm_head.weight.deinit();
    }

    pub fn forward(
        self: Model,
        tokens_: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        rng: zml.Tensor.Rng,
    ) struct { zml.Tensor, KvCache, zml.Tensor.Rng } {
        const tokens = tokens_.withPartialTags(.{.s});
        const hidden, const new_kv_cache = self.text_model.forward(tokens, token_index, kv_cache);
        const next_tokens, const new_rng = self.sampleTokens(hidden, rng);
        return .{ next_tokens.convert(tokens.dtype()).reuseBuffer(tokens), new_kv_cache, new_rng };
    }

    pub fn sampleTokens(self: Model, hidden: zml.Tensor, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        const next_tokens, const new_rng, _ = self.sampler().sampleTokens(hidden, rng, null);
        return .{ next_tokens, new_rng };
    }

    pub fn sampler(self: Model) Sampler {
        return .{
            .norm = self.text_model.norm,
            .lm_head = self.lm_head,
            .gen_options = self.gen_options,
        };
    }
};

pub const Sampler = struct {
    norm: RmsNorm,
    lm_head: zml.nn.Linear,
    gen_options: Model.GenOptions,

    pub fn sampleTokens(
        self: Sampler,
        hidden: zml.Tensor,
        rng: zml.Tensor.Rng,
        token_index: ?zml.Tensor,
    ) struct { zml.Tensor, zml.Tensor.Rng, ?zml.Tensor } {
        const x = self.norm.forward(hidden, .d);
        const logits = self.lm_head.forward(x.withPartialTags(.{.d})).rename(.{ .dout = .voc });
        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, self.gen_options.sampling_strategy, rng);
        if (token_index) |idx| {
            return .{ next_tokens.convert(.u32), new_rng, idx.addConstant(1) };
        }
        return .{ next_tokens.convert(.u32), new_rng, null };
    }
};

// The equivalent of a struct called "Step3p5Flash"
pub const TextModel = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    norm: RmsNorm,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, model_partitions: i64) !TextModel {
        const num_main_layers = default_config.numMainLayers();
        const layers = try allocator.alloc(TransformerLayer, @intCast(num_main_layers));
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = try .init(store.withPrefix("layers").withLayer(i), i, model_partitions);
        }

        return .{
            .embed_tokens = .{
                .weight = store.createTensor(
                    "embed_tokens.weight",
                    .{ .voc, .d },
                    .{ .voc = .replicated, .d = .model },
                ),
            },
            .layers = layers,
            .norm = .init(store.withPrefix("norm"), 1e-5),
        };
    }

    pub fn deinit(self: TextModel, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TextModel), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| TransformerLayer.unloadBuffers(layer);
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
    }

    pub fn forward(
        self: TextModel,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { zml.Tensor, KvCache } {
        var hidden = self.embed_tokens.forward(tokens);

        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden, updated_kv_cache = layer.forward(
                layer,
                hidden,
                token_index,
                updated_kv_cache.atLayer(i),
                attention_metadata,
                attention_parameters,
            );
        }

        return .{ hidden, updated_kv_cache.reuseBuffer(kv_cache) };
    }
};

pub const Ffn = union(enum) {
    /// Layers [0, 1, 2] are a single SwiGLU MLP
    mlp: Mlp,
    /// Layers 3 - 45 are MoE layers, consisting of routed experts + a per-token shared expert. Remaining layers are MTP
    moe: struct {
        experts: Moe,
        shared: Mlp,
    },

    pub fn forward(self: Ffn, x: zml.Tensor) zml.Tensor {
        return switch (self) {
            .mlp => |mlp| mlp.forward(x),
            .moe => |m| m.experts.forward(x).add(m.shared.forward(x)),
        };
    }
};

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    attn: Attn,
    ffn: Ffn,
    post_attention_layernorm: RmsNorm,

    pub fn init(store: zml.io.TensorStore.View, layer_idx: usize, model_partitions: i64) !TransformerLayer {
        const is_moe = std.mem.indexOfScalar(u32, default_config.moe_layers_enum.layers, @intCast(layer_idx)) != null;

        const shared_limit: f32 = if (layer_idx < default_config.swiglu_limits_shared.len)
            default_config.swiglu_limits_shared[layer_idx]
        else
            0.0;

        const ffn: Ffn = if (is_moe) .{
            .moe = .{
                .experts = try .init(store.withPrefix("moe"), layer_idx),
                .shared = .init(store.withPrefix("share_expert"), shared_limit),
            },
        } else .{
            .mlp = .init(store.withPrefix("mlp"), shared_limit),
        };

        return .{
            .input_layernorm = .init(store.withPrefix("input_layernorm"), 1e-5),
            .attn = try .init(store.withPrefix("self_attn"), layer_idx, model_partitions),
            .ffn = ffn,
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), 1e-5),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        Attn.unloadBuffers(&self.attn);
        switch (self.ffn) {
            .mlp => |*mlp| {
                mlp.up_proj.weight.deinit();
                mlp.gate_proj.weight.deinit();
                mlp.down_proj.weight.deinit();
            },
            .moe => |*m| {
                Moe.deinit(&m.experts);
                Router.unloadBuffers(&m.experts.router);
                m.shared.up_proj.weight.deinit();
                m.shared.gate_proj.weight.deinit();
                m.shared.down_proj.weight.deinit();
            },
        }
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
    }

    pub fn forward(
        self: TransformerLayer,
        x0: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { zml.Tensor, KvCache } {
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {f}", .{x0});

        // Attention block.
        const attn_input = self.input_layernorm.forward(x0, .d);
        const attn_delta, const updated_kv_cache = self.attn.forward(
            attn_input,
            token_index,
            kv_cache,
            attention_metadata,
            attention_parameters,
        );
        const x1 = x0.add(attn_delta);

        // FFN block
        const ffn_input = self.post_attention_layernorm.forward(x1, .d);
        const ffn_delta = self.ffn.forward(ffn_input);

        return .{ x1.add(ffn_delta).reuseBuffer(x0), updated_kv_cache };
    }
};

/// Temporary deep copy of `config.json` for Step 3.5 Flash. TODO: run config through Model struct and remove this deep copy
pub const default_config: Config = .{
    .architectures = &.{"Step3p5ForCausalLM"},
    .model_type = "step3p5",
    .auto_map = .{
        .AutoConfig = "configuration_step3p5.Step3p5Config",
        .AutoModelForCausalLM = "modeling_step3p5.Step3p5ForCausalLM",
    },
    .rope_scaling = .{
        .rope_type = "llama3",
        .factor = 2.0,
        .original_max_position_embeddings = 131072,
        .low_freq_factor = 1.0,
        .high_freq_factor = 32.0,
    },
    .yarn_only_types = &.{"full_attention"},
    .hidden_size = 4096,
    .intermediate_size = 11264,
    .num_hidden_layers = 48,
    .max_seq_len = 262144,
    .vocab_size = 128896,
    .torch_dtype = "bfloat16",
    .use_qk_norm = true,
    .moe_layers_enum = .{ .layers = &.{
        3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44,
    } },
    .num_attention_heads = 64,
    .num_attention_groups = 8,
    .head_dim = 128,
    .use_moe = true,
    .moe_num_experts = 288,
    .moe_top_k = 8,
    .moe_intermediate_size = 1280,
    .share_expert_dim = 1280,
    .moe_layer_offset = 0,
    .moe_every_n_layer = 1,
    .norm_expert_weight = true,
    .moe_router_activation = "sigmoid",
    .moe_router_scaling_factor = 3.0,
    .att_impl_type = "GQA",
    .tie_word_embeddings = false,
    .rope_theta = &.{
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
    },
    .use_head_wise_attn_gate = true,
    .sliding_window = 512,
    .use_moe_router_bias = true,
    .need_fp32_gate = true,
    .sink = false,
    .layer_types = &.{
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
    },
    .use_rope_layers = &.{},
    .num_nextn_predict_layers = 3,
    .partial_rotary_factors = &.{
        0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,
        0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,
        0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,
        0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,
        0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,
        0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,
    },
    .attention_other_setting = .{
        .attention_type = "sliding_attention",
        .num_attention_heads = 96,
        .num_attention_groups = 8,
        .head_dim = 128,
        .true_head_dim = 128,
    },
    .swiglu_limits = &.{
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 7.0, 7.0, 0.0, 0.0, 0.0,
    },
    .swiglu_limits_shared = &.{
        0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0,
    },
    .zero_centered = true,
    .max_position_embeddings = 262144,
};

pub const Attn = struct {
    layer_idx: usize,
    enable_sliding_window: bool,
    model_partitions: i64,

    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,
    g_proj: zml.nn.Linear,

    q_norm: RmsNorm,
    k_norm: RmsNorm,

    q_size: i64,
    kv_size: i64,

    head_dim: i64,
    scaling: f32,

    num_q_heads: i64,
    num_kv_heads: i64,
    num_kv_groups: i64,
    rotary_dim: i64,
    rotary_emb: TextRotaryEmbedding,

    fn initProj(store: zml.io.TensorStore.View, partitions: anytype, bias_partitions: anytype) zml.nn.Linear {
        return .init(
            store.createTensor("weight", .{ .dout, .d }, partitions),
            store.maybeCreateTensor("bias", .{.dout}, bias_partitions),
            .d,
        );
    }

    pub fn init(store: zml.io.TensorStore.View, layer_idx: usize, model_partitions: i64) !Attn {
        // Layers past the configured count are MTP/speculative blocks and use the SWA shape.
        const kind: AttnType = default_config.layer_types[layer_idx];

        const num_q_heads: i64 = @intCast(if (kind == .full_attention)
            default_config.num_attention_heads
        else
            (default_config.attention_other_setting orelse unreachable).num_attention_heads);

        const num_kv_heads: i64 = @intCast(default_config.num_attention_groups);
        const head_dim: i64 = @intCast(default_config.head_dim);

        const rotary_idx = @min(layer_idx, default_config.partial_rotary_factors.len - 1);
        const rotary_dim: i64 = @intFromFloat(
            default_config.partial_rotary_factors[rotary_idx] * @as(f32, @floatFromInt(default_config.head_dim)),
        );

        const rope_idx = @min(layer_idx, default_config.rope_theta.len - 1);
        const layer_theta = default_config.rope_theta[rope_idx];

        // HF patched modeling: a layer gets the scaled rope ONLY if its layer_types[idx] is listed in `yarn_only_types`
        const layer_type_name: []const u8 = switch (kind) {
            .full_attention => "full_attention",
            .sliding_attention => "sliding_attention",
        };
        const apply_yarn = blk: {
            for (default_config.yarn_only_types) |t| {
                if (std.mem.eql(u8, t, layer_type_name)) break :blk true;
            }
            break :blk false;
        };
        const rs = default_config.rope_scaling;
        const rope_scaling: zml.nn.RopeOpts.Scaling = if (apply_yarn) .{ .llama3 = .{
            .factor = rs.factor,
            .high_freq_factor = rs.high_freq_factor,
            .low_freq_factor = rs.low_freq_factor,
            .original_max_position_embeddings = rs.original_max_position_embeddings,
            .rope_theta = layer_theta,
        } } else .{ .default = .{ .rope_theta = layer_theta } };

        return .{
            .layer_idx = layer_idx,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .enable_sliding_window = !(kind == .full_attention),
            .model_partitions = model_partitions,
            .head_dim = head_dim,
            .num_kv_groups = @divExact(num_q_heads, num_kv_heads),
            .rotary_dim = rotary_dim,
            .rotary_emb = .init(rotary_dim, rope_scaling, 1.0),
            .q_size = num_q_heads * head_dim,
            .kv_size = num_kv_heads * head_dim,
            .scaling = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            .q_proj = initProj(store.withPrefix("q_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .k_proj = initProj(store.withPrefix("k_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .v_proj = initProj(store.withPrefix("v_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .o_proj = initProj(store.withPrefix("o_proj"), .{ .dout = .replicated, .d = .model }, .{ .dout = .replicated }),
            .g_proj = initProj(store.withPrefix("g_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            // Step 3.5 doesn't expose rms_norm_eps in its config; HF reference uses 1e-5.
            .q_norm = .init(store.withPrefix("q_norm"), 1e-5),
            .k_norm = .init(store.withPrefix("k_norm"), 1e-5),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Attn)) void {
        self.q_proj.weight.deinit();
        if (self.q_proj.bias) |*b| b.deinit();
        self.k_proj.weight.deinit();
        if (self.k_proj.bias) |*b| b.deinit();
        self.v_proj.weight.deinit();
        if (self.v_proj.bias) |*b| b.deinit();
        self.o_proj.weight.deinit();
        if (self.o_proj.bias) |*b| b.deinit();
        self.g_proj.weight.deinit();
        if (self.g_proj.bias) |*b| b.deinit();
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    fn projectQAndGate(self: Attn, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const q = self.q_proj.forward(x)
            .splitAxis(.dout, .{ .h = self.num_q_heads, .hd = self.head_dim });
        const gate = self.g_proj.forward(x).rename(.{ .dout = .h });

        return .{ q, gate };
    }

    fn projectKV(self: Attn, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const k = self.k_proj.forward(x)
            .splitAxis(.dout, .{
            .h = self.num_kv_heads,
            .hd = self.head_dim,
        });

        const v = self.v_proj.forward(x)
            .splitAxis(.dout, .{
            .h = self.num_kv_heads,
            .hd = self.head_dim,
        });

        return .{ k, v };
    }

    pub fn forward(
        self: Attn,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { zml.Tensor, KvCache } {
        const input_raw = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });
        const input = input_raw.withPartitioning(.{ .d = .replicated });

        const replicate_kv_heads, const repeat_factor = kvHeadsAreReplicated(self.num_kv_heads, self.model_partitions);

        var q, var gate = self.projectQAndGate(input);
        var k, var v = self.projectKV(input);

        q = q.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        gate = gate.withPartitioning(.{ .s = .replicated, .h = .model });
        k = partitionProjectedKv(k, replicate_kv_heads, repeat_factor);
        v = partitionProjectedKv(v, replicate_kv_heads, repeat_factor);

        q = self.q_norm.forward(q, .hd);
        k = self.k_norm.forward(k, .hd);

        const dtype = q.dtype();

        // Position ids must be absolute and not relative to current chunk. token_index is the cache_position vector; its first element is the absolute start, and positions = arange(S) + start.
        const position_ids = blk: {
            const base = zml.Tensor.arange(.{ .end = input.dim(.s) }, .i64).withTags(.{.s});
            const start = token_index.slice1d(0, .{ .start = 0, .end = 1 }).squeeze(0).convert(.i64);
            const offset = start.broad(base.shape());
            break :blk base.add(offset);
        };

        const cos, const sin = self.rotary_emb.getCosAndSin(position_ids, dtype);

        q = self.rotary_emb.applyRope(q, cos, sin)
            .withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        k = partitionProjectedKv(self.rotary_emb.applyRope(k, cos, sin), replicate_kv_heads, repeat_factor);

        const cache_start = token_index.convert(.u32).slice1d(0, .{ .start = 0, .end = 1 }).squeeze(0);
        const new_kv_cache = kv_cache.update(k, v, cache_start);
        const k_full = new_kv_cache.keys().convert(dtype)
            .withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });
        const v_full = new_kv_cache.values().convert(dtype)
            .withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });

        q = q.rename(.{ .s = .q }).withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated });

        // Take first element of cache_positions array to match zml.attention.attention signature
        const attn_start = token_index.slice1d(0, .{ .start = 0, .end = 1 });

        const attn_output = zml.attention.attention.attention(
            q,
            k_full,
            v_full,
            attn_start,
            attention_metadata,
            attention_parameters,
        ).withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated });

        // Head-wise gate is {b, s, h}
        const gate_b = gate.sigmoid().rename(.{ .s = .q }).broad(attn_output.shape());
        const gated_attn = attn_output.mul(gate_b);

        const projected_output = self.o_proj.forward(
            gated_attn.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s }),
        ).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });

        return .{ projected_output, new_kv_cache };
    }

    // TODO: candidate for removal
    pub const Stages = struct {
        q_proj: zml.Tensor,
        k_proj: zml.Tensor,
        v_proj: zml.Tensor,
        g_proj: zml.Tensor,
        q_norm: zml.Tensor,
        k_norm: zml.Tensor,
        q_pre_rope_hf: zml.Tensor,
        k_pre_rope_hf: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
        q_rope_hf: zml.Tensor,
        k_rope_hf: zml.Tensor,
        attn: zml.Tensor,
        gate_sig: zml.Tensor,
        gated: zml.Tensor,
        o_proj_in: zml.Tensor,
        out: zml.Tensor,
    };

    pub fn forwardTemp(
        self: Attn,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
    ) struct { Stages, KvCache } {
        const input = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });

        const q_proj_raw = self.q_proj.forward(input);
        const k_proj_raw = self.k_proj.forward(input);
        const v_proj_raw = self.v_proj.forward(input);
        const g_proj_raw = self.g_proj.forward(input);

        var q = q_proj_raw.splitAxis(.dout, .{ .h = self.num_q_heads, .hd = self.head_dim });
        var k = k_proj_raw.splitAxis(.dout, .{ .h = self.num_kv_heads, .hd = self.head_dim });
        const v = v_proj_raw.splitAxis(.dout, .{ .h = self.num_kv_heads, .hd = self.head_dim });
        const gate = g_proj_raw.rename(.{ .dout = .h });

        const q_after_norm = self.q_norm.forward(q, .hd);
        const k_after_norm = self.k_norm.forward(k, .hd);
        q = q_after_norm;
        k = k_after_norm;

        const q_pre_rope_hf = q.transpose(.{ .b, .h, .s, .hd });
        const k_pre_rope_hf = k.transpose(.{ .b, .h, .s, .hd });

        const dtype = q.dtype();
        const position_ids = blk: {
            const base = zml.Tensor.arange(.{ .end = input.dim(.s) }, .i64).withTags(.{.s});
            const start = token_index.slice1d(0, .{ .start = 0, .end = 1 }).squeeze(0).convert(.i64);
            const offset = start.broad(base.shape());
            break :blk base.add(offset);
        };
        const cos_raw, const sin_raw = self.rotary_emb.getCosAndSin(position_ids, dtype);
        const cos = cos_raw.insertAxes(0, .{.b});
        const sin = sin_raw.insertAxes(0, .{.b});

        const q_rope = self.rotary_emb.applyRope(q, cos_raw, sin_raw);
        const k_rope = self.rotary_emb.applyRope(k, cos_raw, sin_raw);

        const q_rope_hf = q_rope.transpose(.{ .b, .h, .s, .hd });
        const k_rope_hf = k_rope.transpose(.{ .b, .h, .s, .hd });

        const new_kv_cache = blk: {
            const cache_start = token_index.convert(.u32).slice1d(0, .{ .start = 0, .end = 1 }).squeeze(0);
            break :blk kv_cache.update(k_rope, v, cache_start);
        };
        const k_full = new_kv_cache.keys().convert(dtype);
        const v_full = new_kv_cache.values().convert(dtype);

        const q_for_attn = q_rope.rename(.{ .s = .q });

        const attn_start = token_index.slice1d(0, .{ .start = 0, .end = 1 });
        const attn_output = zml.attention.attention.attention(
            q_for_attn,
            k_full,
            v_full,
            attn_start,
            zml.attention.attention.Metadata.init(.fromBackend(.cuda_fa2, input.dim(.s), self.num_q_heads)),
            zml.attention.attention.Parameters.init(.fromBackend(.cuda_fa2)),
        );

        const gate_sig = gate.sigmoid().rename(.{ .s = .q });
        const gated = attn_output.mul(gate_sig.broad(attn_output.shape()));

        const o_proj_in = gated.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const out = self.o_proj.forward(o_proj_in);

        return .{
            .{
                .q_proj = q_proj_raw,
                .k_proj = k_proj_raw,
                .v_proj = v_proj_raw,
                .g_proj = g_proj_raw,
                .q_norm = q_after_norm,
                .k_norm = k_after_norm,
                .q_pre_rope_hf = q_pre_rope_hf,
                .k_pre_rope_hf = k_pre_rope_hf,
                .cos = cos,
                .sin = sin,
                .q_rope_hf = q_rope_hf,
                .k_rope_hf = k_rope_hf,
                .attn = attn_output,
                .gate_sig = gate_sig,
                .gated = gated,
                .o_proj_in = o_proj_in,
                .out = out,
            },
            new_kv_cache,
        };
    }
};

pub const TextRotaryEmbedding = struct {
    rotary_dim: i64,
    scaling: zml.nn.RopeOpts.Scaling,
    attention_scaling: f32 = 1.0,

    pub fn init(rotary_dimension: i64, scaling: zml.nn.RopeOpts.Scaling, attention_scaling: f32) TextRotaryEmbedding {
        return .{
            .rotary_dim = rotary_dimension,
            .scaling = scaling,
            .attention_scaling = attention_scaling,
        };
    }

    /// Expects position_ids to be tagged {.b, .s} and returns cos and sin tagged {.b, .s, .hd}
    pub fn getCosAndSin(self: TextRotaryEmbedding, position_ids: zml.Tensor, dtype: zml.DataType) struct { zml.Tensor, zml.Tensor } {
        const opts: zml.nn.RopeOpts = .{ .scaling = self.scaling };
        const inv_freq = zml.nn.invFreq(self.rotary_dim, opts).withTags(.{.hd});

        const freqs_t = position_ids.convert(.f32).outer(inv_freq);
        const emb = zml.Tensor.concatenate(&.{ freqs_t, freqs_t }, .hd);

        const cos = emb.cos().scale(self.attention_scaling).convert(dtype);
        const sin = emb.sin().scale(self.attention_scaling).convert(dtype);
        return .{ cos, sin };
    }

    pub fn rotateHalf(x: zml.Tensor) zml.Tensor {
        const half_dim = @divExact(x.dim(.hd), 2);
        const x1 = x.slice1d(.hd, .{ .start = 0, .end = half_dim });
        const x2 = x.slice1d(.hd, .{ .start = half_dim, .end = x.dim(.hd) });
        return zml.Tensor.concatenate(&.{ x2.negate(), x1 }, .hd);
    }

    /// Expects q and k to be tagged {.b, .s, .h, .hd} and cos and sin to be tagged {.b, .s, .hd}
    pub fn applyRope(
        self: TextRotaryEmbedding,
        x: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    ) zml.Tensor {
        const x_rot = x.slice1d(.hd, .{ .start = 0, .end = self.rotary_dim });

        // Insert head axis so cos/sin broadcast over .h.
        const cos_b = cos.insertAxes(.hd, .{.h}).broad(x_rot.shape());
        const sin_b = sin.insertAxes(.hd, .{.h}).broad(x_rot.shape());

        const x_rotated = x_rot.mul(cos_b).add(rotateHalf(x_rot).mul(sin_b));

        if (self.rotary_dim == x.dim(.hd)) {
            return x_rotated;
        }

        const x_pass = x.slice1d(.hd, .{ .start = self.rotary_dim, .end = x.dim(.hd) });

        return zml.Tensor.concatenate(&.{ x_rotated, x_pass }, .hd);
    }
};

pub const KvCache = struct {
    k: zml.Tensor,
    v: zml.Tensor,
    layer_index: zml.Tensor,

    pub const Buffer = zml.Bufferized(KvCache);

    pub fn init(kv_shape: zml.Shape) KvCache {
        return .{
            .k = .fromShape(kv_shape),
            .v = .fromShape(kv_shape),
            .layer_index = .init(.{}, .u32),
        };
    }

    /// Zero-initialized so that masked-out positions in the attention cannot hold NaN bit patterns.
    pub fn initBuffer(kv: KvCache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !Buffer {
        const k_bytes = try allocator.alloc(u8, kv.k.shape().byteSize());
        defer allocator.free(k_bytes);
        @memset(k_bytes, 0);
        const v_bytes = try allocator.alloc(u8, kv.v.shape().byteSize());
        defer allocator.free(v_bytes);
        @memset(v_bytes, 0);
        return .{
            .k = try zml.Buffer.fromBytes(io, platform, kv.k.shape(), sharding, k_bytes),
            .v = try zml.Buffer.fromBytes(io, platform, kv.v.shape(), sharding, v_bytes),
            .layer_index = try zml.Buffer.scalar(io, platform, @as(u32, 0), .u32),
        };
    }

    pub fn deinitBuffer(kv: *Buffer) void {
        kv.k.deinit();
        kv.v.deinit();
        kv.layer_index.deinit();
    }

    pub fn keys(kv: KvCache) zml.Tensor {
        return kv.k.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = kv.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn values(kv: KvCache) zml.Tensor {
        return kv.v.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = kv.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn update(kv: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: ?zml.Tensor) KvCache {
        const k_shape = kv.k.shape().drop(.layer);
        var layer = kv.layer_index;

        // KV cache uses .k instead of .s, so we change here so caller doesn't need to be aware of this naming scheme.
        const k_renamed = if (new_k.shape().hasTag(.s) != null) new_k.rename(.{ .s = .k }) else new_k;
        const v_renamed = if (new_v.shape().hasTag(.s) != null) new_v.rename(.{ .s = .k }) else new_v;
        const k_in = k_renamed.convert(kv.k.dtype()).transpose(k_shape);
        const v_in = v_renamed.convert(kv.v.dtype()).transpose(k_shape);

        return if (token_index) |idx| blk: {
            layer = layer.broad(idx.shape());
            break :blk .{
                .k = kv.k.scatterSlices(.{ .layer = layer, .k = idx }, k_in, .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.k),
                .v = kv.v.scatterSlices(.{ .layer = layer, .k = idx }, v_in, .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.v),
                .layer_index = kv.layer_index,
            };
        } else .{
            .k = kv.k.scatterSlices(.{ .layer = layer }, k_in, .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.k),
            .v = kv.v.scatterSlices(.{ .layer = layer }, v_in, .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.v),
            .layer_index = kv.layer_index,
        };
    }

    pub fn atLayer(kv: KvCache, layer_index: usize) KvCache {
        return .{
            .k = kv.k,
            .v = kv.v,
            .layer_index = .scalar(layer_index, .u32),
        };
    }

    pub fn reuseBuffer(kv: KvCache, other: KvCache) KvCache {
        return .{
            .k = kv.k.reuseBuffer(other.k),
            .v = kv.v.reuseBuffer(other.v),
            .layer_index = kv.layer_index.reuseBuffer(other.layer_index),
        };
    }
};

pub const Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,
    limit: f32,

    pub fn init(store: zml.io.TensorStore.View, swiglu_limit: f32) Mlp {
        return .{
            .up_proj = .init(
                store.createTensor("up_proj.weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                null,
                .d,
            ),
            .gate_proj = .init(
                store.createTensor("gate_proj.weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                null,
                .d,
            ),
            .down_proj = .init(
                store.createTensor("down_proj.weight", .{ .d, .dout }, .{ .d = .replicated, .dout = .model }),
                null,
                .dout,
            ),
            .limit = swiglu_limit,
        };
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const input = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });

        var up_proj = self.up_proj.forward(input);
        var gate = self.gate_proj.forward(input);
        gate = gate.silu();

        // Step 3.5 Flash clamps gate projection asymmetrically.
        if (self.limit != 0) {
            const max_t = zml.Tensor.scalar(self.limit, gate.dtype());
            const min_t = zml.Tensor.scalar(-self.limit, gate.dtype());
            gate = gate.minimum(max_t);
            up_proj = up_proj.clamp(min_t, max_t);
        }
        return self.down_proj.forward(gate.mul(up_proj));
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

    /// L2 normalization of input tensor along the given axis
    pub fn forward(self: RmsNorm, input: zml.Tensor, comptime axis: anytype) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{axis});
        const ax = x.axis(axis);
        const xf32 = x.convert(.f32);
        const variance = xf32.mul(xf32).mean(ax);
        const rsqrt = zml.Tensor.rsqrt(variance.addConstant(self.eps));
        const normalized_f32 = xf32.mul(rsqrt.broad(xf32.shape()));
        const scale_f32 = self.weight.convert(.f32).addConstant(1).withTags(.{axis});
        return normalized_f32.mul(scale_f32.broad(normalized_f32.shape())).convert(x.dtype());
    }
};

pub const Router = struct {
    gate: zml.nn.Linear, // no bias inside the Linear; router_bias is applied post-sigmoid
    router_bias: zml.Tensor,
    num_experts_per_tok: u32,
    routed_scaling_factor: f32,

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
    pub fn forward(self: Router, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const tagged = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });
        const input = tagged.convert(.f32);

        const gate_f32: zml.nn.Linear = .init(
            self.gate.weight.convert(.f32),
            null,
            .d,
        );

        const logits = gate_f32.forward(input);

        const expert_scores = logits.sigmoid();
        const expert_scores_with_bias = expert_scores.add(self.router_bias.convert(.f32).broad(expert_scores.shape()));

        // with N=288 total experts, pick top_k (k=8)
        const topk = expert_scores_with_bias.topK(.{ .topk = .expert }, self.num_experts_per_tok, .{});
        const topk_ids = topk.indices.convert(.i32);

        const router_scores = expert_scores.gather(.{ .expert = topk_ids }, .{});

        const denom = router_scores.sum(.topk).addConstant(1e-20);
        const normalized = router_scores.div(denom);

        return .{ normalized, topk_ids };
    }
};

pub const Moe = struct {
    gate_up_proj: zml.Tensor,
    down_proj: zml.Tensor,
    router: Router,
    layer_idx: usize,
    limit: ?f32,

    pub fn init(store: zml.io.TensorStore.View, layer_idx: usize) !Moe {
        const gate_up_proj_tensor = store.createTensor(
            "gate_up_proj.weight",
            .{ .expert, .dout, .d },
            .{ .expert = .experts, .dout = .replicated, .d = .replicated },
        );

        const down_proj_tensor = store.createTensor(
            "down_proj.weight",
            .{ .expert, .dout, .d },
            .{ .expert = .experts, .dout = .replicated, .d = .replicated },
        );

        const limit: ?f32 = blk: {
            if (layer_idx >= default_config.swiglu_limits.len) break :blk null;
            const v = default_config.swiglu_limits[layer_idx];
            break :blk if (v == 0) null else v;
        };

        return .{
            .gate_up_proj = gate_up_proj_tensor,
            .down_proj = down_proj_tensor,
            .router = .init(store, default_config.moe_top_k, default_config.moe_router_scaling_factor),
            .layer_idx = layer_idx,
            .limit = limit,
        };
    }

    pub fn deinit(self: *zml.Bufferized(Moe)) void {
        self.gate_up_proj.deinit();
        self.down_proj.deinit();
    }

    pub fn forward(self: Moe, x: zml.Tensor) zml.Tensor {
        if (self.layer_idx >= 43 and self.layer_idx <= 44) {
            return self.forwardLoop(x);
        }
        return self.forwardTriton(x);
    }

    fn forwardTriton(self: Moe, x: zml.Tensor) zml.Tensor {
        const input = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });
        // collect topk weights and indices (renormalized, unscaled)
        const routing_scores, const topk_ids = self.router.forward(input);
        const scaled = routing_scores.scale(self.router.routed_scaling_factor);

        const topk_ids_tensor = topk_ids.rename(.{ .topk = .top_expert });
        const scaled_tensor = scaled.rename(.{ .topk = .top_expert });

        const gate_up_proj = self.gate_up_proj.rename(.{ .dout = .out, .d = .in });
        const down_proj = self.down_proj.rename(.{ .dout = .out, .d = .in });

        // TODO: hardcoded zml.moe.metadata, zml.moe.parameters
        const moe_metadata = zml.moe.Metadata.init(.{ .triton = .{} });
        const moe_parameters = zml.moe.Parameters.init(.{ .triton = .{ .num_experts_per_tok = self.router.num_experts_per_tok, .activation = .silu } });

        // NOTE: swiglu limit not considered. Must edit
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

        // zml.moe.forwardMoe returns shape {.b, .s, .d}
        return moe_output;
    }

    fn forwardLoop(self: Moe, x: zml.Tensor) zml.Tensor {
        const input = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });

        const routing_scores, const topk_ids = self.router.forward(input);
        const routing_scaled = routing_scores.scale(self.router.routed_scaling_factor);

        const x_f32 = input.convert(.f32);
        const gate_up_w = self.gate_up_proj.convert(.f32);
        const mid = @divFloor(gate_up_w.dim(.dout), 2);
        const gate_w = gate_up_w.slice1d(.dout, .{ .end = mid });
        const up_w = gate_up_w.slice1d(.dout, .{ .start = mid });
        const down_w = self.down_proj.convert(.f32);

        const gate_all = x_f32.dot(gate_w, .d);
        const up_all = x_f32.dot(up_w, .d);

        var gate_act = gate_all.silu();
        var up_act = up_all;
        if (self.limit) |lim| {
            gate_act = gate_act.minimum(.scalar(lim, .f32));
            up_act = up_act.clamp(.scalar(-lim, .f32), .scalar(lim, .f32));
        }
        const act_d = gate_act.mul(up_act).rename(.{ .dout = .d });
        const down_all = act_d.dot(down_w, .d).transpose(.{ .b, .s, .expert, .dout });
        const down_topk = down_all.gather(.{ .expert = topk_ids }, .{});
        const routing = routing_scaled.convert(.f32).broad(down_topk.shape());
        const weighted = down_topk.mul(routing).convert(x.dtype());
        const sort_order = topk_ids.argsort(.topk, .{ .descending = false })
            .rename(.{ .topk = .topk_sorted });
        const weighted_sorted = weighted
            .gather(.{ .topk = sort_order }, .{})
            .rename(.{ .topk_sorted = .topk });

        var acc = weighted_sorted.slice1d(.topk, .single(0));
        inline for (1..8) |i| {
            const contrib = weighted_sorted.slice1d(.topk, .single(@as(i64, @intCast(i))));
            acc = acc.add(contrib);
        }
        return acc.rename(.{ .dout = .d });
    }
};

pub const LoadedModel = struct {
    pub const SpecialTokens = struct {
        im_start_token_id: u32,
        im_end_token_id: u32,
        eos_token_id: u32,
    };

    inner: Model,
    parsed_config: std.json.Parsed(Config),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
        generation: common.GenerationOptions,
        shardings: common.Shardings,
    ) !LoadedModel {
        const parsed_config = try common.parseConfig(Config, allocator, io, repo);
        errdefer parsed_config.deinit();

        const options: Options = .{
            .sampling_strategy = generation.sampling_strategy,
            .max_seq_len = parsed_config.value.max_position_embeddings,
        };

        const model_partitions = shardings.model.numPartitionsForLogicalAxis(.model);

        return .{
            .inner = try .init(allocator, store, parsed_config.value, options, model_partitions),
            .parsed_config = parsed_config,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        self.parsed_config.deinit();
    }

    pub fn specialTokens(self: *const LoadedModel, tokenizer: zml.tokenizer.Tokenizer) !SpecialTokens {
        const im_start = self.parsed_config.value.im_start_token_id orelse tokenizer.tokenId("<|im_start|>") orelse return error.NoSuchToken;
        const im_end = self.parsed_config.value.im_end_token_id orelse tokenizer.tokenId("<|im_end|>") orelse return error.NoSuchToken;
        const eos = if (self.parsed_config.value.eos_token_id) |eos_token_id| switch (eos_token_id.value) {
            .int => |id| id,
            .ints => |ids| if (ids.len > 0) ids[0] else im_end,
        } else tokenizer.tokenId("<|endoftext|>") orelse im_end;
        return .{
            .im_start_token_id = im_start,
            .im_end_token_id = im_end,
            .eos_token_id = eos,
        };
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
        Model.unloadBuffers(buffers, allocator);
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.attention.Backend,
        shardings: common.Shardings,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !inference.CompiledModel {
        _ = backend;
        const params = inference.CompilationParameters.init(self.inner, self.parsed_config.value, @intCast(seqlen), .vanilla, shardings);
        return inference.CompiledModel.init(allocator, io, platform, self, self.inner, params, progress);
    }
};

pub const Buffers = zml.Bufferized(Model);
