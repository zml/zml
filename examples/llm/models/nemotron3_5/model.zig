const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const common = @import("../common.zig");
const inference = @import("inference.zig");

const log = std.log.scoped(.nemotron3_5_transcribe);

pub const Config = struct {
    llm_config: TextConfig,
};

pub const TextConfig = struct {
    // General
    hidden_size: i64,
    intermediate_size: i64,
    max_position_embeddings: i64,
    vocab_size: i64,
    layers_block_type: []const LayerType,
    layer_norm_epsilon: f32,
    tie_word_embeddings: bool,
    residual_in_fp32: bool,
    use_bias: bool,
    bos_token_id: u32,
    eos_token_id: u32,
    pad_token_id: u32,

    // Self attention
    head_dim: i64,
    num_attention_heads: i64,
    num_key_value_heads: i64,
    attention_bias: bool,
    sliding_window: ?i64,

    // Mamba-2
    conv_kernel: i64,
    mamba_head_dim: i64,
    mamba_num_heads: i64,
    mamba_hidden_act: []const u8,
    mamba_ssm_cache_dtype: []const u8,
    n_groups: i64,
    ssm_state_size: i64,
    use_conv_bias: bool,

    // MoE
    n_routed_experts: i64,
    n_shared_experts: i64,
    num_experts_per_tok: u32,
    moe_intermediate_size: i64,
    moe_latent_size: ?i64,
    moe_shared_expert_intermediate_size: i64,
    mlp_bias: bool,
    mlp_hidden_act: []const u8,
    n_group: i64,
    topk_group: i64,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,

    pub fn numHiddenLayers(self: TextConfig) usize {
        return self.layers_block_type.len;
    }
};

pub const LayerType = enum {
    linear_attention,
    full_attention,
    moe,
    mlp,
    mamba,
    attention,
};

fn partitionProjectedKv(kv: zml.Tensor, kv_head_sharding: zml.Sharding.DimSharding) zml.Tensor {
    return switch (kv_head_sharding) {
        .sharded => |heads| blk: {
            const sharded_kv = if (heads.factor != 1) kv.stutter1d(kv.axis(.h), heads.factor) else kv;
            break :blk sharded_kv.withPartitioning(.model, .{ .s = .replicated, .h = .model, .hd = .replicated });
        },
        .replicated => kv.withPartitioning(.model, .{ .s = .replicated, .h = .replicated, .hd = .replicated }),
    };
}

fn partitionCachedKv(tensor: zml.Tensor, kv_head_sharding: zml.Sharding.DimSharding) zml.Tensor {
    var kv = tensor.rename(.{ .s = .k });
    return switch (kv_head_sharding) {
        .sharded => |heads| blk: {
            if (heads.factor != 1) {
                kv = kv.stutter1d(kv.axis(.h), heads.factor);
            }
            break :blk kv.withPartitioning(.model, .{ .k = .replicated, .h = .model, .hd = .replicated });
        },
        .replicated => kv.withPartitioning(.model, .{ .k = .replicated, .h = .replicated, .hd = .replicated }),
    };
}

pub const LoadedModel = struct {
    inner: Model,
    parsed_config: std.json.Parsed(Config),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
        generation: common.GenerationOptions,
    ) !LoadedModel {
        const parsed_config = try common.parseConfig(Config, allocator, io, repo);
        errdefer parsed_config.deinit();

        const options: Model.GenOptions = .{
            .sampling_strategy = generation.sampling_strategy,
            .max_seq_len = parsed_config.value.llm_config.max_position_embeddings,
        };

        return .{
            .inner = try .init(allocator, store, parsed_config.value, options),
            .parsed_config = parsed_config,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        self.parsed_config.deinit();
    }

    pub fn loadBuffers(
        self: *LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
    ) !Buffers {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);

        var buffers = try zml.mem.bufferize(allocator, Model, &self.inner);
        errdefer self.unloadBuffers(&buffers, allocator);

        var loader: zml.io.Loader = try .init(allocator, platform, .{
            .dma_chunks = 32,
            .dma_chunk_size = 256 * zml.MiB,
            .parallelism = 16,
        });
        defer loader.deinit();

        try loader.load(io, Model, &self.inner, &buffers, store, .{ .progress = progress });
        try loader.await(io);

        const took = now.untilNow(io, .awake);
        const total_bytes: u64 = loader.bytes_loaded.raw;
        const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
        log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });

        return buffers;
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        _ = self;
        Backbone.unloadBuffers(&buffers.backbone, allocator);
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        _: zml.attention.Backend,
        meshes: common.Meshes,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !inference.CompiledModel {
        const moe_backend: zml.moe.Backend = .stablehlo;
        const params = inference.CompilationParameters.init(self.inner, self.parsed_config.value, @intCast(seqlen), moe_backend, meshes);
        return inference.CompiledModel.init(allocator, io, platform, self, self.inner, params, progress);
    }
};

pub const Buffers = zml.Bufferized(Model);

pub const Model = struct {
    pub const GenOptions = struct { sampling_strategy: zml.nn.SamplingStrategy = .{}, max_seq_len: i64 };

    pub const SpecialTokens = struct {
        im_start_token_id: u32,
        im_end_token_id: u32,
        end_of_text_token_id: u32,
    };

    backbone: Backbone,

    config: Config,
    special_tokens: SpecialTokens = .{
        .im_start_token_id = 248045,
        .im_end_token_id = 248046,
        .end_of_text_token_id = 248044,
    },

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, gen_options: GenOptions) !Model {
        return .{
            .backbone = try .init(allocator, store.withPrefix("backbone"), config, gen_options),
            .config = config,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        self.backbone.deinit(allocator);
    }

    pub fn load(
        self: *const Model,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        meshes: []const *const zml.Mesh,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(Model) {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);

        var buffers = try zml.mem.bufferize(allocator, Model, self);
        errdefer Model.unloadBuffers(&buffers, allocator);

        var loader: zml.io.Loader = try .init(allocator, platform, .{
            .dma_chunks = 32,
            .dma_chunk_size = 256 * zml.MiB,
            .parallelism = 16,
        });
        defer loader.deinit();

        loader.load(io, Model, self, &buffers, store, meshes);
        try loader.await(io);

        const took = now.untilNow(io, .awake);
        const total_bytes: u64 = loader.bytes_loaded.raw;
        const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
        log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model), allocator: std.mem.Allocator) void {
        Backbone.unloadBuffers(&self.backbone, allocator);
    }
    pub fn forward(
        self: Model,
        tokens_: zml.Tensor,
        token_index: zml.Tensor,
        active_length: zml.Tensor,
        kv_cache: KvCache,
        rng: zml.Tensor.Rng,
        moe_backend: zml.moe.Backend,
    ) struct { zml.Tensor, KvCache, zml.Tensor.Rng } {
        const tokens = tokens_.withPartialTags(.{.s});
        const new_tokens, const updated_kv_cache, const new_rng = self.backbone.forward(tokens, token_index, active_length, kv_cache, self.config, rng, moe_backend);
        return .{ new_tokens, updated_kv_cache, new_rng };
    }
};

// Use this intermediate struct to compile sampleTokens without taking the whole Backbone as input
pub const Sampler = struct {
    norm: RmsNorm,
    lm_head: zml.nn.Linear,
    gen_options: Model.GenOptions,

    pub const Input = struct {
        sampler: Sampler,
        hidden: zml.Tensor,
        tokens: zml.Tensor,
        active_length: zml.Tensor,
        rng: zml.Tensor.Rng,
        token_index: zml.Tensor,
    };

    pub const Output = struct {
        tokens: zml.Tensor,
        rng: zml.Tensor.Rng,
        token_index: zml.Tensor,
    };

    pub fn sampleTokens(input: Input) Output {
        const self = input.sampler;
        const last_index = input.active_length.convert(.i32).addConstant(-1);
        const x = self.norm.forward(input.hidden.slice(.s, .dyn(last_index, 1)));
        const logits = self.lm_head.forward(x.withPartialTags(.{.d}), x.dtype()).rename(.{ .dout = .voc });
        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, self.gen_options.sampling_strategy, input.rng);
        const tokens = next_tokens.convert(.u32);
        return .{
            .tokens = if (input.tokens.dim(.s) == 1) tokens.reuseBuffer(input.tokens) else tokens,
            .rng = new_rng,
            .token_index = input.token_index.addConstant(1).reuseBuffer(input.token_index),
        };
    }
};

pub const EmbedTokens = struct {
    pub const Input = struct {
        embedding: zml.nn.TokenEmbedding,
        tokens: zml.Tensor,
    };

    pub const Output = struct {
        hidden: zml.Tensor,
    };

    pub fn forward(input: Input) Output {
        return .{ .hidden = input.embedding.forward(input.tokens) };
    }
};

pub const Backbone = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []BackboneLayer,
    norm: RmsNorm,
    lm_head: zml.nn.Linear,
    gen_options: Model.GenOptions,

    pub fn init(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: Config,
        gen_options: Model.GenOptions,
    ) !Backbone {
        const lm_head_prefix = if (store.root().hasKey("lm_head.weight")) "lm_head" else "model.language_model.embed_tokens";

        const layers = try allocator.alloc(BackboneLayer, config.llm_config.numHiddenLayers());
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            errdefer for (layers[0..i]) |previous_layer| previous_layer.deinit(allocator);
            layer.* = try .init(allocator, store.withPrefix("layers").withLayer(i), config, i);
        }

        return .{
            .embed_tokens = .{
                .weight = store.createTensor(
                    "embeddings.weight",
                    .{ .voc, .d },
                    .model,
                    .{ .voc = .replicated, .d = .model },
                ),
            },
            .layers = layers,
            .norm = RmsNorm.init(store.withPrefix("norm_f"), config.llm_config.layer_norm_epsilon),
            .lm_head = .init(store.root().withPrefix(lm_head_prefix).createTensor("weight", .{ .dout, .d }, .model, .{ .dout = .model, .d = .replicated }), null, .d),
            .gen_options = gen_options,
        };
    }

    pub fn deinit(self: Backbone, allocator: std.mem.Allocator) void {
        for (self.layers) |layer| {
            layer.deinit(allocator);
        }
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Backbone), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            BackboneLayer.unloadBuffers(layer, allocator);
        }
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
        zml.nn.Linear.unloadBuffers(&self.lm_head);
    }

    pub fn sampler(self: Backbone) Sampler {
        return .{
            .norm = self.norm,
            .lm_head = self.lm_head,
            .gen_options = self.gen_options,
        };
    }

    pub fn forward(
        self: Backbone,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        active_length: zml.Tensor,
        kv_cache: KvCache,
        config: Config,
        rng: zml.Tensor.Rng,
        moe_backend: zml.moe.Backend,
    ) struct { zml.Tensor, KvCache, zml.Tensor.Rng } {
        var hidden_states = EmbedTokens.forward(.{
            .embedding = self.embed_tokens,
            .tokens = tokens,
        }).hidden;

        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden_states, updated_kv_cache = layer.forward(hidden_states, token_index, active_length, updated_kv_cache.atLayer(i), config, moe_backend);
        }

        const result = Sampler.sampleTokens(.{
            .sampler = self.sampler(),
            .hidden = hidden_states,
            .tokens = tokens,
            .active_length = active_length,
            .rng = rng,
            .token_index = token_index,
        });
        return .{ result.tokens, updated_kv_cache.reuseBuffer(kv_cache), result.rng };
    }
};

pub const BackboneLayer = struct {
    norm: RmsNorm,
    mixer: union(enum) {
        attn: SelfAttn,
        mamba: Mamba2,
        moe: Moe,
    },

    pub const SelfAttnInput = struct {
        layer: BackboneLayer,
        hidden: zml.Tensor,
        token_index: zml.Tensor,
        cache: KvCache.SelfAttnCache,
    };

    pub const SelfAttnOutput = struct {
        hidden: zml.Tensor,
        cache: KvCache.SelfAttnCache,
    };

    pub const MambaInput = struct {
        layer: BackboneLayer,
        hidden: zml.Tensor,
        cache: Mamba2.Cache,
        active_length: zml.Tensor,
    };

    pub const MambaOutput = struct {
        hidden: zml.Tensor,
        cache: Mamba2.Cache,
    };

    pub const MoeInput = struct {
        layer: BackboneLayer,
        hidden: zml.Tensor,
        moe_backend: zml.moe.Backend,
    };

    pub const MoeOutput = struct {
        hidden: zml.Tensor,
    };

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, layer_index: usize) !BackboneLayer {
        return .{
            .norm = RmsNorm.init(store.withPrefix("norm"), config.llm_config.layer_norm_epsilon),
            .mixer = switch (config.llm_config.layers_block_type[layer_index]) {
                .attention, .full_attention => .{
                    .attn = try SelfAttn.init(store.withPrefix("mixer"), config),
                },
                .mamba, .linear_attention => .{
                    .mamba = Mamba2.init(store.withPrefix("mixer"), config),
                },
                .moe => .{
                    .moe = try Moe.init(allocator, store.withPrefix("mixer"), config),
                },
                .mlp => return error.MlpNotImplemented,
            },
        };
    }

    pub fn deinit(self: BackboneLayer, allocator: std.mem.Allocator) void {
        switch (self.mixer) {
            .moe => |moe| moe.deinit(allocator),
            .attn, .mamba => {},
        }
    }

    pub fn unloadBuffers(self: *zml.Bufferized(BackboneLayer), allocator: std.mem.Allocator) void {
        RmsNorm.unloadBuffers(&self.norm);
        switch (self.mixer) {
            .attn => |*attn| SelfAttn.unloadBuffers(attn),
            .mamba => |*mamba| Mamba2.unloadBuffers(mamba),
            .moe => |*moe| Moe.unloadBuffers(moe, allocator),
        }
    }

    pub fn forwardSelfAttn(input: SelfAttnInput) SelfAttnOutput {
        const self = input.layer;
        const x0 = input.hidden;
        const x0_replicated = x0.withPartitioning(.model, .{ .d = .replicated });
        const normalized_x0 = self.norm.forward(x0_replicated);
        const self_attn = switch (self.mixer) {
            .attn => |attn| attn,
            .moe, .mamba => unreachable,
        };
        const attention_output, const updated_kv_cache = self_attn.forward(normalized_x0, input.token_index, input.cache);

        return .{
            .hidden = attention_output.add(x0_replicated).withPartitioning(.model, .{ .d = .replicated }).reuseBuffer(x0),
            .cache = updated_kv_cache,
        };
    }

    pub fn forwardMamba(input: MambaInput) MambaOutput {
        const self = input.layer;
        const x0 = input.hidden;
        const x0_replicated = x0.withPartitioning(.model, .{ .d = .replicated });
        const normalized_x0 = self.norm.forward(x0_replicated);
        const mamba = switch (self.mixer) {
            .mamba => |mamba| mamba,
            .attn, .moe => unreachable,
        };
        const mamba_output, const updated_cache = mamba.forward(normalized_x0, input.cache, input.active_length);

        return .{
            .hidden = mamba_output.add(x0_replicated).withPartitioning(.model, .{ .d = .replicated }).reuseBuffer(x0),
            .cache = updated_cache.reuseBuffer(input.cache),
        };
    }

    pub fn forwardMoe(input: MoeInput) MoeOutput {
        const self = input.layer;
        const x0 = input.hidden;
        const x0_replicated = x0.withPartitioning(.model, .{ .d = .replicated });
        const normalized_x0 = self.norm.forward(x0_replicated);
        const moe = switch (self.mixer) {
            .moe => |moe| moe,
            .attn, .mamba => unreachable,
        };
        const moe_output = moe.forward(normalized_x0, input.moe_backend);

        return .{
            .hidden = moe_output.add(x0_replicated).withPartitioning(.model, .{ .d = .replicated }).reuseBuffer(x0),
        };
    }

    pub fn forward(
        self: BackboneLayer,
        x0: zml.Tensor,
        token_index: zml.Tensor,
        active_length: zml.Tensor,
        kv_cache: KvCache.LayerView,
        config: Config,
        moe_backend: zml.moe.Backend,
    ) struct { zml.Tensor, KvCache } {
        _ = active_length;
        _ = config;
        var updated_kv_cache = kv_cache.parent;
        return switch (self.mixer) {
            .mamba => stdx.debug.panic("Mamba-2 recurrent cache is not integrated into KvCache yet", .{}),
            .attn => blk: {
                const result = forwardSelfAttn(.{
                    .layer = self,
                    .hidden = x0,
                    .token_index = token_index,
                    .cache = kv_cache.cache.self_attn,
                });
                updated_kv_cache.self_attn = result.cache;
                break :blk .{ result.hidden, updated_kv_cache };
            },
            .moe => .{
                forwardMoe(.{
                    .layer = self,
                    .hidden = x0,
                    .moe_backend = moe_backend,
                }).hidden,
                updated_kv_cache,
            },
        };
    }
};

pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    q_proj_scale: ?zml.Tensor,
    k_proj: zml.nn.Linear,
    k_proj_scale: ?zml.Tensor,
    v_proj: zml.nn.Linear,
    v_proj_scale: ?zml.Tensor,

    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    o_proj: zml.nn.Linear,

    o_proj_scale: ?zml.Tensor,

    fn initProj(store: zml.io.TensorStore.View, partitions: anytype, bias_partitions: anytype) zml.nn.Linear {
        return .init(
            store.createTensor("weight", .{ .dout, .d }, .model, partitions),
            store.maybeCreateTensor("bias", .{.dout}, .model, bias_partitions),
            .d,
        );
    }

    pub fn init(store: zml.io.TensorStore.View, config: Config) !SelfAttn {
        return .{
            .q_proj = initProj(store.withPrefix("q_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .q_proj_scale = null,
            .k_proj = initProj(store.withPrefix("k_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .k_proj_scale = null,
            .v_proj = initProj(store.withPrefix("v_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .v_proj_scale = null,
            .o_proj = initProj(store.withPrefix("o_proj"), .{ .dout = .replicated, .d = .model }, .{ .dout = .replicated }),
            .o_proj_scale = null,
            .num_heads = config.llm_config.num_attention_heads,
            .num_kv_heads = config.llm_config.num_key_value_heads,
            .head_dim = config.llm_config.head_dim,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SelfAttn)) void {
        zml.Buffer.deinitAll(SelfAttn, self);
    }

    fn projectQ(self: SelfAttn, x: zml.Tensor) zml.Tensor {
        return self.q_proj.forward(x, x.dtype()).splitAxis(.dout, .{ .h = self.num_heads, .hd = self.head_dim });
    }

    fn projectKV(self: SelfAttn, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;
        const k = self.k_proj.forward(x, x.dtype()).splitAxis(.dout, .{ .h = num_kv_heads, .hd = self.head_dim });
        const v = self.v_proj.forward(x, x.dtype()).splitAxis(.dout, .{ .h = num_kv_heads, .hd = self.head_dim });
        return .{ k, v };
    }

    pub fn forward(
        self: SelfAttn,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache.SelfAttnCache,
    ) struct { zml.Tensor, KvCache.SelfAttnCache } {
        const tp = zml.Compiler.current().mesh(.model);
        const x_qkv = x.withPartitioning(tp, .{ .d = .replicated });
        var q = self.projectQ(x_qkv);
        var k, var v = self.projectKV(x_qkv);
        const kv_head_sharding = tp.shardableDim(k.dim(.h), .model, q.dim(.h));

        q = q.withPartitioning(tp, .{ .s = .replicated, .h = .model, .hd = .replicated });
        k = partitionProjectedKv(k, kv_head_sharding);
        v = partitionProjectedKv(v, kv_head_sharding);

        const dtype = q.dtype();
        const new_kv_cache = kv_cache.update(k, v, token_index.convert(.u32));
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);
        q = q.rename(.{ .s = .q }).withPartitioning(tp, .{ .q = .replicated, .h = .model, .hd = .replicated });
        k = partitionCachedKv(k, kv_head_sharding);
        v = partitionCachedKv(v, kv_head_sharding);

        const head_partitioning: zml.attention.Partitioning = switch (kv_head_sharding) {
            .sharded => .head(.model),
            .replicated => .head(.replicated),
        };
        const attn_output = zml.attention.attention(
            q,
            k,
            v,
            token_index,
            zml.attention.Metadata.init(.fromBackend(.stablehlo)),
            zml.attention.Parameters.init(.fromBackend(.stablehlo)),
            zml.Compiler.current().mesh(.model),
            head_partitioning,
        ).rename(.{ .q = .s }).merge(.{ .d_out_proj = .{ .h, .hd } });

        const projected_output = self.o_proj.forward(attn_output.rename(.{ .d_out_proj = .d }), attn_output.dtype()).rename(.{ .dout = .d }).withPartitioning(tp, .{ .d = .replicated });

        return .{ projected_output, new_kv_cache };
    }
};

pub const Mlp = struct {
    up_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View, config: TextConfig) !Mlp {
        if (config.mlp_bias or !std.mem.eql(u8, config.mlp_hidden_act, "relu2")) return error.UnsupportedMoeConfiguration;
        return .{
            .up_proj = try initProjection(store.withPrefix("up_proj")),
            .down_proj = try initProjection(store.withPrefix("down_proj")),
        };
    }

    fn initProjection(store: zml.io.TensorStore.View) !zml.nn.Linear {
        const shape = store.getShape("weight") orelse return error.MissingMoeTensor;
        if (shape.rank() != 2) return error.InvalidMoeShape;
        var linear: zml.nn.Linear = .init(store.createReplicatedTensor("weight", .{ .dout, .d }), null, .d);
        if (shape.dtype().isFloat() and shape.dtype().bitSizeOf() == 8) {
            const scale_shape = store.getShape("weight_scale") orelse return error.MissingMoeScale;
            if (scale_shape.count() != 1 or zml.Quantization.Scheme.classify(shape, scale_shape) != .fp8_per_tensor) return error.UnsupportedMoeQuantization;
            linear.quantization = .{
                .scheme = .fp8_per_tensor,
                .scales = store.createReplicatedTensor("weight_scale", null),
                .input_scale = if (store.getShape("input_scale")) |input_shape| blk: {
                    if (input_shape.count() != 1) return error.UnsupportedMoeQuantization;
                    break :blk .{ .value = store.createReplicatedTensor("input_scale", null), .operation = .multiply };
                } else null,
            };
        }
        return linear;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        zml.Buffer.deinitAll(Mlp, self);
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const activated = self.up_proj.forward(x, x.dtype()).relu();
        const hidden = activated.mul(activated).rename(.{ .dout = .d });
        return self.down_proj.forward(hidden, x.dtype());
    }

    fn dequantizeWeight(linear: zml.nn.Linear, dtype: zml.DataType) zml.Tensor {
        const weight = linear.weight.convert(dtype);
        const quantization = linear.quantization orelse return weight;
        return weight.mul(quantization.scales.asScalar().convert(dtype).broad(weight.shape()));
    }
};

const Router = struct {
    router: zml.nn.Linear,
    e_score_correction_bias: zml.Tensor,
    num_experts_per_tok: u32,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,

    pub fn init(store: zml.io.TensorStore.View, config: TextConfig) !Router {
        const weight_shape = store.getShape("weight") orelse return error.MissingMoeTensor;
        const bias_shape = store.getShape("e_score_correction_bias") orelse return error.MissingMoeTensor;
        if (weight_shape.rank() != 2 or weight_shape.dim(0) != config.n_routed_experts or weight_shape.dim(1) != config.hidden_size or bias_shape.rank() != 1 or bias_shape.dim(0) != config.n_routed_experts) return error.InvalidMoeShape;
        return .{
            .router = .init(store.createReplicatedTensor("weight", .{ .expert, .d }), null, .d),
            .e_score_correction_bias = store.createReplicatedTensor("e_score_correction_bias", .{.expert}),
            .num_experts_per_tok = config.num_experts_per_tok,
            .norm_topk_prob = config.norm_topk_prob,
            .routed_scaling_factor = config.routed_scaling_factor,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Router)) void {
        zml.Buffer.deinitAll(Router, self);
    }

    pub fn forward(self: Router, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const scores = self.router.forward(x.convert(.f32), .f32).sigmoid();
        // The correction bias selects experts; their contribution uses the unbiased scores.
        const choices = scores.add(self.e_score_correction_bias.convert(.f32).broad(scores.shape()));
        const topk_ids = choices.topK(.{ .topk = .expert }, self.num_experts_per_tok, .{}).indices.convert(.i32);
        var weights = scores.gather(.{ .expert = topk_ids }, .{});
        if (self.norm_topk_prob) weights = weights.div(weights.sum(.topk).addConstant(1e-20).broad(weights.shape()));
        return .{ weights.scale(self.routed_scaling_factor), topk_ids };
    }
};

pub const Moe = struct {
    experts: []Mlp,
    shared_experts: Mlp,
    router: Router,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Moe {
        const cfg = config.llm_config;
        if (cfg.n_routed_experts <= 0 or cfg.n_shared_experts != 1 or cfg.num_experts_per_tok == 0 or cfg.num_experts_per_tok > cfg.n_routed_experts or cfg.n_group != 1 or cfg.topk_group != 1 or cfg.moe_latent_size != null) return error.UnsupportedMoeConfiguration;
        const experts = try allocator.alloc(Mlp, @intCast(cfg.n_routed_experts));
        errdefer allocator.free(experts);
        const experts_store = store.withPrefix("experts");
        for (experts, 0..) |*expert, i| {
            expert.* = try Mlp.init(experts_store.withLayer(i), cfg);
            try validateExpert(expert.*, cfg.hidden_size, cfg.moe_intermediate_size);
        }
        const shared_experts = try Mlp.init(store.withPrefix("shared_experts"), cfg);
        try validateExpert(shared_experts, cfg.hidden_size, cfg.moe_shared_expert_intermediate_size);
        return .{
            .experts = experts,
            .shared_experts = shared_experts,
            .router = try Router.init(store.withPrefix("gate"), cfg),
        };
    }

    fn validateExpert(expert: Mlp, hidden_size: i64, intermediate_size: i64) !void {
        if (expert.up_proj.weight.dim(.d) != hidden_size or expert.up_proj.weight.dim(.dout) != intermediate_size or expert.down_proj.weight.dim(.d) != intermediate_size or expert.down_proj.weight.dim(.dout) != hidden_size) return error.InvalidMoeShape;
    }

    pub fn deinit(self: Moe, allocator: std.mem.Allocator) void {
        allocator.free(self.experts);
    }

    pub fn forward(self: Moe, x: zml.Tensor, _: zml.moe.Backend) zml.Tensor {
        const routing_scores, const topk_ids = self.router.forward(x);
        // Reference path: the shared fused MoE kernels do not support ReLU-squared yet.
        const up = self.stackWeights("up_proj", x.dtype()).gather(.{ .expert = topk_ids }, .{});
        const activated = x.dot(up, .d).relu();
        const hidden = activated.mul(activated).rename(.{ .dout = .ff });
        const down = self.stackWeights("down_proj", x.dtype())
            .rename(.{ .d = .ff })
            .rename(.{ .dout = .d })
            .gather(.{ .expert = topk_ids }, .{});
        const projected = hidden.dot(down, .ff);
        const routed = projected.convert(.f32).mul(routing_scores.broad(projected.shape().withDtype(.f32)))
            .sum(.topk).squeeze(.topk).convert(x.dtype());
        const shared = self.shared_experts.forward(x).rename(.{ .dout = .d });
        return routed.add(shared);
    }

    fn stackWeights(self: Moe, comptime projection: []const u8, dtype: zml.DataType) zml.Tensor {
        const allocator = zml.Compiler.current().arena.allocator();
        const weights = allocator.alloc(zml.Tensor, self.experts.len) catch @panic("OOM");
        defer allocator.free(weights);
        for (self.experts, weights) |expert, *weight| {
            weight.* = Mlp.dequantizeWeight(@field(expert, projection), dtype);
        }
        return zml.Tensor.stack(weights, 0, .expert);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Moe), allocator: std.mem.Allocator) void {
        for (self.experts) |*expert| Mlp.unloadBuffers(expert);
        allocator.free(self.experts);
        Mlp.unloadBuffers(&self.shared_experts);
        Router.unloadBuffers(&self.router);
    }
};

pub const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32 = 1e-6,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{ .weight = store.createTensor("weight", .{.d}, .model, .{ .d = .replicated }), .eps = eps };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNorm, x: zml.Tensor) zml.Tensor {
        const x_f32 = x.convert(.f32);
        const weight_f32 = self.weight.convert(.f32);

        const normalized = zml.nn.rmsNorm(x_f32, .d, self.eps);
        return normalized.mul(weight_f32.broad(x.shape())).add(normalized).convert(x.dtype());
    }
};

pub const KvCache = struct {
    layer_types: []const LayerType,
    self_attn: SelfAttnCache,

    pub const SelfAttnCache = struct {
        k: zml.Tensor,
        v: zml.Tensor,
        layer_index: zml.Tensor,

        pub fn init(config: Config, batch_dim: i64, max_seq_len: i64, dtype: zml.DataType, model_meshe: *const zml.Mesh) SelfAttnCache {
            const kv_shape = zml.Shape.init(.{
                .b = batch_dim,
                .layer = countAttentionLayers(config.llm_config.layers_block_type),
                .s = max_seq_len,
                .h = config.llm_config.num_key_value_heads,
                .hd = config.llm_config.head_dim,
            }, dtype);
            const kv_head_sharding = model_meshe.shardableDim(kv_shape.dim(.h), .model, config.llm_config.num_attention_heads);
            const sharded_kv_shape = switch (kv_head_sharding) {
                .sharded => |heads| kv_shape.setDim(.h, heads.dim).withPartitioning(model_meshe, .{ .h = .model }),
                .replicated => kv_shape.withPartitioning(model_meshe, .{ .h = .replicated }),
            };
            return .{
                .k = zml.Tensor.fromShape(sharded_kv_shape),
                .v = zml.Tensor.fromShape(sharded_kv_shape),
                .layer_index = .init(.{}, .u32),
            };
        }

        pub fn initBuffer(self: SelfAttnCache, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(SelfAttnCache) {
            return .{
                .k = try zml.Buffer.uninitialized(io, platform, self.k.shape(), .{}),
                .v = try zml.Buffer.uninitialized(io, platform, self.v.shape(), .{}),
                .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32),
            };
        }

        pub fn deinitBuffer(self: *zml.Bufferized(SelfAttnCache)) void {
            self.k.deinit();
            self.v.deinit();
            self.layer_index.deinit();
        }

        pub fn keys(self: SelfAttnCache) zml.Tensor {
            return self.k.slice(.layer, .dynSingle(self.layer_index));
        }

        pub fn values(self: SelfAttnCache) zml.Tensor {
            return self.v.slice(.layer, .dynSingle(self.layer_index));
        }

        pub fn update(self: SelfAttnCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: ?zml.Tensor) SelfAttnCache {
            const k_shape = self.k.shape().drop(.layer);
            var layer = self.layer_index;
            layer = if (token_index) |idx| layer.broad(idx.shape()) else layer;

            return if (token_index) |idx| .{
                .k = self.k.scatterSlices(
                    .{ .layer = layer, .s = idx },
                    new_k.convert(self.k.dtype()).transpose(k_shape),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(self.k),
                .v = self.v.scatterSlices(
                    .{ .layer = layer, .s = idx },
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

        pub fn atLayer(self: SelfAttnCache, layer_index: usize) SelfAttnCache {
            return .{
                .k = self.k,
                .v = self.v,
                .layer_index = zml.Tensor.scalar(layer_index, .u32),
            };
        }

        pub fn reuseBuffer(self: SelfAttnCache, other: SelfAttnCache) SelfAttnCache {
            return .{
                .k = self.k.reuseBuffer(other.k),
                .v = self.v.reuseBuffer(other.v),
                .layer_index = self.layer_index.reuseBuffer(other.layer_index),
            };
        }
    };

    pub fn init(
        config: Config,
        batch_dim: i64,
        max_seq_len: i64,
        cache_dtype: zml.DataType,
        _: zml.DataType,
        model_meshe: *const zml.Mesh,
    ) KvCache {
        return .{
            .layer_types = config.llm_config.layers_block_type,
            .self_attn = SelfAttnCache.init(config, batch_dim, max_seq_len, cache_dtype, model_meshe),
        };
    }

    pub fn initBuffer(self: KvCache, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(KvCache) {
        return .{
            .self_attn = try self.self_attn.initBuffer(io, platform),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(KvCache)) void {
        SelfAttnCache.deinitBuffer(&self.self_attn);
    }

    pub const LayerView = struct {
        parent: KvCache,
        cache: union(enum) {
            self_attn: SelfAttnCache,
            none,
        },
    };

    pub fn atLayer(self: KvCache, layer_index: usize) LayerView {
        return .{
            .parent = self,
            .cache = switch (self.layer_types[layer_index]) {
                .attention, .full_attention => .{ .self_attn = self.self_attn.atLayer(@intCast(countAttentionLayers(self.layer_types[0..layer_index]))) },
                .moe, .mlp => .none,
                .mamba, .linear_attention => stdx.debug.panic("Mamba-2 recurrent cache is not integrated into KvCache yet", .{}),
            },
        };
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{
            .layer_types = self.layer_types,
            .self_attn = self.self_attn.reuseBuffer(other.self_attn),
        };
    }

    fn countAttentionLayers(layer_types: []const LayerType) i64 {
        var count: i64 = 0;
        for (layer_types) |layer_type| {
            switch (layer_type) {
                .attention, .full_attention => count += 1,
                else => {},
            }
        }
        return count;
    }
};

pub const GroupedRmsNormGated = struct {
    weight: zml.Tensor,
    eps: f32,
    num_groups: i64,

    pub fn init(store: zml.io.TensorStore.View, eps: f32, num_groups: i64) GroupedRmsNormGated {
        return .{
            .weight = store.createReplicatedTensor("weight", .{.d}),
            .eps = eps,
            .num_groups = num_groups,
        };
    }

    pub fn forward(self: GroupedRmsNormGated, x: zml.Tensor, gate: zml.Tensor) zml.Tensor {
        stdx.debug.assert(self.num_groups > 0 and @rem(x.dim(.d), self.num_groups) == 0, "Mamba2 normalization groups must divide the inner width", .{});
        stdx.debug.assert(self.eps > 0 and self.weight.dim(.d) == x.dim(.d), "Invalid Mamba2 normalization parameters", .{});
        stdx.debug.assert(std.mem.eql(i64, x.dims(), gate.dims()), "Mamba2 gate and normalization input must have matching shapes", .{});

        // Nemotron gates before normalizing; Qwen's RmsNormGated does the reverse.
        const gated = x.convert(.f32).mul(gate.convert(.f32).silu());
        const groups = gated.splitAxis(.d, .{ .group = self.num_groups, .element = @divExact(x.dim(.d), self.num_groups) });
        const normalized = zml.nn.rmsNorm(groups, .element, self.eps)
            .merge(.{ .d = .{ .group, .element } })
            .convert(x.dtype());
        return normalized.mul(self.weight.convert(x.dtype()).broad(normalized.shape()));
    }
};

pub const Mamba2 = struct {
    in_proj: zml.nn.Linear,
    out_proj: zml.nn.Linear,

    conv1d_weight: zml.Tensor,
    conv1d_bias: ?zml.Tensor,

    dt_bias: zml.Tensor,
    aLog: zml.Tensor,
    d: zml.Tensor,

    norm: GroupedRmsNormGated,

    num_heads: i64,
    num_groups: i64,
    head_dim: i64,
    state_size: i64,
    conv_kernel_size: i64,

    pub const Cache = struct {
        conv_state: zml.Tensor,
        recurrent_state: zml.Tensor,

        pub const Buffers = zml.Bufferized(Cache);

        pub fn init(config: Config, batch_dim: i64, conv_dtype: zml.DataType) Cache {
            const cfg = config.llm_config;
            validateConfig(cfg);
            stdx.debug.assert(batch_dim > 0, "Mamba2 batch size must be positive", .{});
            const conv_dim = cfg.mamba_num_heads * cfg.mamba_head_dim + 2 * cfg.n_groups * cfg.ssm_state_size;
            return .{
                .conv_state = .init(.{ .b = batch_dim, .s = cfg.conv_kernel - 1, .mix = conv_dim }, conv_dtype),
                .recurrent_state = .init(.{ .b = batch_dim, .h = cfg.mamba_num_heads, .p = cfg.mamba_head_dim, .n = cfg.ssm_state_size }, .f32),
            };
        }

        pub fn zeroes(self: Cache) Cache {
            return .{
                .conv_state = .zeroes(self.conv_state.shape()),
                .recurrent_state = .zeroes(self.recurrent_state.shape()),
            };
        }

        pub fn initBuffer(self: Cache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !Cache.Buffers {
            var conv_state = try zeroBuffer(allocator, io, platform, self.conv_state.shape());
            errdefer conv_state.deinit();
            return .{
                .conv_state = conv_state,
                .recurrent_state = try zeroBuffer(allocator, io, platform, self.recurrent_state.shape()),
            };
        }

        pub fn deinitBuffer(buffers: *Cache.Buffers) void {
            zml.Buffer.deinitAll(Cache, buffers);
        }

        pub fn reuseBuffer(self: Cache, other: Cache) Cache {
            return .{
                .conv_state = self.conv_state.reuseBuffer(other.conv_state),
                .recurrent_state = self.recurrent_state.reuseBuffer(other.recurrent_state),
            };
        }

        fn zeroBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape) !zml.Buffer {
            const bytes = try allocator.alloc(u8, shape.byteSize());
            defer allocator.free(bytes);
            @memset(bytes, 0);
            return zml.Buffer.fromBytes(io, platform, shape, bytes);
        }
    };

    pub const ScanOutput = struct {
        outputs: zml.Tensor,
        state: zml.Tensor,
    };

    pub fn init(store: zml.io.TensorStore.View, config: Config) Mamba2 {
        const cfg = config.llm_config;
        validateConfig(cfg);
        const conv_store = store.withPrefix("conv1d");
        const result: Mamba2 = .{
            // The fused [gate | x | B | C | dt] layout cannot be naively flat-sharded
            .in_proj = initProj(store.withPrefix("in_proj"), cfg.use_bias),
            .out_proj = initProj(store.withPrefix("out_proj"), cfg.use_bias),
            .conv1d_weight = conv_store.createReplicatedTensor("weight", .{ .out, .in, .kernel_size }),
            .conv1d_bias = if (cfg.use_conv_bias) conv_store.createReplicatedTensor("bias", .{.out}) else null,
            .dt_bias = initStateParameter(store, "dt_bias"),
            .aLog = initStateParameter(store, "A_log"),
            .d = initStateParameter(store, "D"),
            .norm = .init(store.withPrefix("norm"), cfg.layer_norm_epsilon, cfg.n_groups),
            .num_heads = cfg.mamba_num_heads,
            .num_groups = cfg.n_groups,
            .head_dim = cfg.mamba_head_dim,
            .state_size = cfg.ssm_state_size,
            .conv_kernel_size = cfg.conv_kernel,
        };
        const inner_dim = cfg.mamba_num_heads * cfg.mamba_head_dim;
        const conv_dim = inner_dim + 2 * cfg.n_groups * cfg.ssm_state_size;
        stdx.debug.assert(result.in_proj.weight.dim(.d) == cfg.hidden_size and result.in_proj.weight.dim(.dout) == inner_dim + conv_dim + cfg.mamba_num_heads, "Mamba2 input projection shape does not match config", .{});
        stdx.debug.assert(result.out_proj.weight.dim(.d) == inner_dim and result.out_proj.weight.dim(.dout) == cfg.hidden_size, "Mamba2 output projection shape does not match config", .{});
        return result;
    }

    fn validateConfig(cfg: TextConfig) void {
        stdx.debug.assert(cfg.hidden_size > 0 and cfg.mamba_num_heads > 0 and cfg.mamba_head_dim > 0 and cfg.n_groups > 0 and cfg.ssm_state_size > 0 and cfg.conv_kernel > 0, "Mamba2 dimensions must be positive", .{});
        stdx.debug.assert(@rem(cfg.mamba_num_heads, cfg.n_groups) == 0, "Mamba2 groups must divide the number of heads", .{});
        stdx.debug.assert(std.mem.eql(u8, cfg.mamba_hidden_act, "silu"), "Mamba2 only supports the silu activation", .{});
        stdx.debug.assert(std.mem.eql(u8, cfg.mamba_ssm_cache_dtype, "float32"), "Mamba2 recurrent cache must use float32", .{});
        stdx.debug.assert(cfg.layer_norm_epsilon > 0, "Mamba2 normalization epsilon must be positive", .{});
    }

    fn initProj(store: zml.io.TensorStore.View, use_bias: bool) zml.nn.Linear {
        return .init(
            store.createReplicatedTensor("weight", .{ .dout, .d }),
            if (use_bias) store.createReplicatedTensor("bias", .{.dout}) else null,
            .d,
        );
    }

    fn initStateParameter(store: zml.io.TensorStore.View, comptime name: []const u8) zml.Tensor {
        const key = if (store.getShape("_fp32_params." ++ name) != null) "_fp32_params." ++ name else name;
        return store.createReplicatedTensor(key, .{.h});
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mamba2)) void {
        zml.Buffer.deinitAll(Mamba2, self);
    }

    /// FP32 reference scan. B/C are already expanded to heads; dt is already positive
    /// A is already -exp(A_log). Both outputs and final state are FP32
    pub fn selectiveScan(
        x: zml.Tensor,
        b: zml.Tensor,
        c: zml.Tensor,
        dt: zml.Tensor,
        a: zml.Tensor,
        d: zml.Tensor,
        initial_state: zml.Tensor,
        active_length: zml.Tensor,
    ) ScanOutput {
        stdx.debug.assert(x.rank() == 4 and x.shape().hasTags(.{ .b, .s, .h, .p }), "Mamba2 scan expects x {{b, s, h, p}}", .{});
        stdx.debug.assert(b.rank() == 4 and b.shape().hasTags(.{ .b, .s, .h, .n }) and std.mem.eql(i64, b.dims(), c.dims()), "Mamba2 scan expects matching B/C {{b, s, h, n}}", .{});
        stdx.debug.assert(b.dim(.b) == x.dim(.b) and b.dim(.s) == x.dim(.s) and b.dim(.h) == x.dim(.h), "Mamba2 scan B/C dimensions do not match x", .{});
        stdx.debug.assert(dt.rank() == 3 and dt.dim(.b) == x.dim(.b) and dt.dim(.s) == x.dim(.s) and dt.dim(.h) == x.dim(.h), "Mamba2 scan dt dimensions do not match x", .{});
        stdx.debug.assert(a.rank() == 1 and d.rank() == 1 and a.dim(.h) == x.dim(.h) and d.dim(.h) == x.dim(.h), "Mamba2 scan A/D dimensions do not match heads", .{});
        stdx.debug.assert(initial_state.rank() == 4 and initial_state.dim(.b) == x.dim(.b) and initial_state.dim(.h) == x.dim(.h) and initial_state.dim(.p) == x.dim(.p) and initial_state.dim(.n) == b.dim(.n), "Mamba2 scan initial state must have shape {{b, h, p, n}}", .{});

        const scan: SelectiveScan = .{
            .x = x.convert(.f32),
            .b = b.convert(.f32),
            .c = c.convert(.f32),
            .dt = dt.convert(.f32),
            .a = a.convert(.f32),
            .d = d.convert(.f32),
            .active_length = clampActiveLength(active_length, x.dim(.s)),
        };
        const initial: SelectiveScan.State = .{
            .ssm = initial_state.convert(.f32),
            .outputs = .zeroes(x.shape().withDtype(.f32)),
            .step = .scalar(0, .i64),
        };
        if (x.dim(.s) == 0) return .{ .outputs = initial.outputs, .state = initial.ssm };
        const final = if (x.dim(.s) == 1) blk: {
            // Avoid the while-loop overhead for token-by-token decoding.
            const next = scan.body(initial);
            const run = scan.cond(initial);
            break :blk SelectiveScan.State{
                .ssm = run.broad(initial.ssm.shape()).select(next.ssm, initial.ssm),
                .outputs = run.broad(initial.outputs.shape()).select(next.outputs, initial.outputs),
                .step = run.select(next.step, initial.step),
            };
        } else zml.ops.@"while"(SelectiveScan, scan, initial);
        return .{ .outputs = final.outputs, .state = final.ssm };
    }

    pub fn forward(self: Mamba2, x: zml.Tensor, cache: Cache, active_length: zml.Tensor) struct { zml.Tensor, Cache } {
        stdx.debug.assert(self.num_heads > 0 and self.num_groups > 0 and @rem(self.num_heads, self.num_groups) == 0 and self.head_dim > 0 and self.state_size > 0 and self.conv_kernel_size > 0, "Invalid Mamba2 dimensions", .{});
        stdx.debug.assert(x.rank() == 3 and x.shape().hasTags(.{ .b, .s, .d }) and x.dim(.s) > 0, "Mamba2 expects a nonempty input {{b, s, d}}", .{});

        const inner_dim = self.num_heads * self.head_dim;
        const bc_dim = self.num_groups * self.state_size;
        const conv_dim = inner_dim + 2 * bc_dim;
        const history = self.conv_kernel_size - 1;

        stdx.debug.assert(cache.conv_state.rank() == 3 and cache.conv_state.dim(.b) == x.dim(.b) and cache.conv_state.dim(.s) == history and cache.conv_state.dim(.mix) == conv_dim, "Mamba2 convolution cache shape does not match config", .{});
        stdx.debug.assert(cache.recurrent_state.dtype() == .f32, "Mamba2 recurrent cache must be FP32", .{});
        stdx.debug.assert(self.conv1d_weight.dim(.out) == conv_dim and self.conv1d_weight.dim(.in) == 1 and self.conv1d_weight.dim(.kernel_size) == self.conv_kernel_size, "Mamba2 depthwise convolution shape does not match config", .{});
        stdx.debug.assert(self.norm.num_groups == self.num_groups, "Mamba2 normalization and B/C groups must match", .{});

        const length = clampActiveLength(active_length, x.dim(.s));

        const projected = self.in_proj.forward(x, x.dtype()).rename(.{ .dout = .mix });
        const gate = projected.slice(.mix, .{ .end = inner_dim }).rename(.{ .mix = .d });
        const xbc = projected.slice(.mix, .{ .start = inner_dim, .end = inner_dim + conv_dim });
        const dt_raw = projected.slice(.mix, .{ .start = inner_dim + conv_dim, .end = inner_dim + conv_dim + self.num_heads }).rename(.{ .mix = .h });

        // A valid convolution over cached history works for decode AND continuation chunks.
        const conv_input = if (history == 0) xbc else zml.Tensor.concatenate(&.{ cache.conv_state.convert(xbc.dtype()), xbc }, .s);
        var mixed = zml.Tensor.conv1d(conv_input, self.conv1d_weight.convert(xbc.dtype()), .{
            .input_batch_dimension = 0,
            .input_feature_dimension = 2,
            .input_spatial_dimensions = 1,
            .kernel_output_feature_dimension = 0,
            .kernel_input_feature_dimension = 1,
            .kernel_spatial_dimensions = 2,
            .output_batch_dimension = 0,
            .output_feature_dimension = 2,
            .output_spatial_dimensions = 1,
            .feature_group_count = conv_dim,
        });

        if (self.conv1d_bias) |bias|
            mixed = mixed.add(bias.convert(mixed.dtype()).rename(.{ .out = .mix }).broad(mixed.shape()));

        mixed = mixed.silu();

        const values = mixed.slice(.mix, .{ .end = inner_dim }).splitAxis(.mix, .{ .h = self.num_heads, .p = self.head_dim });
        const b = self.expandGroups(mixed.slice(.mix, .{ .start = inner_dim, .end = inner_dim + bc_dim }));
        const c = self.expandGroups(mixed.slice(.mix, .{ .start = inner_dim + bc_dim, .end = conv_dim }));
        const dt = zml.nn.softplus(dt_raw.convert(.f32).add(self.dt_bias.convert(.f32).broad(dt_raw.shape().withDtype(.f32))), .{});
        const a = self.aLog.convert(.f32).exp().negate();
        const scan = selectiveScan(values, b, c, dt, a, self.d, cache.recurrent_state, length);
        const hidden = scan.outputs.convert(x.dtype()).merge(.{ .d = .{ .h, .p } });
        const normalized = self.norm.forward(hidden, gate);
        const projected_output = self.out_proj.forward(normalized, x.dtype()).rename(.{ .dout = .d });
        const positions = zml.Tensor.iota(zml.Shape.init(.{ .s = x.dim(.s) }, .i32), .s).convert(.i64);
        const valid = positions.cmp(.LT, length.broad(positions.shape()));
        const output = valid.broad(projected_output.shape()).select(projected_output, zml.Tensor.zeroes(projected_output.shape()));

        // Select the tail after valid tokens, not the tail of the padded input.
        const updated_cache: Cache = .{
            .conv_state = if (history == 0) cache.conv_state else conv_input.slice(.s, .dyn(length, history)).convert(cache.conv_state.dtype()),
            .recurrent_state = scan.state,
        };

        return .{ output, updated_cache };
    }

    fn expandGroups(self: Mamba2, x: zml.Tensor) zml.Tensor {
        const grouped = x.splitAxis(.mix, .{ .group = self.num_groups, .n = self.state_size });
        return grouped.stutter1d(grouped.axis(.group), @intCast(@divExact(self.num_heads, self.num_groups)))
            .rename(.{ .group = .h });
    }
};

fn clampActiveLength(active_length: zml.Tensor, seqlen: i64) zml.Tensor {
    stdx.debug.assert(active_length.rank() == 0 and active_length.dtype().isInteger(), "Mamba2 active_length must be an integer scalar", .{});
    return active_length.convert(.i64).maximum(.scalar(0, .i64)).minimum(.scalar(seqlen, .i64));
}

const SelectiveScan = struct {
    x: zml.Tensor,
    b: zml.Tensor,
    c: zml.Tensor,
    dt: zml.Tensor,
    a: zml.Tensor,
    d: zml.Tensor,
    active_length: zml.Tensor,

    pub const State = struct {
        ssm: zml.Tensor,
        outputs: zml.Tensor,
        step: zml.Tensor,
    };

    pub fn cond(self: SelectiveScan, state: State) zml.Tensor {
        return state.step.cmp(.LT, self.active_length);
    }

    pub fn body(self: SelectiveScan, state: State) State {
        const x = self.x.slice(.s, .dynSingle(state.step));
        const b = self.b.slice(.s, .dynSingle(state.step));
        const c = self.c.slice(.s, .dynSingle(state.step));
        const dt = self.dt.slice(.s, .dynSingle(state.step));
        const decay = dt.mul(self.a.broad(dt.shape())).exp();
        const delta_x = x.mul(dt.broad(x.shape()));
        const write = delta_x.broad(state.ssm.shape()).mul(b.broad(state.ssm.shape()));
        const next_state = state.ssm.mul(decay.broad(state.ssm.shape())).add(write);
        const y = next_state.dot(c, .n).add(x.mul(self.d.broad(x.shape())));
        return .{
            .ssm = next_state,
            .outputs = state.outputs.dynamicUpdateSlice(.{ .s = state.step }, y),
            .step = state.step.addConstant(1),
        };
    }
};
