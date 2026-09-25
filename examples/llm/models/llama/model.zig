const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const common = @import("../common.zig");
const inference = @import("inference.zig");
const weight_packing = @import("packed_weights.zig");

const log = std.log.scoped(.llama);

pub const Config = struct {
    bos_token_id: u32,
    eos_token_id: EosTokens,
    head_dim: ?u32 = null,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    rope_theta: f32,
    max_position_embeddings: u32,
    rms_norm_eps: f32,
    hf_rope_impl: bool = true,
    tie_word_embeddings: bool = false,
    zml_weight_storage: ?enum { fp8_e4m3fn_pow2_channel } = null,
    rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = .{} },

    pub const EosTokens = union(enum) {
        int: u32,
        ints: []u32,

        const Helpers = stdx.json.UnionHelpers(@This());
        pub const jsonParse = Helpers.jsonParse;
        pub const jsonParseFromValue = Helpers.jsonParseFromValue;
        pub const jsonStringify = Helpers.jsonStringify;
    };
};

const Options = struct {
    sampling_strategy: ?zml.nn.SamplingStrategy,
    max_seq_len: u32,
};

pub const LoadedModel = struct {
    inner: Model,
    packing: weight_packing.Plan,
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
        if (parsed_config.value.zml_weight_storage != null) {
            log.info("FP8 transformer weight storage; activations, embeddings and output head remain BF16", .{});
        }

        const options: Options = .{
            .sampling_strategy = generation.sampling_strategy,
            .max_seq_len = parsed_config.value.max_position_embeddings,
        };

        var inner = try Model.init(allocator, store, parsed_config.value, options);
        errdefer inner.deinit(allocator);
        return .{
            .inner = inner,
            .packing = try .init(allocator, inner),
            .parsed_config = parsed_config,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.packing.deinit();
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
    ) !weight_packing.Buffers {
        return self.packing.load(allocator, io, platform, store, progress, &shardings.all());
    }

    pub fn unloadBuffers(_: *const LoadedModel, buffers: *weight_packing.Buffers, allocator: std.mem.Allocator) void {
        weight_packing.Plan.unload(buffers, allocator);
    }

    // Component tests retain the checkpoint's individual weight buffers as an
    // independent reference for the packed whole-model inference path.
    pub fn loadUnpackedBuffers(
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

        var buffers = try zml.mem.bufferize(allocator, Model, &self.inner);
        errdefer self.unloadUnpackedBuffers(&buffers, allocator);

        var loader: zml.io.Loader = try .init(allocator, platform, .{
            .dma_chunks = 32,
            .dma_chunk_size = 256 * zml.MiB,
            .parallelism = 16,
        });
        defer loader.deinit();

        const all_shardings = shardings.all();
        try loader.load(io, Model, &self.inner, &buffers, store, &all_shardings, .{ .progress = progress });
        try loader.await(io);

        const took = now.untilNow(io, .awake);
        const total_bytes: u64 = loader.bytes_loaded.raw;
        const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
        log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });

        return buffers;
    }

    pub fn unloadUnpackedBuffers(_: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        if (buffers.lm_head) |*lm_head| Projection.unloadBuffers(lm_head);
        Llama.unloadBuffers(&buffers.model, allocator);
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.Backend,
        shardings: common.Shardings,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !inference.CompiledModel {
        const params = inference.CompilationParameters.init(self.inner, self.parsed_config.value, @intCast(seqlen), backend, shardings);

        return inference.CompiledModel.init(allocator, io, platform, self, params, progress);
    }
};

pub const Buffers = zml.Bufferized(Model);

/// Llama architecture, using huggingface transformers naming.
/// Dimensions of activations: {.b, .s, .d}
pub const Model = struct {
    lm_head: ?Projection,
    model: Llama,

    gen_opts: zml.nn.SamplingStrategy = .{},
    config: Config,

    pub fn init(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: Config,
        options: Options,
    ) !Model {
        const lm_head = Projection.maybeInit(store.withPrefix("lm_head"), .{ .dout = .model, .d = .replicated });

        return .{
            .lm_head = lm_head,
            .model = try .init(allocator, store.withPrefix("model"), config),
            .gen_opts = options.sampling_strategy orelse .{},
            .config = config,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
    }

    pub fn loadBuffers(
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

        var buffers = try zml.mem.bufferize(allocator, Model, self);
        errdefer Model.unloadBuffers(&buffers, allocator);

        var loader: zml.io.Loader = try .init(allocator, platform, .{
            .dma_chunks = 32,
            .dma_chunk_size = 256 * zml.MiB,
            .parallelism = 16,
        });
        defer loader.deinit();

        loader.load(io, Model, self, &buffers, store, shardings);
        try loader.await(io);

        const took = now.untilNow(io, .awake);
        const total_bytes: u64 = loader.bytes_loaded.raw;
        const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
        log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model), allocator: std.mem.Allocator) void {
        if (self.lm_head) |*lm_head| Projection.unloadBuffers(lm_head);
        Llama.unloadBuffers(&self.model, allocator);
    }

    fn lmHead(self: Model) LmHead {
        return .init(self);
    }

    /// Predicts the token at `token_index` position.
    /// Returns:
    ///  - updated `tokens`,
    ///  - updated KV cache
    ///  - a Rng state to allow for probabilistic generation
    pub fn forward(
        self: Model,
        tokens_: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        rng: zml.Tensor.Rng,
        attention_metadata: zml.attention.Metadata,
        attention_parameters: zml.attention.Parameters,
    ) struct { zml.Tensor, KvCache, zml.Tensor.Rng } {
        const tokens = tokens_.withPartialTags(.{.s});

        const out, const updated_kv_cache = self.model.forward(
            tokens,
            token_index,
            kv_cache,
            attention_metadata,
            attention_parameters,
        );

        const sample = LmHead.forward(.{
            .lm_head = self.lmHead(),
            .hidden = out,
            .tokens = tokens,
            .rng = rng,
        });

        return .{ sample.tokens, updated_kv_cache, sample.rng };
    }
};

const Llama = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    norm: RmsNorm,
    layers: []TransformerLayer,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Llama {
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = try .init(store.withPrefix("layers").withLayer(i), config);
        }

        return .{
            .embed_tokens = .{ .weight = store.createTensor(
                "embed_tokens.weight",
                .{ .voc, .d },
                .{ .voc = .replicated, .d = .model },
            ) },
            .norm = .{
                .weight = store.withPrefix("norm").createTensor("weight", .{.d}, .{ .d = .replicated }),
                .eps = config.rms_norm_eps,
            },
            .layers = layers,
        };
    }

    pub fn deinit(self: Llama, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Llama), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        RmsNorm.unloadBuffers(&self.norm);
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
    }

    /// Forward one token, using KV cache for previous tokens.
    /// Returns result and updated KV cache.
    pub fn forward(
        self: Llama,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        attention_metadata: zml.attention.Metadata,
        attention_parameters: zml.attention.Parameters,
    ) struct { zml.Tensor, KvCache } {
        const embeds = self.embed_tokens.forward(tokens).withPartialTags(.{.d});
        var hidden = embeds;
        var updated_kv_cache = kv_cache;
        var kv_cache_index = zml.Tensor.scalar(0, .u32);

        for (self.layers) |layer| {
            const result = TransformerLayer.forward(.{
                .layer = layer,
                .hidden = hidden,
                .token_index = token_index,
                .kv_cache = updated_kv_cache,
                .kv_cache_index = kv_cache_index,
                .attention_metadata = attention_metadata,
                .attention_parameters = attention_parameters,
            });
            hidden = result.hidden;
            updated_kv_cache = result.kv_cache;
            kv_cache_index = kv_cache_index.add(zml.Tensor.scalar(@as(u32, 1), .u32));
        }

        // LmHead applies the final normalization before the output projection.
        return .{ hidden, updated_kv_cache.reuseBuffer(kv_cache) };
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

    pub fn forward(input: Input) Output {
        const tokens = input.tokens.withPartialTags(.{.s});
        return .{ .hidden = input.embedding.embed_tokens.forward(tokens)
            .withPartialTags(.{.d})
            .withPartitioning(.{ .d = .replicated }) };
    }
};

pub const LmHead = struct {
    lm_head: ?Projection,
    embed_tokens: zml.nn.TokenEmbedding,
    norm: RmsNorm,
    gen_opts: zml.nn.SamplingStrategy,

    pub fn init(mdl: Model) LmHead {
        return .{
            .lm_head = mdl.lm_head,
            .embed_tokens = mdl.model.embed_tokens,
            .norm = mdl.model.norm,
            .gen_opts = mdl.gen_opts,
        };
    }

    pub const Input = struct {
        lm_head: LmHead,
        hidden: zml.Tensor,
        tokens: zml.Tensor,
        rng: zml.Tensor.Rng,
    };

    pub const Output = struct {
        tokens: zml.Tensor,
        rng: zml.Tensor.Rng,
    };

    pub fn forward(input: Input) Output {
        const self = input.lm_head;
        const tokens = input.tokens.withPartialTags(.{.s});
        const hidden = self.norm.forward(input.hidden.withPartialTags(.{ .s, .d }));

        var logits = blk: {
            if (self.lm_head) |lm_head| {
                break :blk lm_head.forward(hidden).rename(.{ .dout = .d });
            } else {
                break :blk self.embed_tokens.weight.withTags(.{ .voc, .d }).dot(hidden, .d);
            }
        };

        if (logits.shape().hasTag(.voc) == null)
            logits = logits.rename(.{ .d = .voc });

        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, self.gen_opts, input.rng);
        return .{ .tokens = next_tokens.convert(tokens.dtype()).reuseBuffer(tokens), .rng = new_rng };
    }
};

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: SelfAttn,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,

    pub fn init(store: zml.io.TensorStore.View, config: Config) !TransformerLayer {
        return .{
            .input_layernorm = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps),
            .self_attn = try .init(store.withPrefix("self_attn"), config),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps),
            .mlp = .init(store.withPrefix("mlp")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        SelfAttn.unloadBuffers(&self.self_attn);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
        Mlp.unloadBuffers(&self.mlp);
    }

    pub const Input = struct {
        layer: TransformerLayer,
        hidden: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        kv_cache_index: zml.Tensor,
        attention_metadata: zml.attention.Metadata,
        attention_parameters: zml.attention.Parameters,
    };

    pub const Output = struct {
        hidden: zml.Tensor,
        kv_cache: KvCache,
    };

    pub fn forward(input: Input) Output {
        const self = input.layer;
        const x0 = input.hidden;
        // Self Attention
        //log.debug("TransformerLayer({f}) -> {f}", .{ x0, self.input_layernorm.forward(x0) });
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {f}", .{x0});

        // Keep the residual stream replicated to avoid repeated gathers before q/k/v.
        const x0_replicated = x0.withPartitioning(.{ .d = .replicated });
        const x0_normalized = self.input_layernorm.forward(x0_replicated);
        const delta0, const updated_kv_cache = self.self_attn.forward(
            x0_normalized,
            input.token_index,
            input.kv_cache,
            input.kv_cache_index,
            input.attention_metadata,
            input.attention_parameters,
        );

        // Fully Connected
        const x1 = x0_replicated.add(delta0).withPartitioning(.{ .d = .replicated });
        const x1_normalized = self.post_attention_layernorm.forward(x1);
        const x2 = self.mlp.forward(x1_normalized)
            .rename(.{ .dout = .d })
            .withPartitioning(.{ .d = .replicated })
            .add(x1)
            .withPartitioning(.{ .d = .replicated });

        // Furiosa computes the residual into a separate native output. Returning
        // that buffer avoids a device copy back into x0 after every layer.
        // Standalone layer callers own the previous hidden buffer; within the
        // whole forward graph this is an internal compiler-managed value.
        const hidden = if (zml.Compiler.current().platform.target == .furiosa) x2 else x2.reuseBuffer(x0);
        return .{ .hidden = hidden, .kv_cache = updated_kv_cache };
    }
};

pub const TransformerBlock = struct {
    pub const Input = struct {
        layers: []const TransformerLayer,
        hidden: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        kv_cache_index: zml.Tensor,
        attention_metadata: zml.attention.Metadata,
        attention_parameters: zml.attention.Parameters,
    };

    pub fn forward(input: Input) TransformerLayer.Output {
        stdx.debug.assert(input.layers.len > 0, "Transformer block must contain at least one layer", .{});
        var result: TransformerLayer.Output = .{ .hidden = input.hidden, .kv_cache = input.kv_cache };
        for (input.layers, 0..) |layer, i| {
            result = TransformerLayer.forward(.{
                .layer = layer,
                .hidden = result.hidden,
                .token_index = input.token_index,
                .kv_cache = result.kv_cache,
                .kv_cache_index = if (i == 0) input.kv_cache_index else input.kv_cache_index.addConstant(i),
                .attention_metadata = input.attention_metadata,
                .attention_parameters = input.attention_parameters,
            });
        }
        return result;
    }
};

const RmsNorm = struct {
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

    /// L2 normalization of input tensor along `.d` axis.
    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const normalized = zml.nn.rmsNorm(x, .d, self.eps);
        return normalized.mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }
};

/// Optional per-output-channel scales for FP8 checkpoint weights. Activations,
/// dot results and the rest of the model retain their existing dtype.
const Projection = struct {
    linear: zml.nn.Linear,
    weight_scale: ?zml.Tensor = null,

    fn maybeInit(store: zml.io.TensorStore.View, partitioning: anytype) ?Projection {
        const weight = store.maybeCreateTensor("weight", .{ .dout, .d }, partitioning) orelse return null;
        const scale = store.maybeCreateTensor("weight_scale", .{.dout}, .{ .dout = .model });
        if (scale) |value| {
            stdx.debug.assert(weight.dtype() == .f8e4m3fn and value.dtype() == .f32 and value.dim(.dout) == weight.dim(.dout), "Expected E4M3FN weights with an F32 scale per output channel", .{});
        }
        return .{ .linear = .init(weight, null, .d), .weight_scale = scale };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Projection)) void {
        zml.Buffer.deinitAll(Projection, self);
    }

    pub fn forward(self: Projection, input: zml.Tensor) zml.Tensor {
        const dtype = input.dtype();
        const result = self.linear.forward(input, dtype);
        const scale = self.weight_scale orelse return result;
        // The conversion utility uses power-of-two scales, so scaling the BF16
        // result does not add rounding except at the dtype's range boundaries.
        return result.convert(.f32).mul(scale.broad(result.shape().withDtype(.f32))).convert(dtype);
    }
};

const Mlp = struct {
    up_proj: Projection, // (dim -> hidden_dim)
    gate_proj: Projection, // (dim -> hidden_dim)
    down_proj: Projection, // (hidden_dim -> dim)

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .up_proj = Projection.maybeInit(store.withPrefix("up_proj"), .{ .dout = .model }).?,
            .gate_proj = Projection.maybeInit(store.withPrefix("gate_proj"), .{ .dout = .model }).?,
            .down_proj = Projection.maybeInit(store.withPrefix("down_proj"), .{ .d = .model }).?,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        zml.Buffer.deinitAll(Mlp, self);
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const proj = self.up_proj.forward(x);
        var output = self.gate_proj.forward(x);
        output = output.silu().mul(proj).rename(.{ .dout = .d });
        return self.down_proj.forward(output);
    }
};

const SelfAttn = struct {
    q_proj: Projection,
    k_proj: Projection,
    v_proj: Projection,

    q_norm: ?RmsNorm,
    k_norm: ?RmsNorm,

    o_proj: Projection,
    num_heads: i64 = undefined,
    num_kv_heads: i64 = 0,
    rope_opts: zml.nn.RopeOpts = undefined,

    pub fn init(store: zml.io.TensorStore.View, config: Config) !SelfAttn {
        var rope_scaling = config.rope_scaling;
        rope_scaling.setRopeTheta(config.rope_theta);
        return .{
            .q_proj = Projection.maybeInit(store.withPrefix("q_proj"), .{ .dout = .model }).?,
            .k_proj = Projection.maybeInit(store.withPrefix("k_proj"), .{ .dout = .model }).?,
            .v_proj = Projection.maybeInit(store.withPrefix("v_proj"), .{ .dout = .model }).?,
            .o_proj = Projection.maybeInit(store.withPrefix("o_proj"), .{ .d = .model }).?,
            // TODO(Corentin): fix that
            .q_norm = null,
            .k_norm = null,
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
            .rope_opts = .{
                .layout = if (config.hf_rope_impl) .real_im_pass else .interleaved,
                .scaling = rope_scaling,
            },
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SelfAttn)) void {
        zml.Buffer.deinitAll(SelfAttn, self);
    }

    /// Self zml.attention.
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
        kv_cache_index: zml.Tensor,
        attention_metadata: zml.attention.Metadata,
        attention_parameters: zml.attention.Parameters,
    ) struct { zml.Tensor, KvCache } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;

        // Make hidden state replicated once and reuse it across q/k/v projections.
        // This avoids paying gather-style collectives independently for each projection.
        const x_qkv = x.withPartitioning(.{ .d = .replicated });

        var q = self.q_proj.forward(x_qkv).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = self.k_proj.forward(x_qkv).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        var v = self.v_proj.forward(x_qkv).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });

        // In self-attention, .s axis is used both for keys and queries.
        const pos_index = b: {
            const temp = zml.Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };

        if (self.q_norm) |norm| q = norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        if (self.k_norm) |norm| k = norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        var rope_opts = self.rope_opts;
        if (zml.Compiler.current().platform.target == .furiosa) {
            // Valid query positions fit the KV cache. Reuse precomputed
            // rotations instead of reducing sine/cosine arguments every layer.
            rope_opts.cache_length = @intCast(kv_cache.k.dim(.k));
        }
        q = zml.nn.rope(q, pos_index, rope_opts);
        k = zml.nn.rope(k, pos_index, rope_opts);
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const dtype = q.dtype();
        const new_kv_cache = kv_cache.updateAt(k, v, token_index, kv_cache_index);
        k = new_kv_cache.keysAt(kv_cache_index).convert(dtype);
        v = new_kv_cache.valuesAt(kv_cache_index).convert(dtype);

        const layer_attention_metadata: zml.attention.Metadata = switch (attention_parameters) {
            .attnd => .{ .attnd = .{
                .layer_id = kv_cache_index.convert(.u16),
                .conversation_id = attention_metadata.attnd.conversation_id,
                .num_tokens = attention_metadata.attnd.num_tokens,
            } },
            .vanilla => attention_metadata,
            .cuda_fa2 => attention_metadata,
            .cuda_fa3 => attention_metadata,
            .nki => attention_metadata,
            .metal_fa => attention_metadata,
        };

        const attn_output = zml.attention.attention(
            q,
            k,
            v,
            token_index,
            layer_attention_metadata,
            attention_parameters,
        );

        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const delta = self.o_proj.forward(attn)
            .rename(.{ .dout = .d })
            .withPartitioning(.{ .d = .replicated });
        return .{ delta, new_kv_cache };
    }
};

pub const KvCache = struct {
    k: zml.Tensor,
    v: zml.Tensor,

    pub const Buffer = zml.Bufferized(KvCache);

    pub fn init(kv_shape: zml.Shape) KvCache {
        const sharded_shape = kv_shape.withPartitioning(.{ .h = .model });

        return .{
            .k = .fromShape(sharded_shape),
            .v = .fromShape(sharded_shape),
        };
    }

    pub fn initBuffer(kv: KvCache, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !Buffer {
        return .{
            .k = try zml.Buffer.uninitialized(io, platform, kv.k.shape(), sharding, .{}),
            .v = try zml.Buffer.uninitialized(io, platform, kv.v.shape(), sharding, .{}),
        };
    }

    pub fn deinitBuffer(kv: *Buffer) void {
        kv.k.deinit();
        kv.v.deinit();
    }

    pub fn keysAt(kv: KvCache, layer_index: zml.Tensor) zml.Tensor {
        return kv.k.slice(.layer, .dynSingle(layer_index));
    }

    pub fn valuesAt(kv: KvCache, layer_index: zml.Tensor) zml.Tensor {
        return kv.v.slice(.layer, .dynSingle(layer_index));
    }

    pub fn updateAt(kv: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: zml.Tensor, layer_index: zml.Tensor) KvCache {
        const k_shape = kv.k.shape().drop(.layer);
        return .{
            .k = kv.k.scatterSlices(.{ .layer = layer_index, .k = token_index }, new_k.convert(kv.k.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.k),
            .v = kv.v.scatterSlices(.{ .layer = layer_index, .k = token_index }, new_v.convert(kv.v.dtype()).transpose(kv.v.shape().drop(.layer)), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.v),
        };
    }

    pub fn reuseBuffer(kv: KvCache, other: KvCache) KvCache {
        return .{
            .k = kv.k.reuseBuffer(other.k),
            .v = kv.v.reuseBuffer(other.v),
        };
    }
};
