const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const common = @import("../common.zig");
const inference = @import("inference.zig");

const log = std.log.scoped(.llama);

pub const Config = struct {
    bos_token_id: u32,
    eos_token_id: stdx.json.Union(union(enum) {
        int: u32,
        ints: []u32,
    }),
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
    rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = .{} },
};

const Options = struct {
    sampling_strategy: ?zml.nn.SamplingStrategy,
    max_seq_len: u32,
};

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

        const options: Options = .{
            .sampling_strategy = generation.sampling_strategy,
            .max_seq_len = parsed_config.value.max_position_embeddings,
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
        errdefer self.unloadBuffers(&buffers, allocator);

        var loader: zml.io.Loader = try .init(allocator, platform, .{
            .dma_chunks = 32,
            .dma_chunk_size = 256 * zml.MiB,
            .parallelism = 16,
        });
        defer loader.deinit();

        const all_shardings = shardings.all();
        const load_opts: zml.io.Loader.LoadOpts = .{ .progress = progress };
        loader.load(io, @FieldType(Model, "lm_head"), &self.inner.lm_head, &buffers.lm_head, store, &all_shardings, load_opts);
        try self.inner.model.loadBuffers(io, &loader, &buffers.model, store, &all_shardings, load_opts);
        try loader.await(io);

        const took = now.untilNow(io, .awake);
        const total_bytes: u64 = loader.bytes_loaded.raw;
        const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
        log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });

        return buffers;
    }

    pub fn unloadBuffers(_: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        if (buffers.lm_head) |*lm_head| lm_head.weight.deinit();
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

        return inference.CompiledModel.init(allocator, io, platform, self, self.inner, params, progress);
    }
};

pub const Buffers = zml.Bufferized(Model);

/// Llama architecture, using huggingface transformers naming.
/// Dimensions of activations: {.b, .s, .d}
pub const Model = struct {
    lm_head: ?zml.nn.Linear,
    model: Llama,

    gen_opts: zml.nn.SamplingStrategy = .{},
    config: Config,

    pub fn init(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: Config,
        options: Options,
    ) !Model {
        const lm_head: ?zml.nn.Linear = if (store.withPrefix("lm_head").maybeCreateTensor(
            "weight",
            .{ .dout, .d },
            .{ .dout = .model, .d = .replicated },
        )) |weight|
            .init(weight, null, .d)
        else
            null;

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

        const new_tokens, const new_rng = self.lmHead().forward(out, tokens, rng);

        return .{ new_tokens, updated_kv_cache, new_rng };
    }
};

const Llama = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    norm: RmsNorm,
    layers: []TransformerLayer,

    pub const Buffers = zml.Bufferized(@This());

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

    pub fn loadBuffers(
        self: *const Llama,
        io: std.Io,
        loader: *zml.io.Loader,
        buffers: *Llama.Buffers,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        opts: zml.io.Loader.LoadOpts,
    ) !void {
        loader.load(io, @FieldType(Llama, "embed_tokens"), &self.embed_tokens, &buffers.embed_tokens, store, shardings, opts);
        loader.load(io, @FieldType(Llama, "norm"), &self.norm, &buffers.norm, store, shardings, opts);
        for (self.layers, buffers.layers) |*layer, *layer_buffers| {
            try layer.loadBuffers(io, loader, layer_buffers, store, shardings, opts);
        }
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
            hidden, updated_kv_cache, kv_cache_index = layer.forward(
                hidden,
                token_index,
                updated_kv_cache,
                kv_cache_index,
                attention_metadata,
                attention_parameters,
            );
        }

        return .{ self.norm.forward(hidden), updated_kv_cache.reuseBuffer(kv_cache) };
    }
};

pub const LmHead = struct {
    lm_head: ?zml.nn.Linear,
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

    pub fn forward(self: LmHead, hidden_: zml.Tensor, tokens_: zml.Tensor, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        const tokens = tokens_.withPartialTags(.{.s});
        const hidden = self.norm.forward(hidden_.withPartialTags(.{ .s, .d }));

        var logits = blk: {
            if (self.lm_head) |lm_head| {
                break :blk lm_head.forward(hidden).rename(.{ .dout = .d });
            } else {
                break :blk self.embed_tokens.weight.withTags(.{ .voc, .d }).dot(hidden, .d);
            }
        };

        if (logits.shape().hasTag(.voc) == null)
            logits = logits.rename(.{ .d = .voc });

        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, self.gen_opts, rng);
        return .{ next_tokens.convert(tokens.dtype()).reuseBuffer(tokens), new_rng };
    }
};

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: SelfAttn,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,

    pub const Buffers = zml.Bufferized(@This());

    pub fn init(store: zml.io.TensorStore.View, config: Config) !TransformerLayer {
        return .{
            .input_layernorm = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps),
            .self_attn = try .init(store.withPrefix("self_attn"), config),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps),
            .mlp = .init(store.withPrefix("mlp")),
        };
    }

    pub fn loadBuffers(
        self: *const TransformerLayer,
        io: std.Io,
        loader: *zml.io.Loader,
        buffers: *TransformerLayer.Buffers,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        opts: zml.io.Loader.LoadOpts,
    ) !void {
        loader.load(io, @FieldType(@This(), "input_layernorm"), &self.input_layernorm, &buffers.input_layernorm, store, shardings, opts);
        try self.self_attn.loadBuffers(io, loader, &buffers.self_attn, store, shardings, opts);
        loader.load(io, @FieldType(@This(), "post_attention_layernorm"), &self.post_attention_layernorm, &buffers.post_attention_layernorm, store, shardings, opts);
        loader.load(io, @FieldType(@This(), "mlp"), &self.mlp, &buffers.mlp, store, shardings, opts);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        SelfAttn.unloadBuffers(&self.self_attn);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
        Mlp.unloadBuffers(&self.mlp);
    }

    pub fn forward(
        self: TransformerLayer,
        x0: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        kv_cache_index: zml.Tensor,
        attention_metadata: zml.attention.Metadata,
        attention_parameters: zml.attention.Parameters,
    ) struct { zml.Tensor, KvCache, zml.Tensor } {
        // Self Attention
        //log.debug("TransformerLayer({f}) -> {f}", .{ x0, self.input_layernorm.forward(x0) });
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {f}", .{x0});

        // Keep the residual stream replicated to avoid repeated gathers before q/k/v.
        const x0_replicated = x0.withPartitioning(.{ .d = .replicated });
        const x0_normalized = self.input_layernorm.forward(x0_replicated);
        const delta0, const updated_kv_cache = self.self_attn.forward(
            x0_normalized,
            token_index,
            kv_cache,
            kv_cache_index,
            attention_metadata,
            attention_parameters,
        );
        const updated_kv_cache_index = kv_cache_index.add(zml.Tensor.scalar(@as(u32, 1), .u32));

        // Fully Connected
        const x1 = x0_replicated.add(delta0).withPartitioning(.{ .d = .replicated });
        const x1_normalized = self.post_attention_layernorm.forward(x1);
        const x2 = self.mlp.forward(x1_normalized)
            .rename(.{ .dout = .d })
            .withPartitioning(.{ .d = .replicated })
            .add(x1)
            .withPartitioning(.{ .d = .replicated });

        return .{ x2.reuseBuffer(x0), updated_kv_cache.reuseBuffer(kv_cache), updated_kv_cache_index.reuseBuffer(kv_cache_index) };
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

const Mlp = struct {
    up_proj: zml.nn.Linear, // (dim -> hidden_dim)
    gate_proj: zml.nn.Linear, // (dim -> hidden_dim)
    down_proj: zml.nn.Linear, // (hidden_dim -> dim)

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .up_proj = .init(store.createTensor("up_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .gate_proj = .init(store.createTensor("gate_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .down_proj = .init(store.createTensor("down_proj.weight", .{ .dout, .d }, .{ .d = .model }), null, .d),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        self.up_proj.weight.deinit();
        if (self.up_proj.bias) |*bias| bias.deinit();
        self.gate_proj.weight.deinit();
        if (self.gate_proj.bias) |*bias| bias.deinit();
        self.down_proj.weight.deinit();
        if (self.down_proj.bias) |*bias| bias.deinit();
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

    q_norm: ?RmsNorm,
    k_norm: ?RmsNorm,

    qk_hadamar: zml.Tensor,
    o_proj: zml.nn.Linear,
    num_heads: i64 = undefined,
    num_kv_heads: i64 = 0,
    rope_opts: zml.nn.RopeOpts = undefined,

    pub const Buffers = zml.Bufferized(@This());

    pub fn init(store: zml.io.TensorStore.View, config: Config) !SelfAttn {
        var rope_scaling = config.rope_scaling;
        rope_scaling.setRopeTheta(config.rope_theta);
        const q_proj = store.createTensor("q_proj.weight", .{ .dout, .d }, .{ .dout = .model });
        const hd = @divExact(q_proj.dim(.dout), config.num_attention_heads);
        return .{
            .q_proj = .init(q_proj, null, .d),
            .k_proj = .init(store.createTensor("k_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .v_proj = .init(store.createTensor("v_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .o_proj = .init(store.createTensor("o_proj.weight", .{ .dout, .d }, .{ .d = .model }), null, .d),
            .qk_hadamar = store.maybeCreateBinding(&.{}, .init(.{ .dout = hd, .hd = hd }, .i8)).?,

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

    pub fn loadBuffers(
        self: *const SelfAttn,
        io: std.Io,
        loader: *zml.io.Loader,
        buffers: *SelfAttn.Buffers,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        opts: zml.io.Loader.LoadOpts,
    ) !void {
        loader.load(io, zml.nn.Linear, &self.q_proj, &buffers.q_proj, store, shardings, opts);
        loader.load(io, zml.nn.Linear, &self.k_proj, &buffers.k_proj, store, shardings, opts);
        loader.load(io, zml.nn.Linear, &self.v_proj, &buffers.v_proj, store, shardings, opts);
        loader.load(io, zml.nn.Linear, &self.o_proj, &buffers.o_proj, store, shardings, opts);
        loader.load(io, ?RmsNorm, &self.q_norm, &buffers.q_norm, store, shardings, opts);
        loader.load(io, ?RmsNorm, &self.k_norm, &buffers.k_norm, store, shardings, opts);

        const hd: i64 = @divFloor(self.q_proj.weight.dim(.d), self.num_heads);
        const exe: *const zml.Exe = try getHadamarExe(io, loader.platform, hd);
        var arena: std.heap.ArenaAllocator = .init(loader.allocator);
        defer arena.deinit();
        try loader.loadExecute(arena.allocator(), io, self.qk_hadamar, &buffers.qk_hadamar, store, shardings, exe, opts);
    }

    var hadamar_exe_lock: std.Io.RwLock = .init;
    var hadamar_exe_hd: i64 = 0;
    var hadamar_exe: ?zml.Exe = null;

    fn getHadamarExe(io: std.Io, platform: *const zml.Platform, hd: i64) !*const zml.Exe {
        hadamar_exe_lock.lockUncancelable(io);
        defer hadamar_exe_lock.unlock(io);
        if (hadamar_exe) |*exe| {
            if (hadamar_exe_hd == hd) return exe;

            exe.deinit();
        }

        // Since hadamar_exe is a global, use a global allocator for its data.
        var h_exe = io.async(zml.Compiler(zml.nn.walshHadamard).compile, .{ std.heap.page_allocator, io, platform, .{}, .{@intCast(hd)} });
        hadamar_exe = try h_exe.await(io);
        hadamar_exe_hd = hd;
        log.warn("Compiled hadamar generator: {f}", .{hadamar_exe.?});
        return &(hadamar_exe.?);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SelfAttn)) void {
        self.q_proj.weight.deinit();
        if (self.q_proj.bias) |*bias| bias.deinit();
        self.k_proj.weight.deinit();
        if (self.k_proj.bias) |*bias| bias.deinit();
        self.v_proj.weight.deinit();
        if (self.v_proj.bias) |*bias| bias.deinit();
        self.o_proj.weight.deinit();
        if (self.o_proj.bias) |*bias| bias.deinit();

        if (self.q_norm) |*q_norm| RmsNorm.unloadBuffers(q_norm);
        if (self.k_norm) |*k_norm| RmsNorm.unloadBuffers(k_norm);
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

        const dtype = q.dtype();
        if (self.q_norm) |norm| q = norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        if (self.k_norm) |norm| k = norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        q = q.dot(self.qk_hadamar.convert(dtype), .hd).divByConst(q.dim(.hd)).rename(.{ .dout = .hd });
        k = k.dot(self.qk_hadamar.convert(dtype), .hd).rename(.{ .dout = .hd });

        const new_kv_cache = kv_cache.updateAt(k, v, token_index, kv_cache_index);
        k = new_kv_cache.keysAt(kv_cache_index);
        v = new_kv_cache.valuesAt(kv_cache_index);

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
    k_scale: zml.Tensor,

    pub const Buffer = zml.Bufferized(KvCache);

    pub fn init(kv_shape: zml.Shape) KvCache {
        const sharded_shape = kv_shape.withPartitioning(.{ .h = .model });

        return .{
            .k = .fromShape(sharded_shape.withDtype(.i8)),
            .k_scale = .fromShape(sharded_shape.setDim(.hd, 1)),
            .v = .fromShape(sharded_shape),
        };
    }

    pub fn initBuffer(kv: KvCache, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !Buffer {
        return .{
            .k = try .uninitialized(io, platform, kv.k.shape(), sharding, .{}),
            .k_scale = try .uninitialized(io, platform, kv.k_scale.shape(), sharding, .{}),
            .v = try .uninitialized(io, platform, kv.v.shape(), sharding, .{}),
        };
    }

    pub fn deinitBuffer(kv: *Buffer) void {
        kv.k.deinit();
        kv.k_scale.deinit();
        kv.v.deinit();
    }

    pub fn keysAt(kv: KvCache, layer_index: zml.Tensor) zml.Tensor {
        const k_scale = kv.k_scale.slice2(.layer, .single(layer_index));
        const k_quant = kv.k.slice2(.layer, .single(layer_index));

        return k_quant.convert(k_scale.dtype()).mul(k_scale);
    }

    pub fn valuesAt(kv: KvCache, layer_index: zml.Tensor) zml.Tensor {
        return kv.v.slice2(.layer, .single(layer_index));
    }

    pub fn updateAt(kv: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: zml.Tensor, layer_index: zml.Tensor) KvCache {
        const scatter_opts: zml.Tensor.ScatterOpts = .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override };

        const k_shape = kv.k.shape().drop(.layer);
        const k_scale = new_k.abs().max(.hd).divByConst(128.0).transpose(k_shape);
        const k = new_k.transpose(k_shape).div(k_scale).convert(kv.k.dtype());
        const v = new_v.convert(kv.v.dtype()).transpose(kv.v.shape().drop(.layer));
        return .reuseBuffer(
            .{
                .k = .scatterSlices(kv.k, .{ .layer = layer_index, .k = token_index }, k, scatter_opts),
                .k_scale = .scatterSlices(kv.k_scale, .{ .layer = layer_index, .k = token_index }, k_scale, scatter_opts),
                .v = .scatterSlices(kv.v, .{ .layer = layer_index, .k = token_index }, v, scatter_opts),
            },
            kv,
        );
    }

    pub fn reuseBuffer(kv: KvCache, source: KvCache) KvCache {
        return .{
            .k = .reuseBuffer(kv.k, source.k),
            .k_scale = .reuseBuffer(kv.k_scale, source.k_scale),
            .v = .reuseBuffer(kv.v, source.v),
        };
    }
};
