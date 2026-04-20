const std = @import("std");

const zml = @import("zml");
const CompilationContext = zml.module.CompilationContext;

const stdx = zml.stdx;

const common = @import("../common.zig");
const inference = @import("inference.zig");

const log = std.log.scoped(.qwen3_5);

const LoadCtx = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *const zml.io.TensorStore,
    shardings: []const zml.sharding.Sharding,
    replicated_sharding: zml.sharding.Sharding,
    dma_allocators: []zml.mem.DmaAllocator,
    pinned_buffer_pools: []zml.mem.DynamicBufferPool,
    total_bytes: usize = 0,

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !LoadCtx {
        const dma_allocators = try allocator.alloc(zml.mem.DmaAllocator, platform.devices.len);
        errdefer allocator.free(dma_allocators);
        for (platform.devices, 0..) |*device, i| {
            dma_allocators[i] = .init(allocator, device);
        }

        const pinned_buffer_pools = try allocator.alloc(zml.mem.DynamicBufferPool, platform.devices.len);
        errdefer allocator.free(pinned_buffer_pools);
        for (pinned_buffer_pools) |*pool| {
            pool.* = .init(32, 128 * zml.MiB);
        }

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .store = store,
            .shardings = shardings,
            .replicated_sharding = try zml.sharding.replicatedSharding(platform),
            .dma_allocators = dma_allocators,
            .pinned_buffer_pools = pinned_buffer_pools,
        };
    }

    fn deinit(self: *LoadCtx) void {
        for (self.pinned_buffer_pools, 0..) |*pool, i| {
            pool.deinit(self.dma_allocators[i].allocator());
        }
        self.allocator.free(self.pinned_buffer_pools);
        self.allocator.free(self.dma_allocators);
    }

    fn loadBuf(self: *LoadCtx, tensor: *const zml.Tensor, progress: ?*std.Progress.Node) !zml.Buffer {
        var reader = try self.store.getReaderById(tensor.id, self.io, &.{});
        defer reader.deinit();

        const shape = reader.tensor.shape;
        const sharding = zml.sharding.pickSharding(self.shardings, shape, .explicit_axis_binding) orelse self.replicated_sharding;

        var buffer: zml.Buffer = undefined;
        var writer = try zml.io.MemoryWriter.init(
            self.allocator,
            self.io,
            self.platform,
            self.pinned_buffer_pools,
            self.dma_allocators,
            shape,
            sharding,
            &buffer,
        );
        defer writer.deinit(self.allocator);

        const scale = 1024;
        if (progress) |p| {
            var node = p.start(reader.tensor.name, reader.tensor.shape.byteSize() / scale);
            defer node.end();
            writer.setProgress(&node);
            defer writer.setProgress(null);
            var progress_writer: zml.io.ProgressWriter = .init(writer.interface(), &node, .{ .scale = scale });
            const total = try reader.interface.streamRemaining(&progress_writer.interface);
            try progress_writer.interface.flush();
            self.total_bytes += total;
        } else {
            const total = try reader.interface.streamRemaining(writer.interface());
            try writer.interface().flush();
            self.total_bytes += total;
        }

        return buffer;
    }

    fn loadOptionalBuf(self: *LoadCtx, tensor: ?zml.Tensor, progress: ?*std.Progress.Node) !?zml.Buffer {
        if (tensor) |*t| {
            return try self.loadBuf(t, progress);
        }
        return null;
    }

    fn loadLinearBuf(
        self: *LoadCtx,
        linear: *const zml.nn.Linear,
        progress: ?*std.Progress.Node,
    ) !zml.Bufferized(zml.nn.Linear) {
        var weight = try self.loadBuf(&linear.weight, progress);
        errdefer weight.deinit();

        var bias: ?zml.Buffer = null;
        errdefer if (bias) |*b| b.deinit();
        if (linear.bias) |*tensor| {
            bias = try self.loadBuf(tensor, progress);
        }

        return .{
            .weight = weight,
            .bias = bias,
        };
    }
};

pub const Config = struct {
    text_config: TextConfig,
};

pub const TextConfig = struct {
    // General
    num_hidden_layers: i64,
    layer_types: []const LayerType,
    hidden_size: i64,
    max_position_embeddings: i64,
    rms_norm_eps: f32,
    // Self attention
    head_dim: i64,
    num_attention_heads: i64,
    num_key_value_heads: i64,
    rope_parameters: RopeParameters,
    // Linear attention
    linear_conv_kernel_dim: i64,
    linear_key_head_dim: i64,
    linear_num_key_heads: i64,
    linear_num_value_heads: i64,
    linear_value_head_dim: i64,
    // MoE
    num_experts: ?i64 = null,
    num_experts_per_tok: ?u32 = null,
};

// Each layer uses either: full attention (SelfAttn) or linear attention (GatedDeltaNet).
pub const LayerType = enum {
    linear_attention,
    full_attention,
};

pub const RopeParameters = struct {
    mrope_section: [3]i64,
    partial_rotary_factor: f32,
    rope_theta: f32,
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

        const options: Model.GenOptions = .{
            .sampling_strategy = generation.sampling_strategy,
            .max_seq_len = parsed_config.value.text_config.max_position_embeddings,
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
        shardings: common.Shardings,
    ) !Buffers {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);
        const all_shardings = shardings.all();
        var ctx = try LoadCtx.init(allocator, io, platform, store, &all_shardings);
        defer ctx.deinit();
        defer {
            const took = now.untilNow(io, .awake);
            const bytes_per_sec: u64 = @intFromFloat(
                @as(f64, @floatFromInt(ctx.total_bytes)) /
                    (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s),
            );
            log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ ctx.total_bytes, took, bytes_per_sec });
        }

        return try self.inner.loadBuffers(&ctx, store.view(), progress);
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
        const moe_dtype = self.inner.text_model.layers[0].moe.gate_up_proj.dtype();
        log.info("Moe dtype : {}", .{moe_dtype});
        const moe_backend = try zml.moe.Backend.auto(platform, moe_dtype);
        const params = inference.CompilationParameters.init(self.inner, self.parsed_config.value, @intCast(seqlen), moe_backend, shardings);
        return inference.CompiledModel.init(allocator, io, platform, self, self.inner, params, progress);
    }
};

const MoeRepacker = struct {
    fn packExperts(tensors: []const zml.Tensor) zml.Tensor {
        return Moe.repackExperts(tensors, 0, .expert);
    }

    fn packGateUp(gate_tensors: []const zml.Tensor, up_tensors: []const zml.Tensor) zml.Tensor {
        const gate_proj = Moe.repackExperts(gate_tensors, 0, .expert);
        const up_proj = Moe.repackExperts(up_tensors, 0, .expert);
        return zml.Tensor.concatenate(&.{ gate_proj, up_proj }, .dout);
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

    text_model: TextModel,

    config: Config,
    special_tokens: SpecialTokens = .{
        .im_start_token_id = 248045,
        .im_end_token_id = 248046,
        .end_of_text_token_id = 248044,
    },

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, gen_options: GenOptions) !Model {
        // For some Qwen3.5 versions, the output projection lm_head has a standalone weight tensor, while for others it's the same as the input embedding layer
        return .{
            .text_model = try .init(allocator, store.withPrefix("model.language_model"), config, gen_options),
            .config = config,
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
        shardings: []const zml.sharding.Sharding,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(Model) {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);
        var total_bytes: usize = 0;
        defer {
            const took = now.untilNow(io, .awake);
            const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
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

    pub fn loadBuffers(
        self: *Model,
        ctx: *LoadCtx,
        store: zml.io.TensorStore.View,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(Model) {
        return .{
            .text_model = try self.text_model.loadBuffers(ctx, store.withPrefix("model.language_model"), progress),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model), allocator: std.mem.Allocator) void {
        TextModel.unloadBuffers(&self.text_model, allocator);
    }
    pub fn forward(
        self: Model,
        tokens_: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        rng: zml.Tensor.Rng,
        moe_metadata: zml.moe.Metadata,
        moe_parameters: zml.moe.Parameters,
    ) struct { zml.Tensor, KvCache, zml.Tensor.Rng } {
        const tokens = tokens_.withPartialTags(.{.s});
        const new_tokens, const updated_kv_cache, const new_rng = self.text_model.forward(tokens, token_index, kv_cache, self.config, rng, moe_metadata, moe_parameters);
        return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), updated_kv_cache, new_rng };
    }
};

//========================Text model========================

// Use this intermediate struct to compile sampleTokens without taking the whole TextModel as input
pub const Sampler = struct {
    norm: RmsNorm,
    lm_head: zml.nn.Linear,
    gen_options: Model.GenOptions,

    pub fn sampleTokens(
        self: Sampler,
        out: zml.Tensor,
        rng: zml.Tensor.Rng,
        token_index: ?zml.Tensor,
    ) struct { zml.Tensor, zml.Tensor.Rng, ?zml.Tensor } {
        const x = self.norm.forward(out);
        const logits = self.lm_head.forward(x.withPartialTags(.{.d})).rename(.{ .dout = .voc });
        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, self.gen_options.sampling_strategy, rng);
        if (token_index) |token_idx| {
            return .{ next_tokens.convert(.u32), new_rng, token_idx.addConstant(1) };
        }
        return .{ next_tokens.convert(.u32), new_rng, null };
    }
};

pub const TextModel = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    norm: RmsNorm,
    lm_head: zml.nn.Linear,
    gen_options: Model.GenOptions,

    pub fn init(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: Config,
        gen_options: Model.GenOptions,
    ) !TextModel {
        const lm_head_prefix = if (store.root().hasKey("lm_head.weight")) "lm_head" else "model.language_model.embed_tokens";

        const layers = try allocator.alloc(TransformerLayer, @intCast(config.text_config.num_hidden_layers));
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            errdefer for (layers[0..i]) |previous_layer| previous_layer.deinit(allocator);
            layer.* = try .init(allocator, store.withPrefix("layers").withLayer(i), config, i);
        }

        return .{
            .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, .{ .voc = .replicated, .d = .model }) },
            .layers = layers,
            .norm = RmsNorm.init(store.withPrefix("norm"), config.text_config.rms_norm_eps),
            .lm_head = .init(store.root().withPrefix(lm_head_prefix).createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }), null, .d),
            .gen_options = gen_options,
        };
    }

    pub fn deinit(self: TextModel, allocator: std.mem.Allocator) void {
        for (self.layers) |layer| {
            layer.deinit(allocator);
        }
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TextModel), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer, allocator);
        }
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
        self.lm_head.weight.deinit();
    }

    pub fn loadBuffers(
        self: *TextModel,
        ctx: *LoadCtx,
        store: zml.io.TensorStore.View,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(TextModel) {
        var embed_tokens: zml.Bufferized(zml.nn.TokenEmbedding) = .{
            .weight = try ctx.loadBuf(&self.embed_tokens.weight, progress),
        };
        errdefer embed_tokens.weight.deinit();

        const layers = try ctx.allocator.alloc(zml.Bufferized(TransformerLayer), self.layers.len);
        errdefer ctx.allocator.free(layers);
        var loaded_layers: usize = 0;
        errdefer for (layers[0..loaded_layers]) |*layer| {
            TransformerLayer.unloadBuffers(layer, ctx.allocator);
        };

        for (self.layers, 0..) |*layer, i| {
            layers[i] = try layer.loadBuffers(ctx, store.withPrefix("layers").withLayer(i), progress);
            loaded_layers = i + 1;
        }

        var norm = try self.norm.loadBuffers(ctx, store.withPrefix("norm"), progress);
        errdefer RmsNorm.unloadBuffers(&norm);

        var lm_head = try ctx.loadLinearBuf(&self.lm_head, progress);
        errdefer {
            lm_head.weight.deinit();
            if (lm_head.bias) |*bias| bias.deinit();
        }

        return .{
            .embed_tokens = embed_tokens,
            .layers = layers,
            .norm = norm,
            .lm_head = lm_head,
        };
    }

    pub fn sampler(self: TextModel) Sampler {
        return .{
            .norm = self.norm,
            .lm_head = self.lm_head,
            .gen_options = self.gen_options,
        };
    }

    pub fn forward(
        self: TextModel,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        config: Config,
        rng: zml.Tensor.Rng,
        moe_metadata: zml.moe.Metadata,
        moe_parameters: zml.moe.Parameters,
    ) struct { zml.Tensor, KvCache, zml.Tensor.Rng } {
        var hidden_states = self.embed_tokens.weight.gather(.{ .voc = tokens }, .{});

        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden_states, updated_kv_cache = layer.forward(hidden_states, token_index, updated_kv_cache.atLayer(i), config, moe_metadata, moe_parameters);
        }

        const new_tokens, const new_rng = self.sampleTokens(hidden_states, rng);
        return .{ new_tokens, updated_kv_cache.reuseBuffer(kv_cache), new_rng };
    }

    pub fn sampleTokens(
        self: TextModel,
        out: zml.Tensor,
        rng: zml.Tensor.Rng,
        token_index: ?zml.Tensor,
    ) struct { zml.Tensor, zml.Tensor.Rng, ?zml.Tensor } {
        return self.sampler().sampleTokens(out, rng, token_index);
    }
};

pub const TransformerLayer = struct {
    const Attn = union(enum) {
        self_attn: SelfAttn,
        linear_attn: GatedDeltaNet,
    };

    input_layernorm: RmsNorm,
    attn: Attn,
    moe: Moe,
    post_attention_layernorm: RmsNorm,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, layer_index: usize) !TransformerLayer {
        const is_full_attention = config.text_config.layer_types[layer_index] == .full_attention;
        return .{
            .input_layernorm = RmsNorm.init(store.withPrefix("input_layernorm"), config.text_config.rms_norm_eps),
            .attn = if (is_full_attention)
                .{ .self_attn = try .init(store.withPrefix("self_attn"), config) }
            else
                .{ .linear_attn = .init(store.withPrefix("linear_attn"), config) },
            .moe = try .init(allocator, store.withPrefix("mlp"), config),
            .post_attention_layernorm = RmsNorm.init(store.withPrefix("post_attention_layernorm"), config.text_config.rms_norm_eps),
        };
    }

    pub fn deinit(self: TransformerLayer, allocator: std.mem.Allocator) void {
        self.moe.deinit(allocator);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer), allocator: std.mem.Allocator) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        switch (self.attn) {
            .self_attn => |*self_attn| SelfAttn.unloadBuffers(self_attn),
            .linear_attn => |*linear_attn| GatedDeltaNet.unloadBuffers(linear_attn),
        }
        Moe.unloadBuffers(&self.moe, allocator);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
    }

    pub fn loadBuffers(
        self: *TransformerLayer,
        ctx: *LoadCtx,
        store: zml.io.TensorStore.View,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(TransformerLayer) {
        var input_layernorm = try self.input_layernorm.loadBuffers(ctx, store.withPrefix("input_layernorm"), progress);
        errdefer RmsNorm.unloadBuffers(&input_layernorm);

        var attn: zml.Bufferized(Attn) = undefined;
        switch (self.attn) {
            .self_attn => |*self_attn| {
                attn = .{ .self_attn = try self_attn.loadBuffers(ctx, store.withPrefix("self_attn"), progress) };
            },
            .linear_attn => |*linear_attn| {
                attn = .{ .linear_attn = try linear_attn.loadBuffers(ctx, store.withPrefix("linear_attn"), progress) };
            },
        }
        errdefer switch (attn) {
            .self_attn => |*self_attn| SelfAttn.unloadBuffers(self_attn),
            .linear_attn => |*linear_attn| GatedDeltaNet.unloadBuffers(linear_attn),
        };

        var moe = try self.moe.loadBuffers(ctx, store.withPrefix("mlp"), progress);
        errdefer Moe.unloadBuffers(&moe, ctx.allocator);

        var post_attention_layernorm = try self.post_attention_layernorm.loadBuffers(ctx, store.withPrefix("post_attention_layernorm"), progress);
        errdefer RmsNorm.unloadBuffers(&post_attention_layernorm);

        return .{
            .input_layernorm = input_layernorm,
            .attn = attn,
            .moe = moe,
            .post_attention_layernorm = post_attention_layernorm,
        };
    }

    pub fn forwardSelfAttn(
        self: TransformerLayer,
        x0: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache.SelfAttnCache,
        config: Config,
        moe_metadata: zml.moe.Metadata,
        moe_parameters: zml.moe.Parameters,
    ) struct { zml.Tensor, KvCache.SelfAttnCache } {
        _ = config;
        const residual0 = x0;
        const normalized_x0 = self.input_layernorm.forward(x0);

        const self_attn = switch (self.attn) {
            .self_attn => |self_attn| self_attn,
            .linear_attn => unreachable,
        };
        const attention_output, const updated_kv_cache = self_attn.forward(normalized_x0, token_index, kv_cache);

        const x1 = attention_output.add(residual0);
        const residual1 = x1;
        const normalized_hidden = self.post_attention_layernorm.forward(x1);

        const moe_output = self.moe.forward(normalized_hidden, moe_metadata, moe_parameters);

        return .{ moe_output.add(residual1).reuseBuffer(x0), updated_kv_cache };
    }

    pub fn forwardLinearAttn(
        self: TransformerLayer,
        x0: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache.GatedDeltaNetCache,
        config: Config,
        moe_metadata: zml.moe.Metadata,
        moe_parameters: zml.moe.Parameters,
    ) struct { zml.Tensor, KvCache.GatedDeltaNetCache } {
        _ = config;
        _ = token_index;
        const residual0 = x0;
        const normalized_x0 = self.input_layernorm.forward(x0);

        const linear_attn = switch (self.attn) {
            .linear_attn => |linear_attn| linear_attn,
            .self_attn => unreachable,
        };
        const attention_output, const updated_kv_cache = linear_attn.forward(normalized_x0, kv_cache);

        const x1 = attention_output.add(residual0);
        const residual1 = x1;
        const normalized_hidden = self.post_attention_layernorm.forward(x1);

        const moe_output = self.moe.forward(normalized_hidden, moe_metadata, moe_parameters);

        return .{ moe_output.add(residual1).reuseBuffer(x0), updated_kv_cache };
    }

    pub fn forward(
        self: TransformerLayer,
        x0: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache.LayerView,
        config: Config,
        moe_metadata: zml.moe.Metadata,
        moe_parameters: zml.moe.Parameters,
    ) struct { zml.Tensor, KvCache } {
        _ = config;
        const residual0 = x0;
        const normalized_x0 = self.input_layernorm.forward(x0);

        var attention_output: zml.Tensor = undefined;
        var updated_kv_cache: KvCache = kv_cache.parent;
        switch (self.attn) {
            .self_attn => |*self_attn| {
                const result = self_attn.forward(normalized_x0, token_index, kv_cache.cache.self_attn);
                attention_output = result[0];
                updated_kv_cache.self_attn = result[1];
            },
            .linear_attn => |*linear_attn| {
                const result = linear_attn.forward(normalized_x0, kv_cache.cache.linear_attn);
                attention_output = result[0];
                updated_kv_cache.gated_delta_net = result[1];
            },
        }

        const x1 = attention_output.add(residual0);
        const residual1 = x1;
        const normalized_hidden = self.post_attention_layernorm.forward(x1);

        const moe_output = self.moe.forward(normalized_hidden, moe_metadata, moe_parameters);

        return .{ moe_output.add(residual1), updated_kv_cache };
    }
};

pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    q_proj_scale: ?zml.Tensor,
    k_proj: zml.nn.Linear,
    k_proj_scale: ?zml.Tensor,
    v_proj: zml.nn.Linear,
    v_proj_scale: ?zml.Tensor,

    q_norm: RmsNorm,
    k_norm: RmsNorm,

    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    rotary_dim: i64,
    rotary_embed: TextRotaryEmbedding,
    o_proj: zml.nn.Linear,

    o_proj_scale: ?zml.Tensor,

    fn initProj(store: zml.io.TensorStore.View, partitions: anytype, bias_partitions: anytype) zml.nn.Linear {
        return .init(
            store.createTensor("weight", .{ .dout, .d }, partitions),
            store.maybeCreateTensor("bias", .{.dout}, bias_partitions),
            .d,
        );
    }

    pub fn init(store: zml.io.TensorStore.View, config: Config) !SelfAttn {
        const rotary_dim: i64 = @intFromFloat(@as(f32, @floatFromInt(config.text_config.head_dim)) *
            config.text_config.rope_parameters.partial_rotary_factor);
        return .{
            .q_proj = initProj(store.withPrefix("q_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .q_proj_scale = store.withPrefix("q_proj").maybeCreateTensor("weight_scale_inv", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
            .k_proj = initProj(store.withPrefix("k_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .k_proj_scale = store.withPrefix("k_proj").maybeCreateTensor("weight_scale_inv", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
            .v_proj = initProj(store.withPrefix("v_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .v_proj_scale = store.withPrefix("v_proj").maybeCreateTensor("weight_scale_inv", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
            .o_proj = initProj(store.withPrefix("o_proj"), .{ .dout = .replicated, .d = .model }, .{ .dout = .replicated }),
            .o_proj_scale = store.withPrefix("o_proj").maybeCreateTensor("weight_scale_inv", .{ .dout, .d }, .{ .dout = .replicated, .d = .model }),
            .q_norm = RmsNorm.init(store.withPrefix("q_norm"), config.text_config.rms_norm_eps),
            .k_norm = RmsNorm.init(store.withPrefix("k_norm"), config.text_config.rms_norm_eps),
            .num_heads = config.text_config.num_attention_heads,
            .num_kv_heads = config.text_config.num_key_value_heads,
            .head_dim = config.text_config.head_dim,
            .rotary_dim = rotary_dim,
            .rotary_embed = .init(rotary_dim, config.text_config.rope_parameters.rope_theta, config.text_config.rope_parameters.mrope_section),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SelfAttn)) void {
        self.q_proj.weight.deinit();
        if (self.q_proj.bias) |*bias| bias.deinit();
        if (self.q_proj_scale) |*scale| scale.deinit();
        self.k_proj.weight.deinit();
        if (self.k_proj.bias) |*bias| bias.deinit();
        if (self.k_proj_scale) |*scale| scale.deinit();
        self.v_proj.weight.deinit();
        if (self.v_proj.bias) |*bias| bias.deinit();
        if (self.v_proj_scale) |*scale| scale.deinit();
        self.o_proj.weight.deinit();
        if (self.o_proj.bias) |*bias| bias.deinit();
        if (self.o_proj_scale) |*scale| scale.deinit();
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    pub fn loadBuffers(
        self: *SelfAttn,
        ctx: *LoadCtx,
        store: zml.io.TensorStore.View,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(SelfAttn) {
        var q_proj = try ctx.loadLinearBuf(&self.q_proj, progress);
        errdefer {
            q_proj.weight.deinit();
            if (q_proj.bias) |*bias| bias.deinit();
        }
        var k_proj = try ctx.loadLinearBuf(&self.k_proj, progress);
        errdefer {
            k_proj.weight.deinit();
            if (k_proj.bias) |*bias| bias.deinit();
        }
        var v_proj = try ctx.loadLinearBuf(&self.v_proj, progress);
        errdefer {
            v_proj.weight.deinit();
            if (v_proj.bias) |*bias| bias.deinit();
        }
        var o_proj = try ctx.loadLinearBuf(&self.o_proj, progress);
        errdefer {
            o_proj.weight.deinit();
            if (o_proj.bias) |*bias| bias.deinit();
        }

        var q_proj_scale = try ctx.loadOptionalBuf(self.q_proj_scale, progress);
        errdefer if (q_proj_scale) |*buffer| buffer.deinit();
        var k_proj_scale = try ctx.loadOptionalBuf(self.k_proj_scale, progress);
        errdefer if (k_proj_scale) |*buffer| buffer.deinit();
        var v_proj_scale = try ctx.loadOptionalBuf(self.v_proj_scale, progress);
        errdefer if (v_proj_scale) |*buffer| buffer.deinit();
        var o_proj_scale = try ctx.loadOptionalBuf(self.o_proj_scale, progress);
        errdefer if (o_proj_scale) |*buffer| buffer.deinit();

        var q_norm = try self.q_norm.loadBuffers(ctx, store.withPrefix("q_norm"), progress);
        errdefer RmsNorm.unloadBuffers(&q_norm);
        var k_norm = try self.k_norm.loadBuffers(ctx, store.withPrefix("k_norm"), progress);
        errdefer RmsNorm.unloadBuffers(&k_norm);

        return .{
            .q_proj = q_proj,
            .q_proj_scale = q_proj_scale,
            .k_proj = k_proj,
            .k_proj_scale = k_proj_scale,
            .v_proj = v_proj,
            .v_proj_scale = v_proj_scale,
            .q_norm = q_norm,
            .k_norm = k_norm,
            .o_proj = o_proj,
            .o_proj_scale = o_proj_scale,
        };
    }

    fn projectQAndGate(self: SelfAttn, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const q_proj = forwardLinearDequantizeFp8(x, self.q_proj, self.q_proj_scale)
            .splitAxis(.dout, .{ .h = self.num_heads, .hd = 2 * self.head_dim });
        const q, var gate = q_proj.chunkExact(.hd, 2);
        gate = gate.merge(.{ .d_out_proj = .{ .h, .hd } });
        return .{ q, gate };
    }

    fn projectKV(self: SelfAttn, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;
        const k = forwardLinearDequantizeFp8(x, self.k_proj, self.k_proj_scale)
            .splitAxis(.dout, .{ .h = num_kv_heads, .hd = self.head_dim });
        const v = forwardLinearDequantizeFp8(x, self.v_proj, self.v_proj_scale)
            .splitAxis(.dout, .{ .h = num_kv_heads, .hd = self.head_dim });
        return .{ k, v };
    }

    pub fn forward(
        self: SelfAttn,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache.SelfAttnCache,
    ) struct { zml.Tensor, KvCache.SelfAttnCache } {
        const x_qkv = x.withPartitioning(.{ .d = .replicated });

        var q, var gate = self.projectQAndGate(x_qkv);
        var k, var v = self.projectKV(x_qkv);
        q = q.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        gate = gate.withPartitioning(.{ .s = .replicated, .d_out_proj = .model });
        k = k.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        const dtype = q.dtype();
        const position_ids = zml.Tensor.arange(.{ .end = x.dim(.s) }, .i64)
            .withTags(.{.s}).insertAxes(.s, .{.b}).broad(zml.Shape.init(.{ .b = x.dim(.b), .s = x.dim(.s) }, .i64))
            .add(token_index.convert(.i64).broad(zml.Shape.init(.{ .b = x.dim(.b), .s = x.dim(.s) }, .i64)));

        const cos, const sin = self.rotary_embed.getCosAndSin(position_ids, dtype);
        q = self.rotary_embed.applyRope(q, cos, sin);
        k = self.rotary_embed.applyRope(k, cos, sin);
        q = q.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        k = k.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });

        const new_kv_cache = kv_cache.update(k, v, token_index.convert(.u32));
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);
        q = q.rename(.{ .s = .q }).withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated });
        k = k.rename(.{ .s = .k }).withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });
        v = v.rename(.{ .s = .k }).withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });

        const attn_output = zml.attention.attention.attention(
            q,
            k,
            v,
            token_index,
            zml.attention.attention.Metadata.init(.fromBackend(.vanilla, x.dim(.s), self.num_heads)),
            zml.attention.attention.Parameters.init(.fromBackend(.vanilla)),
        ).withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated }).rename(.{ .q = .s }).merge(.{ .d_out_proj = .{ .h, .hd } });

        const gated_output = attn_output.mul(gate.sigmoid());
        const projected_output = forwardLinearDequantizeFp8(
            gated_output.rename(.{ .d_out_proj = .d }),
            self.o_proj,
            self.o_proj_scale,
        ).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });

        return .{ projected_output, new_kv_cache };
    }
};

pub const Mlp = struct {
    up_proj: zml.nn.Linear,
    up_proj_scale: ?zml.Tensor,
    gate_proj: zml.nn.Linear,
    gate_proj_scale: ?zml.Tensor,
    down_proj: zml.nn.Linear,
    down_proj_scale: ?zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .up_proj = .init(
                store.withPrefix("up_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                store.withPrefix("up_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }),
                .d,
            ),
            .up_proj_scale = store.withPrefix("up_proj").maybeCreateTensor("weight_scale_inv", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
            .gate_proj = .init(
                store.withPrefix("gate_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                store.withPrefix("gate_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }),
                .d,
            ),
            .gate_proj_scale = store.withPrefix("gate_proj").maybeCreateTensor("weight_scale_inv", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
            .down_proj = .init(
                store.withPrefix("down_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .model }),
                store.withPrefix("down_proj").maybeCreateTensor("bias", .{.d}, .{ .d = .replicated }),
                .d,
            ),
            .down_proj_scale = store.withPrefix("down_proj").maybeCreateTensor("weight_scale_inv", .{ .dout, .d }, .{ .dout = .replicated, .d = .model }),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        self.up_proj.weight.deinit();
        if (self.up_proj.bias) |*bias| bias.deinit();
        if (self.up_proj_scale) |*scale| scale.deinit();
        self.gate_proj.weight.deinit();
        if (self.gate_proj.bias) |*bias| bias.deinit();
        if (self.gate_proj_scale) |*scale| scale.deinit();
        self.down_proj.weight.deinit();
        if (self.down_proj.bias) |*bias| bias.deinit();
        if (self.down_proj_scale) |*scale| scale.deinit();
    }

    pub fn loadBuffers(
        self: *Mlp,
        ctx: *LoadCtx,
        store: zml.io.TensorStore.View,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(Mlp) {
        _ = store;
        var up_proj = try ctx.loadLinearBuf(&self.up_proj, progress);
        errdefer {
            up_proj.weight.deinit();
            if (up_proj.bias) |*bias| bias.deinit();
        }
        var gate_proj = try ctx.loadLinearBuf(&self.gate_proj, progress);
        errdefer {
            gate_proj.weight.deinit();
            if (gate_proj.bias) |*bias| bias.deinit();
        }
        var down_proj = try ctx.loadLinearBuf(&self.down_proj, progress);
        errdefer {
            down_proj.weight.deinit();
            if (down_proj.bias) |*bias| bias.deinit();
        }

        var up_proj_scale = try ctx.loadOptionalBuf(self.up_proj_scale, progress);
        errdefer if (up_proj_scale) |*scale| scale.deinit();
        var gate_proj_scale = try ctx.loadOptionalBuf(self.gate_proj_scale, progress);
        errdefer if (gate_proj_scale) |*scale| scale.deinit();
        var down_proj_scale = try ctx.loadOptionalBuf(self.down_proj_scale, progress);
        errdefer if (down_proj_scale) |*scale| scale.deinit();

        return .{
            .up_proj = up_proj,
            .up_proj_scale = up_proj_scale,
            .gate_proj = gate_proj,
            .gate_proj_scale = gate_proj_scale,
            .down_proj = down_proj,
            .down_proj_scale = down_proj_scale,
        };
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const up_projed = forwardLinearDequantizeFp8(x, self.up_proj, self.up_proj_scale);
        const gate = forwardLinearDequantizeFp8(x, self.gate_proj, self.gate_proj_scale);
        const hidden = gate.silu().mul(up_projed).rename(.{ .dout = .d });

        const output = forwardLinearDequantizeFp8(hidden, self.down_proj, self.down_proj_scale);
        return output;
    }
};

const Router = struct {
    router: zml.nn.Linear,
    num_experts_per_tok: u32,

    pub fn init(store: zml.io.TensorStore.View, num_experts_per_tok: u32) Router {
        return .{
            .router = .init(store.createTensor("weight", .{ .expert, .d }, null), store.maybeCreateTensor("bias", .{.expert}, null), .d),
            .num_experts_per_tok = num_experts_per_tok,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Router)) void {
        self.router.weight.deinit();
        if (self.router.bias) |*bias| bias.deinit();
    }

    pub fn loadBuffers(
        self: *Router,
        ctx: *LoadCtx,
        store: zml.io.TensorStore.View,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(Router) {
        _ = store;
        return .{
            .router = try ctx.loadLinearBuf(&self.router, progress),
        };
    }

    pub fn forward(self: Router, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const router_logits = self.router.forward(x).convert(.f32);
        const routing = router_logits.topK(.{ .top_expert = .expert }, self.num_experts_per_tok, .{});
        const topk_ids = routing.indices.convert(.i32);
        const router_scores = routing.values.softmax(.top_expert);
        return .{ router_scores, topk_ids };
    }
};

pub const Moe = struct {
    const Layout = enum { packed_weights, per_expert };

    pub const PackedWeights = struct {
        gate_up_proj: zml.Tensor,
        gate_up_proj_scale: zml.Tensor,
        down_proj: zml.Tensor,
        down_proj_scale: zml.Tensor,
    };

    shared_expert: Mlp,
    shared_expert_gate: zml.nn.Linear,
    gate_up_proj: zml.Tensor,
    gate_up_proj_scale: ?zml.Tensor,
    down_proj: zml.Tensor,
    down_proj_scale: ?zml.Tensor,
    router: Router,
    layout: Layout,
    num_experts: usize,

    pub fn init(_: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Moe {
        const experts_store = store.withPrefix("experts");
        const num_experts_i64 = config.text_config.num_experts orelse return error.MissingNumExperts;
        const num_experts: usize = @intCast(num_experts_i64);

        var layout: Layout = .packed_weights;
        var gate_up_proj = experts_store.maybeCreateTensor("gate_up_proj", .{ .expert, .dout, .d }, null);
        var down_proj = experts_store.maybeCreateTensor("down_proj", .{ .expert, .d, .dout }, null);
        var gate_up_proj_scale = experts_store.maybeCreateTensor("gate_up_proj_scale", .{ .expert, .dout, .d }, null);
        var down_proj_scale = experts_store.maybeCreateTensor("down_proj_scale", .{ .expert, .d, .dout }, null);

        if (gate_up_proj == null or down_proj == null) {
            layout = .per_expert;

            var gate_shape = experts_store.withLayer(0).withPrefix("gate_proj").getShape("weight") orelse return error.MissingGateProj;
            var up_shape = experts_store.withLayer(0).withPrefix("up_proj").getShape("weight") orelse return error.MissingUpProj;
            var down_shape = experts_store.withLayer(0).withPrefix("down_proj").getShape("weight") orelse return error.MissingDownProj;

            gate_shape = gate_shape.withTags(.{ .dout, .d });
            up_shape = up_shape.withTags(.{ .dout, .d });
            down_shape = down_shape.withTags(.{ .d, .dout });

            stdx.debug.assert(gate_shape.dim(.d) == up_shape.dim(.d), "gate/up input dims mismatch: {f} vs {f}", .{ gate_shape, up_shape });
            const gate_up_shape = zml.Shape.init(.{
                .expert = @as(i64, @intCast(num_experts)),
                .dout = gate_shape.dim(.dout) + up_shape.dim(.dout),
                .d = gate_shape.dim(.d),
            }, gate_shape.dtype());

            const down_packed_shape = zml.Shape.init(.{
                .expert = @as(i64, @intCast(num_experts)),
                .d = down_shape.dim(.d),
                .dout = down_shape.dim(.dout),
            }, down_shape.dtype());

            gate_up_proj = zml.Tensor.fromShape(gate_up_shape);
            down_proj = zml.Tensor.fromShape(down_packed_shape);

            const gate_scale_shape = experts_store.withLayer(0).withPrefix("gate_proj").getShape("weight_scale_inv");
            const up_scale_shape = experts_store.withLayer(0).withPrefix("up_proj").getShape("weight_scale_inv");
            const down_scale_shape = experts_store.withLayer(0).withPrefix("down_proj").getShape("weight_scale_inv");
            const has_scales = gate_scale_shape != null and up_scale_shape != null and down_scale_shape != null;

            if (has_scales) {
                const gate_scale = gate_scale_shape.?.withTags(.{ .dout, .d });
                const up_scale = up_scale_shape.?.withTags(.{ .dout, .d });
                const down_scale = down_scale_shape.?.withTags(.{ .d, .dout });
                stdx.debug.assert(gate_scale.dim(.d) == up_scale.dim(.d), "gate/up scale input dims mismatch: {f} vs {f}", .{ gate_scale, up_scale });
                gate_up_proj_scale = zml.Tensor.fromShape(zml.Shape.init(.{
                    .expert = @as(i64, @intCast(num_experts)),
                    .dout = gate_scale.dim(.dout) + up_scale.dim(.dout),
                    .d = gate_scale.dim(.d),
                }, gate_scale.dtype()));
                down_proj_scale = zml.Tensor.fromShape(zml.Shape.init(.{
                    .expert = @as(i64, @intCast(num_experts)),
                    .d = down_scale.dim(.d),
                    .dout = down_scale.dim(.dout),
                }, down_scale.dtype()));
            } else {
                gate_up_proj_scale = null;
                down_proj_scale = null;
            }
        }

        return .{
            .shared_expert = Mlp.init(store.withPrefix("shared_expert")),
            .shared_expert_gate = .init(store.withPrefix("shared_expert_gate").createTensor("weight", .{ .dout, .d }, null), store.withPrefix("shared_expert_gate").maybeCreateTensor("bias", .{.dout}, null), .d),
            .gate_up_proj = gate_up_proj.?,
            .gate_up_proj_scale = gate_up_proj_scale,
            .down_proj = down_proj.?,
            .down_proj_scale = down_proj_scale,
            .router = Router.init(store.withPrefix("gate"), config.text_config.num_experts_per_tok.?),
            .layout = layout,
            .num_experts = num_experts,
        };
    }

    pub fn deinit(_: Moe, _: std.mem.Allocator) void {}

    fn packExpertTensor(
        ctx: *LoadCtx,
        experts_store: zml.io.TensorStore.View,
        num_experts: usize,
        projection_name: []const u8,
        tensor_name: []const u8,
        tags: anytype,
        packed_shape: zml.Shape,
    ) !zml.Buffer {
        var packed_slice: zml.Slice = try .alloc(ctx.allocator, packed_shape);
        defer packed_slice.free(ctx.allocator);
        var offset: usize = 0;

        for (0..num_experts) |i| {
            const tensor = experts_store.withLayer(i).withPrefix(projection_name).createTensor(tensor_name, tags, null);
            var src = try ctx.loadBuf(&tensor, null);
            defer src.deinit();

            var src_slice: zml.Slice = try .alloc(ctx.allocator, tensor.shape());
            defer src_slice.free(ctx.allocator);
            try src.toSlice(ctx.io, src_slice);

            const bytes = src_slice.data();
            @memcpy(packed_slice.data()[offset .. offset + bytes.len], bytes);
            offset += bytes.len;
        }

        stdx.debug.assert(offset == packed_slice.data().len, "packed tensor byte size mismatch: packed={d} copied={d}", .{ packed_slice.data().len, offset });
        const sharding = zml.sharding.pickSharding(ctx.shardings, packed_shape, .explicit_axis_binding) orelse ctx.replicated_sharding;
        return zml.Buffer.fromSlice(ctx.io, ctx.platform, packed_slice, sharding);
    }

    fn packGateUpExperts(
        ctx: *LoadCtx,
        experts_store: zml.io.TensorStore.View,
        num_experts: usize,
        packed_shape: zml.Shape,
    ) !zml.Buffer {
        var packed_slice: zml.Slice = try .alloc(ctx.allocator, packed_shape);
        defer packed_slice.free(ctx.allocator);
        var offset: usize = 0;

        for (0..num_experts) |i| {
            const gate = experts_store.withLayer(i).withPrefix("gate_proj").createTensor("weight", .{ .dout, .d }, null);
            const up = experts_store.withLayer(i).withPrefix("up_proj").createTensor("weight", .{ .dout, .d }, null);

            var gate_buf = try ctx.loadBuf(&gate, null);
            defer gate_buf.deinit();
            var up_buf = try ctx.loadBuf(&up, null);
            defer up_buf.deinit();

            var gate_slice: zml.Slice = try .alloc(ctx.allocator, gate.shape());
            defer gate_slice.free(ctx.allocator);
            try gate_buf.toSlice(ctx.io, gate_slice);
            @memcpy(packed_slice.data()[offset .. offset + gate_slice.data().len], gate_slice.data());
            offset += gate_slice.data().len;

            var up_slice: zml.Slice = try .alloc(ctx.allocator, up.shape());
            defer up_slice.free(ctx.allocator);
            try up_buf.toSlice(ctx.io, up_slice);
            @memcpy(packed_slice.data()[offset .. offset + up_slice.data().len], up_slice.data());
            offset += up_slice.data().len;
        }

        stdx.debug.assert(offset == packed_slice.data().len, "packed gate_up byte size mismatch: packed={d} copied={d}", .{ packed_slice.data().len, offset });
        const sharding = zml.sharding.pickSharding(ctx.shardings, packed_shape, .explicit_axis_binding) orelse ctx.replicated_sharding;
        return zml.Buffer.fromSlice(ctx.io, ctx.platform, packed_slice, sharding);
    }

    fn packGateUpScaleExperts(
        ctx: *LoadCtx,
        experts_store: zml.io.TensorStore.View,
        num_experts: usize,
        packed_shape: zml.Shape,
    ) !zml.Buffer {
        var packed_slice: zml.Slice = try .alloc(ctx.allocator, packed_shape);
        defer packed_slice.free(ctx.allocator);
        var offset: usize = 0;

        for (0..num_experts) |i| {
            const gate = experts_store.withLayer(i).withPrefix("gate_proj").createTensor("weight_scale_inv", .{ .dout, .d }, null);
            const up = experts_store.withLayer(i).withPrefix("up_proj").createTensor("weight_scale_inv", .{ .dout, .d }, null);

            var gate_buf = try ctx.loadBuf(&gate, null);
            defer gate_buf.deinit();
            var up_buf = try ctx.loadBuf(&up, null);
            defer up_buf.deinit();

            var gate_slice: zml.Slice = try .alloc(ctx.allocator, gate.shape());
            defer gate_slice.free(ctx.allocator);
            try gate_buf.toSlice(ctx.io, gate_slice);
            @memcpy(packed_slice.data()[offset .. offset + gate_slice.data().len], gate_slice.data());
            offset += gate_slice.data().len;

            var up_slice: zml.Slice = try .alloc(ctx.allocator, up.shape());
            defer up_slice.free(ctx.allocator);
            try up_buf.toSlice(ctx.io, up_slice);
            @memcpy(packed_slice.data()[offset .. offset + up_slice.data().len], up_slice.data());
            offset += up_slice.data().len;
        }

        stdx.debug.assert(offset == packed_slice.data().len, "packed gate_up scale byte size mismatch: packed={d} copied={d}", .{ packed_slice.data().len, offset });
        const sharding = zml.sharding.pickSharding(ctx.shardings, packed_shape, .explicit_axis_binding) orelse ctx.replicated_sharding;
        return zml.Buffer.fromSlice(ctx.io, ctx.platform, packed_slice, sharding);
    }

    pub fn loadBuffers(
        self: *Moe,
        ctx: *LoadCtx,
        store: zml.io.TensorStore.View,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(Moe) {
        var shared_expert = try self.shared_expert.loadBuffers(ctx, store.withPrefix("shared_expert"), progress);
        errdefer Mlp.unloadBuffers(&shared_expert);

        var shared_expert_gate = try ctx.loadLinearBuf(&self.shared_expert_gate, progress);
        errdefer {
            shared_expert_gate.weight.deinit();
            if (shared_expert_gate.bias) |*bias| bias.deinit();
        }

        var router = try self.router.loadBuffers(ctx, store.withPrefix("gate"), progress);
        errdefer Router.unloadBuffers(&router);

        switch (self.layout) {
            .packed_weights => {
                var gate_up_proj = try ctx.loadBuf(&self.gate_up_proj, progress);
                errdefer gate_up_proj.deinit();
                var gate_up_proj_scale = try ctx.loadOptionalBuf(self.gate_up_proj_scale, progress);
                errdefer if (gate_up_proj_scale) |*buffer| buffer.deinit();
                var down_proj = try ctx.loadBuf(&self.down_proj, progress);
                errdefer down_proj.deinit();
                var down_proj_scale = try ctx.loadOptionalBuf(self.down_proj_scale, progress);
                errdefer if (down_proj_scale) |*buffer| buffer.deinit();

                return .{
                    .shared_expert = shared_expert,
                    .shared_expert_gate = shared_expert_gate,
                    .gate_up_proj = gate_up_proj,
                    .gate_up_proj_scale = gate_up_proj_scale,
                    .down_proj = down_proj,
                    .down_proj_scale = down_proj_scale,
                    .router = router,
                };
            },
            .per_expert => {
                const experts_store = store.withPrefix("experts");
                var gate_up_proj = try packGateUpExperts(ctx, experts_store, self.num_experts, self.gate_up_proj.shape());
                errdefer gate_up_proj.deinit();
                var down_proj = try packExpertTensor(
                    ctx,
                    experts_store,
                    self.num_experts,
                    "down_proj",
                    "weight",
                    .{ .d, .dout },
                    self.down_proj.shape(),
                );
                errdefer down_proj.deinit();

                var gate_up_proj_scale: ?zml.Buffer = null;
                errdefer if (gate_up_proj_scale) |*buffer| buffer.deinit();
                var down_proj_scale: ?zml.Buffer = null;
                errdefer if (down_proj_scale) |*buffer| buffer.deinit();

                if (self.gate_up_proj_scale != null and self.down_proj_scale != null) {
                    gate_up_proj_scale = try packGateUpScaleExperts(ctx, experts_store, self.num_experts, self.gate_up_proj_scale.?.shape());
                    down_proj_scale = try packExpertTensor(
                        ctx,
                        experts_store,
                        self.num_experts,
                        "down_proj",
                        "weight_scale_inv",
                        .{ .d, .dout },
                        self.down_proj_scale.?.shape(),
                    );
                }

                self.layout = .packed_weights;
                self.gate_up_proj = zml.Tensor.fromShape(gate_up_proj.shape());
                self.gate_up_proj_scale = if (gate_up_proj_scale) |buffer| zml.Tensor.fromShape(buffer.shape()) else null;
                self.down_proj = zml.Tensor.fromShape(down_proj.shape());
                self.down_proj_scale = if (down_proj_scale) |buffer| zml.Tensor.fromShape(buffer.shape()) else null;

                return .{
                    .shared_expert = shared_expert,
                    .shared_expert_gate = shared_expert_gate,
                    .gate_up_proj = gate_up_proj,
                    .gate_up_proj_scale = gate_up_proj_scale,
                    .down_proj = down_proj,
                    .down_proj_scale = down_proj_scale,
                    .router = router,
                };
            },
        }
    }

    pub fn forward(self: Moe, x: zml.Tensor, moe_metadata: zml.moe.Metadata, moe_parameters: zml.moe.Parameters) zml.Tensor {
        return switch (self.layout) {
            .packed_weights => if (self.gate_up_proj_scale != null and self.down_proj_scale != null)
                self.forwardQuantized(x, moe_metadata, moe_parameters)
            else
                self.forwardUnquantized(x, moe_metadata, moe_parameters),
            .per_expert => self.forwardQuantized(x, moe_metadata, moe_parameters),
        };
    }

    pub fn forwardUnquantized(self: Moe, x: zml.Tensor, moe_metadata: zml.moe.Metadata, moe_parameters: zml.moe.Parameters) zml.Tensor {
        const routing_scores, const topk_ids = self.router.forward(x);

        const moe_output = zml.moe.forwardMoe(
            x,
            topk_ids,
            routing_scores,
            self.gate_up_proj,
            null,
            null,
            self.down_proj,
            null,
            null,
            moe_metadata,
            moe_parameters,
        ) catch |err| stdx.debug.panic("moe backend failed: {}", .{err});

        const shared_gate = self.shared_expert_gate.forward(x).sigmoid().broad(x.shape());
        const shared = self.shared_expert.forward(x).mul(shared_gate);

        return moe_output.add(shared);
    }

    pub fn forwardQuantized(self: Moe, x: zml.Tensor, moe_metadata: zml.moe.Metadata, moe_parameters: zml.moe.Parameters) zml.Tensor {
        stdx.debug.assert(self.layout == .packed_weights, "quantized forward expects packed MoE weights", .{});

        const routing_scores, const topk_ids = self.router.forward(x);

        const moe_output = zml.moe.forwardMoe(
            x,
            topk_ids,
            routing_scores,
            self.gate_up_proj,
            self.gate_up_proj_scale,
            null,
            self.down_proj,
            self.down_proj_scale,
            null,
            moe_metadata,
            moe_parameters,
        ) catch |err| stdx.debug.panic("moe backend failed: {}", .{err});

        const shared_gate = self.shared_expert_gate.forward(x).sigmoid().broad(x.shape());
        const shared = self.shared_expert.forward(x).mul(shared_gate);

        return moe_output.add(shared);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Moe), allocator: std.mem.Allocator) void {
        Mlp.unloadBuffers(&self.shared_expert);
        self.shared_expert_gate.weight.deinit();
        if (self.shared_expert_gate.bias) |*bias| bias.deinit();
        self.gate_up_proj.deinit();
        if (self.gate_up_proj_scale) |*gate_up_proj_scale| gate_up_proj_scale.deinit();
        self.down_proj.deinit();
        if (self.down_proj_scale) |*down_proj_scale| down_proj_scale.deinit();
        _ = allocator;
        Router.unloadBuffers(&self.router);
    }
};

fn forwardLinearDequantizeFp8(x: zml.Tensor, linear: zml.nn.Linear, scale_inv: ?zml.Tensor) zml.Tensor {
    if (scale_inv == null) {
        return linear.forward(x);
    }

    const weight_scale_inv = scale_inv.?;
    const num_tokens = x.dim(.b) * x.dim(.s);

    stdx.debug.assert(x.rank() == 3, "expected a rank-3 activation tensor, got {f}", .{x.shape()});
    stdx.debug.assert(linear.tag == zml.Shape.toTag(.d), "blocked FP8 linear only supports contraction on .d, got {any}", .{linear.tag});
    stdx.debug.assert(linear.weight.rank() == 2, "expected a rank-2 weight tensor, got {f}", .{linear.weight.shape()});
    stdx.debug.assert(linear.weight.dtype() == .f8e4m3fn, "expected FP8 weights for blocked FP8 path, got {}", .{linear.weight.dtype()});
    stdx.debug.assert(@mod(linear.weight.dim(.dout), 128) == 0, "output dim must be divisible by 128, got {d}", .{linear.weight.dim(.dout)});
    stdx.debug.assert(@mod(linear.weight.dim(.d), 128) == 0, "weight input dim must be divisible by 128, got {d}", .{linear.weight.dim(.d)});
    stdx.debug.assert(linear.weight.dim(.d) == x.dim(-1), "activation and weight input dims must match, got {f} and {f}", .{ x.shape(), linear.weight.shape() });
    stdx.debug.assert(weight_scale_inv.rank() == 2, "expected a rank-2 FP8 scale tensor, got {f}", .{weight_scale_inv.shape()});
    stdx.debug.assert(weight_scale_inv.dim(.dout) * 128 == linear.weight.dim(.dout), "scale output blocks must match weight output dim, got {f} and {f}", .{ weight_scale_inv.shape(), linear.weight.shape() });
    stdx.debug.assert(weight_scale_inv.dim(.d) * 128 == linear.weight.dim(.d), "scale input blocks must match weight input dim, got {f} and {f}", .{ weight_scale_inv.shape(), linear.weight.shape() });

    const blocked_weight = linear.weight.reshape(.{
        .dout_block = @divExact(linear.weight.dim(.dout), 128),
        .dout_chunk = 128,
        .d_block = @divExact(linear.weight.dim(.d), 128),
        .d_chunk = 128,
    });
    const blocked_scale = weight_scale_inv.reshape(.{
        .dout_block = weight_scale_inv.dim(.dout),
        .dout_chunk = 1,
        .d_block = weight_scale_inv.dim(.d),
        .d_chunk = 1,
    });

    const dequantized_weight = blocked_weight
        .convert(.bf16)
        .mul(blocked_scale.convert(.bf16))
        .merge(.{ .dout = .{ .dout_block, .dout_chunk }, .d = .{ .d_block, .d_chunk } });

    const flat_x = x.reshape(.{ .token = num_tokens, .d = x.dim(-1) });
    var y = flat_x.dotGeneral(
        dequantized_weight,
        &.{
            .{ @intCast(flat_x.axis(.d)), @intCast(dequantized_weight.axis(.d)) },
        },
        &.{},
    );

    y = y.reshape(.{ .b = x.dim(.b), .s = x.dim(.s), .dout = linear.weight.dim(.dout) });
    return if (linear.bias) |bias| y.add(bias.broad(y.shape())) else y;
}

pub const TextRotaryEmbedding = struct {
    rope_opts: zml.nn.RopeOpts,
    rotary_dim: i64,
    mrope_section: [3]i64,

    pub fn init(rotary_dim: i64, theta: f32, mrope_section: [3]i64) TextRotaryEmbedding {
        return .{
            .rope_opts = .{
                .layout = .sequential,
                .scaling = .{ .default = .{ .rope_theta = theta } },
            },
            .rotary_dim = rotary_dim,
            .mrope_section = mrope_section,
        };
    }

    pub fn getCosAndSin(self: TextRotaryEmbedding, position_ids: zml.Tensor, dtype: zml.DataType) struct { zml.Tensor, zml.Tensor } {
        const inv_freq = zml.nn.invFreq(self.rotary_dim, self.rope_opts).withTags(.{.hd});

        const freqs_t = position_ids.convert(.f32).outer(inv_freq);

        const emb = zml.Tensor.concatenate(&.{ freqs_t, freqs_t }, -1);
        const cos = emb.cos().convert(dtype);
        const sin = emb.sin().convert(dtype);

        return .{ cos, sin };
    }

    pub fn getCosAndSinInterleaved(self: TextRotaryEmbedding, position_ids: zml.Tensor, dtype: zml.DataType) struct { zml.Tensor, zml.Tensor } { // To be used later in image extension
        const stacked_position_ids = zml.Tensor.stack(&.{ position_ids, position_ids, position_ids }, 0, .g).convert(.f32);
        const inv_freq = zml.nn.InvFreq(self.rotary_dim, self.rope_opts).withTags(.{.hd});

        var freqs = stacked_position_ids.outer(inv_freq);
        var freqs_t, var freqs_h, var freqs_w = freqs.chunkExact(.g, 3);
        freqs_t = freqs_t.squeeze(.g);
        freqs_h = freqs_h.squeeze(.g);
        freqs_w = freqs_w.squeeze(.g);

        const h_indices = zml.Tensor.iota(zml.Shape.init(.{ .h = self.mrope_section[1] }, .i32), .h).scale(3).addConstant(1);
        const w_indices = zml.Tensor.iota(zml.Shape.init(.{ .h = self.mrope_section[2] }, .i32), .h).scale(3).addConstant(2);

        const h_input = freqs_h.gather(.{ .dh = h_indices }, .{ .indices_are_sorted = true });
        const w_input = freqs_w.gather(.{ .dh = w_indices }, .{ .indices_are_sorted = true });
        freqs_t = freqs_t.scatterSlices(.{ .dh = h_indices }, h_input, .{ .update_fn = zml.Tensor.ScatterOpts.override });
        freqs = freqs_t.scatterSlices(.{ .dh = w_indices }, w_input, .{ .update_fn = zml.Tensor.ScatterOpts.override });

        const emb = zml.Tensor.concatenate(&.{ freqs, freqs }, -1);
        const cos = emb.cos().convert(dtype);
        const sin = emb.sin().convert(dtype);

        return .{ cos, sin };
    }

    fn rotateHalf(x: zml.Tensor) zml.Tensor {
        const half_dim = @divExact(x.dim(-1), 2);
        const x1 = x.slice1d(-1, .{ .start = 0, .end = half_dim });
        const x2 = x.slice1d(-1, .{ .start = half_dim, .end = x.dim(-1) });
        return zml.Tensor.concatenate(&.{ x2.negate(), x1 }, -1);
    }

    pub fn applyRope(self: TextRotaryEmbedding, x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const x_rot = x.slice1d(-1, .{ .start = 0, .end = self.rotary_dim });
        const x_pass = x.slice1d(-1, .{ .start = self.rotary_dim, .end = x.dim(-1) });

        const cos_x = cos.insertAxes(.hd, .{.h}).broad(x_rot.shape());
        const sin_x = sin.insertAxes(.hd, .{.h}).broad(x_rot.shape());

        const rotated = x_rot.mul(cos_x).add(rotateHalf(x_rot).mul(sin_x));

        return zml.Tensor.concatenate(&.{ rotated, x_pass }, -1);
    }
};

pub const GatedDeltaNet = struct {
    in_proj_qkv: zml.nn.Linear,
    in_proj_qkv_scale: ?zml.Tensor,
    in_proj_z: zml.nn.Linear,
    in_proj_z_scale: ?zml.Tensor,
    in_proj_b: zml.nn.Linear,
    in_proj_a: zml.nn.Linear,
    out_proj: zml.nn.Linear,
    out_proj_scale: ?zml.Tensor,
    conv1d_weight: zml.Tensor,
    dt_bias: zml.Tensor,
    aLog: zml.Tensor,
    norm: RmsNormGated,

    num_k_heads: i64,
    num_v_heads: i64,
    qk_head_repetition: i64,
    head_k_dim: i64,
    head_v_dim: i64,
    conv_kernel_size: i64,

    fn initProj(store: zml.io.TensorStore.View, partitions: anytype) zml.nn.Linear {
        return .init(store.createTensor("weight", .{ .dout, .d }, partitions), null, .d);
    }

    pub fn init(store: zml.io.TensorStore.View, config: Config) GatedDeltaNet {
        const qk_head_repetition =
            @divExact(config.text_config.linear_num_value_heads, config.text_config.linear_num_key_heads);
        return .{
            .in_proj_qkv = initProj(store.withPrefix("in_proj_qkv"), .{ .dout = .model, .d = .replicated }),
            .in_proj_qkv_scale = store.withPrefix("in_proj_qkv").maybeCreateTensor("weight_scale_inv", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
            .in_proj_z = initProj(store.withPrefix("in_proj_z"), .{ .dout = .model, .d = .replicated }),
            .in_proj_z_scale = store.withPrefix("in_proj_z").maybeCreateTensor("weight_scale_inv", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
            .in_proj_b = initProj(store.withPrefix("in_proj_b"), .{ .dout = .model, .d = .replicated }),
            .in_proj_a = initProj(store.withPrefix("in_proj_a"), .{ .dout = .model, .d = .replicated }),
            .out_proj = initProj(store.withPrefix("out_proj"), .{ .dout = .replicated, .d = .model }),
            .out_proj_scale = store.withPrefix("out_proj").maybeCreateTensor("weight_scale_inv", .{ .dout, .d }, .{ .dout = .replicated, .d = .model }),
            .conv1d_weight = store.withPrefix("conv1d").createTensor("weight", .{ .out, .in, .kernel_size }, .{ .out = .model, .in = .replicated, .kernel_size = .replicated }),
            .dt_bias = store.createTensor("dt_bias", .{.vh}, .{ .vh = .model }),
            .aLog = store.createTensor("A_log", .{.vh}, .{ .vh = .model }),
            .norm = RmsNormGated.init(store.withPrefix("norm"), config.text_config.rms_norm_eps),
            .num_k_heads = config.text_config.linear_num_key_heads,
            .num_v_heads = config.text_config.linear_num_value_heads,
            .qk_head_repetition = qk_head_repetition,
            .head_k_dim = config.text_config.linear_key_head_dim,
            .head_v_dim = config.text_config.linear_value_head_dim,
            .conv_kernel_size = config.text_config.linear_conv_kernel_dim,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(GatedDeltaNet)) void {
        self.in_proj_qkv.weight.deinit();
        if (self.in_proj_qkv_scale) |*scale| scale.deinit();
        self.in_proj_z.weight.deinit();
        if (self.in_proj_z_scale) |*scale| scale.deinit();
        self.in_proj_b.weight.deinit();
        self.in_proj_a.weight.deinit();
        self.out_proj.weight.deinit();
        if (self.out_proj_scale) |*scale| scale.deinit();
        self.conv1d_weight.deinit();
        self.dt_bias.deinit();
        self.aLog.deinit();
        RmsNormGated.unloadBuffers(&self.norm);
    }

    pub fn loadBuffers(
        self: *GatedDeltaNet,
        ctx: *LoadCtx,
        store: zml.io.TensorStore.View,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(GatedDeltaNet) {
        var in_proj_qkv = try ctx.loadLinearBuf(&self.in_proj_qkv, progress);
        errdefer in_proj_qkv.weight.deinit();
        var in_proj_z = try ctx.loadLinearBuf(&self.in_proj_z, progress);
        errdefer in_proj_z.weight.deinit();
        var in_proj_b = try ctx.loadLinearBuf(&self.in_proj_b, progress);
        errdefer in_proj_b.weight.deinit();
        var in_proj_a = try ctx.loadLinearBuf(&self.in_proj_a, progress);
        errdefer in_proj_a.weight.deinit();
        var out_proj = try ctx.loadLinearBuf(&self.out_proj, progress);
        errdefer out_proj.weight.deinit();

        var in_proj_qkv_scale = try ctx.loadOptionalBuf(self.in_proj_qkv_scale, progress);
        errdefer if (in_proj_qkv_scale) |*scale| scale.deinit();
        var in_proj_z_scale = try ctx.loadOptionalBuf(self.in_proj_z_scale, progress);
        errdefer if (in_proj_z_scale) |*scale| scale.deinit();
        var out_proj_scale = try ctx.loadOptionalBuf(self.out_proj_scale, progress);
        errdefer if (out_proj_scale) |*scale| scale.deinit();

        var conv1d_weight = try ctx.loadBuf(&self.conv1d_weight, progress);
        errdefer conv1d_weight.deinit();
        var dt_bias = try ctx.loadBuf(&self.dt_bias, progress);
        errdefer dt_bias.deinit();
        var aLog = try ctx.loadBuf(&self.aLog, progress);
        errdefer aLog.deinit();
        var norm = try self.norm.loadBuffers(ctx, store.withPrefix("norm"), progress);
        errdefer RmsNormGated.unloadBuffers(&norm);

        return .{
            .in_proj_qkv = in_proj_qkv,
            .in_proj_qkv_scale = in_proj_qkv_scale,
            .in_proj_z = in_proj_z,
            .in_proj_z_scale = in_proj_z_scale,
            .in_proj_b = in_proj_b,
            .in_proj_a = in_proj_a,
            .out_proj = out_proj,
            .out_proj_scale = out_proj_scale,
            .conv1d_weight = conv1d_weight,
            .dt_bias = dt_bias,
            .aLog = aLog,
            .norm = norm,
        };
    }

    fn recurrent_gated_delta_rule(query: zml.Tensor, key: zml.Tensor, value: zml.Tensor, g: zml.Tensor, beta: zml.Tensor, initial_state: ?zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(query.dim(.khd))));
        const query_norm = zml.nn.normalizeL2(query.rename(.{ .kh = .vh }), 1e-6);
        const key_norm = zml.nn.normalizeL2(key.rename(.{ .kh = .vh }), 1e-6);

        const query_f32, const key_f32, const value_f32, const alpha_f32, const beta_f32 = .{
            query_norm.convert(.f32).scale(scale).rename(.{ .vh = .h, .khd = .k }),
            key_norm.convert(.f32).rename(.{ .vh = .h, .khd = .k }),
            value.convert(.f32).rename(.{ .vh = .h, .vhd = .v }),
            g.convert(.f32).exp().rename(.{ .vh = .h }),
            beta.convert(.f32).rename(.{ .vh = .h }),
        };

        const initial_recurrent_state = if (initial_state) |state|
            state.convert(.f32).transpose(.{ .b, .vh, .vhd, .khd }).rename(.{ .vh = .h, .vhd = .v, .khd = .k })
        else
            zml.Tensor.constant(zml.DataType.zero(.f32)).broad(zml.Shape.init(.{
                .b = value.dim(.b),
                .h = value.dim(.vh),
                .v = value.dim(.vhd),
                .k = query.dim(.khd),
            }, .f32));

        const result = zml.nn.GatedDeltaNet.forward(
            query_f32,
            key_f32,
            value_f32,
            alpha_f32,
            beta_f32,
            .{ .s = initial_recurrent_state },
        );

        return .{
            result.outputs.rename(.{ .h = .vh, .v = .vhd }).convert(query.dtype()),
            result.state.s.transpose(.{ .b, .h, .k, .v }).rename(.{ .h = .vh, .k = .khd, .v = .vhd }),
        };
    }

    fn buildUpdatedConvState(input: zml.Tensor, left_pad: i64) zml.Tensor {
        const copy_len = @min(input.dim(.s), left_pad);
        const tail = input.slice1d(.s, .{ .start = input.dim(.s) - copy_len, .end = input.dim(.s) });
        if (copy_len == left_pad) return tail;

        const padding_shape = zml.Shape.init(.{ .b = input.dim(.b), .s = left_pad - copy_len, .mix = input.dim(.mix) }, input.dtype());
        const padding = zml.Tensor.constant(input.dtype().zero()).broad(padding_shape);
        return zml.Tensor.concatenate(&.{ padding, tail }, .s);
    }

    pub fn forward(self: GatedDeltaNet, x: zml.Tensor, cache: KvCache.GatedDeltaNetCache) struct { zml.Tensor, KvCache.GatedDeltaNetCache } {
        const key_dim = self.num_k_heads * self.head_k_dim;
        const value_dim = self.num_v_heads * self.head_v_dim;
        const conv_dim = 2 * key_dim + value_dim;
        const left_pad = self.conv_kernel_size - 1;

        const x_in = x.withPartitioning(.{ .d = .replicated });
        const projected_qkv = forwardLinearDequantizeFp8(x_in, self.in_proj_qkv, self.in_proj_qkv_scale)
            .rename(.{ .dout = .mix }).withPartitioning(.{ .s = .replicated, .mix = .model });
        const use_cached_state = x.dim(.s) == 1 and left_pad > 0;
        const conv_input = if (use_cached_state)
            zml.Tensor.concatenate(&.{ cache.convState(), projected_qkv }, .s)
        else
            projected_qkv;

        const kernel = self.conv1d_weight;
        var mixed_qkv = zml.Tensor.conv1d(
            conv_input,
            kernel,
            .{
                .padding = &.{ left_pad, 0 },
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
            },
        )
            .silu();

        if (use_cached_state) {
            mixed_qkv = mixed_qkv.slice1d(.s, .{ .start = mixed_qkv.dim(.s) - 1, .end = mixed_qkv.dim(.s) });
        }
        mixed_qkv = mixed_qkv.withPartitioning(.{ .s = .replicated, .mix = .model });

        const z = forwardLinearDequantizeFp8(x_in, self.in_proj_z, self.in_proj_z_scale)
            .splitAxis(.dout, .{ .vh = self.num_v_heads, .vhd = self.head_v_dim })
            .withPartitioning(.{ .s = .replicated, .vh = .model, .vhd = .replicated });
        const b = self.in_proj_b.forward(x_in).rename(.{ .dout = .vh }).withPartitioning(.{ .s = .replicated, .vh = .model });
        const a = self.in_proj_a.forward(x_in).rename(.{ .dout = .vh }).withPartitioning(.{ .s = .replicated, .vh = .model });

        const query = mixed_qkv
            .slice1d(.mix, .{ .start = 0, .end = key_dim })
            .splitAxis(.mix, .{ .kh = self.num_k_heads, .khd = self.head_k_dim })
            .withPartitioning(.{ .s = .replicated, .kh = .model, .khd = .replicated });
        const key = mixed_qkv
            .slice1d(.mix, .{ .start = key_dim, .end = 2 * key_dim })
            .splitAxis(.mix, .{ .kh = self.num_k_heads, .khd = self.head_k_dim })
            .withPartitioning(.{ .s = .replicated, .kh = .model, .khd = .replicated });
        const value = mixed_qkv
            .slice1d(.mix, .{ .start = 2 * key_dim, .end = 2 * key_dim + value_dim })
            .splitAxis(.mix, .{ .vh = self.num_v_heads, .vhd = self.head_v_dim })
            .withPartitioning(.{ .s = .replicated, .vh = .model, .vhd = .replicated });

        const beta = b.sigmoid();
        const aLog_type = self.aLog.dtype();
        const g = self.aLog.broad(a.shape()).exp().mul(softplus(a.convert(aLog_type).add(self.dt_bias.convert(aLog_type).broad(a.shape())))).negate();

        const query_for_rule = if (self.qk_head_repetition == 1) query else query.stutter1d(@intCast(query.axis(.kh)), @intCast(self.qk_head_repetition));
        const key_for_rule = if (self.qk_head_repetition == 1) key else key.stutter1d(@intCast(key.axis(.kh)), @intCast(self.qk_head_repetition));

        const core_attn_out, const last_recurrent_state = recurrent_gated_delta_rule(
            query_for_rule,
            key_for_rule,
            value,
            g,
            beta,
            if (use_cached_state) cache.recurrentState() else null,
        );

        const core_attn_out_normed = self.norm
            .forward(
                core_attn_out.rename(.{ .vhd = .d }),
                z.rename(.{ .vhd = .d }),
            )
            .rename(.{ .d = .vhd })
            .withPartitioning(.{ .s = .replicated, .vh = .model, .vhd = .replicated });

        const output = forwardLinearDequantizeFp8(
            core_attn_out_normed.merge(.{ .d = .{ .vh, .vhd } }),
            self.out_proj,
            self.out_proj_scale,
        ).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
        const updated_cache = cache.update(
            buildUpdatedConvState(conv_input, left_pad),
            last_recurrent_state,
        );
        return .{ output, updated_cache };
    }
};

pub const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32 = 1e-6,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{ .weight = store.createTensor("weight", .{.d}, .{ .d = .replicated }), .eps = eps };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    pub fn loadBuffers(
        self: *RmsNorm,
        ctx: *LoadCtx,
        store: zml.io.TensorStore.View,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(RmsNorm) {
        _ = store;
        return .{
            .weight = try ctx.loadBuf(&self.weight, progress),
        };
    }

    pub fn forward(self: RmsNorm, x: zml.Tensor) zml.Tensor {
        const x_f32 = x.convert(.f32);
        const weight_f32 = self.weight.convert(.f32);

        const normalized = zml.nn.rmsNorm(x_f32, .d, self.eps);
        return normalized.mul(weight_f32.broad(x.shape())).add(normalized).convert(x.dtype());
    }
};

pub const RmsNormGated = struct {
    weight: zml.Tensor,
    eps: f32 = 1e-6,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNormGated {
        return .{ .weight = store.createTensor("weight", .{.d}, .{ .d = .replicated }), .eps = eps };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNormGated)) void {
        self.weight.deinit();
    }

    pub fn loadBuffers(
        self: *RmsNormGated,
        ctx: *LoadCtx,
        store: zml.io.TensorStore.View,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(RmsNormGated) {
        _ = store;
        return .{
            .weight = try ctx.loadBuf(&self.weight, progress),
        };
    }

    pub fn forward(self: RmsNormGated, x: zml.Tensor, gate: zml.Tensor) zml.Tensor {
        const x_f32 = x.convert(.f32);
        const gate_f32 = gate.convert(.f32);

        const normalized = zml.nn.rmsNorm(x_f32, .d, self.eps);
        const output = normalized.mul(self.weight.broad(x.shape()));

        const gated_output = output.mul(gate_f32.silu());
        return gated_output.convert(x.dtype());
    }
};

pub const KvCache = struct {
    layer_types: []const LayerType,
    self_attn: SelfAttnCache,
    gated_delta_net: GatedDeltaNetCache,

    pub const SelfAttnCache = struct {
        k: zml.Tensor,
        v: zml.Tensor,
        layer_index: zml.Tensor,

        pub fn init(config: Config, batch_dim: i64, max_seq_len: i64, dtype: zml.DataType) SelfAttnCache {
            const num_self_attn_layers = countLayers(config.text_config.layer_types, .full_attention);
            const kv_shape = zml.Shape.init(.{
                .b = batch_dim,
                .layer = num_self_attn_layers,
                .s = max_seq_len,
                .h = config.text_config.num_key_value_heads,
                .hd = config.text_config.head_dim,
            }, dtype);
            const sharded_kv_shape = kv_shape.withPartitioning(.{ .h = .model });
            return .{
                .k = .fromShape(sharded_kv_shape),
                .v = .fromShape(sharded_kv_shape),
                .layer_index = .init(.{}, .u32),
            };
        }

        pub fn initBuffer(self: SelfAttnCache, io: std.Io, platform: *const zml.Platform, sharding: zml.sharding.Sharding) !zml.Bufferized(SelfAttnCache) {
            return .{
                .k = try zml.Buffer.uninitialized(io, platform, self.k.shape(), sharding, .{}),
                .v = try zml.Buffer.uninitialized(io, platform, self.v.shape(), sharding, .{}),
                .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32, sharding),
            };
        }

        pub fn deinitBuffer(self: *zml.Bufferized(SelfAttnCache)) void {
            self.k.deinit();
            self.v.deinit();
            self.layer_index.deinit();
        }

        pub fn keys(self: SelfAttnCache) zml.Tensor {
            return self.k.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
        }

        pub fn values(self: SelfAttnCache) zml.Tensor {
            return self.v.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
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

    pub const GatedDeltaNetCache = struct {
        conv_state: zml.Tensor,
        recurrent_state: zml.Tensor,
        layer_index: zml.Tensor,

        pub fn init(config: Config, batch_dim: i64, conv_dtype: zml.DataType, recurrent_dtype: zml.DataType) GatedDeltaNetCache {
            const num_linear_attn_layers = countLayers(config.text_config.layer_types, .linear_attention);
            const conv_dim = 2 * config.text_config.linear_num_key_heads * config.text_config.linear_key_head_dim + config.text_config.linear_num_value_heads * config.text_config.linear_value_head_dim;
            const conv_state_shape = zml.Shape.init(.{
                .b = batch_dim,
                .layer = num_linear_attn_layers,
                .s = config.text_config.linear_conv_kernel_dim - 1,
                .mix = conv_dim,
            }, conv_dtype);
            const recurrent_state_shape = zml.Shape.init(.{
                .b = batch_dim,
                .layer = num_linear_attn_layers,
                .vh = config.text_config.linear_num_value_heads,
                .khd = config.text_config.linear_key_head_dim,
                .vhd = config.text_config.linear_value_head_dim,
            }, recurrent_dtype);
            const sharded_conv_state_shape = conv_state_shape.withPartitioning(.{ .mix = .model });
            const sharded_recurrent_state_shape = recurrent_state_shape.withPartitioning(.{ .vh = .model });
            return .{
                .conv_state = .fromShape(sharded_conv_state_shape),
                .recurrent_state = .fromShape(sharded_recurrent_state_shape),
                .layer_index = .init(.{}, .u32),
            };
        }

        pub fn initBuffer(self: GatedDeltaNetCache, io: std.Io, platform: *const zml.Platform, sharding: zml.sharding.Sharding) !zml.Bufferized(GatedDeltaNetCache) {
            return .{
                .conv_state = try zml.Buffer.uninitialized(io, platform, self.conv_state.shape(), sharding, .{}),
                .recurrent_state = try zml.Buffer.uninitialized(io, platform, self.recurrent_state.shape(), sharding, .{}),
                .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32, sharding),
            };
        }

        pub fn deinitBuffer(self: *zml.Bufferized(GatedDeltaNetCache)) void {
            self.conv_state.deinit();
            self.recurrent_state.deinit();
            self.layer_index.deinit();
        }

        pub fn convState(self: GatedDeltaNetCache) zml.Tensor {
            return self.conv_state.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
        }

        pub fn recurrentState(self: GatedDeltaNetCache) zml.Tensor {
            return self.recurrent_state.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
        }

        pub fn update(self: GatedDeltaNetCache, new_conv_state: ?zml.Tensor, new_recurrent_state: ?zml.Tensor) GatedDeltaNetCache {
            const conv_state = if (new_conv_state) |state|
                self.conv_state.scatterSlices(
                    .{ .layer = self.layer_index },
                    state.convert(self.conv_state.dtype()).transpose(self.conv_state.shape().drop(.layer)),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(self.conv_state)
            else
                self.conv_state;

            const recurrent_state = if (new_recurrent_state) |state|
                self.recurrent_state.scatterSlices(
                    .{ .layer = self.layer_index },
                    state.convert(self.recurrent_state.dtype()).transpose(self.recurrent_state.shape().drop(.layer)),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(self.recurrent_state)
            else
                self.recurrent_state;

            return .{
                .conv_state = conv_state,
                .recurrent_state = recurrent_state,
                .layer_index = self.layer_index,
            };
        }

        pub fn atLayer(self: GatedDeltaNetCache, layer_index: usize) GatedDeltaNetCache {
            return .{
                .conv_state = self.conv_state,
                .recurrent_state = self.recurrent_state,
                .layer_index = zml.Tensor.scalar(layer_index, .u32),
            };
        }

        pub fn reuseBuffer(self: GatedDeltaNetCache, other: GatedDeltaNetCache) GatedDeltaNetCache {
            return .{
                .conv_state = self.conv_state.reuseBuffer(other.conv_state),
                .recurrent_state = self.recurrent_state.reuseBuffer(other.recurrent_state),
                .layer_index = self.layer_index.reuseBuffer(other.layer_index),
            };
        }
    };

    pub fn init(
        config: Config,
        batch_dim: i64,
        max_seq_len: i64,
        cache_dtype: zml.DataType,
        recurrent_dtype: zml.DataType,
    ) KvCache {
        return .{
            .layer_types = config.text_config.layer_types,
            .self_attn = SelfAttnCache.init(config, batch_dim, max_seq_len, cache_dtype),
            .gated_delta_net = GatedDeltaNetCache.init(config, batch_dim, cache_dtype, recurrent_dtype),
        };
    }

    pub fn initBuffer(self: KvCache, io: std.Io, platform: *const zml.Platform, sharding: zml.sharding.Sharding) !zml.Bufferized(KvCache) {
        return .{
            .self_attn = try self.self_attn.initBuffer(io, platform, sharding),
            .gated_delta_net = try self.gated_delta_net.initBuffer(io, platform, sharding),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(KvCache)) void {
        SelfAttnCache.deinitBuffer(&self.self_attn);
        GatedDeltaNetCache.deinitBuffer(&self.gated_delta_net);
    }

    pub const LayerView = struct {
        parent: KvCache,
        cache: union(enum) {
            self_attn: SelfAttnCache,
            linear_attn: GatedDeltaNetCache,
        },
    };

    pub fn atLayer(self: KvCache, layer_index: usize) LayerView {
        return switch (getDenseIndex(self.layer_types, layer_index)) {
            .full_attention => |dense_index| .{
                .parent = self,
                .cache = .{ .self_attn = self.self_attn.atLayer(dense_index.layer_dense_index) },
            },
            .linear_attention => |dense_index| .{
                .parent = self,
                .cache = .{ .linear_attn = self.gated_delta_net.atLayer(dense_index.layer_dense_index) },
            },
        };
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{
            .layer_types = self.layer_types,
            .self_attn = self.self_attn.reuseBuffer(other.self_attn),
            .gated_delta_net = self.gated_delta_net.reuseBuffer(other.gated_delta_net),
        };
    }

    fn countLayers(layer_types: []const LayerType, layer_type: LayerType) i64 {
        var count: i64 = 0;
        for (layer_types) |registered_layer_type| {
            if (registered_layer_type == layer_type) count += 1;
        }
        return count;
    }

    fn getDenseIndex(layer_types: []const LayerType, layer_index: usize) union(enum) {
        full_attention: struct { layer_dense_index: usize },
        linear_attention: struct { layer_dense_index: usize },
    } {
        var self_attn_layer_index: usize = 0;
        var linear_attn_layer_index: usize = 0;
        for (layer_types[0..layer_index]) |layer_type| {
            switch (layer_type) {
                .full_attention => self_attn_layer_index += 1,
                .linear_attention => linear_attn_layer_index += 1,
            }
        }
        return switch (layer_types[layer_index]) {
            .full_attention => .{ .full_attention = .{ .layer_dense_index = self_attn_layer_index } },
            .linear_attention => .{ .linear_attention = .{ .layer_dense_index = linear_attn_layer_index } },
        };
    }
};

//========================Utils========================

fn softplus(x: zml.Tensor) zml.Tensor {
    return x.exp().addConstant(1).log();
}
