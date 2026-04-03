const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const common = @import("../common.zig");
const inference = @import("inference.zig");

const log = std.log.scoped(.qwen3_5);

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

fn kvHeadsAreReplicated(axis_size: i64, model_partitions: i64) bool {
    return axis_size > 0 and axis_size < model_partitions and @rem(model_partitions, axis_size) == 0;
}

fn partitionKvCacheShape(kv_shape: zml.Shape, kv_heads: i64, model_partitions: i64) zml.Shape {
    return if (kvHeadsAreReplicated(kv_heads, model_partitions))
        kv_shape.withPartitioning(.{ .h = .replicated })
    else
        kv_shape.withPartitioning(.{ .h = .model });
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
            log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });
        }

        const all_shardings = shardings.all();
        return zml.io.load(Model, &self.inner, allocator, io, platform, store, .{
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = progress,
            .shardings = &all_shardings,
            .parallelism = 16,
            .total_bytes = &total_bytes,
        });
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        _ = self;
        TextModel.unloadBuffers(&buffers.text_model, allocator);
        buffers.lm_head.weight.deinit();
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
        _ = backend;
        const params = inference.CompilationParameters.init(self.inner, self.parsed_config.value, @intCast(seqlen), shardings);
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

    text_model: TextModel,
    lm_head: zml.nn.Linear,

    config: Config,
    gen_options: GenOptions,
    special_tokens: SpecialTokens = .{
        .im_start_token_id = 248045,
        .im_end_token_id = 248046,
        .end_of_text_token_id = 248044,
    },

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, gen_options: GenOptions) !Model {
        // For some Qwen3.5 versions, the output projection lm_head has a standalone weight tensor, while for others it's the same as the input embedding layer
        const lm_head_prefix = if (store.hasKey("lm_head.weight")) "lm_head" else "model.language_model.embed_tokens";
        return .{
            .text_model = try .init(allocator, store.withPrefix("model.language_model"), config),
            .lm_head = .init(
                store.withPrefix(lm_head_prefix).createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
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
        const text_model_output, const updated_kv_cache = self.text_model.forward(tokens, token_index, kv_cache);
        const new_tokens, const new_rng = self.sampleTokens(text_model_output, rng);
        return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), updated_kv_cache, new_rng };
    }

    pub fn sampleTokens(
        self: Model,
        out: zml.Tensor,
        rng: zml.Tensor.Rng,
    ) struct { zml.Tensor, zml.Tensor.Rng } {
        const logits = self.lm_head.forward(out.withPartialTags(.{.d})).rename(.{ .dout = .voc });
        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, self.gen_options.sampling_strategy, rng);
        return .{ next_tokens, new_rng };
    }
};

//========================Text model========================

pub const TextModel = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    norm: RmsNorm,

    pub fn init(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: Config,
    ) !TextModel {
        const layers = try allocator.alloc(TransformerLayer, @intCast(config.text_config.num_hidden_layers));
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = try .init(store.withPrefix("layers").withLayer(i), config, i);
        }

        return .{
            .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, .{ .voc = .replicated, .d = .model }) },
            .layers = layers,
            .norm = RmsNorm.init(store.withPrefix("norm"), config.text_config.rms_norm_eps),
        };
    }

    pub fn deinit(self: TextModel, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TextModel), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
    }

    pub fn forward(
        self: TextModel,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
    ) struct { zml.Tensor, KvCache } {
        var hidden_states = self.embed_tokens.weight.gather(.{ .voc = tokens }, .{});

        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden_states, updated_kv_cache = layer.forward(hidden_states, token_index, updated_kv_cache.atLayer(i));
        }

        hidden_states = self.norm.forward(hidden_states);
        return .{ hidden_states, updated_kv_cache.reuseBuffer(kv_cache) };
    }
};

pub const TransformerLayer = struct {
    const Attn = union(enum) {
        self_attn: SelfAttn,
        linear_attn: GatedDeltaNet,
    };

    input_layernorm: RmsNorm,
    attn: Attn,
    mlp: Mlp,
    post_attention_layernorm: RmsNorm,

    pub fn init(store: zml.io.TensorStore.View, config: Config, layer_index: usize) !TransformerLayer {
        const is_full_attention = config.text_config.layer_types[layer_index] == .full_attention;
        return .{
            .input_layernorm = RmsNorm.init(store.withPrefix("input_layernorm"), config.text_config.rms_norm_eps),
            .attn = if (is_full_attention)
                .{ .self_attn = try .init(store.withPrefix("self_attn"), config) }
            else
                .{ .linear_attn = .init(store.withPrefix("linear_attn"), config) },
            .mlp = .init(store.withPrefix("mlp")),
            .post_attention_layernorm = RmsNorm.init(store.withPrefix("post_attention_layernorm"), config.text_config.rms_norm_eps),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        switch (self.attn) {
            .self_attn => |*self_attn| SelfAttn.unloadBuffers(self_attn),
            .linear_attn => |*linear_attn| GatedDeltaNet.unloadBuffers(linear_attn),
        }
        Mlp.unloadBuffers(&self.mlp);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
    }

    pub fn forward(
        self: TransformerLayer,
        x0: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache.LayerView,
    ) struct { zml.Tensor, KvCache } {
        const x0_replicated = x0.withPartitioning(.{ .d = .replicated });
        const normalized_x0 = self.input_layernorm.forward(x0_replicated);

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

        const x1 = attention_output.add(x0_replicated).withPartitioning(.{ .d = .replicated });
        const normalized_hidden = self.post_attention_layernorm.forward(x1);

        const mlp_output = self.mlp.forward(normalized_hidden).withPartitioning(.{ .d = .replicated });

        return .{ mlp_output.add(x1).withPartitioning(.{ .d = .replicated }).reuseBuffer(x0), updated_kv_cache };
    }
};

pub const Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .up_proj = .init(
                store.withPrefix("up_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                store.withPrefix("up_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }),
                .d,
            ),
            .gate_proj = .init(
                store.withPrefix("gate_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                store.withPrefix("gate_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }),
                .d,
            ),
            .down_proj = .init(
                store.withPrefix("down_proj").createTensor("weight", .{ .d, .dout }, .{ .d = .replicated, .dout = .model }),
                store.withPrefix("down_proj").maybeCreateTensor("bias", .{.d}, .{ .d = .replicated }),
                .dout,
            ),
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
        const up_projed = self.up_proj.forward(x);
        const gate = self.gate_proj.forward(x);
        const hidden = gate.silu().mul(up_projed);

        const output = self.down_proj.forward(hidden);
        return output;
    }
};

pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,

    q_norm: RmsNorm,
    k_norm: RmsNorm,

    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    rotary_dim: i64,
    rotary_embed: TextRotaryEmbedding,
    o_proj: zml.nn.Linear,

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
            .k_proj = initProj(store.withPrefix("k_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .v_proj = initProj(store.withPrefix("v_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .o_proj = initProj(store.withPrefix("o_proj"), .{ .dout = .replicated, .d = .model }, .{ .dout = .replicated }),
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
        self.k_proj.weight.deinit();
        if (self.k_proj.bias) |*bias| bias.deinit();
        self.v_proj.weight.deinit();
        if (self.v_proj.bias) |*bias| bias.deinit();
        self.o_proj.weight.deinit();
        if (self.o_proj.bias) |*bias| bias.deinit();
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    fn projectQAndGate(self: SelfAttn, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const q_proj = self.q_proj.forward(x).splitAxis(.dout, .{ .h = self.num_heads, .hd = 2 * self.head_dim });
        const q, var gate = q_proj.chunkExact(.hd, 2);
        gate = gate.merge(.{ .d_out_proj = .{ .h, .hd } });
        return .{ q, gate };
    }

    fn projectKV(self: SelfAttn, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;
        const k = self.k_proj.forward(x).splitAxis(.dout, .{ .h = num_kv_heads, .hd = self.head_dim });
        const v = self.v_proj.forward(x).splitAxis(.dout, .{ .h = num_kv_heads, .hd = self.head_dim });
        return .{ k, v };
    }

    fn partitionProjectedKv(tensor: zml.Tensor, replicate_kv_heads: bool) zml.Tensor {
        return if (replicate_kv_heads)
            tensor.withPartitioning(.{ .s = .replicated, .h = .replicated, .hd = .replicated })
        else
            tensor.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
    }

    fn partitionCachedKv(tensor: zml.Tensor, replicate_kv_heads: bool) zml.Tensor {
        return if (replicate_kv_heads)
            tensor.rename(.{ .s = .k }).withPartitioning(.{ .k = .replicated, .h = .replicated, .hd = .replicated })
        else
            tensor.rename(.{ .s = .k }).withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });
    }

    pub fn forward(
        self: SelfAttn,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache.SelfAttnCache,
    ) struct { zml.Tensor, KvCache.SelfAttnCache } {
        const model_partitions = zml.module.CompilationContext.current().partitioning.numPartitionsForLogicalAxis(self.q_proj.weight.shape(), .model) catch unreachable;
        const replicate_kv_heads = kvHeadsAreReplicated(self.num_kv_heads, model_partitions);
        const x_qkv = x.withPartitioning(.{ .d = .replicated });

        var q, var gate = self.projectQAndGate(x_qkv);
        var k, var v = self.projectKV(x_qkv);
        q = q.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        gate = gate.withPartitioning(.{ .s = .replicated, .d_out_proj = .model });
        k = partitionProjectedKv(k, replicate_kv_heads);
        v = partitionProjectedKv(v, replicate_kv_heads);
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
        k = partitionProjectedKv(k, replicate_kv_heads);
        v = partitionProjectedKv(v, replicate_kv_heads);

        const new_kv_cache = kv_cache.update(k, v, token_index.convert(.u32));
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);
        q = q.rename(.{ .s = .q }).withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated });
        k = partitionCachedKv(k, replicate_kv_heads);
        v = partitionCachedKv(v, replicate_kv_heads);

        const attn_output = zml.attention.causal(
            q,
            k,
            v,
            token_index,
            zml.attention.Metadata.init(.fromBackend(.vanilla, x.dim(.s), self.num_heads)),
            zml.attention.Parameters.init(.fromBackend(.vanilla)),
        ).withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated }).rename(.{ .q = .s }).merge(.{ .d_out_proj = .{ .h, .hd } });

        const gated_output = attn_output.mul(gate.sigmoid());
        const projected_output = self.o_proj.forward(gated_output.rename(.{ .d_out_proj = .d })).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });

        return .{ projected_output, new_kv_cache };
    }
};

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
    in_proj_z: zml.nn.Linear,
    in_proj_b: zml.nn.Linear,
    in_proj_a: zml.nn.Linear,
    out_proj: zml.nn.Linear,
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
            .in_proj_z = initProj(store.withPrefix("in_proj_z"), .{ .dout = .model, .d = .replicated }),
            .in_proj_b = initProj(store.withPrefix("in_proj_b"), .{ .dout = .model, .d = .replicated }),
            .in_proj_a = initProj(store.withPrefix("in_proj_a"), .{ .dout = .model, .d = .replicated }),
            .out_proj = initProj(store.withPrefix("out_proj"), .{ .dout = .replicated, .d = .model }),
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
        self.in_proj_z.weight.deinit();
        self.in_proj_b.weight.deinit();
        self.in_proj_a.weight.deinit();
        self.out_proj.weight.deinit();
        self.conv1d_weight.deinit();
        self.dt_bias.deinit();
        self.aLog.deinit();
        RmsNormGated.unloadBuffers(&self.norm);
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
        const projected_qkv = self.in_proj_qkv.forward(x_in).rename(.{ .dout = .mix }).withPartitioning(.{ .s = .replicated, .mix = .model });
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

        const z = self.in_proj_z.forward(x_in).splitAxis(.dout, .{ .vh = self.num_v_heads, .vhd = self.head_v_dim }).withPartitioning(.{ .s = .replicated, .vh = .model, .vhd = .replicated });
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

        const output = self.out_proj.forward(core_attn_out_normed.merge(.{ .d = .{ .vh, .vhd } })).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
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

        pub fn init(config: Config, batch_dim: i64, max_seq_len: i64, dtype: zml.DataType, model_partitions: i64) SelfAttnCache {
            const num_self_attn_layers = countLayers(config.text_config.layer_types, .full_attention);
            const kv_shape = zml.Shape.init(.{
                .b = batch_dim,
                .layer = num_self_attn_layers,
                .s = max_seq_len,
                .h = config.text_config.num_key_value_heads,
                .hd = config.text_config.head_dim,
            }, dtype);
            const sharded_kv_shape = partitionKvCacheShape(kv_shape, config.text_config.num_key_value_heads, model_partitions);
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
        model_partitions: i64,
    ) KvCache {
        return .{
            .layer_types = config.text_config.layer_types,
            .self_attn = SelfAttnCache.init(config, batch_dim, max_seq_len, cache_dtype, model_partitions),
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
