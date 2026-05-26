const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;
const tt_nn = @import("platforms/tt/nn");

const common = @import("../common.zig");
const inference = @import("inference.zig");

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

pub const Options = struct {
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
        var total_bytes: usize = 0;
        defer {
            const took = now.untilNow(io, .awake);
            const bytes_per_sec: u64 = @intFromFloat(
                @as(f64, @floatFromInt(total_bytes)) /
                    (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s),
            );
            std.log.scoped(.llama).info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });
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
        Llama.unloadBuffers(&buffers.model, allocator);
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.attention.Backend,
        shardings: common.Shardings,
        seqlen: usize,
        batch_size: u32,
        progress: *std.Progress.Node,
    ) !inference.CompiledModel {
        const params = try inference.CompilationParameters.init(allocator, self.inner, self.parsed_config.value, @intCast(seqlen), batch_size, backend, shardings);
        errdefer params.deinit();
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
            .{ .dout = .replicated, .d = .model },
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
        var total_bytes: usize = 0;

        defer {
            const took = now.untilNow(io, .awake);
            const bytes_per_sec: u64 = @intFromFloat(
                @as(f64, @floatFromInt(total_bytes)) /
                    (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s),
            );
            std.log.scoped(.llama).info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });
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
        if (self.lm_head) |*lm_head| lm_head.weight.deinit();
        Llama.unloadBuffers(&self.model, allocator);
    }
    /// Predicts the token at `token_index` position.
    /// Returns:
    ///  - updated `tokens`,
    ///  - updated KV cache
    ///  - a Rng state to allow for probabilistic generation
    pub fn forward(
        self: Model,
        tokens_: zml.Tensor,
        token_index_: zml.Tensor,
        last_token_index: zml.Tensor,
        kv_cache: KvCache,
        rng: zml.Tensor.Rng,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { zml.Tensor, KvCache } {
        const tokens = tokens_.withPartialTags(.{.s});
        // TT-FIX: rank-1 token_index — tt-mlir's `CacheFillUpdatePattern`
        // hands this block argument directly to `update_cache`.
        const token_index = token_index_.squeeze(.pos);

        // TT-FIX: draw Gumbel noise up front (outside the trace body) and
        // don't return the advanced rng — `ttnn.rand` is non-hoistable, and
        // returning rng strands `mesh_shard` mid-graph (`Non-hoistable op
        // found in the middle of hoistable ops`).
        var gumbel_noise: ?zml.Tensor = null;
        if (self.gen_opts.topk > 1) {
            const noise_shape = zml.Shape.init(
                .{ .b = tokens.dim(.b), .s = 1, .topk = self.gen_opts.topk },
                .f32,
            );
            gumbel_noise = rng.gumbel(noise_shape)[1];
        }

        const out, const updated_kv_cache = self.model.forward(
            tokens,
            token_index,
            kv_cache,
            attention_metadata,
            attention_parameters,
        );
        const new_tokens = blk: {
            // TT-FIX: one-hot `dot` over `.s` instead of `dynamic_slice` —
            // dynamic_slice confuses tt-mlir's SPMD partitioner (sharded `.d`
            // operand with global `slice_sizes`).
            const out_last = if (out.dim(.s) == 1)
                out
            else sel: {
                const sel_shape = zml.Shape.init(.{ .s = out.dim(.s) }, last_token_index.dtype());
                const onehot = zml.Tensor.iota(sel_shape, .s)
                    .cmp(.EQ, last_token_index.broad(sel_shape))
                    .convert(out.dtype());
                break :sel out.dot(onehot, .s).insertAxes(.d, .{.s});
            };
            const sampled = self.sampleTokens(self.lm_head, out_last, gumbel_noise, self.gen_opts);
            const toks = if (out.dim(.s) == 1)
                sampled.convert(tokens.dtype())
            else tok: {
                // TT-FIX: one-hot `select` instead of `dynamic_update_slice` —
                // the latter has no TTNN lowering and would CPU-hoist.
                const idx_shape = tokens.shape().withDtype(last_token_index.dtype());
                const at_last = zml.Tensor.iota(idx_shape, .s)
                    .cmp(.EQ, last_token_index.broad(idx_shape));
                break :tok at_last.select(
                    sampled.convert(tokens.dtype()).broad(tokens.shape()),
                    tokens,
                );
            };
            break :blk toks;
        };

        return .{ new_tokens.reuseBuffer(tokens), updated_kv_cache };
    }

    pub fn sampleTokens(
        self: Model,
        lm_head_: ?zml.nn.Linear,
        out_: zml.Tensor,
        gumbel_noise: ?zml.Tensor,
        opts: zml.nn.SamplingStrategy,
    ) zml.Tensor {
        const out = out_.withPartialTags(.{ .s, .d });

        var logits = blk: {
            // TT-FIX: `.s`-leading transpose around lm_head — rank-3
            // [B,S=1,N] otherwise pads to [1,B,1,N] for reduce_scatter where
            // `.s=1` next-to-last tile-pads each batch row to 32 → 32x blowup.
            const out_t = out.transpose(.{ .s, .b, .d });
            if (lm_head_) |lm_head| {
                break :blk lm_head.forward(out_t).rename(.{ .dout = .d }).transpose(.{ .b, .s, .d });
            } else {
                break :blk out_t.dot(self.model.embed_tokens.weight.withTags(.{ .voc, .d }), .d).transpose(.{ .b, .s, .voc });
            }
        };

        if (logits.shape().hasTag(.voc) == null)
            logits = logits.rename(.{ .d = .voc });

        if (opts.topk <= 1) return logits.argMax(.voc).indices.squeeze(.voc);

        const topk = logits.topK(.{ .topk = .voc }, opts.topk, .{});
        var x = topk.values;
        if (opts.temperature != 1.0) {
            x = x.scale(1 / opts.temperature);
        }
        x = x.add(gumbel_noise.?.convert(x.dtype()).transpose(x.shape()));
        const topk_idx = x.argMax(.topk).indices;
        return topk.indices.gather(.{ .topk = topk_idx.squeeze(.topk) }, .{});
    }
};

pub const Llama = struct {
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
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { zml.Tensor, KvCache } {
        const embeds = embed_tokensForward(self.embed_tokens, tokens);
        const b_size = embeds.dim(.b);
        const s_size = embeds.dim(.s);
        // Flatten the residual stream to rank-2 {.tokens, .d} so it matches
        // tt-mlir's 2D-native `ttnn.matmul` / `ttnn.rms_norm` lowering —
        // `ttnn.add` is then the same rank as the norm and the
        // `to_memory_config + reshape + to_memory_config` bridge that
        // otherwise sits between every residual add and the next norm goes
        // away. We split back to {.b, .s, .d} after the final norm so the
        // caller (sampling) sees the original rank-3 shape.
        var hidden = embeds.merge(.{ .tokens = .{ .b, .s } });
        var updated_kv_cache = kv_cache;

        for (self.layers, 0..) |layer, i| {
            const layer_attention_metadata: zml.attention.attention.Metadata = switch (attention_parameters) {
                .attnd => .{ .attnd = .{
                    .layer_id = zml.Tensor.scalar(i, .u16),
                    .conversation_id = attention_metadata.attnd.conversation_id,
                    .num_tokens = attention_metadata.attnd.num_tokens,
                } },
                .vanilla => attention_metadata,
                .cuda_fa2 => attention_metadata,
                .cuda_fa3 => attention_metadata,
                .tenstorrent => attention_metadata,
            };
            hidden, updated_kv_cache = layer.forward(
                hidden,
                b_size,
                s_size,
                token_index,
                updated_kv_cache.atLayer(i),
                layer_attention_metadata,
                attention_parameters,
            );
        }

        const normed = self.norm.forward(hidden);
        return .{
            normed.splitAxis(.tokens, .{ .b = b_size, .s = s_size }),
            updated_kv_cache.reuseBuffer(kv_cache),
        };
    }
};

fn embed_tokensForward(embed_tokens_: zml.nn.TokenEmbedding, tokens_: zml.Tensor) zml.Tensor {
    return embed_tokens_.forward(tokens_).withPartialTags(.{.d});
}

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

    pub fn forward(
        self: TransformerLayer,
        x0: zml.Tensor,
        b_size: i64,
        s_size: i64,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { zml.Tensor, KvCache } {
        // TT-FIX: residual stays rank-2 {.tokens,.d} matching ttnn's 2D-native
        // matmul/rms_norm — splits to {.b,.s,.d} only around SelfAttn.
        stdx.debug.assert(x0.rank() == 2 and x0.shape().hasTags(.{ .tokens, .d }), "TransformerLayer expected flattened input shape: {{.tokens, .d}}, received: {f}", .{x0});

        // Keep the residual stream replicated to avoid repeated gathers before q/k/v.
        const x0_replicated = x0.withPartitioning(.{ .d = .replicated });
        const x0_normalized = self.input_layernorm.forward(x0_replicated);

        const x0_norm_bsd = x0_normalized.splitAxis(.tokens, .{ .b = b_size, .s = s_size });
        const delta_bsd, const updated_kv_cache = self.self_attn.forward(
            x0_norm_bsd,
            token_index,
            kv_cache,
            attention_metadata,
            attention_parameters,
        );
        const delta0 = delta_bsd.merge(.{ .tokens = .{ .b, .s } });

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

    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const weight = self.weight.withTags(.{.d});
        const ctx = zml.module.CompilationContext.current();
        switch (ctx.platform.target) {
            // TT-FIX: fused custom_call — math-decomposed RMSNorm trips
            // shardy's `duplicate axis ref: "link"` on rank-2 + sharded weight.
            .tt => return tt_nn.rmsNormFused(x, weight, self.eps),
            .cpu, .cuda, .rocm, .tpu, .neuron => {
                const normalized = zml.nn.rmsNorm(x, -1, self.eps);
                return normalized.mul(weight.convert(x.dtype()).broad(x.shape()));
            },
        }
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

pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,

    q_norm: ?RmsNorm,
    k_norm: ?RmsNorm,

    o_proj: zml.nn.Linear,
    num_heads: i64 = undefined,
    num_kv_heads: i64 = 0,
    rope_opts: zml.nn.RopeOpts = undefined,

    pub fn init(store: zml.io.TensorStore.View, config: Config) !SelfAttn {
        var rope_scaling = config.rope_scaling;
        rope_scaling.setRopeTheta(config.rope_theta);
        return .{
            .q_proj = .init(store.createTensor("q_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .k_proj = .init(store.createTensor("k_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .v_proj = .init(store.createTensor("v_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .o_proj = .init(store.createTensor("o_proj.weight", .{ .dout, .d }, .{ .d = .model }), null, .d),
            // TODO(Corentin): fix that
            .q_norm = null,
            .k_norm = null,
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
            .rope_opts = .{
                .layout = if (config.hf_rope_impl) .sequential else .interleaved,
                .scaling = rope_scaling,
            },
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

        if (self.q_norm) |*q_norm| RmsNorm.unloadBuffers(q_norm);
        if (self.k_norm) |*k_norm| RmsNorm.unloadBuffers(k_norm);
    }

    /// Self zml.attention.attention.
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
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { zml.Tensor, KvCache } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;

        // Make hidden state replicated once and reuse it across q/k/v projections.
        // This avoids paying gather-style collectives independently for each projection.
        const x_qkv = x.withPartitioning(.{ .d = .replicated });

        var q = self.q_proj.forward(x_qkv).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto }).transpose(.{ .h, .b, .s, .hd });
        var k = self.k_proj.forward(x_qkv).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto }).transpose(.{ .h, .b, .s, .hd });
        var v = self.v_proj.forward(x_qkv).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto }).transpose(.{ .h, .b, .s, .hd });
        q = q.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        k = k.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });

        // In self-attention, .s axis is used both for keys and queries.
        const pos_index = b: {
            const temp = zml.Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };

        if (self.q_norm) |norm| q = norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        if (self.k_norm) |norm| k = norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const dtype = q.dtype();
        const new_kv_cache = kv_cache.update(k, v, token_index);
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);
        k = k.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });

        const attn_output = zml.attention.attention.attention(
            q,
            k,
            v,
            token_index,
            attention_metadata,
            attention_parameters,
        ).withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated });

        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        // TT-FIX: `.s`-leading transpose around o_proj — at batch=32 the
        // rank-3 [B,S=1,N] reduce_scatter-pads to [1,B,1,N] where `.s=1`
        // next-to-last tile-pads each batch row to 32 → 32x collective bloat.
        const attn_t = attn.transpose(.{ .s, .b, .d });
        const delta = self.o_proj.forward(attn_t)
            .rename(.{ .dout = .d })
            .transpose(.{ .b, .s, .d })
            .withPartitioning(.{ .d = .replicated });
        return .{ delta, new_kv_cache };
    }
};

/// TT-FIX: per-layer rank-4 `{.b,.h,.k,.hd}` tensors instead of a shared
/// rank-5 cache. tt-mlir's `update_cache` requires the cache operand to have
/// exactly one user (MLIR/SSA sense) — a shared tensor is read by one layer
/// and written by the next, so the op match fails.
pub const KvCache = struct {
    k: []zml.Tensor,
    v: []zml.Tensor,
    layer_index: ?u32,

    pub const Buffer = zml.Bufferized(KvCache);

    pub fn init(allocator: std.mem.Allocator, layer_shape: zml.Shape, num_layers: u32) !KvCache {
        const sharded_shape = layer_shape.withPartitioning(.{ .h = .model });
        const k = try allocator.alloc(zml.Tensor, num_layers);
        errdefer allocator.free(k);
        const v = try allocator.alloc(zml.Tensor, num_layers);
        for (k, v) |*kt, *vt| {
            kt.* = .fromShape(sharded_shape);
            vt.* = .fromShape(sharded_shape);
        }
        return .{ .k = k, .v = v, .layer_index = null };
    }

    pub fn deinit(kv: KvCache, allocator: std.mem.Allocator) void {
        allocator.free(kv.k);
        allocator.free(kv.v);
    }

    pub fn initBuffer(kv: KvCache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !Buffer {
        const k = try allocator.alloc(zml.Buffer, kv.k.len);
        errdefer allocator.free(k);
        const v = try allocator.alloc(zml.Buffer, kv.v.len);
        errdefer allocator.free(v);
        for (kv.k, k) |src, *dst| dst.* = try zml.Buffer.uninitialized(io, platform, src.shape(), sharding, .{});
        for (kv.v, v) |src, *dst| dst.* = try zml.Buffer.uninitialized(io, platform, src.shape(), sharding, .{});
        return .{ .k = k, .v = v };
    }

    pub fn deinitBuffer(kv: *Buffer, allocator: std.mem.Allocator) void {
        for (kv.k) |*b| b.deinit();
        for (kv.v) |*b| b.deinit();
        allocator.free(kv.k);
        allocator.free(kv.v);
    }

    pub fn keys(kv: KvCache) zml.Tensor {
        return kv.k[kv.layer_index orelse @panic("forgot to call atLayer")];
    }

    pub fn values(kv: KvCache) zml.Tensor {
        return kv.v[kv.layer_index orelse @panic("forgot to call atLayer")];
    }

    pub fn update(kv: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: ?zml.Tensor) KvCache {
        const layer = kv.layer_index orelse @panic("forgot to call atLayer");
        const arena = zml.module.CompilationContext.current().arena.allocator();
        const k = arena.dupe(zml.Tensor, kv.k) catch @panic("OOM duplicating KV cache");
        const v = arena.dupe(zml.Tensor, kv.v) catch @panic("OOM duplicating KV cache");

        const pos: zml.Tensor = token_index orelse zml.Tensor.scalar(0, .u32);
        k[layer] = zml.attention.attention.updateKvCache(kv.k[layer], new_k, pos).reuseBuffer(kv.k[layer]);
        v[layer] = zml.attention.attention.updateKvCache(kv.v[layer], new_v, pos).reuseBuffer(kv.v[layer]);
        return .{ .k = k, .v = v, .layer_index = kv.layer_index };
    }

    pub fn atLayer(kv: KvCache, layer_index: usize) KvCache {
        return .{ .k = kv.k, .v = kv.v, .layer_index = @intCast(layer_index) };
    }

    pub fn reuseBuffer(kv: KvCache, other: KvCache) KvCache {
        const arena = zml.module.CompilationContext.current().arena.allocator();
        const k = arena.dupe(zml.Tensor, kv.k) catch @panic("OOM duplicating KV cache");
        const v = arena.dupe(zml.Tensor, kv.v) catch @panic("OOM duplicating KV cache");
        for (k, other.k) |*t, o| t.* = t.reuseBuffer(o);
        for (v, other.v) |*t, o| t.* = t.reuseBuffer(o);
        return .{ .k = k, .v = v, .layer_index = null };
    }
};
