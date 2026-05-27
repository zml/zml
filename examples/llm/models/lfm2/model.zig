const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;
const attention = zml.attention.attention;
const tt_nn = @import("platforms/tt/nn");

const common = @import("../common.zig");
const inference = @import("inference.zig");

const log = std.log.scoped(.lfm);

pub const Config = struct {
    architectures: []const []const u8,
    block_auto_adjust_ff_dim: bool,
    block_dim: u32,
    block_ff_dim: u32,
    block_ffn_dim_multiplier: f32,
    block_mlp_init_scale: f32,
    block_multiple_of: u32,
    block_norm_eps: f32,
    block_out_init_scale: f32,
    block_use_swiglu: bool,
    block_use_xavier_init: bool,
    bos_token_id: u32,
    conv_L_cache: u32,
    conv_bias: bool,
    conv_dim: u32,
    conv_use_xavier_init: bool,
    dtype: []const u8,
    eos_token_id: u32,
    hidden_size: u32,
    initializer_range: f32,
    intermediate_size: u32,
    layer_types: []const []const u8,
    max_position_embeddings: u32,
    model_type: []const u8,
    norm_eps: f32,
    num_attention_heads: u32,
    num_heads: u32,
    num_hidden_layers: u32,
    num_key_value_heads: u32,
    pad_token_id: u32,
    rope_theta: f32,
    tie_embedding: bool,
    transformers_version: []const u8,
    use_cache: bool,
    use_pos_enc: bool,
    vocab_size: u32,
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

        return .{
            .inner = .init(allocator, store, parsed_config.value, generation),
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
        // TT-FIX: force single-kernel mode. Composed mode hands inter-kernel
        // buffers across PJRT executes which trips a mesh-dispatch assertion
        // on multi-chip (`Can't get a single buffer from host storage
        // distributed over mesh shape MeshShape([1,2])`).
        const params = try inference.CompilationParameters.init(allocator, self.inner, self.parsed_config.value, @intCast(seqlen), backend, true, shardings);
        errdefer params.deinit();
        return inference.CompiledModel.init(allocator, io, @constCast(platform), self, self.inner, params, progress);
    }
};

pub const Buffers = zml.Bufferized(Model);

pub const Model = struct {
    embed_tokens: TokenEmbedding,
    lm_head: LmHead,
    layers: []DecoderLayer,
    num_attention_layers: usize,
    num_conv_layers: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        root_store: zml.io.TensorStore.View,
        config: Config,
        generation: common.GenerationOptions,
    ) Model {
        const store = root_store.withPrefix("model");
        stdx.debug.assert(config.layer_types.len == config.num_hidden_layers, "Expected layer_types len {} to match num_hidden_layers {}", .{ config.layer_types.len, config.num_hidden_layers });

        const layers = allocator.alloc(DecoderLayer, config.num_hidden_layers) catch unreachable;
        var num_attention_layers: usize = 0;
        var num_conv_layers: usize = 0;
        for (layers, 0..) |*layer, i| {
            const layer_store = store.withPrefix("layers").withLayer(i);
            const kind = DecoderLayer.parseOperatorKind(config.layer_types[i]);
            switch (kind) {
                .conv => num_conv_layers += 1,
                .full_attention => num_attention_layers += 1,
            }
            layer.* = DecoderLayer.init(config, layer_store, kind);
        }

        return .{
            .embed_tokens = .init(store.withPrefix("embed_tokens")),
            .lm_head = LmHead.init(store, config, generation.sampling_strategy),
            .layers = layers,
            .num_attention_layers = num_attention_layers,
            .num_conv_layers = num_conv_layers,
        };
    }

    pub fn loadBuffers(
        self: *const Model,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: common.Shardings,
    ) !zml.Bufferized(Model) {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);
        var total_bytes: usize = 0;
        defer {
            const took = now.untilNow(io, .awake);
            const took_ns: usize = @max(1, @as(usize, @intCast(took.toNanoseconds())));
            log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{
                total_bytes,
                took,
                total_bytes * std.time.ns_per_s / took_ns,
            });
        }
        const all_shardings = shardings.all();
        return zml.io.load(Model, self, allocator, io, platform, store, .{
            .dma_chunks = 8,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = progress,
            .parallelism = 16,
            .total_bytes = &total_bytes,
            .shardings = &all_shardings,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        LmHead.unloadBuffers(&self.lm_head);
        for (self.layers) |*layer| {
            DecoderLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
    }

    pub fn forward(
        self: Model,
        tokens: zml.Tensor,
        tokens_position_offset: zml.Tensor,
        actual_seq_len: zml.Tensor,
        rng: zml.Tensor.Rng,
        cache_: Cache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
        conv_parameters: ConvParameters,
    ) struct { zml.Tensor, Cache } {
        stdx.debug.assert(tokens.shape().hasTags(.{ .batch, .seq }), "Tokens should have tags {{.batch, .seq}}, got {f}", .{tokens.shape()});

        const embeds = self.embed_tokens.forward(tokens);

        var hidden = embeds;
        var cache = cache_;
        var conv_idx: usize = 0;
        var attn_idx: usize = 0;
        for (self.layers) |layer| {
            hidden, cache = layer.forward(
                hidden,
                tokens_position_offset,
                actual_seq_len,
                cache,
                conv_idx,
                attn_idx,
                attention_metadata,
                attention_parameters,
                conv_parameters,
            );
            switch (layer.operator) {
                .conv => conv_idx += 1,
                .self_attn => attn_idx += 1,
            }
        }
        // TT-FIX: drop the advanced rng from the return value. Returning it
        // tags a `to_layout` after the epilogue's `mesh_shard` and trips
        // TTNNTraceHoistTransform with "Non-hoistable op found in middle of
        // hoistable ops". Matches llama.
        const new_tokens, _ = self.lm_head.forward(hidden, self.embed_tokens, tokens, rng);
        return .{ new_tokens, cache.reuseBuffer(cache_) };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }
};

pub const TokenEmbedding = struct {
    weight: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) TokenEmbedding {
        // Keep replicated. With `.d = .model` shardy's verifier rejects
        // the gather's `dynamic_slice slice_size 2048 > dim 1024` on the
        // decode compile (it doesn't rewrite slice_sizes when sharding
        // the operand). We instead work around the host-vector
        // `ttnn::slice` runtime path by handling tokens specially below.
        return .{ .weight = store.createTensor("weight", .{ .voc, .d }, .replicated) };
    }

    pub fn forward(self: TokenEmbedding, tokens: zml.Tensor) zml.Tensor {
        stdx.debug.assert(tokens.dtype().isInteger(), "TokenEmbedding expects an integer input, received: {f}", .{tokens});
        stdx.debug.assert(self.weight.rank() == 2, "TokenEmbedding expects it's weight zml.Tensor to be a 2D matrix, got {f}", .{self.weight});
        // TT-FIX: single-token gather (decode `seq=1`) lowers to
        // `ttnn.slice_dynamic` whose runtime path reads the index as a
        // host vector — fails on a replicated index with N>1 shards.
        // Multi-token gather (`seq>=2`) lowers to `ttnn.embedding` which
        // runs on device. Pad to seq=2 and slice the first row back out.
        if (tokens.dim(.seq) != 1) {
            return self.weight.gather(.{ .voc = tokens }, .{});
        }
        const padded = zml.Tensor.concatenate(&.{ tokens, tokens }, tokens.axis(.seq));
        return self.weight.gather(.{ .voc = padded }, .{}).slice1d(.seq, .{ .end = 1 });
    }

    pub fn unembed(self: TokenEmbedding, embeds: zml.Tensor) zml.Tensor {
        stdx.debug.assert(embeds.shape().hasTags(.{.d}), "TokenEmbedding expects the input embeds to have a .d tag, got {f}", .{embeds.shape()});
        return self.weight.dot(embeds, .d);
    }
};

pub const LmHead = struct {
    embedding_norm: RmsNorm,
    sampling_strategy: zml.nn.SamplingStrategy,

    pub fn init(store: zml.io.TensorStore.View, config: Config, sampling_strategy: zml.nn.SamplingStrategy) LmHead {
        return .{
            .embedding_norm = RmsNorm.init(store.withPrefix("embedding_norm"), config.norm_eps, .d),
            .sampling_strategy = sampling_strategy,
        };
    }

    pub fn forward(self: LmHead, hidden: zml.Tensor, embed_tokens: TokenEmbedding, tokens: zml.Tensor, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
        const ctx = zml.module.CompilationContext.current();
        switch (ctx.platform.target) {
            .tt => {
                // TT-FIX: rename `batch`/`seq` -> `b`/`s` to match tt-mlir's
                // expected axis names, then do the tied-embedding matmul
                // with `s` leading so the unembedding stays on the optimized
                // matmul path (mirrors the llama 3B tied-embed fix).
                const normed = self.embedding_norm.forward(hidden).rename(.{ .batch = .b, .seq = .s });
                const normed_t = normed.transpose(.{ .s, .b, .d });
                var logits = normed_t.dot(embed_tokens.weight.withTags(.{ .voc, .d }), .d).transpose(.{ .b, .s, .voc });
                if (logits.shape().hasTag(.voc) == null) logits = logits.rename(.{ .d = .voc });

                if (self.sampling_strategy.topk <= 1 or logits.dim(.s) != 1) {
                    // argMax works at any seq length (prefill takes the last).
                    // TT-FIX: tt.sampling kernel is rank-2 `{.b,.topk}` only —
                    // only usable on decode (seq=1).
                    const new_tokens = logits.argMax(.voc).indices.squeeze(.voc).rename(.{ .b = .batch, .s = .seq });
                    return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), rng };
                }
                const topk = logits.topK(.{ .topk = .voc }, self.sampling_strategy.topk, .{});
                const tok = tt_nn.sampling(topk.values.squeeze(.s), topk.indices.squeeze(.s), self.sampling_strategy)
                    .insertAxes(.last, .{.s})
                    .rename(.{ .b = .batch, .s = .seq });
                return .{ tok.convert(tokens.dtype()).reuseBuffer(tokens), rng };
            },
            .cpu, .cuda, .rocm, .tpu, .neuron => {
                const logits = embed_tokens.unembed(self.embedding_norm.forward(hidden));
                const new_tokens, const new_rng = zml.nn.sampleTokens(logits, self.sampling_strategy, rng);
                return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), new_rng };
            },
        }
    }

    pub fn unloadBuffers(self: *zml.Bufferized(LmHead)) void {
        RmsNorm.unloadBuffers(&self.embedding_norm);
    }
};

pub const OperatorKind = enum { conv, full_attention };
const Operator = union(enum) { conv: ShortConv, self_attn: Attention };

pub const DecoderLayer = struct {
    operator: Operator,
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
    feed_forward: Mlp,

    pub fn parseOperatorKind(layer_type: []const u8) OperatorKind {
        return std.meta.stringToEnum(OperatorKind, layer_type) orelse {
            stdx.debug.assert(false, "Unsupported layer type {s}", .{layer_type});
            unreachable;
        };
    }

    pub fn init(config: Config, store: zml.io.TensorStore.View, kind: OperatorKind) DecoderLayer {
        const operator_norm = RmsNorm.init(store.withPrefix("operator_norm"), config.norm_eps, .d);
        const ffn_norm = RmsNorm.init(store.withPrefix("ffn_norm"), config.norm_eps, .d);
        const feed_forward = Mlp.init(store.withPrefix("feed_forward"));
        const operator: Operator = switch (kind) {
            .conv => .{ .conv = ShortConv.init(config, store.withPrefix("conv")) },
            .full_attention => .{ .self_attn = Attention.init(config, store.withPrefix("self_attn")) },
        };

        return .{
            .operator = operator,
            .operator_norm = operator_norm,
            .ffn_norm = ffn_norm,
            .feed_forward = feed_forward,
        };
    }

    pub fn forward(
        self: DecoderLayer,
        input: zml.Tensor,
        tokens_position_offset: zml.Tensor,
        actual_seq_len: zml.Tensor,
        cache_: Cache,
        conv_idx: usize,
        attn_idx: usize,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
        conv_parameters: ConvParameters,
    ) struct { zml.Tensor, Cache } {
        var cache = cache_;
        // TT-FIX: pin the residual stream `.d = .replicated` at each layer
        // boundary. Matches llama. The embedding output is `.d`-sharded,
        // so without this every conv layer (whose weights are replicated)
        // would force shardy to reshard between `.d`-sharded input and
        // replicated weights — that's the `duplicate axis ref "link"`
        // pattern. Forcing the residual replicated puts the reshard /
        // all-gather at one well-defined spot per layer.
        const input_repl = input.withPartitioning(.{ .d = .replicated });
        const residual = switch (self.operator) {
            .conv => |operator| b: {
                const residual, const updated_conv_cache = operator.forward(
                    self.operator_norm.forward(input_repl),
                    tokens_position_offset,
                    actual_seq_len,
                    cache.conv.atLayer(conv_idx),
                    conv_parameters,
                );
                cache.conv = updated_conv_cache;
                break :b residual;
            },
            .self_attn => |operator| b: {
                const residual, const updated_kv_cache = operator.forward(
                    self.operator_norm.forward(input_repl),
                    tokens_position_offset,
                    cache.kv.atLayer(attn_idx),
                    attention_metadata,
                    attention_parameters,
                );
                cache.kv = updated_kv_cache;
                break :b residual;
            },
        };

        const x = input_repl.add(residual).withPartitioning(.{ .d = .replicated });
        const out = x.add(self.feed_forward.forward(self.ffn_norm.forward(x))).withPartitioning(.{ .d = .replicated });

        return .{
            out.reuseBuffer(input),
            cache.reuseBuffer(cache_),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(DecoderLayer)) void {
        switch (self.operator) {
            .conv => |*operator| ShortConv.unloadBuffers(operator),
            .self_attn => |*operator| Attention.unloadBuffers(operator),
        }
        RmsNorm.unloadBuffers(&self.operator_norm);
        RmsNorm.unloadBuffers(&self.ffn_norm);
        Mlp.unloadBuffers(&self.feed_forward);
    }
};

pub const ConvParameters = struct { is_prefill: bool };

pub const ShortConv = struct {
    in_proj: Linear,
    out_proj: Linear,
    kernel: zml.Tensor,
    config: Config,

    pub fn init(config: Config, store: zml.io.TensorStore.View) ShortConv {
        stdx.debug.assert(!config.conv_bias, "conv_bias is not supported.", .{});
        // Conv path stays replicated — depthwise conv1d can't be split.
        return .{
            .in_proj = initLinear(store.withPrefix("in_proj"), .d, .replicated),
            .out_proj = initLinear(store.withPrefix("out_proj"), .d, .replicated),
            .kernel = store.createTensor("conv.weight", .{ .out, .in, .kernel_size }, .replicated),
            .config = config,
        };
    }

    pub fn forward(self: ShortConv, input: zml.Tensor, tokens_position_offset: zml.Tensor, actual_seq_len: zml.Tensor, cache_: ConvCache, parameters: ConvParameters) struct { zml.Tensor, ConvCache } {
        const cache = cache_;
        const BCx = self.in_proj.forward(input);

        const B, const C, const x = BCx.chunkExact(.d, 3);
        const Bx = B.mul(x);

        // TT-FIX: per-layer rank-3 conv state (no `.layer` axis); scatter
        // operates directly on this layer's `[batch, seq, d]` buffer.
        const layer_state = cache.current();

        const conv_out, const new_state = if (parameters.is_prefill) b: {
            const actual_seq_len_i32 = actual_seq_len.convert(.i32);
            const start = actual_seq_len_i32.sub(zml.Tensor.scalar(@as(i32, @intCast(self.config.conv_L_cache)), .i32)).maximum(zml.Tensor.scalar(@as(i32, 0), .i32));
            const cache_seq_indices = lkp: {
                const n = actual_seq_len_i32.sub(start);
                const left_pad = zml.Tensor.scalar(@as(i32, @intCast(self.config.conv_L_cache)), .i32).sub(n);
                const sh = tokens_position_offset.shape().insert(.last, .{ .seq = self.config.conv_L_cache });
                break :lkp zml.Tensor.iota(sh, .seq).convert(.u32).add(left_pad.convert(.u32).broad(sh)).broad(sh);
            };
            // TT-FIX: avoid `stablehlo.dynamic_slice` with a runtime index
            // here — tt-metal's `ttnn::slice` reads the index as a host
            // vector (`host_buffer::get_host_buffer`) and trips
            // `buffers.size() == 1` because a replicated scalar on a
            // multi-chip mesh has N host shards. `stablehlo.gather` with
            // a precomputed index tensor lowers to `ttnn.embedding`
            // instead and runs entirely on device.
            const scatter_data = sd: {
                // Indices tagged `.window` (not `.seq`) so `gather` can
                // attach them to the `.seq` axis of `Bx` without a tag clash.
                const sh = tokens_position_offset.shape().insert(.last, .{ .window = self.config.conv_L_cache });
                const bx_seq_indices = zml.Tensor.iota(sh, .window).convert(.u32)
                    .add(start.convert(.u32).broad(sh));
                break :sd Bx.gather(.{ .seq = bx_seq_indices }, .{}).rename(.{ .window = .seq });
            };
            const updated = layer_state.scatterSlices(.{ .seq = cache_seq_indices }, scatter_data, .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override });
            const padding = self.config.conv_L_cache - 1;
            const conv_out_ = zml.Tensor.conv1d(Bx, self.kernel, .{
                .padding = &.{ padding, 0 },
                .input_batch_dimension = Bx.axis(.batch),
                .input_feature_dimension = Bx.axis(.d),
                .input_spatial_dimensions = Bx.axis(.seq),
                .kernel_input_feature_dimension = self.kernel.axis(.in),
                .kernel_output_feature_dimension = self.kernel.axis(.out),
                .kernel_spatial_dimensions = self.kernel.axis(.kernel_size),
                .output_batch_dimension = Bx.axis(.batch),
                .output_feature_dimension = Bx.axis(.d),
                .output_spatial_dimensions = Bx.axis(.seq),
                .feature_group_count = Bx.dim(.d),
            });
            break :b .{ conv_out_.slice1d(.seq, .{ .end = Bx.dim(.seq) }), updated };
        } else b: {
            // TT-FIX: `rollRight1d + scatterSlices` lowered to ~8 small
            // `ttnn.scatter` ops per layer (~80 total across 10 conv layers).
            // Replace with `concat(cache[1:K], Bx)` — 1 slice + 1 concat. Math
            // is identical: with K=3 we always want the last K values as a
            // sliding window. The cache is zero-initialized so early positions
            // (pos < K-1) work correctly without the `pos.clamp(0, K-1)`
            // scatter index.
            const K_dim: i64 = @intCast(self.config.conv_L_cache);
            const updated = zml.Tensor.concatenate(&.{
                layer_state.slice1d(.seq, .{ .start = 1, .end = K_dim }),
                Bx,
            }, layer_state.axis(.seq));
            const kernel = self.kernel.squeeze(.in).rename(.{ .out = .d, .kernel_size = .seq });
            const sh = kernel.shape().insert(.last, .{ .batch = tokens_position_offset.dim(.batch) });
            // Read back this layer's full window for the elementwise conv.
            break :b .{ updated.mul(kernel.broad(sh).transpose(.{ .batch, .seq, .d })).sum(.seq), updated };
        };

        const y = C.mul(conv_out);

        const output = self.out_proj.forward(y);
        return .{ output.reuseBuffer(input), cache.replace(new_state) };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(ShortConv)) void {
        Linear.unloadBuffers(&self.in_proj);
        Linear.unloadBuffers(&self.out_proj);
        self.kernel.deinit();
    }
};

pub const Attention = struct {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    q_layernorm: RmsNorm,
    k_layernorm: RmsNorm,
    head_dim: usize,
    num_key_value_groups: usize,
    rope_opts: zml.nn.RopeOpts,

    pub fn init(config: Config, store: zml.io.TensorStore.View) Attention {
        const head_dim = config.hidden_size / config.num_attention_heads;
        const num_key_value_groups = config.num_attention_heads / config.num_key_value_heads;
        return .{
            // TT-FIX: q/k/v shard output (head columns) on `.model`; out_proj
            // shards input (sharded `.d` from merged heads). Matches llama.
            .q_proj = initLinear(store.withPrefix("q_proj"), .d, .{ .out = .model }),
            .k_proj = initLinear(store.withPrefix("k_proj"), .d, .{ .out = .model }),
            .v_proj = initLinear(store.withPrefix("v_proj"), .d, .{ .out = .model }),
            .out_proj = initLinear(store.withPrefix("out_proj"), .d, .{ .d = .model }),
            .q_layernorm = RmsNorm.init(store.withPrefix("q_layernorm"), config.norm_eps, .hd),
            .k_layernorm = RmsNorm.init(store.withPrefix("k_layernorm"), config.norm_eps, .hd),
            .head_dim = head_dim,
            .num_key_value_groups = num_key_value_groups,
            .rope_opts = .{ .layout = .sequential, .scaling = .{ .default = .{ .rope_theta = config.rope_theta } } },
        };
    }

    pub fn forward(
        self: Attention,
        x: zml.Tensor,
        tokens_position_offset: zml.Tensor,
        kv_cache: KvCache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { zml.Tensor, KvCache } {
        // TT-FIX: replicate input `.d` so q/k/v share a gather; partition
        // the projected heads `.h = .model` to match the sharded KvCache.
        // TT-FIX: rename `.batch` → `.b` so `zml.attention.attention` finds
        // the batch tag and stays in the rank-4 path. Then transpose
        // `.{.h, .b, .seq, .hd}` so the metal kernel's
        // `cos.shape[0]==1 && cos.shape[1]==1` assert passes — required for
        // `ttnn.rotary_embedding` to fuse. Mirrors the Llama path.
        const x_qkv = x.withPartitioning(.{ .d = .replicated }).rename(.{ .batch = .b });
        var q = self.q_proj.forward(x_qkv).splitAxis(-1, .{ .h = .auto, .hd = self.head_dim }).transpose(.{ .h, .b, .seq, .hd });
        var k = self.k_proj.forward(x_qkv).splitAxis(-1, .{ .h = .auto, .hd = self.head_dim }).transpose(.{ .h, .b, .seq, .hd });
        var v = self.v_proj.forward(x_qkv).splitAxis(-1, .{ .h = .auto, .hd = self.head_dim }).transpose(.{ .h, .b, .seq, .hd });
        q = q.withPartitioning(.{ .h = .model, .hd = .replicated });
        k = k.withPartitioning(.{ .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .h = .model, .hd = .replicated });

        q = self.q_layernorm.forward(q);
        k = self.k_layernorm.forward(k);

        const token_positions = b: {
            const sh = tokens_position_offset.shape().rename(.{ .batch = .b }).insert(.last, .{ .seq = x.dim(.seq) });
            // Match the dtype of `tokens_position_offset` (now i32 to satisfy
            // tt-metal's update_cache `update_idxs` INT32 requirement).
            break :b zml.Tensor.iota(sh, .seq).convert(tokens_position_offset.dtype()).add(tokens_position_offset.rename(.{ .batch = .b }).broad(sh));
        };

        q = zml.nn.rope(q, token_positions, self.rope_opts);
        k = zml.nn.rope(k, token_positions, self.rope_opts);

        q = q.rename(.{ .seq = .q });
        k = k.rename(.{ .seq = .k });
        v = v.rename(.{ .seq = .k });

        // Cache update uses `.batch` tag (KV cache stores `.batch`), so rename
        // `.b` → `.batch` only for the cache write, then back to `.b` for SDPA.
        const new_kv_cache = kv_cache.update(k.rename(.{ .b = .batch }), v.rename(.{ .b = .batch }), tokens_position_offset);
        k = new_kv_cache.keys().rename(.{ .batch = .b });
        v = new_kv_cache.values().rename(.{ .batch = .b });

        stdx.debug.assert(q.dim(.b) == 1, "LFM attention currently expects batch size 1, got {}", .{q.dim(.b)});
        const tok_pos_b = tokens_position_offset.rename(.{ .batch = .b });
        const attn = zml.attention.attention.attention(q, k, v, tok_pos_b, attention_metadata, attention_parameters).merge(.{ .d = .{ .h, .hd } }).rename(.{ .b = .batch, .q = .seq });

        // TT-FIX: terminate the head-shard via reduce_scatter at out_proj,
        // then pin `.d = .replicated` so the residual add stays replicated.
        const delta = self.out_proj.forward(attn).withPartitioning(.{ .d = .replicated });
        return .{ delta.reuseBuffer(x), new_kv_cache.reuseBuffer(kv_cache) };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Attention)) void {
        Linear.unloadBuffers(&self.q_proj);
        Linear.unloadBuffers(&self.k_proj);
        Linear.unloadBuffers(&self.v_proj);
        Linear.unloadBuffers(&self.out_proj);
        RmsNorm.unloadBuffers(&self.q_layernorm);
        RmsNorm.unloadBuffers(&self.k_layernorm);
    }
};

/// TT-FIX: per-layer rank-4 KV / rank-3 conv caches (no `.layer` axis on
/// the buffers; selection happens at compile-time via `atLayer(idx)`).
///
/// Two reasons:
///
/// 1. With KvCache `.h = .model` sharding, the cache argument's sharding
///    matches the `.h`-sharded scatter result from q/k/v projection ->
///    shardy doesn't need to reshard between `replicated={"link"}` and
///    `[{?},{?},{"link"},{?}]` and the `'sdy.reshard' op duplicate axis
///    ref: "link"` error in `InsertExplicitReshardsPass` goes away.
///
/// 2. A rank-5 cache forces a runtime `dynamicSlice` on `.layer`, whose
///    `slice_sizes` keeps the unsharded `.h = 8` while shardy shrinks
///    the operand to `.h = 4` per chip -> verifier rejects with
///    `slice_size 8 > dim 4`. Per-layer rank-4 sidesteps `.layer`.
pub const Cache = struct {
    conv: ConvCache,
    kv: KvCache,

    pub fn initBuffers(self: Cache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shardings: common.Shardings) !zml.Bufferized(Cache) {
        return .{
            .conv = try self.conv.initBuffers(allocator, io, platform, .replicated),
            .kv = try self.kv.initBuffers(allocator, io, platform, shardings.model),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Cache), allocator: std.mem.Allocator) void {
        ConvCache.unloadBuffers(&self.conv, allocator);
        KvCache.unloadBuffers(&self.kv, allocator);
    }

    pub fn reuseBuffer(self: Cache, other: Cache) Cache {
        return .{ .conv = self.conv.reuseBuffer(other.conv), .kv = self.kv.reuseBuffer(other.kv) };
    }
};

pub const ConvCache = struct {
    state: []zml.Tensor,
    layer_index: ?u32 = null,

    pub fn init(allocator: std.mem.Allocator, layer_shape: zml.Shape, num_layers: u32) !ConvCache {
        const state = try allocator.alloc(zml.Tensor, num_layers);
        for (state) |*t| t.* = .fromShape(layer_shape);
        return .{ .state = state, .layer_index = null };
    }

    pub fn deinit(self: ConvCache, allocator: std.mem.Allocator) void {
        allocator.free(self.state);
    }

    pub fn initBuffers(self: ConvCache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(ConvCache) {
        const state = try allocator.alloc(zml.Buffer, self.state.len);
        errdefer allocator.free(state);
        for (self.state, state) |src, *dst| {
            const sh = src.shape();
            const host = try allocator.alloc(u8, sh.byteSize());
            defer allocator.free(host);
            @memset(host, 0);
            dst.* = try zml.Buffer.fromBytes(io, platform, sh, sharding, host);
        }
        return .{ .state = state };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(ConvCache), allocator: std.mem.Allocator) void {
        for (self.state) |*b| b.deinit();
        allocator.free(self.state);
    }

    pub fn atLayer(self: ConvCache, idx: usize) ConvCache {
        return .{ .state = self.state, .layer_index = @intCast(idx) };
    }

    pub fn current(self: ConvCache) zml.Tensor {
        return self.state[self.layer_index orelse @panic("forgot to call atLayer")];
    }

    pub fn replace(self: ConvCache, new_state: zml.Tensor) ConvCache {
        const layer = self.layer_index orelse @panic("forgot to call atLayer");
        const arena = zml.module.CompilationContext.current().arena.allocator();
        const state = arena.dupe(zml.Tensor, self.state) catch @panic("OOM dup ConvCache");
        state[layer] = new_state.reuseBuffer(self.state[layer]);
        return .{ .state = state, .layer_index = self.layer_index };
    }

    pub fn reuseBuffer(self: ConvCache, other: ConvCache) ConvCache {
        const arena = zml.module.CompilationContext.current().arena.allocator();
        const state = arena.dupe(zml.Tensor, self.state) catch @panic("OOM dup ConvCache");
        for (state, other.state) |*t, o| t.* = t.reuseBuffer(o);
        return .{ .state = state, .layer_index = null };
    }
};

pub const KvCache = struct {
    k: []zml.Tensor,
    v: []zml.Tensor,
    layer_index: ?u32 = null,

    pub fn init(allocator: std.mem.Allocator, layer_shape: zml.Shape, num_layers: u32) !KvCache {
        // TT-FIX: shard heads `.h = .model` so the cache arg sharding matches
        // the `.h`-sharded scatter result from q/k/v projection (otherwise
        // shardy emits a reshard listing the `.model` axis twice).
        const sharded = layer_shape.withPartitioning(.{ .h = .model });
        const k = try allocator.alloc(zml.Tensor, num_layers);
        errdefer allocator.free(k);
        const v = try allocator.alloc(zml.Tensor, num_layers);
        for (k, v) |*kt, *vt| {
            kt.* = .fromShape(sharded);
            vt.* = .fromShape(sharded);
        }
        return .{ .k = k, .v = v, .layer_index = null };
    }

    pub fn deinit(self: KvCache, allocator: std.mem.Allocator) void {
        allocator.free(self.k);
        allocator.free(self.v);
    }

    pub fn initBuffers(self: KvCache, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !zml.Bufferized(KvCache) {
        const k = try allocator.alloc(zml.Buffer, self.k.len);
        errdefer allocator.free(k);
        const v = try allocator.alloc(zml.Buffer, self.v.len);
        errdefer allocator.free(v);
        for (self.k, k) |src, *dst| dst.* = try zml.Buffer.uninitialized(io, platform, src.shape(), sharding, .{});
        for (self.v, v) |src, *dst| dst.* = try zml.Buffer.uninitialized(io, platform, src.shape(), sharding, .{});
        return .{ .k = k, .v = v };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(KvCache), allocator: std.mem.Allocator) void {
        for (self.k) |*b| b.deinit();
        for (self.v) |*b| b.deinit();
        allocator.free(self.k);
        allocator.free(self.v);
    }

    pub fn atLayer(self: KvCache, idx: usize) KvCache {
        return .{ .k = self.k, .v = self.v, .layer_index = @intCast(idx) };
    }

    pub fn keys(self: KvCache) zml.Tensor {
        return self.k[self.layer_index orelse @panic("forgot to call atLayer")];
    }

    pub fn values(self: KvCache) zml.Tensor {
        return self.v[self.layer_index orelse @panic("forgot to call atLayer")];
    }

    /// Use the canonical TT cache-fill path so the scatter fuses into
    /// `ttir.update_cache` / `ttir.fill_cache`. `pos` must trace back to
    /// a rank-1 function input.
    pub fn update(self: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, pos: zml.Tensor) KvCache {
        const layer = self.layer_index orelse @panic("forgot to call atLayer");
        const arena = zml.module.CompilationContext.current().arena.allocator();
        const k = arena.dupe(zml.Tensor, self.k) catch @panic("OOM dup KvCache.k");
        const v = arena.dupe(zml.Tensor, self.v) catch @panic("OOM dup KvCache.v");
        k[layer] = zml.attention.attention.updateKvCache(self.k[layer], new_k, pos).reuseBuffer(self.k[layer]);
        v[layer] = zml.attention.attention.updateKvCache(self.v[layer], new_v, pos).reuseBuffer(self.v[layer]);
        return .{ .k = k, .v = v, .layer_index = self.layer_index };
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        const arena = zml.module.CompilationContext.current().arena.allocator();
        const k = arena.dupe(zml.Tensor, self.k) catch @panic("OOM dup KvCache.k");
        const v = arena.dupe(zml.Tensor, self.v) catch @panic("OOM dup KvCache.v");
        for (k, other.k) |*t, o| t.* = t.reuseBuffer(o);
        for (v, other.v) |*t, o| t.* = t.reuseBuffer(o);
        return .{ .k = k, .v = v, .layer_index = null };
    }
};

fn initLinear(store: zml.io.TensorStore.View, tag: anytype, sharding: anytype) Linear {
    return .init(store.createTensor("weight", .{ .out, tag }, sharding), null, tag);
}

pub const Linear = struct {
    weight: zml.Tensor,
    bias: ?zml.Tensor = null,
    in_tag: zml.Shape.Tag,
    out_tag: zml.Shape.Tag,

    pub fn init(weight: zml.Tensor, bias: ?zml.Tensor, tag: anytype) Linear {
        stdx.debug.guard(weight.shape().hasTag(tag) != null, @src());
        const axis = weight.shape().axis(tag);
        const out_tag = weight.shape().tag(1 - axis);
        return .{ .weight = weight, .bias = bias, .in_tag = zml.Shape.toTag(tag), .out_tag = out_tag };
    }

    pub fn unloadBuffers(linear: *zml.Bufferized(Linear)) void {
        linear.weight.deinit();
        if (linear.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: Linear, x: zml.Tensor) zml.Tensor {
        var y = x.dot(self.weight, self.in_tag).renameTag(self.out_tag, self.in_tag);
        return if (self.bias) |bias| y.add(bias.broad(y.shape())).reuseBuffer(x) else y;
    }
};

const Mlp = struct {
    w1: Linear,
    w2: Linear,
    w3: Linear,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        // TT-FIX: w1/w3 shard `.out` (ff dim); w2 shards `.d` (ff input).
        // Matches llama's gate_proj/up_proj/down_proj.
        return .{
            .w1 = initLinear(store.withPrefix("w1"), .d, .{ .out = .model }),
            .w2 = initLinear(store.withPrefix("w2"), .d, .{ .d = .model }),
            .w3 = initLinear(store.withPrefix("w3"), .d, .{ .out = .model }),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        Linear.unloadBuffers(&self.w1);
        Linear.unloadBuffers(&self.w2);
        Linear.unloadBuffers(&self.w3);
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        return self.w2.forward(self.w1.forward(x).silu().mul(self.w3.forward(x))).reuseBuffer(x);
    }
};

const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32,
    tag: zml.Shape.Tag,

    pub fn init(store: zml.io.TensorStore.View, eps: f32, tag: anytype) RmsNorm {
        return .{ .weight = store.createTensor("weight", .{tag}, .replicated), .eps = eps, .tag = zml.Shape.toTag(tag) };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const ctx = zml.module.CompilationContext.current();
        switch (ctx.platform.target) {
            // TT-FIX: fused custom_call — math-decomposed RMSNorm trips
            // shardy's `duplicate axis ref` on sharded weight.
            .tt => return tt_nn.rmsNormFused(input, self.weight, self.eps),
            .cpu, .cuda, .rocm, .tpu, .neuron => {
                const normalized = zml.nn.rmsNorm(input, self.tag, self.eps);
                return normalized.mul(self.weight.broad(input.shape())).reuseBuffer(input);
            },
        }
    }
};
