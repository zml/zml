const std = @import("std");

const zml = @import("zml");
const stdx = @import("stdx");

const Tensor = @import("main.zig").Tensor;
const TensorDescriptor = @import("main.zig").TensorDescriptor;
const BufferStore4 = @import("main.zig").BufferStore4;
const BufferStore5 = @import("main.zig").BufferStore5;
const CompilationContext = @import("main.zig").CompilationContext;
const Exe = @import("main.zig").Exe;
const Bufferized = @import("main.zig").Bufferized;
const loadBuffersFromId = @import("main.zig").loadBuffersFromId;
const multiTransfer = @import("main.zig").multiTransfer;
const singleTransfer = @import("main.zig").singleTransfer;
const structTransfer = @import("main.zig").structTransfer;
const bufferTransfer = @import("main.zig").bufferTransfer;
const autoLoad = @import("main.zig").autoLoad;
const initBufferizedFrom = @import("main.zig").initBufferizedFrom;
const compile = @import("main.zig").compile;

pub const LlamaLM = struct {
    pub const Config = struct {
        bos_token_id: u32,
        eos_token_id: stdx.json.Union(union(enum) {
            int: u32,
            ints: []u32,
        }),
        num_hidden_layers: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        rope_theta: f32,
        max_position_embeddings: usize,
        rms_norm_eps: f32,
        hf_rope_impl: bool = true,
        rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = {} },
    };

    pub const Options = struct {
        sampling_strategy: ?zml.nn.SamplingStrategy,
        max_seq_len: usize,
        qkv_type: QkvKind,
    };

    pub const LoadingOptions = struct {
        merge_qkv_exe: ?Exe = null,
        precompute_qkv_exe: ?Exe = null,
    };

    lm_head: ?Linear,
    model: Llama,

    // Options controlling generation
    gen_opts: zml.nn.SamplingStrategy = .{},
    config: Config,

    pub fn init(allocator: std.mem.Allocator, buffer_store: BufferStore5.View, config: LlamaLM.Config, options: Options) !LlamaLM {
        const lm_head: ?Linear = b: {
            const lm_head_weight = buffer_store.withPrefix("lm_head").maybeCreateTensorWithTags("weight", .{ .dout, .d }) orelse break :b null;
            break :b .{ .weight = lm_head_weight, .tag = zml.Shape.toTag(.d) };
        };

        return .{
            .lm_head = lm_head,
            .model = try .init(allocator, buffer_store.withPrefix("model"), config, options),
            .gen_opts = options.sampling_strategy orelse .{},
            .config = config,
        };
    }

    pub fn deinit(self: LlamaLM, allocator: std.mem.Allocator) void {
        allocator.free(self.model.layers);
    }

    pub fn loadBuffers(allocator: std.mem.Allocator, llama: LlamaLM, buffer_store: BufferStore5.View, platform: zml.Platform, options: LoadingOptions) !LlamaLMBufferized {
        return .{
            .lm_head = if (llama.lm_head != null) try loadBuffersFromId(allocator, llama.lm_head.?, buffer_store.withPrefix("lm_head"), platform) else null,
            .model = try Llama.loadBuffers(allocator, llama.model, buffer_store.withPrefix("model"), platform, options),
        };
    }

    /// Predicts the token at `token_index` position.
    /// Returns:
    ///  - updated `tokens`,
    ///  - updated KV cache
    ///  - a Rng state to allow for probabilistic generation
    pub fn forward(
        self: LlamaLM,
        tokens_: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        rng: Tensor.Rng,
    ) struct { Tensor, KvCache, Tensor.Rng } {
        stdx.debug.assert(tokens_.dtype() == .u32 and tokens_.rank() >= 1 and token_index.dtype() == .u32 and token_index.rank() <= 1, "Can't run Llama ! Expected >=1d tokens and 0d token_index, got: {f} and {f}", .{ tokens_, token_index });
        const tokens = tokens_.withPartialTags(.{.s});
        const out, const updated_kv_cache = self.model.forward(tokens, token_index, kv_cache);
        const new_tokens, const new_rng = self.sampleTokens(self.lm_head, out, rng, self.gen_opts);
        return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), updated_kv_cache, new_rng };
    }

    pub fn sampleTokens(
        self: LlamaLM,
        lm_head_: ?Linear,
        out_: Tensor,
        rng: Tensor.Rng,
        opts: zml.nn.SamplingStrategy,
    ) struct { Tensor, Tensor.Rng } {
        const out = out_.withPartialTags(.{ .s, .d });

        var logits = blk: {
            if (lm_head_) |lm_head| {
                break :blk lm_head.forward(out).rename(.{ .dout = .d });
            } else {
                break :blk self.model.embed_tokens.weight.withTags(.{ .voc, .d }).dot(out, .d);
            }
        };

        if (logits.shape().hasTag(.voc) == null)
            logits = logits.rename(.{ .d = .voc });

        const next_tokens, const new_rng = sampleTokens2(logits, opts, rng);
        return .{ next_tokens, new_rng };
    }
};

pub const LlamaLMBufferized = struct {
    lm_head: ?Bufferized(Linear),
    model: Bufferized(Llama),

    pub fn deinit(self: LlamaLMBufferized, allocator: std.mem.Allocator) void {
        allocator.free(self.model.layers);
    }
};

pub const Linear = struct {
    weight: Tensor,
    bias: ?Tensor = null,
    tag: zml.Shape.Tag,

    pub fn initWithTag(buffer_store: BufferStore5.View, tag: anytype) Linear {
        return .{
            .weight = buffer_store.createTensorWithTags("weight", .{ .dout, tag }),
            .bias = buffer_store.maybeCreateTensorWithTags("bias", .{.dout}),
            .tag = zml.Shape.toTag(tag),
        };
    }

    pub fn forward(self: Linear, x: Tensor) Tensor {
        var y = x.dot(self.weight.convert(x.dtype()), self.tag);

        // log.debug("Linear({*}): {d} -> {d} -> {d}", .{ self, x.dims(), y.dims(), if (self.bias) |bias| y.add(bias).dims() else y.dims() });
        return if (self.bias) |bias| y.add(bias.autoBroadcast()) else y;
    }
};

pub const TokenEmbedding = struct {
    weight: Tensor,

    pub fn forward(self: TokenEmbedding, idx: Tensor) Tensor {
        stdx.debug.assert(idx.dtype().isInteger(), "TokenEmbedding expects an integer input, received: {f}", .{idx});
        stdx.debug.assert(self.weight.rank() == 2, "TokenEmbedding expects it's weight Tensor to be a 2D matrix, got {f}", .{self.weight});
        return self.weight.gatherValues(0, idx, .{});
    }
};

pub const Llama = struct {
    embed_tokens: TokenEmbedding,
    norm: RmsNorm,
    layers: []TransformerLayer,

    pub fn init(allocator: std.mem.Allocator, buffer_store: BufferStore5.View, config: LlamaLM.Config, options: LlamaLM.Options) !Llama {
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            //const custom_options = if (i != 0) options else b: {
            //    var custom_options = options;
            //    custom_options.qkv_type = .precomputed;
            //    break :b custom_options;
            //};
            const custom_options = options;
            var buffer: [16]u8 = undefined;
            const number_prefix = try std.fmt.bufPrint(&buffer, "layers.{d}", .{i});
            layer.* = .init(buffer_store.withPrefix(number_prefix), config, custom_options);
        }

        return .{
            .embed_tokens = .{ .weight = buffer_store.createTensorWithTags("embed_tokens.weight", .{ .x, .y }) },
            .norm = .init(buffer_store.withPrefix("norm"), config.rms_norm_eps),
            .layers = layers,
        };
    }

    pub fn loadBuffers(allocator: std.mem.Allocator, model: Llama, buffer_store: BufferStore5.View, platform: zml.Platform, options: LlamaLM.LoadingOptions) !Bufferized(Llama) {
        std.debug.print("Loading Llama\n", .{});
        const layers = try allocator.alloc(Bufferized(TransformerLayer), model.layers.len);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            var buffer: [16]u8 = undefined;
            const number_prefix = try std.fmt.bufPrint(&buffer, "layers.{d}", .{i});
            std.debug.print("Loading layer {d}\n", .{i});
            layer.* = try TransformerLayer.loadBuffers(allocator, model.layers[i], buffer_store.withPrefix(number_prefix), platform, options);
        }

        return .{
            .embed_tokens = try loadBuffersFromId(allocator, model.embed_tokens, buffer_store.withPrefix("embed_tokens"), platform),
            .layers = layers,
            .norm = try loadBuffersFromId(allocator, model.norm, buffer_store.withPrefix("norm"), platform),
        };
    }

    /// Forward one token, using KV cache for previous tokens.
    /// Returns result and updated KV cache.
    pub fn forward(self: Llama, tokens: Tensor, token_index: Tensor, kv_cache: KvCache) struct { Tensor, KvCache } {
        const embeds = embed(self.embed_tokens, tokens);
        var hidden = embeds;

        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            std.debug.print("hidden.shape: {f}\n", .{hidden.shape()});
            hidden, updated_kv_cache = layer.forward(hidden, tokens, token_index, updated_kv_cache.atLayer(i));
        }
        const output = self.norm.forward(hidden);

        return .{ output, updated_kv_cache.reuseBuffer(kv_cache) };
    }

    pub fn embed(embed_tokens_: TokenEmbedding, tokens_: Tensor) Tensor {
        return embed_tokens_.forward(tokens_).withPartialTags(.{.d});
    }
};

pub const RmsNorm = struct {
    weight: Tensor,
    eps: f32 = 1e-5,

    pub fn init(buffer_store: BufferStore5.View, eps: f32) RmsNorm {
        return .{
            .weight = buffer_store.createTensorWithTags("weight", .{.d}),
            .eps = eps,
        };
    }

    /// L2 normalization of input tensor along `.d` axis.
    pub fn forward(self: RmsNorm, input: Tensor) Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const normalized = rmsNorm(x, .d, self.eps);
        return normalized.mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }
};

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: SelfAttn,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,

    pub fn init(buffer_store: BufferStore5.View, config: LlamaLM.Config, options: LlamaLM.Options) TransformerLayer {
        return .{
            .input_layernorm = .init(buffer_store.withPrefix("input_layernorm"), config.rms_norm_eps),
            .self_attn = SelfAttn.init(buffer_store.withPrefix("self_attn"), config, options),
            .post_attention_layernorm = .init(buffer_store.withPrefix("post_attention_layernorm"), config.rms_norm_eps),
            .mlp = .init(buffer_store.withPrefix("mlp")),
        };
    }

    pub fn loadBuffers(allocator: std.mem.Allocator, layer: TransformerLayer, buffer_store: BufferStore5.View, platform: zml.Platform, options: LlamaLM.LoadingOptions) !Bufferized(TransformerLayer) {
        return .{
            .input_layernorm = try loadBuffersFromId(allocator, layer.input_layernorm, buffer_store.withPrefix("input_layernorm"), platform),
            .self_attn = try SelfAttn.loadBuffers(allocator, layer.self_attn, buffer_store.withPrefix("self_attn"), platform, options),
            .post_attention_layernorm = try loadBuffersFromId(allocator, layer.post_attention_layernorm, buffer_store.withPrefix("post_attention_layernorm"), platform),
            .mlp = try loadBuffersFromId(allocator, layer.mlp, buffer_store.withPrefix("mlp"), platform),
        };
    }

    pub fn forward(
        self: TransformerLayer,
        x0: Tensor,
        tokens: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
    ) struct { Tensor, KvCache } {
        // Self Attention
        //log.debug("TransformerLayer({f}) -> {f}", .{ x0, self.input_layernorm.forward(x0) });
        //stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {f}", .{x0});

        const params: SelfAttn.ForwardParams = switch (self.self_attn.proj) {
            .merged => .{ .merged = self.input_layernorm.forward(x0) },
            .separate => .{ .separate = self.input_layernorm.forward(x0) },
            .precomputed => .{ .precomputed = .{ .x = x0, .tokens = tokens } },
        };
        const delta0, const updated_kv_cache = self.self_attn.forward(params, token_index, kv_cache);
        const x1 = x0.add(delta0);

        // Fully Connected
        const x1_normalized = self.post_attention_layernorm.forward(x1);
        const x2 = self.mlp.forward(x1_normalized).add(x1).rename(.{ .dout = .d });

        return .{ x2.reuseBuffer(x0), updated_kv_cache };
    }
};

const QkvKind = enum {
    merged,
    separate,
    precomputed,
};

pub const SeparateQkv = struct {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,

    pub fn init(buffer_store: BufferStore5.View) SeparateQkv {
        const q_proj = Linear.initWithTag(buffer_store.withPrefix("q_proj"), .d);
        const k_proj = Linear.initWithTag(buffer_store.withPrefix("k_proj"), .d);
        const v_proj = Linear.initWithTag(buffer_store.withPrefix("v_proj"), .d);
        return .{ .q_proj = q_proj, .k_proj = k_proj, .v_proj = v_proj };
    }
};

pub const SelfAttn = struct {
    pub const Proj = union(QkvKind) {
        merged: Linear,
        separate: SeparateQkv,
        precomputed: Tensor,
    };
    proj: Proj,

    o_proj: Linear,
    num_heads: i64,
    num_kv_heads: i64,
    rope_opts: zml.nn.RopeOpts = .{
        .layout = .interleaved,
        .freq_base = 10_000,
    },

    pub fn init(buffer_store: BufferStore5.View, config: LlamaLM.Config, options: LlamaLM.Options) SelfAttn {
        const proj: Proj = switch (options.qkv_type) {
            .separate => .{ .separate = SeparateQkv.init(buffer_store) },
            .merged => b: {
                const q_weight_shape = buffer_store.getShape("q_proj.weight").?;
                const k_weight_shape = buffer_store.getShape("k_proj.weight").?;
                const v_weight_shape = buffer_store.getShape("v_proj.weight").?;
                const qkv_weight_shape = zml.Shape.concatenate(&.{ q_weight_shape, k_weight_shape, v_weight_shape }, 0);
                break :b .{ .merged = .{ .weight = Tensor.init(qkv_weight_shape).withTags(.{ .dout, .d }), .tag = zml.Shape.toTag(.d) } };
            },
            .precomputed => b: {
                const q_weight_shape = buffer_store.getShape("q_proj.weight").?;
                const k_weight_shape = buffer_store.getShape("k_proj.weight").?;
                const v_weight_shape = buffer_store.getShape("v_proj.weight").?;
                const embedding_shape = buffer_store.getShapeOpts("model.embed_tokens.weight", .{ .no_prefix = true }).?;
                const qkv_weight_shape = zml.Shape.concatenate(&.{ q_weight_shape, k_weight_shape, v_weight_shape }, 0);
                const precomputed_qkv_weight_shape = zml.Shape.init(.{ embedding_shape.dim(0), qkv_weight_shape.dim(0) }, qkv_weight_shape.dtype());

                break :b .{ .precomputed = Tensor.init(precomputed_qkv_weight_shape).withTags(.{ .voc, .d }) };
            },
        };

        return .{
            .proj = proj,
            .o_proj = .initWithTag(buffer_store.withPrefix("o_proj"), .d),
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
            .rope_opts = .{
                .layout = if (config.hf_rope_impl) .sequential else .interleaved,
                .freq_base = config.rope_theta,
                .scaling = config.rope_scaling,
            },
        };
    }

    pub fn loadBuffers(allocator: std.mem.Allocator, self_attn: SelfAttn, buffer_store: BufferStore5.View, platform: zml.Platform, options: LlamaLM.LoadingOptions) !Bufferized(SelfAttn) {
        return switch (self_attn.proj) {
            .separate => |separate| .{
                .proj = .{
                    .separate = .{
                        .q_proj = try loadBuffersFromId(allocator, separate.q_proj, buffer_store.withPrefix("q_proj"), platform),
                        .k_proj = try loadBuffersFromId(allocator, separate.k_proj, buffer_store.withPrefix("k_proj"), platform),
                        .v_proj = try loadBuffersFromId(allocator, separate.v_proj, buffer_store.withPrefix("v_proj"), platform),
                    },
                },
                .o_proj = try loadBuffersFromId(allocator, self_attn.o_proj, buffer_store.withPrefix("o_proj"), platform),
            },
            .merged => b: {
                const proj: SeparateQkv = .init(buffer_store);

                var args = try options.merge_qkv_exe.?.args(allocator);
                defer args.deinit(allocator);

                var results = try options.merge_qkv_exe.?.results(allocator);
                defer results.deinit(allocator);

                const proj_buffers = try loadBuffersFromId(allocator, proj, buffer_store, platform);

                args.set(.{proj_buffers});
                options.merge_qkv_exe.?.call(args, &results);

                var qkv_proj: Bufferized(Linear) = undefined;
                initBufferizedFrom(self_attn.proj.merged, &qkv_proj);

                results.fill(&qkv_proj);
                errdefer qkv_proj.weight.deinit();
                errdefer if (qkv_proj.bias != null) qkv_proj.bias.?.deinit();

                const o_proj = try loadBuffersFromId(allocator, self_attn.o_proj, buffer_store.withPrefix("o_proj"), platform);

                break :b .{
                    .proj = .{
                        .merged = qkv_proj,
                    },
                    .o_proj = o_proj,
                };
            },
            .precomputed => b: {
                const proj: SeparateQkv = .init(buffer_store);

                var args = try options.precompute_qkv_exe.?.args(allocator);
                defer args.deinit(allocator);

                var results = try options.precompute_qkv_exe.?.results(allocator);
                defer results.deinit(allocator);

                const proj_buffers = try loadBuffersFromId(allocator, proj, buffer_store, platform);
                const embedding_buffer = try bufferTransfer(allocator, buffer_store.root(), "model.embed_tokens.weight", platform);
                defer embedding_buffer.deinit();

                const input_layernorm_buffer = try bufferTransfer(allocator, buffer_store.parent(), "input_layernorm.weight", platform);
                defer input_layernorm_buffer.deinit();

                args.set(.{ proj_buffers, input_layernorm_buffer, embedding_buffer });
                options.precompute_qkv_exe.?.call(args, &results);

                var precomputed_qkv = results.get(zml.Buffer);
                errdefer precomputed_qkv.deinit();

                const o_proj = try loadBuffersFromId(allocator, self_attn.o_proj, buffer_store.withPrefix("o_proj"), platform);

                break :b .{
                    .proj = .{
                        .precomputed = precomputed_qkv,
                    },
                    .o_proj = o_proj,
                };
            },
        };
    }

    const ForwardParams = union(QkvKind) {
        merged: Tensor,
        separate: Tensor,
        precomputed: struct { x: Tensor, tokens: Tensor },
    };

    /// Self Attention.
    ///   - If token_index is set, x is assumed to be the representation of one new token,
    /// and kv_cache will be read for the previous tokens.
    ///   - If token_index is not set, x is assumed to be the representation of all tokens
    /// since the beginning of the sequence, and kv_cache won't be read.
    /// In both case, kv_cache will be updated with the computed key and value.
    /// x: {.b, .s, .d } -> .{.b, .s, .d}
    pub fn forward(
        self: SelfAttn,
        params: ForwardParams,
        token_index: Tensor,
        kv_cache: KvCache,
    ) struct { Tensor, KvCache } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;
        std.debug.assert(@as(QkvKind, params) == @as(QkvKind, self.proj));
        var q, var k, var v = switch (self.proj) {
            .separate => b: {
                const q = self.proj.separate.q_proj.forward(params.separate).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
                const k = self.proj.separate.k_proj.forward(params.separate).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
                const v = self.proj.separate.v_proj.forward(params.separate).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
                break :b .{ q, k, v };
            },
            .merged => b: {
                const qkv = self.proj.merged.forward(params.merged).splitAxis(-1, .{ .h = self.num_heads + num_kv_heads + num_kv_heads, .hd = .auto });
                const q = qkv.slice1d(.h, .{ .start = 0, .end = self.num_heads });
                const k = qkv.slice1d(.h, .{ .start = self.num_heads, .end = self.num_heads + num_kv_heads });
                const v = qkv.slice1d(.h, .{ .start = self.num_heads + num_kv_heads, .end = self.num_heads + num_kv_heads + num_kv_heads });
                break :b .{ q, k, v };
            },
            .precomputed => b: {
                std.debug.print("shape: {f}\n", .{self.proj.precomputed.shape()});
                std.debug.print("tokens.shape: {f}\n", .{params.precomputed.tokens.shape()});
                const qkv = self.proj.precomputed.gatherValues(0, params.precomputed.tokens, .{}).splitAxis(-1, .{ .h = self.num_heads + num_kv_heads + num_kv_heads, .hd = .auto });
                const q = qkv.slice1d(.h, .{ .start = 0, .end = self.num_heads });
                const k = qkv.slice1d(.h, .{ .start = self.num_heads, .end = self.num_heads + num_kv_heads });
                const v = qkv.slice1d(.h, .{ .start = self.num_heads + num_kv_heads, .end = self.num_heads + num_kv_heads + num_kv_heads });
                break :b .{ q, k, v };
            },
        };

        // Generate the attention mask.
        const seq_len = kv_cache.k.dim(.k);
        var attn_mask = causalAttnMask(.{ .q = seq_len, .k = seq_len }, q.dtype(), null);

        // Note: in Pytorch it would be very inefficient to generate the full attn_mask,
        // then slice into it, but XLA is able to optimize this correctly.
        attn_mask = attn_mask.gatherSlices(zml.Shape.init(.{ .q = q.dim(.s) }, attn_mask.dtype()), token_index.reshape(.{ .coord = 1 }), .{});

        // In self-attention, .s axis is used both for keys and queries.
        const pos_index = b: {
            const temp = Tensor.arange(.{ .end = q.dim(.s) }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = q.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };

        q = rope(q, pos_index, self.rope_opts);
        k = rope(k, pos_index, self.rope_opts);
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const dtype = q.dtype();
        const new_kv_cache = kv_cache.update(k, v, token_index);
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);

        const attn_output = sdpa(q, k, v, .{ .attn_mask = attn_mask, .allow_cudnn = true });
        // const attn_output = zml.nn.sdpaMemEfficient(q, k, v, .{ .attn_mask = attn_mask }, .{ .q_chunk_size = 4096, .k_chunk_size = 1024 });
        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return .{ self.o_proj.forward(attn), new_kv_cache };
    }
};

fn mergeQkv(q: Tensor, k: Tensor, v: Tensor) Tensor {
    return Tensor.concatenate(&.{ q, k, v }, 0);
}

pub fn mergeQkv2(q: Linear, k: Linear, v: Linear) Linear {
    return .{
        .weight = Tensor.concatenate(&.{ q.weight, k.weight, v.weight }, 0),
        .bias = if (q.bias != null) Tensor.concatenate(&.{ q.bias.?, k.bias.?, v.bias.? }, 0) else null,
        .tag = q.tag,
    };
}

pub fn mergeQkv3(proj: SeparateQkv) Linear {
    return .{
        .weight = Tensor.concatenate(&.{ proj.q_proj.weight, proj.k_proj.weight, proj.v_proj.weight }, 0),
        .bias = if (proj.q_proj.bias != null) Tensor.concatenate(&.{ proj.q_proj.bias.?, proj.k_proj.bias.?, proj.v_proj.bias.? }, 0) else null,
        .tag = proj.q_proj.tag,
    };
}

pub fn precomputeQkv0(proj: SeparateQkv, layer_norm: RmsNorm, embedding: Tensor) Tensor {
    const hidden = layer_norm.forward(embedding);
    const qkv_proj: Linear = .{
        .weight = Tensor.concatenate(&.{ proj.q_proj.weight, proj.k_proj.weight, proj.v_proj.weight }, 0),
        .bias = if (proj.q_proj.bias != null) Tensor.concatenate(&.{ proj.q_proj.bias.?, proj.k_proj.bias.?, proj.v_proj.bias.? }, 0) else null,
        .tag = proj.q_proj.tag,
    };

    return qkv_proj.forward(hidden).rename(.{ .dout = .d });
}

const Mlp = struct {
    up_proj: Linear, // (dim -> hidden_dim)
    gate_proj: Linear, // (dim -> hidden_dim)
    down_proj: Linear, // (hidden_dim -> dim)

    pub fn init(buffer_store: BufferStore5.View) Mlp {
        return .{
            .up_proj = .initWithTag(buffer_store.withPrefix("up_proj"), .d),
            .gate_proj = .initWithTag(buffer_store.withPrefix("gate_proj"), .d),
            .down_proj = .initWithTag(buffer_store.withPrefix("down_proj"), .d),
        };
    }

    pub fn forward(self: Mlp, x: Tensor) Tensor {
        const proj = self.up_proj.forward(x);
        var output = self.gate_proj.forward(x);
        output = output.silu().mul(proj).rename(.{ .dout = .d });
        return self.down_proj.forward(output);
    }
};

pub const KvCache = struct {
    k: Tensor,
    v: Tensor,
    layer_index: Tensor,

    pub fn init(kv_shape: zml.Shape) KvCache {
        // The KV-cache is initialized with ones to detect reads of uninitialized memory.
        return .{
            .k = Tensor.init(kv_shape),
            .v = Tensor.init(kv_shape),
            .layer_index = Tensor.init(zml.Shape.init(.{}, .u32)),
        };
    }

    pub fn initBuffer(kv_shape: zml.Shape, platform: zml.Platform) !Bufferized(KvCache) {
        return .{
            .k = try zml.Buffer.constant(platform, kv_shape, 0),
            .v = try zml.Buffer.constant(platform, kv_shape, 0),
            .layer_index = try zml.Buffer.scalar(platform, 0, .u32),
        };
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

        std.debug.print("new_k: {f}\n", .{new_k.shape()});
        std.debug.print("self: {f}\n", .{self.k.shape()});

        return if (token_index) |idx| .{
            .k = self.k.scatterSlices(
                .{ .layer = layer, .k = idx },
                new_k.convert(self.k.dtype()).transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = Tensor.ScatterOpts.override },
            ).reuseBuffer(self.k),
            .v = self.v.scatterSlices(
                .{ .layer = layer, .k = idx },
                new_v.convert(self.v.dtype()).transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = Tensor.ScatterOpts.override },
            ).reuseBuffer(self.v),
            .layer_index = self.layer_index,
        } else .{
            .k = self.k.scatterSlices(
                .{ .layer = layer },
                new_k.convert(self.k.dtype()).transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = Tensor.ScatterOpts.override },
            ).reuseBuffer(self.k),
            .v = self.v.scatterSlices(
                .{ .layer = layer },
                new_v.convert(self.v.dtype()).transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = Tensor.ScatterOpts.override },
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

pub fn sampleTokens2(activations: Tensor, opts: zml.nn.SamplingStrategy, rng: Tensor.Rng) struct { Tensor, Tensor.Rng } {
    _ = opts; // autofix
    //if (opts.topk <= 1) {
    if (true) {
        const next_tokens = activations.argMax(.voc).indices.squeeze(.voc);
        return .{ next_tokens, rng };
    }

    //const topk = activations.topK(opts.topk, .voc, .{});
    //// After the topk, we don't have .voc values, anymore, only topk.
    //var x = topk.values.rename(.{ .voc = .topk });
    //if (opts.temperature != 1.0) {
    //    x = x.scale(1 / opts.temperature);
    //}

    //// Gumbel reparametrization trick:
    //// Adding gumbel noise and taking the argmax is equivalent
    //// to sampling from the categorical distribution produced by the softmax.
    //// https://en.wikipedia.org/wiki/Gumbel_distribution#Gumbel_reparametrization_tricks
    //const next_rng, const gumbel_noise = rng.gumbel(x.shape());
    //x = x.add(gumbel_noise);
    //const topk_idx = x.argMax(.topk).indices;

    //// topk_idx is indices into topk.values ! so in the range [0, topk]
    //// Convert for the original indices from the full [0, voc] range.
    //const next_tokens = topk.indices.gatherValues(.voc, topk_idx.squeeze(.topk), .{});
    //// log.debug("sampleTokens({}) -> {} -> {} -> {}", .{ activations, topk.indices, topk_idx, next_tokens });
    //return .{ next_tokens, next_rng };
}

pub fn rmsNorm(x: Tensor, axis: anytype, eps: f32) Tensor {
    const ax = x.axis(axis);
    // upcast to improve precision
    const xf32 = x.convert(.f32);
    const mean = xf32.mul(xf32).mean(ax);
    const rsqrt = Tensor.rsqrt(mean.addConstant(eps)).convert(x.dtype());
    return x.mul(rsqrt.broad(x.shape()));
}

/// Return causal attention masks for the given shape.
/// The last dimensions are
pub fn causalAttnMask(
    attn_shape_: anytype,
    dtype: zml.DataType,
    attn_window_len: ?u32,
) Tensor {
    const attn_shape = zml.Shape.init(attn_shape_, dtype);
    stdx.debug.assert(attn_shape.rank() == 2, "causalAttnMask({f}) shape need to be exactly 2 axes", .{attn_shape});
    const qlen = attn_shape.dim(-2);
    const q_idx = Tensor.iota(attn_shape, -2);
    const klen = attn_shape.dim(-1);
    const k_idx = Tensor.iota(attn_shape, -1);

    // all elements > main diagonal must be 0
    // (q_idx - window_len < k_idx <= q_idx)
    var mask = k_idx.cmp(.LE, q_idx);
    if (attn_window_len) |window_len| {
        if (qlen >= window_len or klen >= window_len) {
            const window_mask = q_idx.cmp(.LT, k_idx.addConstant(window_len));
            mask = mask.logical(.AND, window_mask);
        }
    }

    if (dtype.isFloat()) {
        const zeros = Tensor.constant(dtype.zero()).broad(mask.shape());
        const minus_inf = Tensor.constant(dtype.minValue()).broad(mask.shape());
        mask = Tensor.select(mask, zeros, minus_inf);
    } else {
        mask = mask.convert(dtype);
    }

    return mask;
}

/// Rotary position embedding modify queries and keys tensor before compute Q * K in self attention.
/// This biases a token to look at token near him.
/// The nice thing with rope is that you can cache the modified queries and keys directly.
/// See: https://paperswithcode.com/method/rope
///
/// Expected shapes of tensor:
/// - x: .{ .s, .hd } where .s is the sequence length and .hd the head dimension
/// - pos_idx: optional tensor which indicates which positions are needed.
///   When not set `rope` return all positions from 0 to x.dim(.s) which is the max seq len.
pub fn rope(x: Tensor, pos_idx: ?Tensor, opts: zml.nn.RopeOpts) Tensor {
    stdx.debug.assert(@mod(x.dim(.hd), 2) == 0, "rope expects a even head dim (.hd), got {f}", .{x});

    const idx = if (pos_idx) |idx| blk: {
        stdx.debug.assert(x.shape().hasTags(.{.hd}), "rope expects x argument to have .hd axes got: rope(x={f}, idx={f})", .{ x, idx });
        break :blk idx;
    } else blk: {
        stdx.debug.assert(x.shape().hasTags(.{ .s, .hd }), "rope expects x argument to have both .s and .hd axes got: rope(x={f})", .{x});
        break :blk Tensor.arange(.{ .end = x.dim(.s) }, .f32).withTags(.{.s});
    };
    const x_real, const x_imag = splitRealImg(x, opts.layout);

    // compute sin and cos in f32 before downcasting to x type.
    const inv_freq = invFreq(x.dim(.hd), opts).withTags(.{.hd});
    const inv_freq_pos = Tensor.outer(idx.convert(.f32), inv_freq);
    const cos = inv_freq_pos.cos().convert(x.dtype()).broad(x_real.shape());
    const sin = inv_freq_pos.sin().convert(x.dtype()).broad(x_real.shape());

    // apply rotation
    const y_real = x_real.mul(cos).sub(x_imag.mul(sin));
    const y_imag = x_real.mul(sin).add(x_imag.mul(cos));

    // flatten last dimensions
    return mergeRealImg(y_real, y_imag, opts.layout);
}

pub fn mergeRealImg(x_real: Tensor, x_imag: Tensor, layout: zml.nn.RopeOpts.Layout) Tensor {
    return switch (layout) {
        .sequential => Tensor.concatenate(&.{ x_real, x_imag }, -1),
        .interleaved => Tensor.concatenate(&.{
            x_real.appendAxes(.{.interleaved_real_img}),
            x_imag.appendAxes(.{.interleaved_real_img}),
        }, -1).flatten(-2),
    };
}

/// {exp( - n * ln(10_000) / N ) | n in [0..N] }
pub fn invFreq(N: i64, opts: zml.nn.RopeOpts) Tensor {
    var arena = std.heap.ArenaAllocator.init(CompilationContext.current().allocator);
    defer arena.deinit();
    const N_half: usize = @intCast(@divExact(N, 2));
    const inv_freq = arena.allocator().alloc(f32, N_half) catch @panic("OOM");
    _invFreq(opts, inv_freq);
    return Tensor.constantTensor(std.mem.sliceAsBytes(inv_freq), zml.Shape.init(.{@divExact(N, 2)}, .f32));
}

fn _invFreq(opts: zml.nn.RopeOpts, inv_freq: []f32) void {
    const N = inv_freq.len;
    // Default frequencies
    for (0.., inv_freq) |n, *f| {
        f.* = @exp(-@log(opts.freq_base) * stdx.math.divFloat(f32, n, N));
    }

    switch (opts.scaling) {
        .default => {},
        .custom => {
            stdx.debug.assert(opts.scaling.custom.len == N, "rope expected custom inv_freq to match half head dimension {d}, got {d}", .{ N, opts.scaling.custom.len });
            @memcpy(inv_freq, opts.scaling.custom);
        },
        .llama3 => |s| {
            // https://arxiv.org/pdf/2309.16039
            // After Llama2 they observed that the rope frequencies where too sharp and hurting long distance attention.
            // In Llama3 they used a higher base freq and also downscaled low frequencies.
            std.debug.assert(s.low_freq_factor < s.high_freq_factor);
            const M: f64 = @floatFromInt(s.original_max_position_embeddings);
            const f_high = s.high_freq_factor * (2 * std.math.pi) / M;
            const f_low = s.low_freq_factor * (2 * std.math.pi) / M;
            const downscaling = 1.0 / s.factor;

            for (0..N, inv_freq) |n, f| {
                if (f > f_high) {
                    // High freq match default implem
                } else if (f < f_low) {
                    // Downscaling for low freq
                    inv_freq[n] *= downscaling;
                } else {
                    // Linear interpolation for middle freq
                    const lerp: f64 = (inv_freq[n] - f_low) / (f_high - f_low);
                    inv_freq[n] *= @floatCast(lerp + (1 - lerp) * downscaling);
                }
            }
        },
    }
}

pub fn splitRealImg(x: Tensor, layout: zml.nn.RopeOpts.Layout) [2]Tensor {
    const n = x.dim(-1);

    return switch (layout) {
        .sequential => .{
            x.slice1d(-1, .{ .end = @divExact(n, 2) }),
            x.slice1d(-1, .{ .start = @divExact(n, 2), .end = n }),
        },
        .interleaved => .{
            x.slice1d(-1, .{ .start = 0, .step = 2 }),
            x.slice1d(-1, .{ .start = 1, .step = 2 }),
        },
    };
}

pub const SdpaOpts = struct {
    attn_mask: ?Tensor = null,
    scale: ?Tensor = null,
    allow_cudnn: bool = true,
    // TODO: put a callback instead of all this field,
    // so that
};

/// Scaled dot product attention.
///
/// **Shapes**:
///   - q, result: .{ .h, .q, .hd }
///   - k, v:      .{ .h, .k, .hd }
///
/// Where:
///   - .h is the number of head
///   - .q is the number of queries
///   - .k is the number of keys
///   - .hd is the head dimension
///
/// .h is allowed to differ from queries and keys as long as the key heads
/// can be repeated to match query heads.
pub fn sdpa(q_: Tensor, k_: Tensor, v_: Tensor, opts: SdpaOpts) Tensor {
    var q, var k, var v = .{ q_, k_, v_ };

    const err_template = "sdpa(q: {f}, k: {f}, v: {f}, attn: {?f}) is invalid ! ";
    const err_args = .{ q, k, v, opts.attn_mask };
    stdx.debug.assert(q.shape().hasTags(.{ .h, .q, .hd }), err_template ++ "q is missing tags {{.h, .q, .hd}}", err_args);
    stdx.debug.assert(k.shape().hasTags(.{ .h, .k, .hd }), err_template ++ "k is missing tags {{.h, .k, .hd}}", err_args);
    stdx.debug.assert(v.shape().hasTags(.{ .h, .k, .hd }), err_template ++ "v is missing tags {{.h, .k, .hd}}", err_args);

    //if (opts.allow_cudnn and cuda.canUseCudnnSdpa(q.shape())) {
    //    return cuda.sdpa(q, k, v, opts);
    //}

    if (q.dim(.h) != k.dim(.h)) {
        stdx.debug.assert(@mod(q.dim(.h), k.dim(.h)) == 0, err_template ++ "Different number of heads for keys and queries, but can't repeat keys.", err_args);
        // Note: we don't try to repeat queries.
        // Repeating keys is the interesting optimisation cause it reduces KV cache memory usage.
        const num_rep: u63 = @intCast(@divExact(q.dim(.h), k.dim(.h)));
        k, v = .{ k.repeat1d(.h, num_rep), v.repeat1d(.h, num_rep) };
    }
    const attn_mask = if (opts.attn_mask) |m| m else null;

    const dims = zml.helpers.collectDims(.{ .h, .q, .k, .hd }, &.{ q, k, v, attn_mask }, .strict) catch {
        stdx.debug.panic(err_template ++ "Inputs have incompatible shapes.", err_args);
    };
    const sqrtHeadDim: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(dims.hd)));
    const head_scaling = if (opts.scale) |s| s else Tensor.scalar(sqrtHeadDim, k.dtype());
    k = k.mul(head_scaling.convert(k.dtype()));
    std.debug.print("k.shape: {f}\n", .{k.shape()});

    var attn_weights = q.dot(k, .hd);
    std.debug.print("attn_weights.shape: {f}\n", .{attn_weights.shape()});
    // log.debug("attn_weights : {}, attn_mask : {?}", .{ attn_weights, attn_mask });
    if (attn_mask) |mask| attn_weights = attn_weights.add(mask.broad(attn_weights.shape()));
    attn_weights = attn_weights.convert(.f32).softmax(.k).convert(q.dtype());

    var attn = attn_weights.dot(v, .k);
    return attn.transpose(q.shape());
}
