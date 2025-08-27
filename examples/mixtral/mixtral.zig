const std = @import("std");
const testing = std.testing;

const stdx = @import("stdx");
const zml = @import("zml");

const log = std.log.scoped(.Mixtral);

/// Mixtral architecture, using huggingface transformers naming.
/// Dimensions of activations: {.b, .s, .d}
pub const MixtralLM = struct {
    pub const Config = struct {
        bos_token_id: u32 = 199998,
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
    };

    lm_head: ?zml.nn.Linear,
    model: Mixtral,

    // Options controlling generation
    gen_opts: zml.nn.SamplingStrategy = .{},
    config: Config,

    pub fn init(allocator: std.mem.Allocator, store: zml.aio.BufferStore, config: Config, options: Options) !MixtralLM {
        var self: MixtralLM = .{
            .config = config,
            .gen_opts = options.sampling_strategy orelse .{},
            .model = .{
                .max_seq_len = @intCast(options.max_seq_len),
                .num_heads = @intCast(config.num_attention_heads),
                .num_kv_heads = @intCast(config.num_key_value_heads),
                .rope_opts = .{
                    .layout = if (config.hf_rope_impl) .sequential else .interleaved,
                    .freq_base = config.rope_theta,
                    .scaling = config.rope_scaling,
                },

                .embed_tokens = .{
                    .weight = store.getTensor("model.embed_tokens.weight").withSharding(.{1}),
                },
                .layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers),
                .norm = .{
                    .weight = store.getTensor("model.norm.weight"),
                    .eps = config.rms_norm_eps,
                },
            },
            .lm_head = .{ .weight = store.getTensor("lm_head.weight").withSharding(.{0}) },
        };

        var prefix: zml.aio.PrefixBuilder = try .initCapacity(allocator, 1024);
        try prefix.push(stdx.noalloc, "model.layers");
        for (self.model.layers, 0..) |*layer, i| {
            try prefix.pushDigit(stdx.noalloc, i);
            defer prefix.pop();

            // TODO: safer layer init
            var self_attn = try zml.aio.populateModelWithPrefix(SelfAttn, allocator, store, prefix.concat("self_attn"));
            {
                self_attn.num_heads = self.model.num_heads;
                self_attn.num_kv_heads = self.model.num_kv_heads;
                self_attn.rope_opts = self.model.rope_opts;
                self_attn.q_proj.weight = self_attn.q_proj.weight.withSharding(.{0});
                self_attn.k_proj.weight = self_attn.k_proj.weight.withSharding(.{0});
                self_attn.v_proj.weight = self_attn.v_proj.weight.withSharding(.{0});
                self_attn.o_proj.weight = self_attn.o_proj.weight.withSharding(.{1});
            }

            const on_disk_moe = try zml.aio.populateModelWithPrefix(MoE.OnDisk, allocator, store, prefix.concat("mlp"));
            var moe = on_disk_moe.rewrite();
            {
                moe.experts.gate_up_proj.blocks = moe.experts.gate_up_proj.blocks.withSharding(.{.expert});
                moe.experts.down_proj.blocks = moe.experts.down_proj.blocks.withSharding(.{.expert});
            }

            layer.* = .{
                .input_layernorm = .{
                    .weight = store.getTensor(prefix.concat("input_layernorm.weight")),
                    .eps = config.rms_norm_eps,
                },
                .post_attention_layernorm = .{
                    .weight = store.getTensor(prefix.concat("post_attention_layernorm.weight")),
                    .eps = config.rms_norm_eps,
                },
                .self_attn = self_attn,
                .mlp = moe,
            };
        }

        // TODO(Corentin): Fix lm_head sharding when top-k sampling is enabled.
        // It currently crashes/compilation fails
        if (self.gen_opts.topk == 1 and self.lm_head != null) {
            self.lm_head.?.weight = self.lm_head.?.weight.withSharding(.{0});
        }

        return self;
    }

    /// Predicts the token at `token_index` position.
    /// Returns:
    ///  - updated `tokens`,
    ///  - updated KV cache
    ///  - a Rng state to allow for probabilistic generation
    pub fn forward(
        self: MixtralLM,
        tokens_: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        rng: zml.Tensor.Rng,
    ) struct { zml.Tensor, KvCache, zml.Tensor.Rng } {
        stdx.debug.assert(tokens_.dtype() == .u32 and tokens_.rank() >= 1 and token_index.dtype() == .u32 and token_index.rank() <= 1, "Can't run Mixtral ! Expected >=1d tokens and 0d token_index, got: {} and {}", .{ tokens_, token_index });
        const tokens = tokens_.withPartialTags(.{.s});
        const out, const updated_kv_cache = zml.call(self.model, .forward, .{ tokens, token_index, kv_cache });
        const new_tokens, const new_rng = self.sampleTokens(self.lm_head, out, rng, self.gen_opts);
        return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), updated_kv_cache, new_rng };
    }

    pub fn sampleTokens(
        self: MixtralLM,
        lm_head_: ?zml.nn.Linear,
        out_: zml.Tensor,
        rng: zml.Tensor.Rng,
        opts: zml.nn.SamplingStrategy,
    ) struct { zml.Tensor, zml.Tensor.Rng } {
        const out = out_.withPartialTags(.{ .s, .d });

        var logits = blk: {
            if (lm_head_) |lm_head| {
                break :blk zml.call(lm_head, .forward, .{out});
            } else {
                break :blk self.model.embed_tokens.weight.withTags(.{ .voc, .d }).dot(out, .{.d});
            }
        };

        if (logits.shape().hasTag(.voc) == null)
            logits = logits.rename(.{ .d = .voc });

        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, opts, rng);
        return .{ next_tokens, new_rng };
    }

    pub fn loadBuffers(self: MixtralLM, allocator: std.mem.Allocator, store: zml.aio.BufferStore, platform: zml.Platform) !zml.Bufferized(MixtralLM) {
        var prefix: zml.aio.PrefixBuilder = try .initCapacity(allocator, 256);
        defer prefix.deinit(allocator);

        const noalloc = stdx.noalloc;
        const loaded: zml.Bufferized(MixtralLM) = .{
            .model = .{
                .embed_tokens = try store.loadModelById(zml.nn.TokenEmbedding, noalloc, self.model.embed_tokens, platform),
                .layers = try allocator.alloc(zml.Bufferized(TransformerLayer), self.model.layers.len),
                .norm = try store.loadModelById(RmsNorm, noalloc, self.model.norm, platform),
            },
            .lm_head = try store.loadModelById(?zml.nn.Linear, noalloc, self.lm_head, platform),
        };

        prefix.push(noalloc, "model.layers") catch unreachable;
        for (loaded.model.layers, self.model.layers, 0..) |*d_layer, layer, layer_id| {
            const ckpt = prefix.checkpoint();
            defer prefix.restore(ckpt);

            prefix.pushDigit(noalloc, layer_id) catch unreachable;
            d_layer.* = .{
                .input_layernorm = try store.loadModelById(RmsNorm, noalloc, layer.input_layernorm, platform),
                .self_attn = try store.loadModelById(SelfAttn, noalloc, layer.self_attn, platform),
                .post_attention_layernorm = try store.loadModelById(RmsNorm, noalloc, layer.post_attention_layernorm, platform),
                .mlp = try store.loadModelById(MoE, noalloc, layer.mlp, platform),
            };
        }

        return loaded;
    }
};

pub const Mixtral = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    norm: RmsNorm,
    layers: []TransformerLayer,

    max_seq_len: u32 = 0,
    num_heads: i64 = 32,
    num_kv_heads: i64 = 32,
    rope_opts: zml.nn.RopeOpts = .{
        .layout = .interleaved,
        .freq_base = 10_000,
    },

    const Shape = struct {
        s: u32,
        layer: u16,
        hd: u16,
        nh: u16,
        nkvh: u16,
        dtype: zml.DataType,
    };

    pub fn shape(self: Mixtral) Shape {
        const key_dim = self.layers[0].self_attn.k_proj.weight.dim(0);
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;

        return .{
            .s = self.max_seq_len,
            .layer = @intCast(self.layers.len),
            .hd = @intCast(@divExact(key_dim, num_kv_heads)),
            .nh = @intCast(self.num_heads),
            .nkvh = @intCast(num_kv_heads),
            .dtype = self.embed_tokens.weight.dtype(),
        };
    }

    /// Forward one token, using KV cache for previous tokens.
    /// Returns result and updated KV cache.
    pub fn forward(self: Mixtral, tokens: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache) struct { zml.Tensor, KvCache } {
        const embeds = embed(self.embed_tokens, tokens);
        var hidden = embeds;

        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden, updated_kv_cache = zml.call(layer, .forward, .{ hidden, token_index, updated_kv_cache.atLayer(i) });
        }
        const output = zml.call(self.norm, .forward, .{hidden});

        return .{ output, updated_kv_cache.reuseBuffer(kv_cache) };
    }

    pub fn embed(embed_tokens_: zml.nn.TokenEmbedding, tokens_: zml.Tensor) zml.Tensor {
        return zml.call(embed_tokens_, .forward, .{tokens_}).withPartialTags(.{.d});
    }
};

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: SelfAttn,
    post_attention_layernorm: RmsNorm,
    mlp: MoE,

    pub fn forward(
        self: TransformerLayer,
        x0: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
    ) struct { zml.Tensor, KvCache } {
        // Self Attention
        //log.debug("TransformerLayer({}) -> {}", .{ x0, self.input_layernorm.forward(x0) });
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {}", .{x0});

        const x0_normalized = zml.call(self.input_layernorm, .forward, .{x0});
        const delta0, const updated_kv_cache = zml.call(self.self_attn, .forward, .{ x0_normalized, token_index, kv_cache });
        const x1 = x0.add(delta0);

        // Fully Connected
        const x1_normalized = zml.call(self.post_attention_layernorm, .forward, .{x1});
        const x2 = zml.call(self.mlp, .forward, .{x1_normalized}).add(x1);

        return .{ x2.reuseBuffer(x0), updated_kv_cache };
    }
};

const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32 = 1e-6,

    /// L2 normalization of input tensor along `.d` axis.
    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        // Note: contrary to Llama here the full layer is done in .f32, not just the variance computation.
        const normalized = zml.nn.rmsNorm(x.convert(.f32), .d, self.eps);
        return normalized.mul(self.weight.convert(.f32).withTags(.{.d}).broad(x.shape())).convert(input.dtype());
    }
};

const MoE = struct {
    router: zml.nn.Linear,
    experts: Mlp,

    pub const OnDisk = struct {
        router: zml.nn.Linear,
        experts: struct {
            down_proj_bias: zml.Tensor,
            down_proj_blocks: zml.Tensor,
            down_proj_scales: zml.Tensor,
            gate_up_proj_bias: zml.Tensor,
            gate_up_proj_blocks: zml.Tensor,
            gate_up_proj_scales: zml.Tensor,
        },

        pub fn rewrite(on_disk: OnDisk) MoE {
            const e = on_disk.experts;
            const experts: Mlp = .{
                .gate_up_proj = .{
                    // We need to bitcast the scale cause safetensors doesn't encode f8 types correctly
                    .scale = e.gate_up_proj_scales.bitCast(.f8e8m0).withTags(.{ .expert, .out, .d }),
                    // We don't bitcast here because PJRT doesn't handle packed host buffers
                    .blocks = e.gate_up_proj_blocks.withTags(.{ .expert, .out, .d, .d_block }),
                    .blocks_dtype = .f4e2m1,
                    .bias = e.gate_up_proj_bias.withTags(.{ .expert, .d }),
                },
                .down_proj = .{
                    .blocks = e.down_proj_blocks.withTags(.{ .expert, .out, .d, .d_block }),
                    .blocks_dtype = .f4e2m1,
                    .scale = e.down_proj_scales.bitCast(.f8e8m0).withTags(.{ .expert, .out, .d }),
                    .bias = e.down_proj_bias.withTags(.{ .expert, .d }),
                },
            };

            var router = on_disk.router;
            router.weight = router.weight.withTags(.{ .expert, .d });

            return .{ .router = router, .experts = experts };
        }
    };

    pub fn forward(self: MoE, input: zml.Tensor) zml.Tensor {
        const gating = self.router.forward(input).convert(.f32).softmax(.expert);
        return zml.nn.mixtureOfExperts(Mlp, self.experts, input, gating, .{ .experts_per_token = 2, .tokens_per_expert_ratio = 1.5 });
    }

    pub fn loadSharded(self: MoE, store: zml.aio.BufferStore, prefix: *zml.aio.PrefixBuilder, platform: zml.Platform) !zml.Bufferized(MoE) {
        prefix.push(stdx.noalloc, "block_sparse_moe.experts") catch unreachable;

        return .{
            .router = try store.loadModelById(zml.nn.Linear, stdx.noalloc, self.router, platform),
            .experts = .{
                .gate_up_proj = .{ .blocks = try loadShardedWeight(store, prefix, "w1", platform, self.experts.gate_up_proj.blocks) },
                .down_proj = .{ .blocks = try loadShardedWeight(store, prefix, "w2", platform, self.experts.down_proj.blocks) },
            },
        };
    }

    fn loadShardedWeight(store: zml.aio.BufferStore, prefix: *zml.aio.PrefixBuilder, name: []const u8, platform: zml.Platform, weight: zml.Tensor) !zml.Buffer {
        const devices = platform.getDevices();
        const num_experts = weight.dim(.expert);
        const num_shards = platform.sharding().num_partitions;

        // Note: this requires num_devices == num_partitions, this is overly restrictive.
        if (devices.len == num_shards and num_experts % num_shards == 0 and num_experts >= num_shards) {
            const num_experts_per_shard = @divExact(num_experts, num_shards);
            const tmp_buf: []u8 = try std.heap.smp_allocator.alloc(u8, 4096);
            defer std.heap.smp_allocator.free(tmp_buf);
            var fba: std.heap.FixedBufferAllocator = .init(tmp_buf);

            const allocator = fba.allocator();
            const transfer = try platform.batchedTransfer(allocator, &.{weight.shape()}, .device);

            for (0..num_experts) |expert_id| {
                const part_id = expert_id % num_experts_per_shard;
                const shard_id = @divFloor(expert_id, num_experts_per_shard);
                const ckpt = prefix.checkpoint();
                defer prefix.restore(ckpt);
                try prefix.pushDigit(stdx.noalloc, expert_id);
                try prefix.push(stdx.noalloc, name);
                try prefix.push(stdx.noalloc, "weight");

                const expert = store.get(prefix.items()) orelse {
                    log.err("Buffer not found: {s}", .{prefix.items()});
                    store.findSimilarBufferKeys(std.heap.smp_allocator, prefix.items());
                    @panic("Buffer not found");
                };

                const ev = transfer.transferData(
                    .{ .buffer_id = 0, .device_id = @intCast(shard_id), .offset = expert.bytes().len * part_id },
                    expert.bytes(),
                    part_id + 1 == num_experts_per_shard,
                );
                _ = ev;
            }
            return transfer.buffers[0];
        } else {
            @panic("TODO");
        }
    }
};

const Mlp = struct {
    gate_up_proj: zml.nn.BlockScaledLinear, // {.out = intermediate_size * 2, .d = hidden_size / block_size, .d_block = block_size }
    down_proj: zml.nn.BlockScaledLinear, // {.out = hidden_size * 2, .d = intermediate_size / block_size, .d_block = block_size }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const dt = x.dtype();
        var gate, var up = zml.nn.splitRealImg(self.gate_up_proj.forward(x), .interleaved);
        gate = .minimum(gate, .scalar(7, dt));
        up = .clamp(up, .scalar(-7, dt), .scalar(7, dt));

        const out = gate.quickGelu().mul(up.addConstant(1));
        return zml.call(self.down_proj, .forward, .{out});
    }
};

pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    sinks: zml.Tensor,

    o_proj: zml.nn.Linear,

    num_heads: i64 = undefined,
    num_kv_heads: i64 = 0,
    rope_opts: zml.nn.RopeOpts = undefined,

    /// Self Attention.
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
    ) struct { zml.Tensor, KvCache } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;
        var q = zml.call(self.q_proj, .forward, .{x}).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto }).withSharding(.{.h});
        var k = zml.call(self.k_proj, .forward, .{x}).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto }).withSharding(.{.h});
        var v = zml.call(self.v_proj, .forward, .{x}).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto }).withSharding(.{.h});

        // Generate the attention mask.
        const seq_len = kv_cache.k.dim(.k);
        var attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, x.dtype(), null);

        // Note: in Pytorch it would be very inefficient to generate the full attn_mask,
        // then slice into it, but XLA is able to optimize this correctly.
        attn_mask = attn_mask.gatherSlices(zml.Shape.init(.{ .q = x.dim(.s) }, attn_mask.dtype()), token_index.reshape(.{ .coord = 1 }), .{});

        // In self-attention, .s axis is used both for keys and queries.
        const pos_index = b: {
            const temp = zml.Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };

        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const dtype = q.dtype();
        const new_kv_cache = kv_cache.update(k, v, token_index);
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);

        const softmax_bias = q.dot(self.sinks.withTags(.{.hd}), .hd);
        std.log.warn("q: {}, sinks: {} -> softmax_bias: {}", .{ q, self.sinks, softmax_bias });
        const attn_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .allow_cudnn = true, .softmax_bias = softmax_bias });
        // const attn_output = zml.nn.sdpaMemEfficient(q, k, v, .{ .attn_mask = attn_mask }, .{ .q_chunk_size = 4096, .k_chunk_size = 1024 });
        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return .{ zml.call(self.o_proj, .forward, .{attn}), new_kv_cache };
    }
};

pub const KvCache = struct {
    k: zml.Tensor,
    v: zml.Tensor,
    layer_index: zml.Tensor,

    pub fn init(kv_shape: zml.Shape) KvCache {
        // The KV-cache is initialized with ones to detect reads of uninitialized memory.
        return .{
            .k = .constant(kv_shape, kv_shape.dtype().one()).withSharding(.{.h}),
            .v = .constant(kv_shape, kv_shape.dtype().one()).withSharding(.{.h}),
            .layer_index = .scalar(-1, .u32),
        };
    }

    pub fn initShape(kv_shape: zml.Shape) zml.ShapeOf(KvCache) {
        return .{
            .k = kv_shape,
            .v = kv_shape,
            .layer_index = zml.Shape.init(.{}, .u32),
        };
    }

    pub fn initBuffer(kv_shape: zml.Shape, platform: zml.Platform) !zml.Bufferized(KvCache) {
        return .{
            .k = try zml.Buffer.uninitialized(platform, kv_shape, .{}),
            .v = try zml.Buffer.uninitialized(platform, kv_shape, .{}),
            .layer_index = try zml.Buffer.uninitialized(platform, .scalar(.u32), .{}),
        };
    }

    pub fn keys(self: KvCache) zml.Tensor {
        return self.k.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn values(self: KvCache) zml.Tensor {
        return self.v.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn update(self: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: ?zml.Tensor) KvCache {
        const k_shape = self.k.shape().drop(.layer);
        var layer = self.layer_index;
        layer = if (token_index) |idx| layer.broad(idx.shape()) else layer;

        return if (token_index) |idx| .{
            .k = self.k.scatterSlices(
                .{ .layer = layer, .k = idx },
                new_k.convert(self.k.dtype()).transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(self.k),
            .v = self.v.scatterSlices(
                .{ .layer = layer, .k = idx },
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

    pub fn atLayer(self: KvCache, layer_index: usize) KvCache {
        return .{
            .k = self.k,
            .v = self.v,
            .layer_index = .scalar(layer_index, .u32),
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
