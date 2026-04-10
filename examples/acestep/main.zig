const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const dialects = @import("mlir/dialects");

const hz_type = .f32;
const cfg: f32 = 1.0;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Acestep = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    norm: RmsNorm,

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
        tie_word_embeddings: bool = false,
        rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = .{} },
    };

    pub const Options = struct {
        seq_len: u32 = 256,
    };

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Acestep {
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| {
            layer.* = try .init(@intCast(i), store.withPrefix("layers").withLayer(i), config);
        }
        return .{
            .embed_tokens = .{ .weight = store.createTensor(
                "embed_tokens.weight",
                .{ .voc, .d },
                .{ .voc = .replicated, .d = .model },
            ) },
            .layers = layers,
            .norm = .init(store.withPrefix("norm"), config),
        };
    }

    pub fn load(
        self: *const Acestep,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(Acestep) {
        return zml.io.load(Acestep, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 128 * 1024 * 1024,
        });
    }

    pub fn deinit(self: *const Acestep, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Acestep)) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }
        RmsNorm.unloadBuffers(&self.norm);
    }

    pub fn forward(
        self: Acestep,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        rng: zml.Tensor.Rng,
        phase_mask: zml.Tensor,
        repetition_pos_penalty: zml.Tensor,
        repetition_neg_penalty: zml.Tensor,
        penalize_repetitions: bool,
    ) struct { zml.Tensor, KvCache, zml.Tensor.Rng, zml.Tensor } {
        const seq_size = tokens.dim(.s);
        var updated_kv_cache = kv_cache;
        // embbed input tokens
        var output = self.embed_tokens.convert(hz_type).forward(tokens.withPartialTags(.{.s})).withPartialTags(.{.d});
        // forward pass on the neural network layers
        for (self.layers, 0..) |layer, i| {
            output, updated_kv_cache = layer.forward(output, token_index, updated_kv_cache.atLayer(i));
        }
        output = self.norm.forward(output);
        // compute logits for the output tokens
        var logits = self.embed_tokens.weight.withTags(.{ .voc, .d }).convert(hz_type).dot(output, .d);
        // apply repetition penalty (only during decode)
        if (penalize_repetitions) {
            const zero = zml.Tensor.scalar(0.0, hz_type);
            const logits_positive = zml.Tensor.cmp(logits, dialects.stablehlo.ComparisonDirection.Direction.GE, zero.broad(logits.shape()));
            const repetition_penalty = zml.Tensor.select(logits_positive, repetition_pos_penalty, repetition_neg_penalty);
            logits = logits.mul(repetition_penalty.convert(hz_type));
        }
        // apply phase mask to select tokens from the valid subset
        const masked_logits = logits.add(phase_mask.convert(hz_type).broad(logits.shape()));
        // return result
        const sample_strat: zml.nn.SamplingStrategy = .{
            .topk = 1,
            .temperature = 0.85,
        };
        const logits_32 = masked_logits.convert(.f32);
        const next_logits = logits_32.choose1d(.s, seq_size - 1);
        const next_tokens, const new_rng = zml.nn.sampleTokens(logits_32, sample_strat, rng);
        return .{ next_tokens.convert(tokens.dtype()).reuseBuffer(tokens), updated_kv_cache.reuseBuffer(kv_cache), new_rng, next_logits };
    }
};

const EmbedWrapper = struct {
    embed: zml.nn.TokenEmbedding,

    pub fn forward(wrapper: EmbedWrapper, input: zml.Tensor) zml.Tensor {
        const tagged_input = input.withTags(.{ .b, .s }).squeeze(.b);
        const output = wrapper.embed.convert(hz_type).forward(tagged_input).withPartialTags(.{.d});
        return output.insertAxes(0, .{ .b });
    }
};

const TransformerLayer = struct {
    id: u8,
    input_norm: RmsNorm,
    att_layer: AttLayer,
    post_att_norm: RmsNorm,
    mlp_layer: MlpLayer,

    pub fn init(id_ : u8, store: zml.io.TensorStore.View, config: Acestep.Config) !TransformerLayer {
        return .{
            .id = id_,
            .input_norm = .init(store.withPrefix("input_layernorm"), config),
            .att_layer = try .init(store.withPrefix("self_attn"), config),
            .post_att_norm = .init(store.withPrefix("post_attention_layernorm"), config),
            .mlp_layer = try .init(store.withPrefix("mlp")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_norm);
        AttLayer.unloadBuffers(&self.att_layer);
        RmsNorm.unloadBuffers(&self.post_att_norm);
        MlpLayer.unloadBuffers(&self.mlp_layer);
    }

    pub fn forward(self: TransformerLayer, x0: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache) struct { zml.Tensor, KvCache } {
        std.log.debug("FWD : transformer layer {d}", .{self.id});

        // Keep the residual stream replicated to avoid repeated gathers before q/k/v.
        const x0_replicated = x0.withPartitioning(.{ .d = .replicated });
        const x0_normalized = self.input_norm.forward(x0_replicated);
        const delta0, const updated_kv_cache = self.att_layer.forward(
            x0_normalized,
            token_index,
            kv_cache,
        );

        // Fully Connected
        const x1 = x0_replicated.add(delta0).withPartitioning(.{ .d = .replicated });
        const x1_normalized = self.post_att_norm.forward(x1);
        const x2 = self.mlp_layer.forward(x1_normalized).withPartitioning(.{ .d = .replicated }).add(x1).withPartitioning(.{ .d = .replicated });

        return .{ x2.reuseBuffer(x0), updated_kv_cache };
    }
};

pub const TransWrapper = struct {
    trans: TransformerLayer,
    tok_id: zml.Tensor,
    kv_cache: KvCache,

    pub fn forward(wrapper: TransWrapper, input: zml.Tensor, att_mask: zml.Tensor, pos_id: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        _ = att_mask;
        _ = pos_id;
        _ = cos;
        _ = sin;
        const tagged_input = input.withTags(.{ .b, .s, .d }).squeeze(.b);
        const output, _ = wrapper.trans.forward(tagged_input, wrapper.tok_id, wrapper.kv_cache.atLayer(0));
        return output.insertAxes(0, .{ .b });
    }
};

const AttLayer = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    o_proj: zml.nn.Linear,
    num_heads: i64 = undefined,
    num_kv_heads: i64 = 0,
    rope_opts: zml.nn.RopeOpts = undefined,

    pub fn init(store: zml.io.TensorStore.View, config: Acestep.Config) !AttLayer {
        var rope_scaling = config.rope_scaling;
        rope_scaling.setRopeTheta(config.rope_theta);
        return .{
            .q_proj = .init(store.createTensor("q_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }), null, .d),
            .k_proj = .init(store.createTensor("k_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }), null, .d),
            .v_proj = .init(store.createTensor("v_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }), null, .d),
            .o_proj = .init(store.createTensor("o_proj.weight", .{ .d_out, .d }, .{ .d = .model }), null, .d),
            .q_norm = .init(store.withPrefix("q_norm"), config),
            .k_norm = .init(store.withPrefix("k_norm"), config),
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
            .rope_opts = .{
                .layout = .sequential,
                .scaling = rope_scaling,
            },
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AttLayer)) void {
        self.q_proj.weight.deinit();
        self.k_proj.weight.deinit();
        self.v_proj.weight.deinit();
        self.o_proj.weight.deinit();
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    /// Self zml.attention.attention.
    ///   - If token_index is set, x is assumed to be the representation of one new token,
    /// and kv_cache will be read for the previous tokens.
    ///   - If token_index is not set, x is assumed to be the representation of all tokens
    /// since the beginning of the sequence, and kv_cache won't be read.
    /// In both case, kv_cache will be updated with the computed key and value.
    /// x: {.b, .s, .d } -> .{.b, .s, .d}
    pub fn forward(
        self: AttLayer,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
    ) struct { zml.Tensor, KvCache } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;

        // Make hidden state replicated once and reuse it across q/k/v projections.
        // This avoids paying gather-style collectives independently for each projection.
        const x_qkv = x.withPartitioning(.{ .d = .replicated });

        var q = self.q_proj.convert(hz_type).forward(x_qkv).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = self.k_proj.convert(hz_type).forward(x_qkv).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        var v = self.v_proj.convert(hz_type).forward(x_qkv).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        q = q.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        k = k.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });

        // In self-attention, .s axis is used both for keys and queries.
        const pos_index = b: {
            const temp = zml.Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });
        q = q.withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated });
        k = k.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });

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
            .vanilla,
            .vanilla,
        ).withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated });

        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const delta = self.o_proj.convert(hz_type).forward(attn)
            .rename(.{ .d_out = .d })
            .withPartitioning(.{ .d = .replicated });
        return .{ delta, new_kv_cache };
    }
};

const AttWrapper = struct {
    att: AttLayer,
    tok_id: zml.Tensor,
    kv_cache: KvCache,

    pub fn forward(wrapper: AttWrapper, input: zml.Tensor, att_mask: zml.Tensor, pos_id: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        _ = att_mask;
        _ = pos_id;
        _ = cos;
        _ = sin;
        const tagged_input = input.withTags(.{ .b, .s, .d }).squeeze(.b);
        const output, _ = wrapper.att.forward(tagged_input, wrapper.tok_id, wrapper.kv_cache);
        return output.insertAxes(0, .{ .b });
    }
};

const MlpLayer = struct {
    up_proj: zml.Tensor,
    gate_proj: zml.Tensor,
    down_proj: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) !MlpLayer {
        return .{
            .up_proj = store.createTensor("up_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }),
            .gate_proj = store.createTensor("gate_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }),
            .down_proj = store.createTensor("down_proj.weight", .{ .d, .d_out }, .{ .d = .model }),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(MlpLayer)) void {
        self.up_proj.deinit();
        self.gate_proj.deinit();
        self.down_proj.deinit();
    }

    pub fn forward(self: MlpLayer, input: zml.Tensor) zml.Tensor {
        std.log.debug("FWD : mlp", .{});
        const up_projection = input.dot(self.up_proj.convert(hz_type), .d);
        const gate_projection = input.dot(self.gate_proj.convert(hz_type), .d);
        const activation = gate_projection.silu().mul(up_projection);
        const output = activation.dot(self.down_proj.convert(hz_type), .d_out);
        return output;
    }
};

const MlpWrapper = struct {
    mlp: MlpLayer,

    pub fn forward(wrapper: MlpWrapper, input: zml.Tensor) zml.Tensor {
        const tagged_input = input.withTags(.{ .b, .s, .d }).squeeze(.b);
        const output = wrapper.mlp.forward(tagged_input);
        return output.insertAxes(0, .{ .b });
    }
};

const TensorWrapper = struct {
    tensor: zml.Tensor,

    pub fn forward(wrapper: TensorWrapper, input: zml.Tensor) zml.Tensor {
        const tagged_input = input.withTags(.{ .b, .s, .d }).squeeze(.b);
        const output = tagged_input.dot(wrapper.tensor.convert(hz_type), .d);
        return output.insertAxes(0, .{ .b });
    }
};

const TensorWrapperI = struct {
    tensor: zml.Tensor,

    pub fn forward(wrapper: TensorWrapperI, input: zml.Tensor) zml.Tensor {
        const tagged_input = input.withTags(.{ .b, .s, .d_out }).squeeze(.b);
        const output = tagged_input.dot(wrapper.tensor.convert(hz_type), .d_out);
        return output.insertAxes(0, .{ .b });
    }
};

const RmsNorm = struct {
    weights: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Acestep.Config) RmsNorm {
        return .{
            .weights = store.createTensor("weight", .{.d_out}, null),
            .eps = config.rms_norm_eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weights.deinit();
    }

    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        //std.log.debug("FWD : norm", .{});
        // normalize along d axis each one of the s input tokens
        const normalized = zml.nn.rmsNorm(input, .d, self.eps);
        // scale with learned weights by broadening weights to match s input tokens
        return normalized.mul(self.weights.convert(hz_type).withTags(.{.d}).broad(input.shape()));
    }
};

pub const RmsWrapper = struct {
    norm: RmsNorm,

    pub fn forward(wrapper: RmsWrapper, input: zml.Tensor) zml.Tensor {
        const tagged_input = input.withTags(.{ .b, .s, .d }).squeeze(.b);
        const output = wrapper.norm.forward(tagged_input);
        return output.insertAxes(0, .{ .b });
    }
};

pub const RmshWrapper = struct {
    norm: RmsNorm,

    pub fn forward(wrapper: RmshWrapper, input: zml.Tensor) zml.Tensor {
        const tagged_input = input.withTags(.{ .b, .s, .h, .d }).squeeze(.b);
        const output = wrapper.norm.forward(tagged_input);
        return output.insertAxes(0, .{ .b });
    }
};

pub const ProjWrapper = struct {
    proj: zml.nn.Linear,

    pub fn forward(wrapper: ProjWrapper, input: zml.Tensor) zml.Tensor {
        const tagged_input = input.withTags(.{ .b, .s, .d }).squeeze(.b);
        const output = wrapper.proj.convert(hz_type).forward(tagged_input);
        return output.insertAxes(0, .{ .b });
    }
};

pub const KvCache = struct {
    k: zml.Tensor,
    v: zml.Tensor,
    layer_index: zml.Tensor,
    io: std.Io,

    pub fn init(kv_shape: zml.Shape, io: std.Io) KvCache {
        const sharded_shape = kv_shape.withPartitioning(.{ .h = .model });

        return .{
            .k = .fromShape(sharded_shape),
            .v = .fromShape(sharded_shape),
            .layer_index = .init(.{}, .u32),
            .io = io,
        };
    }

    pub fn initBuffer(self: KvCache, io: std.Io, platform: *const zml.Platform, sharding: zml.sharding.Sharding) !zml.Bufferized(KvCache) {
        return .{
            .k = try zml.Buffer.uninitialized(io, platform, self.k.shape(), sharding, .{}),
            .v = try zml.Buffer.uninitialized(io, platform, self.v.shape(), sharding, .{}),
            .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32, sharding),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(KvCache)) void {
        self.k.deinit();
        self.v.deinit();
        self.layer_index.deinit();
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

        return if (token_index) |idx|
            .{
                .k = self.k.scatterSlices(.{ .layer = layer, .k = idx }, new_k.convert(self.k.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.k),
                .v = self.v.scatterSlices(.{ .layer = layer, .k = idx }, new_v.convert(self.v.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.v),
                .layer_index = self.layer_index,
                .io = self.io,
            }
        else
            .{
                .k = self.k.scatterSlices(.{ .layer = layer }, new_k.convert(self.k.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.k),
                .v = self.v.scatterSlices(.{ .layer = layer }, new_v.convert(self.v.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.v),
                .layer_index = self.layer_index,
                .io = self.io,
            };
    }

    pub fn atLayer(self: KvCache, layer_index: usize) KvCache {
        return .{
            .k = self.k,
            .v = self.v,
            .layer_index = zml.Tensor.scalar(layer_index, .u32),
            .io = self.io,
        };
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{
            .k = self.k.reuseBuffer(other.k),
            .v = self.v.reuseBuffer(other.v),
            .layer_index = self.layer_index.reuseBuffer(other.layer_index),
            .io = self.io,
        };
    }
};

pub const Shardings = struct {
    replicated: zml.sharding.Sharding,
    model: zml.sharding.Sharding,

    pub fn init(platform: *zml.Platform) !Shardings {
        const model_mesh: zml.sharding.LogicalMesh = try .init("model", .{ .model = .high_bandwidth });
        const model_sharding_strategy: zml.sharding.Strategy = try .suggest(model_mesh, platform.physical_mesh);
        return .{
            .replicated = try zml.sharding.replicatedSharding(platform),
            .model = try .initFromStrategy(platform, model_mesh, model_sharding_strategy),
        };
    }

    pub fn all(self: *const Shardings) [2]zml.sharding.Sharding {
        return .{ self.replicated, self.model };
    }
};

const AcestepParams = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: KvCache,
    phase_mask: zml.Tensor,
    pos_penalty: zml.Tensor,
    neg_penalty: zml.Tensor,
    decode_phase: bool,
    rng: zml.Tensor.Rng,
    shardings: Shardings,
};

const AcestepPhase = struct {
    phase1: bool,
    phase1_mask: []f32,
    phase2_mask: []f32,
    eos_id: u32,

    pub fn deinit(self: AcestepPhase, allocator: std.mem.Allocator) void {
        allocator.free(self.phase1_mask);
        allocator.free(self.phase2_mask);
    }
};

const CompileModelResult = struct {
    prefill_exe: zml.Exe,
    decode_exe: zml.Exe,
};

const AudioMetadata = struct {
    bpm: []const u8,
    caption: []const u8,
    duration: []const u8,
    genres: []const u8,
    keyscale: []const u8,
    language: []const u8,
    timesignature: []const u8,
    lyric: []const u8,

    const field_names = [_][]const u8{
        "bpm:",
        "caption:",
        "duration:",
        "genres:",
        "keyscale:",
        "language:",
        "timesignature:",
        "lyric:",
    };

    fn isFieldStart(line: []const u8) bool {
        const trimmed = std.mem.trimStart(u8, line, " \t");
        for (field_names) |f| {
            if (std.mem.startsWith(u8, trimmed, f)) return true;
        }
        return false;
    }

    fn extractFieldBlock(allocator: std.mem.Allocator, input: []const u8, field: []const u8) ![]const u8 {
        var it = std.mem.splitScalar(u8, input, '\n');
        var found = false;
        var start: usize = 0;
        var end: usize = input.len;
        while (it.next()) |line| {
            const trimmed = std.mem.trimStart(u8, line, " \t");
            if (!found) {
                if (std.mem.startsWith(u8, trimmed, field)) {
                    found = true;
                    // Compute start index
                    const line_offset = @intFromPtr(line.ptr) - @intFromPtr(input.ptr);
                    const trim_offset = @intFromPtr(trimmed.ptr) - @intFromPtr(line.ptr);
                    start = line_offset + trim_offset + field.len;
                }
            } else {
                if (isFieldStart(line)) {
                    end = @intFromPtr(line.ptr) - @intFromPtr(input.ptr);
                    break;
                }
            }
        }
        if (!found) return "";
        const trimmed = std.mem.trim(u8, input[start..end], " \t\r\n");
        const output: []u8 = try allocator.alloc(u8, std.mem.replacementSize(u8, trimmed, "\n  ", " "));
        _ = std.mem.replace(u8, trimmed, "\n  ", " ", output);
        return output;
    }

    pub fn initFromString(allocator: std.mem.Allocator, input: []const u8) !AudioMetadata {
        var it = std.mem.splitSequence(u8, input, "</think>");
        const trimmed = it.first()[7..];
        return .{
            .bpm = try extractFieldBlock(allocator, trimmed, "bpm:"),
            .caption = try extractFieldBlock(allocator, trimmed, "caption:"),
            .duration = try extractFieldBlock(allocator, trimmed, "duration:"),
            .genres = try extractFieldBlock(allocator, trimmed, "genres:"),
            .keyscale = try extractFieldBlock(allocator, trimmed, "keyscale:"),
            .language = try extractFieldBlock(allocator, trimmed, "language:"),
            .timesignature = try extractFieldBlock(allocator, trimmed, "timesignature:"),
            .lyric = try extractFieldBlock(allocator, trimmed, "lyric:"),
        };
    }

    pub fn deinit(self: AudioMetadata, allocator: std.mem.Allocator) void {
        allocator.free(self.bpm);
        allocator.free(self.caption);
        allocator.free(self.duration);
        allocator.free(self.genres);
        allocator.free(self.keyscale);
        allocator.free(self.language);
        allocator.free(self.timesignature);
        allocator.free(self.lyric);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    // Auto-select platform
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    // Parse program args
    const process_args = try init.minimal.args.toSlice(arena.allocator());
    const options: std.Io.Dir.OpenOptions = .{
        .access_sub_paths = true,
        .iterate = false,
        .follow_symlinks = true,
    };
    const model_repo = try std.Io.Dir.openDirAbsolute(io, process_args[1], options);
    const model_path = process_args[2];
    const tokenizer_path = process_args[3];

    // Read model shapes.
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, model_path);
    defer registry.deinit();
    std.log.info("Found {d} activations in {s}", .{ registry.tensors.count(), model_path });

    // Print model shapes
    const tensors: zml.safetensors.Tensors = registry.tensors;
    const data = tensors.entries;
    for (0..data.len) |i| {
        const entry = data.get(i);
        const tensor: zml.safetensors.Tensor = tensors.get(entry.key).?;
        std.log.debug("Tensor(name={s} shape={f} size={d})", .{
            tensor.name,
            tensor.shape,
            tensor.byteSize(),
        });
    }

    // Init model
    const parsed_config = try parseConfig(allocator, io, model_repo);
    defer parsed_config.deinit();
    const config = parsed_config.value;
    const acestep_options: Acestep.Options = .{
        .seq_len = 256,
    };
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();
    const acestep_model: Acestep = try .init(allocator, store.view().withPrefix("model"), config);
    defer acestep_model.deinit(allocator);

    // Specify shapes of input arguments
    const dtype = acestep_model.embed_tokens.weight.dtype();
    const voc_size: usize = @intCast(acestep_model.embed_tokens.weight.dim(.voc));
    const acestep_parameters: AcestepParams = .{
        .prefill_tokens = .init(.{ .s = acestep_options.seq_len }, .u32),
        .decode_tokens = .init(.{ .s = 1 }, .u32),
        .token_index = .init(.{}, .u32),
        .kv_cache = .init(zml.Shape.init(.{
            .layer = acestep_model.layers.len,
            .k = acestep_options.seq_len,
            .h = config.num_key_value_heads,
            .hd = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads),
        }, dtype), io),
        .phase_mask = .init( .{ .voc = voc_size }, .f32),
        .pos_penalty = .init( .{ .voc = voc_size }, .f32),
        .neg_penalty = .init( .{ .voc = voc_size }, .f32),
        .decode_phase = false,
        .rng = .init(),
        .shardings = try .init(platform),
    };

    // Compile the prefill and decode models
    var compiled_model = try compileModel(allocator, io, platform, acestep_model, acestep_parameters);
    defer compiled_model.prefill_exe.deinit();
    defer compiled_model.decode_exe.deinit();
 
    // Fill the buffers with weights
    std.log.info("Load buffers", .{});
    var acestep_buffers = try acestep_model.load(init.arena.allocator(), io, platform, &store, &acestep_parameters.shardings.all());
    var acestep_kv_cache_buffers = try acestep_parameters.kv_cache.initBuffer(io, platform, acestep_parameters.shardings.model);
    defer Acestep.unloadBuffers(&acestep_buffers);
    defer KvCache.deinitBuffer(&acestep_kv_cache_buffers);

    // Test model activations
    //try testModel(allocator, io, platform, acestep_model, acestep_buffers, acestep_kv_cache_buffers, acestep_parameters, process_args[4];);

    // Initialize tokenizer
    std.log.info("Initialize tokenizer", .{});
    var tokenizer: zml.tokenizer.Tokenizer = try .fromFile(allocator, io, tokenizer_path);
    defer tokenizer.deinit();

    // Initialize masks for phases I and II
    std.log.info("Initialize phase masks", .{});
    var acestep_phase = try buildTokenMasks(allocator, tokenizer, voc_size);
    defer acestep_phase.deinit(allocator);

    // Test on one prompt
    const prompt = "a chill guitar melody\n\ninstrumental: true";
    std.log.info("Start inspiration with initial prompt", .{});
    std.log.info("########### prompt start ###########:\n{s}", .{prompt});
    std.log.info("###########  prompt end  ###########", .{});
    const inspi_tokens = try tokenizeInspirationPrompt(allocator, tokenizer, prompt);
    defer allocator.free(inspi_tokens);

    const inspi_result = try generateInspirationText(
        allocator,
        io,
        acestep_buffers,
        compiled_model.prefill_exe,
        compiled_model.decode_exe,
        &acestep_kv_cache_buffers,
        tokenizer,
        inspi_tokens,
        config,
        acestep_options,
        acestep_phase,
        0,
        platform,
        acestep_parameters.shardings.replicated,
    );
    defer allocator.free(inspi_result);

    const metadata: AudioMetadata = try .initFromString(allocator, inspi_result);
    defer metadata.deinit(allocator);

    std.log.info("Start generation", .{});
    acestep_phase.phase1 = false;
    const gen_tokens_cond, const gen_tokens_uncond = try tokenizeGenerationPrompt(allocator, tokenizer, metadata);
    defer allocator.free(gen_tokens_cond);
    defer allocator.free(gen_tokens_uncond);

    std.log.debug("Conditional input tokens {any}", .{gen_tokens_cond});
    std.log.debug("Unconditional input tokens {any}", .{gen_tokens_uncond});
    
    var acestep_kv_cond_cache_buffers = try acestep_parameters.kv_cache.initBuffer(io, platform, acestep_parameters.shardings.model);
    var acestep_kv_uncond_cache_buffers = try acestep_parameters.kv_cache.initBuffer(io, platform, acestep_parameters.shardings.model);
    defer KvCache.deinitBuffer(&acestep_kv_cond_cache_buffers);
    defer KvCache.deinitBuffer(&acestep_kv_uncond_cache_buffers);

    const gen_result = try generateAudioCodes(
        allocator,
        io,
        acestep_buffers,
        compiled_model.prefill_exe,
        compiled_model.decode_exe,
        &acestep_kv_cond_cache_buffers,
        &acestep_kv_uncond_cache_buffers,
        tokenizer,
        gen_tokens_cond,
        gen_tokens_uncond,
        acestep_options,
        acestep_phase,
        0,
        platform,
        acestep_parameters.shardings.replicated,
        metadata
    );
    defer allocator.free(gen_result);

    std.log.info("End of phase II", .{});
}

fn testModel(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, acestep_model: Acestep, acestep_buffers: zml.Bufferized(Acestep), kv_cache_buffers: zml.Bufferized(KvCache), parameters: AcestepParams, activations_path: []const u8) !void {
    std.log.info("Test activations", .{});
    var activations_registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, activations_path);
    defer activations_registry.deinit();
    std.log.info("Found {d} activations in {s}", .{ activations_registry.tensors.count(), activations_path });

    var activations_store: zml.io.TensorStore = .fromRegistry(allocator, &activations_registry);
    defer activations_store.deinit();
    var prefill_token_pos_buffer = try zml.Buffer.scalar(io, platform, 0, .u32, parameters.shardings.model);
    defer prefill_token_pos_buffer.deinit();
    const tok_id = zml.Tensor.fromShape(.init(.{}, .u32));

    std.log.info("Test activations : embed layer", .{});
    const wrapper_embed: EmbedWrapper = .{
        .embed = acestep_model.embed_tokens,
    };
    const wrapper_buffers_embed: zml.Bufferized(EmbedWrapper) = .{
        .embed = acestep_buffers.embed_tokens,
    };
    const layer_embed = "model.model.embed_tokens";
    try zml.testing.testLayer(allocator, io, platform, wrapper_embed, .forward, activations_store.view(), layer_embed, wrapper_buffers_embed, &parameters.shardings.all(), .{});

    std.log.info("Test activations : rms layer input norm", .{});
    const wrapper_rms: RmsWrapper = .{
        .norm = acestep_model.layers[0].input_norm,
    };
    const wrapper_buffers_rms: zml.Bufferized(RmsWrapper) = .{
        .norm = acestep_buffers.layers[0].input_norm,
    };
    const layer_rms = "model.model.layers.0.input_layernorm";
    try zml.testing.testLayer(allocator, io, platform, wrapper_rms, .forward, activations_store.view(), layer_rms, wrapper_buffers_rms, &parameters.shardings.all(), .{});

    std.log.info("Test activations : q_proj", .{});
    const wrapper_q_proj: ProjWrapper = .{
        .proj = acestep_model.layers[0].att_layer.q_proj,
    };
    const wrapper_buffers_q_proj: zml.Bufferized(ProjWrapper) = .{
        .proj = acestep_buffers.layers[0].att_layer.q_proj,
    };
    const layer_q_proj = "model.model.layers.0.self_attn.q_proj";
    try zml.testing.testLayer(allocator, io, platform, wrapper_q_proj, .forward, activations_store.view(), layer_q_proj, wrapper_buffers_q_proj, &parameters.shardings.all(), .{});

    std.log.info("Test activations : k_proj", .{});
    const wrapper_k_proj: ProjWrapper = .{
        .proj = acestep_model.layers[0].att_layer.k_proj,
    };
    const wrapper_buffers_k_proj: zml.Bufferized(ProjWrapper) = .{
        .proj = acestep_buffers.layers[0].att_layer.k_proj,
    };
    const layer_k_proj = "model.model.layers.0.self_attn.k_proj";
    try zml.testing.testLayer(allocator, io, platform, wrapper_k_proj, .forward, activations_store.view(), layer_k_proj, wrapper_buffers_k_proj, &parameters.shardings.all(), .{});

    std.log.info("Test activations : v_proj", .{});
    const wrapper_v_proj: ProjWrapper = .{
        .proj = acestep_model.layers[0].att_layer.v_proj,
    };
    const wrapper_buffers_v_proj: zml.Bufferized(ProjWrapper) = .{
        .proj = acestep_buffers.layers[0].att_layer.v_proj,
    };
    const layer_v_proj = "model.model.layers.0.self_attn.v_proj";
    try zml.testing.testLayer(allocator, io, platform, wrapper_v_proj, .forward, activations_store.view(), layer_v_proj, wrapper_buffers_v_proj, &parameters.shardings.all(), .{});

    std.log.info("Test activations : o_proj", .{});
    const wrapper_o_proj: ProjWrapper = .{
        .proj = acestep_model.layers[0].att_layer.o_proj,
    };
    const wrapper_buffers_o_proj: zml.Bufferized(ProjWrapper) = .{
        .proj = acestep_buffers.layers[0].att_layer.o_proj,
    };
    const layer_o_proj = "model.model.layers.0.self_attn.o_proj";
    try zml.testing.testLayer(allocator, io, platform, wrapper_o_proj, .forward, activations_store.view(), layer_o_proj, wrapper_buffers_o_proj, &parameters.shardings.all(), .{});

    std.log.info("Test activations : k_norm", .{});
    const wrapper_k_norm: RmshWrapper = .{
        .norm = acestep_model.layers[0].att_layer.k_norm,
    };
    const wrapper_buffers_k_norm: zml.Bufferized(RmshWrapper) = .{
        .norm = acestep_buffers.layers[0].att_layer.k_norm,
    };
    const layer_k_norm = "model.model.layers.0.self_attn.k_norm";
    try zml.testing.testLayer(allocator, io, platform, wrapper_k_norm, .forward, activations_store.view(), layer_k_norm, wrapper_buffers_k_norm, &parameters.shardings.all(), .{});

    std.log.info("Test activations : q_norm", .{});
    const wrapper_q_norm: RmshWrapper = .{
        .norm = acestep_model.layers[0].att_layer.q_norm,
    };
    const wrapper_buffers_q_norm: zml.Bufferized(RmshWrapper) = .{
        .norm = acestep_buffers.layers[0].att_layer.q_norm,
    };
    const layer_q_norm = "model.model.layers.0.self_attn.q_norm";
    try zml.testing.testLayer(allocator, io, platform, wrapper_q_norm, .forward, activations_store.view(), layer_q_norm, wrapper_buffers_q_norm, &parameters.shardings.all(), .{});

    std.log.info("Test activations : self attention", .{});
    const wrapper_att: AttWrapper = .{
        .att = acestep_model.layers[0].att_layer,
        .tok_id = tok_id,
        .kv_cache = parameters.kv_cache,
    };
    const wrapper_buffers_att: zml.Bufferized(AttWrapper) = .{
        .att = acestep_buffers.layers[0].att_layer,
        .tok_id = prefill_token_pos_buffer,
        .kv_cache = kv_cache_buffers,
    };
    const layer_att = "model.model.layers.0.self_attn";
    try zml.testing.testLayer(allocator, io, platform, wrapper_att, .forward, activations_store.view(), layer_att, wrapper_buffers_att, &parameters.shardings.all(), .{});

    std.log.info("Test activations : up_proj", .{});
    const wrapper_up_proj: TensorWrapper = .{
        .tensor = acestep_model.layers[0].mlp_layer.up_proj,
    };
    const wrapper_buffers_up_proj: zml.Bufferized(TensorWrapper) = .{
        .tensor = acestep_buffers.layers[0].mlp_layer.up_proj,
    };
    const layer_up_proj = "model.model.layers.0.mlp.up_proj";
    try zml.testing.testLayer(allocator, io, platform, wrapper_up_proj, .forward, activations_store.view(), layer_up_proj, wrapper_buffers_up_proj, &parameters.shardings.all(), .{});

    std.log.info("Test activations : gate_proj", .{});
    const wrapper_gate_proj: TensorWrapper = .{
        .tensor = acestep_model.layers[0].mlp_layer.gate_proj,
    };
    const wrapper_buffers_gate_proj: zml.Bufferized(TensorWrapper) = .{
        .tensor = acestep_buffers.layers[0].mlp_layer.gate_proj,
    };
    const layer_gate_proj = "model.model.layers.0.mlp.gate_proj";
    try zml.testing.testLayer(allocator, io, platform, wrapper_gate_proj, .forward, activations_store.view(), layer_gate_proj, wrapper_buffers_gate_proj, &parameters.shardings.all(), .{});

    std.log.info("Test activations : down_proj", .{});
    const wrapper_down_proj: TensorWrapperI = .{
        .tensor = acestep_model.layers[0].mlp_layer.down_proj,
    };
    const wrapper_buffers_down_proj: zml.Bufferized(TensorWrapperI) = .{
        .tensor = acestep_buffers.layers[0].mlp_layer.down_proj,
    };
    const layer_down_proj = "model.model.layers.0.mlp.down_proj";
    try zml.testing.testLayer(allocator, io, platform, wrapper_down_proj, .forward, activations_store.view(), layer_down_proj, wrapper_buffers_down_proj, &parameters.shardings.all(), .{});

    std.log.info("Test activations : mlp", .{});
    const wrapper_mlp: MlpWrapper = .{
        .mlp = acestep_model.layers[0].mlp_layer,
    };
    const wrapper_buffers_mlp: zml.Bufferized(MlpWrapper) = .{
        .mlp = acestep_buffers.layers[0].mlp_layer,
    };
    const layer_mlp = "model.model.layers.0.mlp";
    try zml.testing.testLayer(allocator, io, platform, wrapper_mlp, .forward, activations_store.view(), layer_mlp, wrapper_buffers_mlp, &parameters.shardings.all(), .{});

    std.log.info("Test activations : transformer layer", .{});
    const wrapper: TransWrapper = .{
        .trans = acestep_model.layers[0],
        .tok_id = tok_id,
        .kv_cache = parameters.kv_cache,
    };
    const wrapper_buffers: zml.Bufferized(TransWrapper) = .{
        .trans = acestep_buffers.layers[0],
        .tok_id = prefill_token_pos_buffer,
        .kv_cache = kv_cache_buffers,
    };
    const layer = "model.model.layers.0";
    try zml.testing.testLayer(allocator, io, platform, wrapper, .forward, activations_store.view(), layer, wrapper_buffers, &parameters.shardings.all(), .{});

}

fn parseConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(Acestep.Config) {
    log.info("Loading model config", .{});
    const parsed_config = blk: {
        const config_json_file = try dir.openFile(io, "config.json", .{});
        defer config_json_file.close(io);
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(io, &config_json_buffer);
        var reader = std.json.Reader.init(allocator, &config_reader.interface);
        defer reader.deinit();
        break :blk try std.json.parseFromTokenSource(Acestep.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
    };
    errdefer parsed_config.deinit();
    return parsed_config;
}

fn compileModel(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, model: Acestep, parameters: AcestepParams) !CompileModelResult {
    log.info("Compiling model", .{});
    const opts: zml.module.CompilationOptions = .{
        .shardings = &parameters.shardings.all(),
    };
    // Compile the model twice, one for prefill, one for generation.
    std.log.info("Compile prefill", .{});
    const prefill_exe = try platform.compile(allocator, io, model, .forward, .{ parameters.prefill_tokens, parameters.token_index, parameters.kv_cache, parameters.rng, parameters.phase_mask, parameters.pos_penalty, parameters.neg_penalty, parameters.decode_phase }, opts);
    std.log.info("Compile decode", .{});
    const decode_exe = try platform.compile(allocator, io, model, .forward, .{ parameters.decode_tokens, parameters.token_index, parameters.kv_cache, parameters.rng, parameters.phase_mask, parameters.pos_penalty, parameters.neg_penalty, parameters.decode_phase }, opts);
    return .{ .prefill_exe = prefill_exe, .decode_exe = decode_exe };
}

fn buildTokenMasks(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, voc_size: usize) !AcestepPhase {
    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();
    var tokens_audio: std.ArrayList(f32) = try .initCapacity(allocator, voc_size);
    var tokens_text: std.ArrayList(f32) = try .initCapacity(allocator, voc_size);
    const token_slice = try allocator.alloc(u32, 1);
    defer allocator.free(token_slice);
    for (0..voc_size) |i| {
        token_slice[0] = @intCast(i);
        const chunk = try tokenizer_decoder.decode(token_slice);
        const l = chunk.len;
        if (l < 16) {
            try tokens_audio.append(allocator, -1e10);
            try tokens_text.append(allocator, 0.0);
        } else {
            if (std.mem.eql(u8, chunk[0..12], "<|audio_code")) {
                try tokens_audio.append(allocator, 0.0);
                try tokens_text.append(allocator, -1e10);
            } else {
                try tokens_audio.append(allocator, -1e10);
                try tokens_text.append(allocator, 0.0);
            }
        }
    }
    const eos_id = tokenizer.tokenToId("<|im_end|>") orelse return error.NoSuchToken;
    // in audio tokens mode, the eos is constrained at pos 5 * duration
    tokens_audio.items[eos_id] = 1e-10;
    return .{
        .phase1 = true,
        .phase1_mask = try tokens_text.toOwnedSlice(allocator),
        .phase2_mask = try tokens_audio.toOwnedSlice(allocator),
        .eos_id = eos_id,
    };
}

pub fn tokenizeInspirationPrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const im_start = tokenizer.tokenToId("<|im_start|>") orelse return error.NoSuchToken;
    const im_end = tokenizer.tokenToId("<|im_end|>") orelse return error.NoSuchToken;
    const newline = try encodeSingleToken(&encoder, "\n");
    const system_token = try encodeSingleToken(&encoder, "system");
    const user_token = try encodeSingleToken(&encoder, "user");
    const assistant_token = try encodeSingleToken(&encoder, "assistant");

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, 10);

    try tokens.append(allocator, im_start);
    try tokens.append(allocator, system_token);
    try tokens.append(allocator, newline);
    try tokens.appendSlice(allocator, try encoder.encode("# Instruction\nExpand the user's input into a more detailed and specific musical description:\n\n"));
    try tokens.append(allocator, im_end);

    try tokens.append(allocator, newline);
    try tokens.append(allocator, im_start);
    try tokens.append(allocator, user_token);
    try tokens.append(allocator, newline);
    try tokens.appendSlice(allocator, try encoder.encode(prompt));
    try tokens.append(allocator, im_end);
    try tokens.append(allocator, newline);

    try tokens.append(allocator, im_start);
    try tokens.append(allocator, assistant_token);
    try tokens.append(allocator, newline);

    return tokens.toOwnedSlice(allocator);
}

pub fn tokenizeGenerationPrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, metadata: AudioMetadata) !struct { []u32, []u32 } {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    var prompt_before_cot: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer prompt_before_cot.deinit(allocator);
    try prompt_before_cot.appendSlice(allocator, "<|im_start|>system\n");
    try prompt_before_cot.appendSlice(allocator, "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
    try prompt_before_cot.appendSlice(allocator, "<|im_end|>\n");
    try prompt_before_cot.appendSlice(allocator, "<|im_start|>user\n");
    try prompt_before_cot.appendSlice(allocator, "# Caption\n");
    try prompt_before_cot.appendSlice(allocator, metadata.caption);
    try prompt_before_cot.appendSlice(allocator, "\n\n");
    try prompt_before_cot.appendSlice(allocator, "# Lyric\n");
    try prompt_before_cot.appendSlice(allocator, "[Instrumental]\n");
    try prompt_before_cot.appendSlice(allocator, "<|im_end|>\n");
    try prompt_before_cot.appendSlice(allocator, "<|im_start|>assistant\n");
    try prompt_before_cot.appendSlice(allocator, "<think>");
    
    var prompt_cot: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer prompt_cot.deinit(allocator);
    try prompt_cot.appendSlice(allocator, "\nbpm: ");
    try prompt_cot.appendSlice(allocator, metadata.bpm);
    try prompt_cot.appendSlice(allocator, "\nduration: ");
    try prompt_cot.appendSlice(allocator, metadata.duration);
    try prompt_cot.appendSlice(allocator, "\nkeyscale: ");
    try prompt_cot.appendSlice(allocator, metadata.keyscale);
    try prompt_cot.appendSlice(allocator, "\nlanguage: ");
    try prompt_cot.appendSlice(allocator, metadata.language);
    try prompt_cot.appendSlice(allocator, "\ntimesignature: ");
    try prompt_cot.appendSlice(allocator, metadata.timesignature);
    
    var prompt_empty_cot: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer prompt_empty_cot.deinit(allocator);
    try prompt_empty_cot.appendSlice(allocator, "\n");
    
    var prompt_after_cot: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    defer prompt_after_cot.deinit(allocator);
    try prompt_after_cot.appendSlice(allocator, "\n</think>\n\n");
    try prompt_after_cot.appendSlice(allocator, "<|im_end|>\n");

    var cond_tokens: std.ArrayList(u32) = try .initCapacity(allocator, 0);
    var uncond_tokens: std.ArrayList(u32) = try .initCapacity(allocator, 0);
    
    // same before
    try cond_tokens.appendSlice(allocator, try encoder.encode(prompt_before_cot.items));
    try uncond_tokens.appendSlice(allocator, try encoder.encode(prompt_before_cot.items));
    // cond has CoT content, uncond has empty think block
    try cond_tokens.appendSlice(allocator, try encoder.encode(prompt_cot.items));
    try uncond_tokens.appendSlice(allocator, try encoder.encode(prompt_empty_cot.items));
    // same after
    try cond_tokens.appendSlice(allocator, try encoder.encode(prompt_after_cot.items));
    try uncond_tokens.appendSlice(allocator, try encoder.encode(prompt_after_cot.items));
    
    return .{ try cond_tokens.toOwnedSlice(allocator), try uncond_tokens.toOwnedSlice(allocator) };
}

fn encodeSingleToken(encoder: *zml.tokenizer.Tokenizer.Encoder, text: []const u8) !u32 {
    const encoded = try encoder.encode(text);
    if (encoded.len != 1) return error.InvalidTokenizerEncoding;
    return encoded[0];
}

pub fn generateInspirationText(
    allocator: std.mem.Allocator,
    io: std.Io,
    acestep_buffers: zml.Bufferized(Acestep),
    prefill_exe: zml.exe.Exe,
    decode_exe: zml.exe.Exe,
    kv_cache_buffers: *zml.Bufferized(KvCache),
    tokenizer: zml.tokenizer.Tokenizer,
    prompt_tok: []const u32,
    config: Acestep.Config,
    options: Acestep.Options,
    phase: AcestepPhase,
    seed: u128,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
) ![]u8 {
    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    std.log.info("Call 5Hz model with formatted prompt", .{});
    std.log.info("########### prompt start ###########\n{s}", .{try tokenizer_decoder.decode(prompt_tok)});
    std.log.info("###########  prompt end  ###########", .{});

    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, seed, io, sharding);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    var generated_token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32));
    defer generated_token_slice.free(allocator);

    var prefill_args = try prefill_exe.args(allocator);
    defer prefill_args.deinit(allocator);
    var prefill_results = try prefill_exe.results(allocator);
    defer prefill_results.deinit(allocator);

    // prepare device buffers for the prefill tokens and their positions
    const prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{options.seq_len}, .u32));
    defer prefill_tokens_slice.free(allocator);
    @memcpy(prefill_tokens_slice.items(u32)[0..prompt_tok.len], prompt_tok);

    var prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice, sharding);
    defer prefill_tokens_buffer.deinit();
    var prefill_token_pos_buffer = try zml.Buffer.scalar(io, platform, 0, .u32, sharding);
    defer prefill_token_pos_buffer.deinit();

    const voc_len = phase.phase1_mask.len;

    const mask_slice: zml.Slice = try .alloc(allocator, .init(.{ voc_len }, .f32));
    defer mask_slice.free(allocator);
    @memcpy(mask_slice.items(f32)[0..phase.phase1_mask.len], if (phase.phase1) phase.phase1_mask else phase.phase2_mask);
    var phase_mask_buffer: zml.Buffer = try .fromSlice(io, platform, mask_slice, sharding);
    defer phase_mask_buffer.deinit();

    const pen_value: f32 = 2.0;
    const pos_pen_values: []f32 = try allocator.alloc(f32, voc_len);
    const neg_pen_values: []f32 = try allocator.alloc(f32, voc_len);
    defer allocator.free(pos_pen_values);
    defer allocator.free(neg_pen_values);
    for (0..voc_len) |i| {
        pos_pen_values[i] = 1.0;
        neg_pen_values[i] = 1.0;
    }
    for (0..prompt_tok.len) |i| {
        const token_id = prompt_tok[i];
        pos_pen_values[token_id] /= pen_value;
        neg_pen_values[token_id] *= pen_value;
    }
    const pos_pen_slice: zml.Slice = try .alloc(allocator, .init(.{ voc_len }, .f32));
    const neg_pen_slice: zml.Slice = try .alloc(allocator, .init(.{ voc_len }, .f32));
    defer pos_pen_slice.free(allocator);
    defer neg_pen_slice.free(allocator);
    @memcpy(pos_pen_slice.items(f32)[0..voc_len], pos_pen_values);
    @memcpy(neg_pen_slice.items(f32)[0..voc_len], neg_pen_values);
    var pos_pen_buffer: zml.Buffer = try .fromSlice(io, platform, pos_pen_slice, sharding);
    var neg_pen_buffer: zml.Buffer = try .fromSlice(io, platform, neg_pen_slice, sharding);
    defer pos_pen_buffer.deinit();
    defer neg_pen_buffer.deinit();

    const logits_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .voc = voc_len }, .f32));
    defer logits_slice.free(allocator);
    var logits_buffer: zml.Buffer = try .fromSlice(io, platform, logits_slice, sharding);
    defer logits_buffer.deinit();

    std.log.info("Run prefill with sequence of {d} tokens", .{prompt_tok.len});
    prefill_args.set(.{ acestep_buffers, prefill_tokens_buffer, prefill_token_pos_buffer, kv_cache_buffers, rng_buffers, phase_mask_buffer, pos_pen_buffer, neg_pen_buffer, true });
    prefill_exe.call(prefill_args, &prefill_results);
    prefill_results.fill(.{ &prefill_tokens_buffer, kv_cache_buffers, &rng_buffers, &logits_buffer });

    try logits_buffer.toSlice(io, logits_slice);
    try prefill_tokens_buffer.toSlice(io, prefill_tokens_slice);
    generated_token_slice.items(u32)[0] = prefill_tokens_slice.items(u32)[prompt_tok.len - 1];

    // Prepare for token-by-token generation,
    std.log.info("Prepare decode", .{});
    var decode_args = try decode_exe.args(allocator);
    defer decode_args.deinit(allocator);
    var decode_results = try decode_exe.results(allocator);
    defer decode_results.deinit(allocator);

    // start with the token generated based on the full prompt.
    var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice, sharding);
    defer current_token_buffer.deinit();
    const output_tokens_len = options.seq_len - prompt_tok.len - 1;
    var num_tokens_generated: usize = 0;
    var result: std.ArrayList(u8) = try .initCapacity(allocator, 0);

    std.log.info("Run decode", .{});
    var stdout = std.Io.File.stdout().writer(io, &.{});
    var writer: *std.Io.Writer = &stdout.interface;
    generation: for (0..output_tokens_len + 1) |i| {
        // collect and print generated sequence
        num_tokens_generated += 1;
        const generated_token = generated_token_slice.items(u32)[0];
        if (try tokenizer_decoder.next(generated_token)) |chunk| {
            try result.appendSlice(allocator, chunk);
            try writer.writeAll(chunk);
            try writer.flush();
        } else {
            std.log.info("ERROR could not decode token: {d}", .{generated_token});
        }
        pos_pen_values[generated_token] /= pen_value;
        neg_pen_values[generated_token] *= pen_value;
        @memcpy(pos_pen_slice.items(f32)[0..voc_len], pos_pen_values);
        @memcpy(neg_pen_slice.items(f32)[0..voc_len], neg_pen_values);

        // check for eos
        if (i == output_tokens_len) break :generation;
        switch (config.eos_token_id.value) {
            .int => |eos| if (generated_token == @as(u32, @intCast(eos))) break :generation,
            .ints => |eos_list| {
                for (eos_list) |eos| {
                    if (generated_token == @as(u32, @intCast(eos))) break :generation;
                }
            },
        }
        // current token pos needs to go into a zml.Buffer
        const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(prompt_tok.len + i)}));
        var token_pos_buffer: zml.Buffer = try .fromSlice(io, platform, token_pos_slice, sharding);
        defer token_pos_buffer.deinit();

        var pos_pen_buff: zml.Buffer = try .fromSlice(io, platform, pos_pen_slice, sharding);
        var neg_pen_buff: zml.Buffer = try .fromSlice(io, platform, neg_pen_slice, sharding);
        defer pos_pen_buff.deinit();
        defer neg_pen_buff.deinit();
        // call to generate the next token
        decode_args.set(.{ acestep_buffers, current_token_buffer, token_pos_buffer, kv_cache_buffers, rng_buffers, phase_mask_buffer, pos_pen_buff, neg_pen_buff, false });
        decode_exe.call(decode_args, &decode_results);
        decode_results.fill(.{ &current_token_buffer, kv_cache_buffers, &rng_buffers, &logits_buffer });
        // extract the generated token from the buffer
        try logits_buffer.toSlice(io, logits_slice);
        try current_token_buffer.toSlice(io, generated_token_slice);
    }
    std.log.info("\nDone, generated {d} tokens", .{num_tokens_generated});
    return result.toOwnedSlice(allocator);
}

pub fn generateAudioCodes(
    allocator: std.mem.Allocator,
    io: std.Io,
    acestep_buffers: zml.Bufferized(Acestep),
    prefill_exe: zml.exe.Exe,
    decode_exe: zml.exe.Exe,
    kv_cache_buffers_cond: *zml.Bufferized(KvCache),
    kv_cache_buffers_uncond: *zml.Bufferized(KvCache),
    tokenizer: zml.tokenizer.Tokenizer,
    cond_prompt_tok: []const u32,
    uncond_prompt_tok: []const u32,
    options: Acestep.Options,
    phase: AcestepPhase,
    seed: u128,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    metadata: AudioMetadata,
) ![]u8 {
    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    std.log.info("Call 5Hz model with formatted prompt : conditional", .{});
    std.log.info("########### prompt start ###########\n{s}", .{try tokenizer_decoder.decode(cond_prompt_tok)});
    std.log.info("###########  prompt end  ###########", .{});
    std.log.info("Call 5Hz model with formatted prompt : unconditional", .{});
    std.log.info("########### prompt start ###########\n{s}", .{try tokenizer_decoder.decode(uncond_prompt_tok)});
    std.log.info("###########  prompt end  ###########", .{});

    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, seed, io, sharding);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    var generated_token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32));
    defer generated_token_slice.free(allocator);

    var prefill_args = try prefill_exe.args(allocator);
    defer prefill_args.deinit(allocator);
    var prefill_results = try prefill_exe.results(allocator);
    defer prefill_results.deinit(allocator);

    // prepare device buffers for the prefill tokens and their positions
    const prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{options.seq_len}, .u32));
    defer prefill_tokens_slice.free(allocator);
    var prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice, sharding);
    defer prefill_tokens_buffer.deinit();
    var prefill_token_pos_buffer = try zml.Buffer.scalar(io, platform, 0, .u32, sharding);
    defer prefill_token_pos_buffer.deinit();

    const voc_len = phase.phase1_mask.len;

    const mask_slice: zml.Slice = try .alloc(allocator, .init(.{ voc_len }, .f32));
    defer mask_slice.free(allocator);
    @memcpy(mask_slice.items(f32)[0..phase.phase1_mask.len], if (phase.phase1) phase.phase1_mask else phase.phase2_mask);
    var phase_mask_buffer: zml.Buffer = try .fromSlice(io, platform, mask_slice, sharding);
    defer phase_mask_buffer.deinit();

    const pen_value: f32 = 1.0;
    const pos_pen_values: []f32 = try allocator.alloc(f32, voc_len);
    const neg_pen_values: []f32 = try allocator.alloc(f32, voc_len);
    defer allocator.free(pos_pen_values);
    defer allocator.free(neg_pen_values);
    for (0..voc_len) |i| {
        pos_pen_values[i] = 1.0;
        neg_pen_values[i] = 1.0;
    }
    for (0..cond_prompt_tok.len) |i| {
        const token_id = cond_prompt_tok[i];
        pos_pen_values[token_id] /= pen_value;
        neg_pen_values[token_id] *= pen_value;
    }
    const pos_pen_slice: zml.Slice = try .alloc(allocator, .init(.{ voc_len }, .f32));
    const neg_pen_slice: zml.Slice = try .alloc(allocator, .init(.{ voc_len }, .f32));
    defer pos_pen_slice.free(allocator);
    defer neg_pen_slice.free(allocator);
    @memcpy(pos_pen_slice.items(f32)[0..voc_len], pos_pen_values);
    @memcpy(neg_pen_slice.items(f32)[0..voc_len], neg_pen_values);
    var pos_pen_buffer: zml.Buffer = try .fromSlice(io, platform, pos_pen_slice, sharding);
    var neg_pen_buffer: zml.Buffer = try .fromSlice(io, platform, neg_pen_slice, sharding);
    defer pos_pen_buffer.deinit();
    defer neg_pen_buffer.deinit();

    const cond_logits_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .voc = voc_len }, .f32));
    const uncond_logits_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .voc = voc_len }, .f32));
    defer cond_logits_slice.free(allocator);
    defer uncond_logits_slice.free(allocator);
    var cond_logits_buffer: zml.Buffer = try .fromSlice(io, platform, cond_logits_slice, sharding);
    var uncond_logits_buffer: zml.Buffer = try .fromSlice(io, platform, uncond_logits_slice, sharding);
    defer cond_logits_buffer.deinit();
    defer uncond_logits_buffer.deinit();

    std.log.info("Run cond prefill with sequence of {d} tokens", .{cond_prompt_tok.len});
    @memcpy(prefill_tokens_slice.items(u32)[0..cond_prompt_tok.len], cond_prompt_tok);
    prefill_args.set(.{ acestep_buffers, prefill_tokens_buffer, prefill_token_pos_buffer, kv_cache_buffers_cond, rng_buffers, phase_mask_buffer, pos_pen_buffer, neg_pen_buffer, true });
    prefill_exe.call(prefill_args, &prefill_results);
    prefill_results.fill(.{ &prefill_tokens_buffer, kv_cache_buffers_cond, &rng_buffers, &cond_logits_buffer });
    
    std.log.info("Run uncond prefill with sequence of {d} tokens", .{uncond_prompt_tok.len});
    @memcpy(prefill_tokens_slice.items(u32)[0..uncond_prompt_tok.len], uncond_prompt_tok);
    prefill_args.set(.{ acestep_buffers, prefill_tokens_buffer, prefill_token_pos_buffer, kv_cache_buffers_uncond, rng_buffers, phase_mask_buffer, pos_pen_buffer, neg_pen_buffer, true });
    prefill_exe.call(prefill_args, &prefill_results);
    prefill_results.fill(.{ &prefill_tokens_buffer, kv_cache_buffers_uncond, &rng_buffers, &uncond_logits_buffer });

    // combine logits
    try cond_logits_buffer.toSlice(io, cond_logits_slice);
    try uncond_logits_buffer.toSlice(io, uncond_logits_slice);
    var clo: []f32 = cond_logits_slice.items(f32);
    var ulo: []f32 = uncond_logits_slice.items(f32);
    var max_logit: f32 = -1e20;
    var argmax_logit: u32 = 0;
    for (0..clo.len) |i| {
        clo[i] = ulo[i] + cfg * (clo[i] - ulo[i]);
        if (clo[i] > max_logit) {
            max_logit = clo[i];
            argmax_logit = @intCast(i);
        }
    }   
    generated_token_slice.items(u32)[0] = argmax_logit;

    // Prepare for token-by-token generation,
    std.log.info("Prepare decode", .{});
    var decode_args = try decode_exe.args(allocator);
    defer decode_args.deinit(allocator);
    var decode_results = try decode_exe.results(allocator);
    defer decode_results.deinit(allocator);
    var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice, sharding);
    defer current_token_buffer.deinit();
    const nb_audio_codes = 5 * try std.fmt.parseInt(u32, metadata.duration, 10);
    const max_output_tokens = options.seq_len - cond_prompt_tok.len;
    var num_tokens_generated: usize = 1;
    var result: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    std.log.info("Run decode", .{});
    var stdout = std.Io.File.stdout().writer(io, &.{});
    var writer: *std.Io.Writer = &stdout.interface;
    for (0..max_output_tokens) |i| {
        // collect and print generated sequence
        const generated_token = generated_token_slice.items(u32)[0];
        if (try tokenizer_decoder.next(generated_token)) |chunk| {
            num_tokens_generated += 1;
            try result.appendSlice(allocator, chunk);
            try writer.writeAll(chunk);
            try writer.flush();
        } else {
            std.log.info("ERROR could not decode token: {d}", .{generated_token});
        }
        if (num_tokens_generated == nb_audio_codes) break;
        
        // update penalty values
        pos_pen_values[generated_token] /= pen_value;
        neg_pen_values[generated_token] *= pen_value;
        @memcpy(pos_pen_slice.items(f32)[0..voc_len], pos_pen_values);
        @memcpy(neg_pen_slice.items(f32)[0..voc_len], neg_pen_values);

        // current token pos needs to go into a zml.Buffer
        const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(cond_prompt_tok.len + i)}));
        var token_pos_buffer: zml.Buffer = try .fromSlice(io, platform, token_pos_slice, sharding);
        defer token_pos_buffer.deinit();

        var pos_pen_buff: zml.Buffer = try .fromSlice(io, platform, pos_pen_slice, sharding);
        var neg_pen_buff: zml.Buffer = try .fromSlice(io, platform, neg_pen_slice, sharding);
        defer pos_pen_buff.deinit();
        defer neg_pen_buff.deinit();
        
        // call to generate the next cond logits
        decode_args.set(.{ acestep_buffers, current_token_buffer, token_pos_buffer, kv_cache_buffers_cond, rng_buffers, phase_mask_buffer, pos_pen_buff, neg_pen_buff, false });
        decode_exe.call(decode_args, &decode_results);
        decode_results.fill(.{ &current_token_buffer, kv_cache_buffers_cond, &rng_buffers, &cond_logits_buffer });
        // call to generate the next uncond logits
        decode_args.set(.{ acestep_buffers, current_token_buffer, token_pos_buffer, kv_cache_buffers_uncond, rng_buffers, phase_mask_buffer, pos_pen_buff, neg_pen_buff, true });
        decode_exe.call(decode_args, &decode_results);
        decode_results.fill(.{ &current_token_buffer, kv_cache_buffers_uncond, &rng_buffers, &uncond_logits_buffer });
        
        // combine logits
        try cond_logits_buffer.toSlice(io, cond_logits_slice);
        try uncond_logits_buffer.toSlice(io, uncond_logits_slice);
        clo = cond_logits_slice.items(f32);
        ulo = uncond_logits_slice.items(f32);
        max_logit = -1e20;
        argmax_logit = 0;
        for (0..clo.len) |ii| {
            clo[ii] = ulo[ii] + cfg * (clo[ii] - ulo[ii]);
            if (clo[ii] > max_logit) {
                max_logit = clo[ii];
                argmax_logit = @intCast(ii);
            }
        }
        generated_token_slice.items(u32)[0] = argmax_logit;
    }
    std.log.info("\nDone, generated {d} tokens", .{num_tokens_generated});
    return result.toOwnedSlice(allocator);
}
