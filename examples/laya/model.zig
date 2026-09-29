//! Laya decision model: a ModernBERT encoder followed by a small transformer
//! decision head that scores option markers in a single forward pass.
//!
//! Reference: https://huggingface.co/convaiinnovations/laya
//! Activations are tagged {.s, .d}; attention uses {.q|.k, .h, .hd}.

const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;
const Tensor = zml.Tensor;

/// `encoder/config.json` (Hugging Face ModernBERT config).
pub const EncoderConfig = struct {
    model_type: []const u8 = "modernbert",
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    hidden_activation: []const u8 = "gelu",
    norm_eps: f32 = 1e-5,
    local_attention: u32 = 128,
    global_attn_every_n_layers: u32 = 3,
    global_rope_theta: f32 = 160000,
    local_rope_theta: f32 = 10000,
    layer_types: ?[]const LayerType = null,
    rope_parameters: ?struct {
        full_attention: ?RopeParameters = null,
        sliding_attention: ?RopeParameters = null,
    } = null,

    pub const LayerType = enum { full_attention, sliding_attention };
    pub const RopeParameters = struct { rope_theta: f32, rope_type: []const u8 = "default" };

    pub fn layerType(self: EncoderConfig, index: usize) LayerType {
        if (self.layer_types) |types| return types[index];
        return if (index % self.global_attn_every_n_layers == 0) .full_attention else .sliding_attention;
    }

    pub fn ropeTheta(self: EncoderConfig, kind: LayerType) f32 {
        const params = self.rope_parameters orelse return switch (kind) {
            .full_attention => self.global_rope_theta,
            .sliding_attention => self.local_rope_theta,
        };
        return switch (kind) {
            .full_attention => if (params.full_attention) |p| p.rope_theta else self.global_rope_theta,
            .sliding_attention => if (params.sliding_attention) |p| p.rope_theta else self.local_rope_theta,
        };
    }

    pub fn validate(self: EncoderConfig) !void {
        if (!std.mem.eql(u8, self.model_type, "modernbert")) return error.UnsupportedEncoder;
        if (!std.mem.eql(u8, self.hidden_activation, "gelu")) return error.UnsupportedActivation;
        if (self.hidden_size % self.num_attention_heads != 0) return error.InvalidHeadDim;
        if (self.layer_types) |types| if (types.len != self.num_hidden_layers) return error.InvalidLayerTypes;
    }
};

/// `rl_agent_config.json`: decision head and calibration settings.
pub const AgentConfig = struct {
    head_layers: u32 = 2,
    max_len: u32 = 512,
    head_max_len: u32 = 192,
    act_costs: std.json.ArrayHashMap(f32) = .{},
    temperature: [3]f32 = .{ 1, 1, 1 },
    temperature_by_options: std.json.ArrayHashMap(f32) = .{},
};

pub const Options = struct {
    /// Activation dtype. Weights are converted on the fly.
    dtype: zml.DataType = .f32,
};

pub const Laya = struct {
    encoder: ModernBert,
    head: []HeadLayer,
    type_emb: Tensor,
    scorer: Scorer,
    act_head: ActHead,
    dtype: zml.DataType,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, encoder_config: EncoderConfig, agent_config: AgentConfig, options: Options) !Laya {
        try encoder_config.validate();

        const head = try allocator.alloc(HeadLayer, agent_config.head_layers);
        errdefer allocator.free(head);
        for (head, 0..) |*layer, i| {
            layer.* = .init(store.withPrefix("head").withPrefix("layers").withLayer(i), encoder_config.hidden_size);
        }

        return .{
            .encoder = try .init(allocator, store.withPrefix("encoder"), encoder_config),
            .head = head,
            .type_emb = store.createTensor("type_emb.weight", .{ .qtype, .d }, .replicated),
            .scorer = .init(store.withPrefix("scorer")),
            .act_head = .init(store.withPrefix("act_head")),
            .dtype = options.dtype,
        };
    }

    pub fn deinit(self: Laya, allocator: std.mem.Allocator) void {
        self.encoder.deinit(allocator);
        allocator.free(self.head);
    }

    pub fn load(
        self: *const Laya,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Laya) {
        var buffers = try zml.mem.bufferize(allocator, Laya, self);
        errdefer unloadBuffers(&buffers, allocator);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Laya, self, &buffers, store, &.{}, .{});
        try loader.await(io);
        return buffers;
    }

    pub fn unloadBuffers(buffers: *zml.Bufferized(Laya), allocator: std.mem.Allocator) void {
        zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Laya, buffers);
        allocator.free(buffers.encoder.layers);
        allocator.free(buffers.head);
    }

    /// Scores every option marker of one prompt.
    ///   - tokens: {.s} u32, padded prompt
    ///   - length: {} u32, number of valid tokens
    ///   - markers: {.m} u32, position of each option's [MASK] marker (padded with 0)
    ///   - n_markers: {} u32, number of valid markers
    ///   - qtype: {} u32, 0=choice, 1=score, 2=noul
    /// Returns raw marker logits {.m} (invalid slots set to -1e4) and act logits {.act}, both f32.
    pub fn forward(self: Laya, tokens: Tensor, length: Tensor, markers: Tensor, n_markers: Tensor, qtype: Tensor) struct { Tensor, Tensor } {
        const seq_len = tokens.dim(.s);
        const masks: AttentionMasks = .init(seq_len, length, self.encoder.local_window, self.dtype);

        var h = self.encoder.forward(tokens, masks, self.dtype);
        const type_bias = self.type_emb.gather(.{ .qtype = qtype }, .{}).convert(self.dtype);
        h = h.add(type_bias.broad(h.shape()));
        for (self.head) |layer| h = layer.forward(h, masks.full);

        // One logit per option, read at the option's [MASK] marker.
        var logits = self.scorer.forward(h.gather(.{ .s = markers }, .{})).convert(.f32);
        const marker_valid = Tensor.arange(.{ .end = markers.dim(.m) }, .u32).withTags(.{.m}).cmp(.LT, n_markers);
        logits = marker_valid.select(logits, Tensor.scalar(-1e4, .f32));

        // Confidence features fed to the act/escalate head, as in the reference runtime.
        const p = logits.softmax(.m);
        const k = n_markers.maximum(Tensor.scalar(2, .u32)).convert(.f32);
        const entropy = p.mul(p.maximum(Tensor.scalar(1e-9, .f32)).log()).sum(.m).squeeze(.m).scale(-1).div(k.log());
        const top = p.sort(.m, .{ .descending = true }).values;
        const top1 = top.slice(.m, .single(0));
        const top2 = top.slice(.m, .single(1));
        const features = Tensor.stack(&.{ top1, top1.sub(top2), entropy, k.scale(1.0 / 255.0) }, 0, .d);

        const cls = h.slice(.s, .single(0)).convert(.f32);
        const pooled = Tensor.concatenate(&.{ cls, features }, .d);
        const act = self.act_head.forward(pooled.convert(self.dtype)).convert(.f32);

        return .{ logits, act };
    }
};

/// Boolean attention masks turned into additive biases {.q, .k}.
/// Sliding layers see keys with |q - k| <= window / 2. Padded queries may see every
/// valid key so that no softmax row is fully masked; their outputs are never read.
const AttentionMasks = struct {
    full: Tensor,
    sliding: Tensor,

    fn init(seq_len: i64, length: Tensor, window: u32, dtype: zml.DataType) AttentionMasks {
        const pos = Tensor.arange(.{ .end = seq_len }, .u32).withTags(.{.s});
        const shape = zml.Shape.init(.{ .q = seq_len, .k = seq_len }, .i32);
        const key_valid = pos.cmp(.LT, length).rename(.{ .s = .k }).broad(shape.withDtype(.bool));
        const query_pad = pos.cmp(.GE, length).rename(.{ .s = .q }).broad(shape.withDtype(.bool));

        const distance = Tensor.iota(shape, .q).sub(Tensor.iota(shape, .k)).abs();
        const near = distance.cmp(.LE, Tensor.scalar(@divFloor(window, 2), .i32));
        const sliding = near.logical(.OR, query_pad).logical(.AND, key_valid);

        return .{ .full = toBias(key_valid, dtype), .sliding = toBias(sliding, dtype) };
    }

    fn toBias(mask: Tensor, dtype: zml.DataType) Tensor {
        return mask.select(Tensor.scalar(0, dtype), Tensor.scalar(-std.math.inf(f32), dtype));
    }

    fn get(self: AttentionMasks, kind: EncoderConfig.LayerType) Tensor {
        return switch (kind) {
            .full_attention => self.full,
            .sliding_attention => self.sliding,
        };
    }
};

const ModernBert = struct {
    tok_embeddings: zml.nn.TokenEmbedding,
    embeddings_norm: LayerNorm,
    layers: []EncoderLayer,
    final_norm: LayerNorm,
    local_window: u32,

    fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: EncoderConfig) !ModernBert {
        const layers = try allocator.alloc(EncoderLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| {
            layer.* = .init(store.withPrefix("layers").withLayer(i), config, i);
        }

        return .{
            .tok_embeddings = .{ .weight = store.createTensor("embeddings.tok_embeddings.weight", .{ .voc, .d }, .replicated) },
            .embeddings_norm = .init(store.withPrefix("embeddings").withPrefix("norm"), config.norm_eps),
            .layers = layers,
            .final_norm = .init(store.withPrefix("final_norm"), config.norm_eps),
            .local_window = config.local_attention,
        };
    }

    fn deinit(self: ModernBert, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    fn forward(self: ModernBert, tokens: Tensor, masks: AttentionMasks, dtype: zml.DataType) Tensor {
        var x = self.tok_embeddings.forward(tokens.withTags(.{.s})).convert(dtype);
        x = self.embeddings_norm.forward(x);
        for (self.layers) |layer| x = layer.forward(x, masks.get(layer.kind));
        return self.final_norm.forward(x);
    }
};

const EncoderLayer = struct {
    /// The first layer has no attention norm: embeddings are already normalized.
    attn_norm: ?LayerNorm,
    attn: EncoderAttention,
    mlp_norm: LayerNorm,
    mlp: EncoderMlp,
    kind: EncoderConfig.LayerType,

    fn init(store: zml.io.TensorStore.View, config: EncoderConfig, index: usize) EncoderLayer {
        const kind = config.layerType(index);
        return .{
            .attn_norm = if (index == 0) null else .init(store.withPrefix("attn_norm"), config.norm_eps),
            .attn = .init(store.withPrefix("attn"), config.num_attention_heads, config.ropeTheta(kind)),
            .mlp_norm = .init(store.withPrefix("mlp_norm"), config.norm_eps),
            .mlp = .init(store.withPrefix("mlp")),
            .kind = kind,
        };
    }

    fn forward(self: EncoderLayer, x: Tensor, mask: Tensor) Tensor {
        const normed = if (self.attn_norm) |norm| norm.forward(x) else x;
        const h = x.add(self.attn.forward(normed, mask));
        return h.add(self.mlp.forward(self.mlp_norm.forward(h)));
    }
};

const EncoderAttention = struct {
    wqkv: zml.nn.Linear,
    wo: zml.nn.Linear,
    num_heads: i64,
    rope_opts: zml.nn.RopeOpts,

    fn init(store: zml.io.TensorStore.View, num_heads: u32, rope_theta: f32) EncoderAttention {
        return .{
            .wqkv = .init(store.createTensor("Wqkv.weight", .{ .dout, .d }, .replicated), null, .d),
            .wo = .init(store.createTensor("Wo.weight", .{ .dout, .d }, .replicated), null, .d),
            .num_heads = num_heads,
            // Hugging Face ModernBERT uses the rotate_half RoPE layout.
            .rope_opts = .{ .layout = .real_im_pass, .scaling = .{ .default = .{ .rope_theta = rope_theta } } },
        };
    }

    fn forward(self: EncoderAttention, x: Tensor, mask: Tensor) Tensor {
        const qkv = self.wqkv.forward(x, x.dtype()).splitAxis(.dout, .{ .qkv = 3, .h = self.num_heads, .hd = .auto });
        const q, const k, const v = qkv.chunkExact(.qkv, 3);
        const q_rot = zml.nn.rope(q.squeeze(.qkv), null, self.rope_opts);
        const k_rot = zml.nn.rope(k.squeeze(.qkv), null, self.rope_opts);
        const out = zml.nn.sdpa(
            q_rot.rename(.{ .s = .q }),
            k_rot.rename(.{ .s = .k }),
            v.squeeze(.qkv).rename(.{ .s = .k }),
            .{ .attn_mask = mask },
        );
        const merged = out.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return self.wo.forward(merged, merged.dtype()).rename(.{ .dout = .d });
    }
};

/// Gated GELU MLP: Wo(gelu(input) * gate) where Wi projects to [input, gate].
const EncoderMlp = struct {
    wi: zml.nn.Linear,
    wo: zml.nn.Linear,

    fn init(store: zml.io.TensorStore.View) EncoderMlp {
        return .{
            .wi = .init(store.createTensor("Wi.weight", .{ .dout, .d }, .replicated), null, .d),
            .wo = .init(store.createTensor("Wo.weight", .{ .dout, .d }, .replicated), null, .d),
        };
    }

    fn forward(self: EncoderMlp, x: Tensor) Tensor {
        const input, const gate = self.wi.forward(x, x.dtype()).chunkExact(.dout, 2);
        const hidden = geluErf(input).mul(gate).rename(.{ .dout = .d });
        return self.wo.forward(hidden, hidden.dtype()).rename(.{ .dout = .d });
    }
};

/// Pre-norm `torch.nn.TransformerEncoderLayer` (ReLU feed-forward, no RoPE).
const HeadLayer = struct {
    norm1: LayerNorm,
    in_proj: zml.nn.Linear,
    out_proj: zml.nn.Linear,
    norm2: LayerNorm,
    linear1: zml.nn.Linear,
    linear2: zml.nn.Linear,
    num_heads: i64,

    fn init(store: zml.io.TensorStore.View, hidden_size: u32) HeadLayer {
        const attn = store.withPrefix("self_attn");
        return .{
            .norm1 = .init(store.withPrefix("norm1"), 1e-5),
            .in_proj = .init(
                attn.createTensor("in_proj_weight", .{ .dout, .d }, .replicated),
                attn.createTensor("in_proj_bias", .{.dout}, .replicated),
                .d,
            ),
            .out_proj = linear(attn.withPrefix("out_proj")),
            .norm2 = .init(store.withPrefix("norm2"), 1e-5),
            .linear1 = linear(store.withPrefix("linear1")),
            .linear2 = linear(store.withPrefix("linear2")),
            .num_heads = @max(1, hidden_size / 64),
        };
    }

    fn forward(self: HeadLayer, x: Tensor, mask: Tensor) Tensor {
        const qkv = self.in_proj.forward(self.norm1.forward(x), x.dtype()).splitAxis(.dout, .{ .qkv = 3, .h = self.num_heads, .hd = .auto });
        const q, const k, const v = qkv.chunkExact(.qkv, 3);
        const attn = zml.nn.sdpa(
            q.squeeze(.qkv).rename(.{ .s = .q }),
            k.squeeze(.qkv).rename(.{ .s = .k }),
            v.squeeze(.qkv).rename(.{ .s = .k }),
            .{ .attn_mask = mask },
        ).merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const h = x.add(self.out_proj.forward(attn, x.dtype()).rename(.{ .dout = .d }));

        const ff = self.linear1.forward(self.norm2.forward(h), x.dtype()).relu().rename(.{ .dout = .d });
        return h.add(self.linear2.forward(ff, x.dtype()).rename(.{ .dout = .d }));
    }
};

/// LayerNorm -> Linear -> GELU -> Linear(1), applied to each marker {.m, .d} -> {.m}.
const Scorer = struct {
    norm: LayerNorm,
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,

    fn init(store: zml.io.TensorStore.View) Scorer {
        return .{
            .norm = .init(store.withLayer(0), 1e-5),
            .fc1 = linear(store.withLayer(1)),
            .fc2 = linear(store.withLayer(3)),
        };
    }

    fn forward(self: Scorer, x: Tensor) Tensor {
        const h = geluErf(self.fc1.forward(self.norm.forward(x), x.dtype()).rename(.{ .dout = .d }));
        return self.fc2.forward(h, h.dtype()).squeeze(.dout);
    }
};

/// Linear -> GELU -> Linear over [CLS state, confidence features]: {.d} -> {.act}.
const ActHead = struct {
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,

    fn init(store: zml.io.TensorStore.View) ActHead {
        return .{ .fc1 = linear(store.withLayer(0)), .fc2 = linear(store.withLayer(2)) };
    }

    fn forward(self: ActHead, x: Tensor) Tensor {
        const h = geluErf(self.fc1.forward(x, x.dtype()).rename(.{ .dout = .d }));
        return self.fc2.forward(h, h.dtype()).rename(.{ .dout = .act });
    }
};

const LayerNorm = struct {
    weight: Tensor,
    bias: ?Tensor,
    eps: f32,

    fn init(store: zml.io.TensorStore.View, eps: f32) LayerNorm {
        return .{
            .weight = store.createTensor("weight", .{.d}, .replicated),
            .bias = store.maybeCreateTensor("bias", .{.d}, .replicated),
            .eps = eps,
        };
    }

    /// Normalizes the last axis, which must be `.d`.
    fn forward(self: LayerNorm, x: Tensor) Tensor {
        stdx.debug.assert(x.axis(.d) == x.rank() - 1, "LayerNorm expects .d to be the last axis, got {f}", .{x});
        const y = zml.nn.normalizeVariance(x, self.eps).mul(self.weight.convert(x.dtype()).broad(x.shape()));
        return if (self.bias) |bias| y.add(bias.convert(x.dtype()).broad(x.shape())) else y;
    }
};

fn linear(store: zml.io.TensorStore.View) zml.nn.Linear {
    return .init(
        store.createTensor("weight", .{ .dout, .d }, .replicated),
        store.maybeCreateTensor("bias", .{.dout}, .replicated),
        .d,
    );
}

/// Exact (erf-based) GELU, as used by PyTorch `nn.GELU()` and Hugging Face "gelu".
/// `Tensor.gelu` is the tanh approximation, which drifts across 28 encoder layers.
/// erf uses Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7), evaluated in f32.
fn geluErf(x_: Tensor) Tensor {
    const x = x_.convert(.f32);
    const z = x.scale(std.math.sqrt1_2);
    const az = z.abs();
    const t = Tensor.scalar(1, .f32).div(az.scale(0.3275911).addConstant(1));
    const coefs = [_]f32{ 1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592 };
    var poly = t.scale(coefs[0]);
    for (coefs[1..]) |c| poly = poly.addConstant(c).mul(t);
    const erf_abs = Tensor.scalar(1, .f32).sub(poly.mul(az.mul(az).scale(-1).exp()));
    const erf = z.cmp(.GE, Tensor.scalar(0, .f32)).select(erf_abs, erf_abs.scale(-1));
    return x.mul(erf.addConstant(1)).scale(0.5).convert(x_.dtype());
}
