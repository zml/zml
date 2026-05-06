// Gemma-3 text encoder for LTX-2.3.
//
// Dense causal self-attention (no KV cache), produces stacked hidden states
// [s, d, layer=49] for the LTX text embedding pipeline.
//
// IREE tokenizer does not apply HuggingFace TemplateProcessing, so BOS
// (id=2) is prepended explicitly. Left-padded to MAX_SEQ_LEN with pad=0.

const std = @import("std");
const zml = @import("zml");

const log = std.log.scoped(.@"ltx/gemma3");

pub const MAX_SEQ_LEN: usize = 1024;
pub const PAD_TOKEN_ID: u32 = 0;
pub const BOS_TOKEN_ID: u32 = 2;
pub const NUM_HIDDEN_STATES: usize = 49; // 1 embedding + 47 raw layers + 1 normed

pub const LayerType = enum { full_attention, sliding_attention };

pub const Config = struct {
    head_dim: u32 = 256,
    hidden_size: u32 = 3840,
    num_hidden_layers: u32 = 48,
    num_attention_heads: u32 = 16,
    num_key_value_heads: u32 = 8,
    intermediate_size: u32 = 15360,
    rms_norm_eps: f32 = 1e-6,
    sliding_window: u32 = 1024,
    sliding_window_pattern: u32 = 6,
    query_pre_attn_scalar: u32 = 256,
    vocab_size: u32 = 262208,
    rope_theta: f32 = 1_000_000,
    rope_local_base_freq: f32 = 10_000,
    // Linear RoPE scaling factor=8.0 for full attention layers.
    rope_scaling: zml.nn.RopeOpts.Scaling = .{ .linear = .{ .factor = 8.0, .rope_theta = 1_000_000 } },

    pub fn layerType(self: Config, i: usize) LayerType {
        return if ((i + 1) % self.sliding_window_pattern != 0) .sliding_attention else .full_attention;
    }

    pub fn ropeOpts(self: Config, layer_type: LayerType) zml.nn.RopeOpts {
        return .{
            .layout = .sequential,
            .scaling = switch (layer_type) {
                .full_attention => self.rope_scaling,
                .sliding_attention => .{ .default = .{ .rope_theta = self.rope_local_base_freq } },
            },
        };
    }
};

// -- Layers --

pub const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{ .weight = store.createTensor("weight", .{.d}, .{}), .eps = eps };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    /// Gemma-3 variant: normalize(x) * (weight + 1), computed in f32.
    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const x = input.convert(.f32);
        const norm = x.mul(zml.Tensor.rsqrt(x.powByConst(2).mean(.d).addConstant(self.eps)));
        return norm.mul(self.weight.convert(.f32).addConstant(1.0).broad(norm.shape())).convert(input.dtype());
    }
};

pub const Mlp = struct {
    gate_proj: zml.nn.Linear,
    up_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .gate_proj = .init(store.withPrefix("gate_proj").createTensor("weight", .{ .dout, .d }, .{}), null, .d),
            .up_proj = .init(store.withPrefix("up_proj").createTensor("weight", .{ .dout, .d }, .{}), null, .d),
            .down_proj = .init(store.withPrefix("down_proj").createTensor("weight", .{ .dout, .d }, .{}), null, .d),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        self.gate_proj.weight.deinit();
        if (self.gate_proj.bias) |*bias| bias.deinit();
        self.up_proj.weight.deinit();
        if (self.up_proj.bias) |*bias| bias.deinit();
        self.down_proj.weight.deinit();
        if (self.down_proj.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const gate = self.gate_proj.forward(x).gelu();
        const up = self.up_proj.forward(x);
        return self.down_proj.forward(gate.mul(up).rename(.{ .dout = .d }));
    }
};

pub const Attention = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rope_opts: zml.nn.RopeOpts,

    pub fn init(store: zml.io.TensorStore.View, config: Config, layer_type: LayerType) Attention {
        const linear = struct {
            fn mk(s: zml.io.TensorStore.View, prefix: []const u8) zml.nn.Linear {
                return .init(s.withPrefix(prefix).createTensor("weight", .{ .dout, .d }, .{}), null, .d);
            }
        };
        return .{
            .q_proj = linear.mk(store, "q_proj"),
            .k_proj = linear.mk(store, "k_proj"),
            .v_proj = linear.mk(store, "v_proj"),
            .o_proj = linear.mk(store, "o_proj"),
            .q_norm = .init(store.withPrefix("q_norm"), config.rms_norm_eps),
            .k_norm = .init(store.withPrefix("k_norm"), config.rms_norm_eps),
            .num_heads = config.num_attention_heads,
            .num_kv_heads = config.num_key_value_heads,
            .head_dim = config.head_dim,
            .rope_opts = config.ropeOpts(layer_type),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Attention)) void {
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

    pub fn forward(self: Attention, x: zml.Tensor, attn_mask: zml.Tensor) zml.Tensor {
        const num_head_groups = @divExact(self.num_heads, self.num_kv_heads);

        var q = self.q_proj.forward(x).splitAxis(-1, .{ .h = self.num_kv_heads, .hg = num_head_groups, .hd = .auto });
        var k = self.k_proj.forward(x).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = .auto });
        var v = self.v_proj.forward(x).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = .auto });

        // Q/K norm then RoPE (pos_idx=null → arange).
        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        q = zml.nn.rope(q, null, self.rope_opts);
        k = zml.nn.rope(k, null, self.rope_opts);

        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        var out = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask });
        return self.o_proj.forward(out.merge(.{ .d = .{ .h, .hg, .hd } }).rename(.{ .q = .s }));
    }
};

pub const DecoderLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    mlp: Mlp,

    pub fn init(store: zml.io.TensorStore.View, config: Config, layer_type: LayerType) DecoderLayer {
        const eps = config.rms_norm_eps;
        return .{
            .input_layernorm = .init(store.withPrefix("input_layernorm"), eps),
            .self_attn = .init(store.withPrefix("self_attn"), config, layer_type),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), eps),
            .pre_feedforward_layernorm = .init(store.withPrefix("pre_feedforward_layernorm"), eps),
            .post_feedforward_layernorm = .init(store.withPrefix("post_feedforward_layernorm"), eps),
            .mlp = .init(store.withPrefix("mlp")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(DecoderLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        Attention.unloadBuffers(&self.self_attn);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
        RmsNorm.unloadBuffers(&self.pre_feedforward_layernorm);
        RmsNorm.unloadBuffers(&self.post_feedforward_layernorm);
        Mlp.unloadBuffers(&self.mlp);
    }

    pub fn forward(self: DecoderLayer, hidden_states: zml.Tensor, attn_mask: zml.Tensor) zml.Tensor {
        var residual = hidden_states;
        var x = self.input_layernorm.forward(hidden_states);
        x = self.self_attn.forward(x, attn_mask);
        x = self.post_attention_layernorm.forward(x.rename(.{ .dout = .d }));
        x = x.add(residual);

        residual = x;
        x = self.pre_feedforward_layernorm.forward(x);
        x = self.mlp.forward(x);
        x = self.post_feedforward_layernorm.forward(x.rename(.{ .dout = .d }));
        return x.add(residual);
    }
};

pub const ScaledWordEmbedding = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    scale: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config) ScaledWordEmbedding {
        return .{
            .embed_tokens = .{ .weight = store.createTensor("weight", .{ .voc, .d }, .{}) },
            .scale = @sqrt(@as(f32, @floatFromInt(config.hidden_size))),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(ScaledWordEmbedding)) void {
        self.embed_tokens.weight.deinit();
    }

    pub fn forward(self: ScaledWordEmbedding, tokens: zml.Tensor) zml.Tensor {
        const embeds = self.embed_tokens.forward(tokens).withPartialTags(.{.d});
        return embeds.mul(.constant(embeds.dtype().constant(self.scale)));
    }
};

// -- Top-level encoder --

pub const Encoder = struct {
    embed_tokens: ScaledWordEmbedding,
    layers: []DecoderLayer,
    norm: RmsNorm,
    config: Config,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Encoder {
        const s = store.withPrefix("language_model").withPrefix("model");
        const layers = try allocator.alloc(DecoderLayer, config.num_hidden_layers);
        for (layers, 0..) |*layer, i| {
            layer.* = DecoderLayer.init(s.withPrefix("layers").withLayer(i), config, config.layerType(i));
        }
        return .{
            .embed_tokens = ScaledWordEmbedding.init(s.withPrefix("embed_tokens"), config),
            .layers = layers,
            .norm = .init(s.withPrefix("norm"), config.rms_norm_eps),
            .config = config,
        };
    }

    pub fn deinit(self: *Encoder, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
        self.layers = &.{};
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Encoder), allocator: std.mem.Allocator) void {
        ScaledWordEmbedding.unloadBuffers(&self.embed_tokens);
        for (self.layers) |*layer| {
            DecoderLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
    }

    /// Causal + padding attention mask. Padding masks KV side only (matching HuggingFace).
    /// Returns bf16 [q, k] with 0 for attend, -inf for masked.
    fn buildAttnMask(attention_mask: zml.Tensor, layer_type: LayerType, sliding_window: u32) zml.Tensor {
        const seq_len = attention_mask.dim(.s);
        const window: ?u32 = if (layer_type == .sliding_attention) sliding_window else null;
        const causal = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, .f32, window);

        // attention_mask: 1=real, 0=pad → 0=real, -inf=pad on KV axis.
        const m = attention_mask.convert(.f32);
        const one = zml.Tensor.constant(zml.DataType.f32.constant(1.0)).broad(m.shape());
        const zero = zml.Tensor.constant(zml.DataType.f32.zero()).broad(m.shape());
        const neg_inf = zml.Tensor.constant(zml.DataType.f32.minValue()).broad(m.shape());
        const pad_mask = zml.Tensor.select(
            m.cmp(.EQ, one),
            zero,
            neg_inf,
        ).rename(.{ .s = .k });

        return causal.add(pad_mask.broad(causal.shape())).convert(.bf16);
    }

    /// Forward pass returning stacked hidden states [s, d, layer=49].
    /// States: embedding + 47 raw decoder layer outputs + final normed output.
    pub fn forward(self: Encoder, input_ids: zml.Tensor, attention_mask: zml.Tensor) zml.Tensor {
        var hidden = self.embed_tokens.forward(input_ids.withTags(.{.s}));
        var stacked = hidden.appendAxes(.{.layer});

        const mask_input = attention_mask.withTags(.{.s});
        const full_mask = buildAttnMask(mask_input, .full_attention, self.config.sliding_window);
        const sliding_mask = buildAttnMask(mask_input, .sliding_attention, self.config.sliding_window);

        for (self.layers, 0..) |layer, i| {
            const mask = if (self.config.layerType(i) == .full_attention) full_mask else sliding_mask;
            hidden = layer.forward(hidden, mask);
            if (i < self.config.num_hidden_layers - 1) {
                stacked = zml.Tensor.concatenate(&.{ stacked, hidden.appendAxes(.{.layer}) }, .layer);
            }
        }

        // 49th entry: final normed output (matches HuggingFace hidden_states[-1]).
        stacked = zml.Tensor.concatenate(&.{ stacked, self.norm.forward(hidden).appendAxes(.{.layer}) }, .layer);
        return stacked;
    }
};

// -- Tokenization --

pub const TokenizeResult = struct {
    input_ids: [MAX_SEQ_LEN]u32,
    attention_mask: [MAX_SEQ_LEN]u32,
    real_token_count: usize,
};

/// Tokenize, prepend BOS if needed, left-pad to MAX_SEQ_LEN.
pub fn tokenizeAndPad(
    allocator: std.mem.Allocator,
    tokenizer: *const zml.tokenizer.Tokenizer,
    text: []const u8,
) !TokenizeResult {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();
    const raw_ids = try encoder.encodeAlloc(allocator, text);
    defer allocator.free(raw_ids);

    const has_bos = raw_ids.len > 0 and raw_ids[0] == BOS_TOKEN_ID;
    const total_len = if (has_bos) raw_ids.len else raw_ids.len + 1;
    const effective_len = @min(total_len, MAX_SEQ_LEN);
    const pad_len = MAX_SEQ_LEN - effective_len;

    var result: TokenizeResult = .{
        .input_ids = undefined,
        .attention_mask = undefined,
        .real_token_count = effective_len,
    };

    @memset(result.input_ids[0..pad_len], PAD_TOKEN_ID);
    if (has_bos) {
        @memcpy(result.input_ids[pad_len..], raw_ids[0..effective_len]);
    } else {
        result.input_ids[pad_len] = BOS_TOKEN_ID;
        const n = effective_len - 1;
        @memcpy(result.input_ids[pad_len + 1 ..][0..n], raw_ids[0..n]);
    }

    @memset(result.attention_mask[0..pad_len], 0);
    @memset(result.attention_mask[pad_len..], 1);

    return result;
}
