const std = @import("std");
const zml = @import("zml");
const stdx = zml.stdx;
const common = @import("../common.zig");

pub const Config = struct {
    architectures: []const []const u8 = &.{},
    model_type: []const u8,

    auto_map: ?AutoMap = null,

    rope_scaling: RopeScaling,
    yarn_only_types: []const []const u8 = &.{},

    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    max_seq_len: u32,
    vocab_size: u32,

    torch_dtype: []const u8 = "bfloat16",

    use_qk_norm: bool = false,

    moe_layers_enum: []const u8 = "",
    num_attention_heads: u32,
    num_attention_groups: u32,
    head_dim: u32,

    use_moe: bool = false,
    moe_num_experts: u32 = 0,
    moe_top_k: u32 = 0,
    moe_intermediate_size: u32 = 0,
    share_expert_dim: u32 = 0,
    moe_layer_offset: u32 = 0,
    moe_every_n_layer: u32 = 1,
    norm_expert_weight: bool = false,
    moe_router_activation: []const u8 = "sigmoid",
    moe_router_scaling_factor: f32 = 1.0,

    att_impl_type: []const u8 = "GQA",
    tie_word_embeddings: bool = false,

    rope_theta: []const f32,

    use_head_wise_attn_gate: bool = false,
    sliding_window: u32 = 0,

    use_moe_router_bias: bool = false,
    need_fp32_gate: bool = false,
    sink: bool = false,

    layer_types: []const []const u8 = &.{},
    use_rope_layers: []const u32 = &.{},

    num_nextn_predict_layers: u32 = 0,
    partial_rotary_factors: []const f32 = &.{},

    attention_other_setting: ?AttentionOtherSetting = null,

    swiglu_limits: []const f32 = &.{},
    swiglu_limits_shared: []const f32 = &.{},

    zero_centered: bool = false,
    max_position_embeddings: u32,

    pub const AutoMap = struct {
        AutoConfig: []const u8,
        AutoModelForCausalLM: []const u8,
    };

    pub const RopeScaling = struct {
        rope_type: []const u8,
        factor: f32,
        original_max_position_embeddings: u32,
        low_freq_factor: f32,
        high_freq_factor: f32,
    };

    pub const AttentionOtherSetting = struct {
        attention_type: []const u8,
        num_attention_heads: u32,
        num_attention_groups: u32,
        head_dim: u32,
        true_head_dim: u32,
    };

    pub fn numKeyValueHeads(self: Config) u32 {
        return self.num_attention_groups;
    }
};

// Options
pub const Options = struct {
    sampling_strategy: ?zml.nn.SamplingStrategy,
    max_seq_len: u32,
};

// LayerType

// Rope
// - parameters

// There are some partitioning functions re: KV Cache

<<<<<<< HEAD
// TextRotaryEmbedding
=======
    input_layernorm: RmsNorm,
    attn: Attn,
    attention_type: AttnType,
    ffn: Ffn,
    post_attention_layernorm: RmsNorm,

    pub fn init(store: zml.io.TensorStore.View, layer_idx: usize) !TransformerLayer {
        const is_moe = std.mem.indexOfScalar(u32, default_config.moe_layers_enum, @intCast(layer_idx)) != null;
        const attention_type = default_config.layer_types[layer_idx];

        return .{
            .input_layernorm = RmsNorm.init(store.withPrefix("input_layernorm")),
            .attn = try Attn.init(store.withPrefix("self_attn"), layer_idx, attention_type),
            .attention_type = attention_type,
            .ffn = if (is_moe) .{ .moe = try .init(store.withPrefix("moe"), layer_idx) } else .{ .mlp = .init(store.withPrefix("mlp"), default_config.swiglu_limits[layer_idx]) },
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), 1e5),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        Attn.unloadBuffers(&self.self_attn);
        Ffn.unloadBuffers(&self.ffn);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
    }

    pub fn forward(
        self: TransformerLayer,
        x0: zml.Tensor,
        token_index: zml.Tensor,
    ) struct { zml.Tensor } {
        _ = self; // autofix
        _ = token_index; // autofix
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {f}", .{x0});

        // const residual = x0;
        // _ = residual; // autofix

        // var hidden_states = self.input_layernorm.forward(x0);
        // _ = hidden_states; // autofix

        // hidden_states, _ = self.attn.forward(
        //     hidden_states,
        //     token_index,
        // );

        // // Fully Connected
        // const x1 = x0.add(delta0);
        // const x1_normalized = self.post_attention_layernorm.forward(x1);
        // const x2 = self.mlp.forward(x1_normalized)
        //     .rename(.{ .dout = .d })
        //     .add(x1);

        // return .{x2.reuseBuffer(x0)};
    }
};

/// Temporary deep copy of `config.json` for Step 3.5 Flash. TODO: run config through Model struct and remove this deep copy
pub const default_config: Config = .{
    .architectures = &.{"Step3p5ForCausalLM"},
    .model_type = "step3p5",
    .auto_map = .{
        .AutoConfig = "configuration_step3p5.Step3p5Config",
        .AutoModelForCausalLM = "modeling_step3p5.Step3p5ForCausalLM",
    },
    .rope_scaling = .{
        .rope_type = "llama3",
        .factor = 2.0,
        .original_max_position_embeddings = 131072,
        .low_freq_factor = 1.0,
        .high_freq_factor = 32.0,
    },
    .yarn_only_types = &.{"full_attention"},
    .hidden_size = 4096,
    .intermediate_size = 11264,
    .num_hidden_layers = 48,
    .max_seq_len = 262144,
    .vocab_size = 128896,
    .torch_dtype = "bfloat16",
    .use_qk_norm = true,
    .moe_layers_enum = &.{
        3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
        39, 40, 41, 42, 43, 44,
    },
    .num_attention_heads = 64,
    .num_attention_groups = 8,
    .head_dim = 128,
    .use_moe = true,
    .moe_num_experts = 288,
    .moe_top_k = 8,
    .moe_intermediate_size = 1280,
    .share_expert_dim = 1280,
    .moe_layer_offset = 0,
    .moe_every_n_layer = 1,
    .norm_expert_weight = true,
    .moe_router_activation = "sigmoid",
    .moe_router_scaling_factor = 3.0,
    .att_impl_type = "GQA",
    .tie_word_embeddings = false,
    .rope_theta = &.{
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
        5_000_000.0, 10_000.0, 10_000.0, 10_000.0,
    },
    .use_head_wise_attn_gate = true,
    .sliding_window = 512,
    .use_moe_router_bias = true,
    .need_fp32_gate = true,
    .sink = false,
    .layer_types = &.{
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
        .full_attention, .sliding_attention, .sliding_attention, .sliding_attention,
    },
    .use_rope_layers = &.{},
    .num_nextn_predict_layers = 3,
    .partial_rotary_factors = &.{
        0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,
        0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,
        0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,
        0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,
        0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,
        0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0,
    },
    .attention_other_setting = .{
        .attention_type = "sliding_attention",
        .num_attention_heads = 96,
        .num_attention_groups = 8,
        .head_dim = 128,
        .true_head_dim = 128,
    },
    .swiglu_limits = &.{
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 7.0, 7.0, 0.0, 0.0, 0.0,
    },
    .swiglu_limits_shared = &.{
        0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0,
    },
    .zero_centered = true,
    .max_position_embeddings = 262144,
};

// TODO: Attention struct wrapping Attn and SwAttn
pub const Attn = struct {
    layer_idx: usize,
    enable_sliding_window: bool,

    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,
    g_proj: zml.nn.Linear,

    q_norm: RmsNorm,
    k_norm: RmsNorm,

    q_size: i64,
    kv_size: i64,

    head_dim: i64,
    scaling: f32,

    num_q_heads: i64,
    num_kv_heads: i64,
    num_kv_groups: i64,
    rotary_dim: i64,
    rotary_emb: TextRotaryEmbedding,

    // do we need initProj
    fn initProj(store: zml.io.TensorStore.View, partitions: anytype, bias_partitions: anytype) zml.nn.Linear {
        return .init(
            store.createTensor("weight", .{ .dout, .d }, partitions),
            store.maybeCreateTensor("bias", .{.dout}, bias_partitions),
            .d,
        );
    }

    pub fn init(store: zml.io.TensorStore.View, layer_idx: usize, kind: AttnType) !Attn {
        const num_q_heads: i64 = @intCast(if (kind == .full_attention)
            default_config.num_attention_heads
        else
            (default_config.attention_other_setting orelse unreachable).num_attention_heads);

        const num_kv_heads: i64 = @intCast(default_config.num_attention_groups);
        const head_dim: i64 = @intCast(default_config.head_dim);

        const rotary_idx = @min(layer_idx, default_config.partial_rotary_factors.len - 1);
        const rotary_dim: i64 = @intFromFloat(
            default_config.partial_rotary_factors[rotary_idx] * @as(f32, @floatFromInt(default_config.head_dim)),
        );

        const rope_idx = @min(layer_idx, default_config.rope_theta.len - 1);
        const rs = default_config.rope_scaling;
        const rope_scaling: zml.nn.RopeOpts.Scaling = .{ .llama3 = .{
            .factor = rs.factor,
            .high_freq_factor = rs.high_freq_factor,
            .low_freq_factor = rs.low_freq_factor,
            .original_max_position_embeddings = rs.original_max_position_embeddings,
            .rope_theta = default_config.rope_theta[rope_idx],
        } };

        return .{
            .layer_idx = layer_idx,
            .num_q_heads = num_q_heads,
            .num_kv_heads = num_kv_heads,
            .enable_sliding_window = !(kind == .full_attention),
            .head_dim = head_dim,
            .num_kv_groups = @divExact(num_q_heads, num_kv_heads),
            .rotary_dim = rotary_dim,
            .rotary_emb = .init(rotary_dim, rope_scaling, 1.0),
            .q_size = num_q_heads * head_dim,
            .kv_size = num_kv_heads * head_dim,
            .scaling = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            .q_proj = initProj(store.withPrefix("q_proj"), .{ .dout = .replicated, .d = .replicated }, .{ .dout = .replicated }),
            .k_proj = initProj(store.withPrefix("k_proj"), .{ .dout = .replicated, .d = .replicated }, .{ .dout = .replicated }),
            .v_proj = initProj(store.withPrefix("v_proj"), .{ .dout = .replicated, .d = .replicated }, .{ .dout = .replicated }),
            .o_proj = initProj(store.withPrefix("o_proj"), .{ .dout = .replicated, .d = .replicated }, .{ .dout = .replicated }),
            .g_proj = initProj(store.withPrefix("g_proj"), .{ .dout = .replicated, .d = .replicated }, .{ .dout = .replicated }),
            // Step 3.5 doesn't expose rms_norm_eps in its config; HF reference uses 1e-5.
            .q_norm = .init(store.withPrefix("q_norm"), 1e-5),
            .k_norm = .init(store.withPrefix("k_norm"), 1e-5),
        };
    }
>>>>>>> 157a1d1e (examples/llm: remove hardcoding of swiglu limit)

<<<<<<< HEAD
// Moe
=======
<<<<<<< HEAD
// Router
=======
>>>>>>> 88deb96c (examples/llm: KV cache)
    pub fn unloadBuffers(self: *zml.Bufferized(Attn)) void {
        self.q_proj.weight.deinit();
        if (self.q_proj.bias) |*b| b.deinit();
        self.k_proj.weight.deinit();
        if (self.k_proj.bias) |*b| b.deinit();
        self.v_proj.weight.deinit();
        if (self.v_proj.bias) |*b| b.deinit();
        self.o_proj.weight.deinit();
        if (self.o_proj.bias) |*b| b.deinit();
        self.g_proj.weight.deinit();
        if (self.g_proj.bias) |*b| b.deinit();
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

<<<<<<< HEAD
    // project Q, Gate
    // Step 3.5 Flash keeps q and gate as separate projections (q_proj and g_proj). q_proj outputs num_q_heads * head_dim (= 12288 for layer 30).
    // g_proj is a head-wise attention gate: one scalar per query head (dout = num_q_heads), broadcast over .hd and applied to the per-head attention output before merging heads.
=======
>>>>>>> 88deb96c (examples/llm: KV cache)
    fn projectQAndGate(self: Attn, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const q = self.q_proj.forward(x)
            .splitAxis(.dout, .{ .h = self.num_q_heads, .hd = self.head_dim });
        const gate = self.g_proj.forward(x).rename(.{ .dout = .h });

        return .{ q, gate };
    }

    fn projectKV(self: Attn, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const k = self.k_proj.forward(x)
            .splitAxis(.dout, .{
            .h = self.num_kv_heads,
            .hd = self.head_dim,
        });

        const v = self.v_proj.forward(x)
            .splitAxis(.dout, .{
            .h = self.num_kv_heads,
            .hd = self.head_dim,
        });

        return .{ k, v };
    }

    pub fn forward(
        self: Attn,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
    ) struct { zml.Tensor, KvCache } {
        const input = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });
        var q, const gate = self.projectQAndGate(input);

        // PREFILL: x has .s=prompt_len, project initial KV
        // DECODE: x has .s=1, retrieve rest from KV cache
        var k, const v = self.projectKV(input);

        q = self.q_norm.forward(q, .hd);
        k = self.k_norm.forward(k, .hd);

        const dtype = q.dtype();
<<<<<<< HEAD
=======

        // TODO: shift by token_index in decode
>>>>>>> 88deb96c (examples/llm: KV cache)
        const position_ids = zml.Tensor.arange(.{ .end = input.dim(.s) }, .i64).withTags(.{.s});

        const cos, const sin = self.rotary_emb.getCosAndSin(position_ids, dtype);

        q = self.rotary_emb.applyRope(q, cos, sin);
        k = self.rotary_emb.applyRope(k, cos, sin);

        // Scatter the new k/v slice into the persistent cache, then read the full
        // history back so attention sees all prior tokens.
        // scatterSlices wants a scalar start offset for the `.k` axis; the slice
        // length comes from the update tensor itself.
        const cache_start = token_index.convert(.u32).slice1d(0, .{ .start = 0, .end = 1 }).squeeze(0);
        const new_kv_cache = kv_cache.update(k, v, cache_start);
        const k_full = new_kv_cache.keys().convert(dtype);
        const v_full = new_kv_cache.values().convert(dtype);

        q = q.rename(.{ .s = .q });

        // Take first element of cache_positions array to match zml.attention.attention signature
        const attn_start = token_index.slice1d(0, .{ .start = 0, .end = 1 });

        const attn_output = zml.attention.attention.attention(
            q,
            k_full,
            v_full,
            attn_start,
            zml.attention.attention.Metadata.init(.fromBackend(.vanilla, input.dim(.s), self.num_q_heads)),
            zml.attention.attention.Parameters.init(.fromBackend(.vanilla)),
        );

        // Head-wise gate is {b, s, h}
        const gate_b = gate.sigmoid().rename(.{ .s = .q }).broad(attn_output.shape());
        const gated_attn = attn_output.mul(gate_b);

        const projected_output = self.o_proj.forward(
            gated_attn.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s }),
        );

        return .{ projected_output, new_kv_cache };
    }

    pub const Stages = struct {
        q_proj: zml.Tensor,
        k_proj: zml.Tensor,
        v_proj: zml.Tensor,
        g_proj: zml.Tensor,
        q_norm: zml.Tensor,
        k_norm: zml.Tensor,
        // HF dumper records q/k pre-rope in [b,h,s,hd] layout (rope.q_in, rope.k_in)
        q_pre_rope_hf: zml.Tensor,
        k_pre_rope_hf: zml.Tensor,
        // cos/sin in HF layout [1,s,hd]
        cos: zml.Tensor,
        sin: zml.Tensor,
        // post-rope, HF layout [b,h,s,hd] (rope.q_embed, rope.k_embed)
        q_rope_hf: zml.Tensor,
        k_rope_hf: zml.Tensor,
        attn: zml.Tensor,
        gate_sig: zml.Tensor,
        gated: zml.Tensor,
        // input to o_proj: merged head output [b,s,h*hd]
        o_proj_in: zml.Tensor,
        out: zml.Tensor,
    };

<<<<<<< HEAD
<<<<<<< HEAD
    pub fn forwardStages(self: Attn, x: zml.Tensor, token_index: zml.Tensor) Stages {
=======
    pub fn forwardTemp(self: Attn, x: zml.Tensor, token_index: zml.Tensor) Stages {
>>>>>>> 88deb96c (examples/llm: KV cache)
=======
    pub fn forwardTemp(
        self: Attn,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
    ) struct { Stages, KvCache } {
>>>>>>> 52f36010 (examples/llm: update temp forward for testing KV cache)
        const input = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });

        const q_proj_raw = self.q_proj.forward(input);
        const k_proj_raw = self.k_proj.forward(input);
        const v_proj_raw = self.v_proj.forward(input);
        const g_proj_raw = self.g_proj.forward(input);

        var q = q_proj_raw.splitAxis(.dout, .{ .h = self.num_q_heads, .hd = self.head_dim });
        var k = k_proj_raw.splitAxis(.dout, .{ .h = self.num_kv_heads, .hd = self.head_dim });
        const v = v_proj_raw.splitAxis(.dout, .{ .h = self.num_kv_heads, .hd = self.head_dim });
        const gate = g_proj_raw.rename(.{ .dout = .h });

        const q_after_norm = self.q_norm.forward(q, .hd);
        const k_after_norm = self.k_norm.forward(k, .hd);
        q = q_after_norm;
        k = k_after_norm;

        const q_pre_rope_hf = q.transpose(.{ .b, .h, .s, .hd });
        const k_pre_rope_hf = k.transpose(.{ .b, .h, .s, .hd });

        const dtype = q.dtype();
        const position_ids = zml.Tensor.arange(.{ .end = input.dim(.s) }, .i64).withTags(.{.s});
        const cos_raw, const sin_raw = self.rotary_emb.getCosAndSin(position_ids, dtype);
        const cos = cos_raw.insertAxes(0, .{.b});
        const sin = sin_raw.insertAxes(0, .{.b});

        const q_rope = self.rotary_emb.applyRope(q, cos_raw, sin_raw);
        const k_rope = self.rotary_emb.applyRope(k, cos_raw, sin_raw);

        const q_rope_hf = q_rope.transpose(.{ .b, .h, .s, .hd });
        const k_rope_hf = k_rope.transpose(.{ .b, .h, .s, .hd });

        const new_kv_cache = kv_cache.update(k_rope, v, token_index.convert(.u32).slice1d(0, .{ .start = 0, .end = 1 }).squeeze(0));
        const k_full = new_kv_cache.keys().convert(dtype);
        const v_full = new_kv_cache.values().convert(dtype);

        const q_for_attn = q_rope.rename(.{ .s = .q });

        const attn_start = token_index.slice1d(0, .{ .start = 0, .end = 1 });
        const attn_output = zml.attention.attention.attention(
            q_for_attn,
            k_full,
            v_full,
            attn_start,
            zml.attention.attention.Metadata.init(.fromBackend(.vanilla, input.dim(.s), self.num_q_heads)),
            zml.attention.attention.Parameters.init(.fromBackend(.vanilla)),
        );

        const gate_sig = gate.sigmoid().rename(.{ .s = .q });
        const gated = attn_output.mul(gate_sig.broad(attn_output.shape()));

        const o_proj_in = gated.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const out = self.o_proj.forward(o_proj_in);

        return .{
            .{
                .q_proj = q_proj_raw,
                .k_proj = k_proj_raw,
                .v_proj = v_proj_raw,
                .g_proj = g_proj_raw,
                .q_norm = q_after_norm,
                .k_norm = k_after_norm,
                .q_pre_rope_hf = q_pre_rope_hf,
                .k_pre_rope_hf = k_pre_rope_hf,
                .cos = cos,
                .sin = sin,
                .q_rope_hf = q_rope_hf,
                .k_rope_hf = k_rope_hf,
                .attn = attn_output,
                .gate_sig = gate_sig,
                .gated = gated,
                .o_proj_in = o_proj_in,
                .out = out,
            },
            new_kv_cache,
        };
    }
};

pub const TextRotaryEmbedding = struct {
    rotary_dim: i64,
    scaling: zml.nn.RopeOpts.Scaling,
    attention_scaling: f32 = 1.0,

    pub fn init(rotary_dimension: i64, scaling: zml.nn.RopeOpts.Scaling, attention_scaling: f32) TextRotaryEmbedding {
        return .{
            .rotary_dim = rotary_dimension,
            .scaling = scaling,
            .attention_scaling = attention_scaling,
        };
    }

    /// Expects `position_ids` tagged `{.b, .s}` and returns cos/sin tagged `{.b, .s, .hd}`
    pub fn getCosAndSin(self: TextRotaryEmbedding, position_ids: zml.Tensor, dtype: zml.DataType) struct { zml.Tensor, zml.Tensor } {
        const opts: zml.nn.RopeOpts = .{ .scaling = self.scaling };
        const inv_freq = zml.nn.invFreq(self.rotary_dim, opts).withTags(.{.hd});

        const freqs_t = position_ids.convert(.f32).outer(inv_freq);
        const emb = zml.Tensor.concatenate(&.{ freqs_t, freqs_t }, .hd);

        const cos = emb.cos().scale(self.attention_scaling).convert(dtype);
        const sin = emb.sin().scale(self.attention_scaling).convert(dtype);
        return .{ cos, sin };
    }

    pub fn rotateHalf(x: zml.Tensor) zml.Tensor {
        const half_dim = @divExact(x.dim(.hd), 2);
        const x1 = x.slice1d(.hd, .{ .start = 0, .end = half_dim });
        const x2 = x.slice1d(.hd, .{ .start = half_dim, .end = x.dim(.hd) });
        return zml.Tensor.concatenate(&.{ x2.negate(), x1 }, .hd);
    }

    /// Expects `q`/`k` tagged `{.b, .s, .h, .hd}` and `cos`/`sin` tagged `{.b, .s, .hd}`.
    pub fn applyRope(
        self: TextRotaryEmbedding,
        x: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    ) zml.Tensor {
        const x_rot = x.slice1d(.hd, .{ .start = 0, .end = self.rotary_dim });

        // Insert head axis so cos/sin broadcast over .h.
        const cos_b = cos.insertAxes(.hd, .{.h}).broad(x_rot.shape());
        const sin_b = sin.insertAxes(.hd, .{.h}).broad(x_rot.shape());

        const x_rotated = x_rot.mul(cos_b).add(rotateHalf(x_rot).mul(sin_b));

        if (self.rotary_dim == x.dim(.hd)) {
            return x_rotated;
        }

        const x_pass = x.slice1d(.hd, .{ .start = self.rotary_dim, .end = x.dim(.hd) });

        return zml.Tensor.concatenate(&.{ x_rotated, x_pass }, .hd);
    }
};

// TODO: KV Cache
pub const KvCache = struct {
    k: zml.Tensor,
    v: zml.Tensor,
    layer_index: ?u32,

    pub const Buffer = zml.Bufferized(KvCache);

    pub fn init(kv_shape: zml.Shape) KvCache {
        return .{
            .k = .fromShape(kv_shape),
            .v = .fromShape(kv_shape),
            .layer_index = null,
        };
    }

    pub fn initBuffer(kv: KvCache, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !Buffer {
        return .{
            .k = try zml.Buffer.uninitialized(io, platform, kv.k.shape(), sharding, .{}),
            .v = try zml.Buffer.uninitialized(io, platform, kv.v.shape(), sharding, .{}),
        };
    }

    pub fn deinitBuffer(kv: *Buffer) void {
        kv.k.deinit();
        kv.v.deinit();
    }

    pub fn keys(kv: KvCache) zml.Tensor {
        return kv.k.slice1d(.layer, .single(kv.layer_index orelse @panic("forgot to call atLayer")));
    }

    pub fn values(kv: KvCache) zml.Tensor {
        return kv.v.slice1d(.layer, .single(kv.layer_index orelse @panic("forgot to call atLayer")));
    }

    pub fn update(kv: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: ?zml.Tensor) KvCache {
        const k_shape = kv.k.shape().drop(.layer);
        const layer: zml.Tensor = .scalar(kv.layer_index orelse @panic("forgot to call atLayer"), .u32);

        // KV cache uses .k instead of .s, so change here so caller doesn't need to be aware of this naming scheme.
        // Accept callers that already provide .k (e.g. synthetic tests).
        const k_renamed = if (new_k.shape().hasTag(.s) != null) new_k.rename(.{ .s = .k }) else new_k;
        const v_renamed = if (new_v.shape().hasTag(.s) != null) new_v.rename(.{ .s = .k }) else new_v;
        const k_in = k_renamed.convert(kv.k.dtype()).transpose(k_shape);
        const v_in = v_renamed.convert(kv.v.dtype()).transpose(k_shape);

        return if (token_index) |idx| .{
            .k = kv.k.scatterSlices(.{ .layer = layer, .k = idx }, k_in, .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.k),
            .v = kv.v.scatterSlices(.{ .layer = layer, .k = idx }, v_in, .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.v),
            .layer_index = kv.layer_index,
        } else .{
            .k = kv.k.scatterSlices(.{ .layer = layer }, k_in, .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.k),
            .v = kv.v.scatterSlices(.{ .layer = layer }, v_in, .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.v),
            .layer_index = kv.layer_index,
        };
    }

    pub fn atLayer(kv: KvCache, layer_index: usize) KvCache {
        return .{
            .k = kv.k,
            .v = kv.v,
            .layer_index = @intCast(layer_index),
        };
    }

    pub fn reuseBuffer(kv: KvCache, other: KvCache) KvCache {
        return .{
            .k = kv.k.reuseBuffer(other.k),
            .v = kv.v.reuseBuffer(other.v),
            .layer_index = null,
        };
    }
};

pub const Mlp = struct {
    up_proj: zml.nn.Linear, // (dim -> hidden_dim)
    gate_proj: zml.nn.Linear, // (dim -> hidden_dim)
    down_proj: zml.nn.Linear, // (hidden_dim -> dim)
    /// `0` means no clamp (matches the sentinel used in `config.swiglu_limits`).
    limit: f32,

    pub fn init(store: zml.io.TensorStore.View, swiglu_limit: f32) Mlp {
        return .{
            .up_proj = .init(
                store.createTensor("up_proj.weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }),
                null,
                .d,
            ),
            .gate_proj = .init(
                store.createTensor("gate_proj.weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }),
                null,
                .d,
            ),
            .down_proj = .init(
                store.createTensor("down_proj.weight", .{ .d, .dout }, .{ .d = .replicated, .dout = .replicated }),
                null,
                .dout,
            ),
            .limit = swiglu_limit,
        };
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        // Add tags to input before providing to our layer.
        const input = x.withTags(.{ .b, .s, .d });

        var up_proj = self.up_proj.forward(input);
        var gate = self.gate_proj.forward(input);
        gate = gate.silu();

        // Step 3.5 Flash clamps gate projection asymmetrically.
        if (self.limit != 0) {
            const max_t = zml.Tensor.scalar(self.limit, gate.dtype());
            const min_t = zml.Tensor.scalar(-self.limit, gate.dtype());
            gate = gate.minimum(max_t);
            up_proj = up_proj.clamp(min_t, max_t);
        }
        return self.down_proj.forward(gate.mul(up_proj));
    }
};

// RmsNorm
pub const RmsNorm = struct {
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

    /// L2 normalization of input tensor along the given axis.
    /// Step 3.5 Flash uses offset-style RMSNorm: the effective scale is `1 + weight`,
    /// not just `weight`.
    pub fn forward(self: RmsNorm, input: zml.Tensor, comptime axis: anytype) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{axis});
        const normalized = zml.nn.rmsNorm(x, axis, self.eps);
        const scale = self.weight.convert(.f32).addConstant(1).convert(x.dtype()).withTags(.{axis});
        return normalized.mul(scale.broad(normalized.shape()));
    }
};

>>>>>>> 157a1d1e (examples/llm: remove hardcoding of swiglu limit)
pub const Router = struct {
    gate: zml.nn.Linear, // no bias inside the Linear; router_bias is applied post-sigmoid
    router_bias: zml.Tensor,
    num_experts_per_tok: u32,
    routed_scaling_factor: f32,

    // k = num_experts_per_tok
    // `store` is the view at the parent `...moe` prefix; HF layout is:
    //   <prefix>.gate.weight
    //   <prefix>.router_bias   (optional)
    pub fn init(
        store: zml.io.TensorStore.View,
        num_experts_per_tok: u32,
        routed_scaling_factor: f32,
    ) Router {
        return .{
            .gate = .init(
                store.createTensor("gate.weight", .{ .expert, .d }, .{ .expert = .replicated, .d = .replicated }),
                null,
                .d,
            ),
            .router_bias = store.createTensor("router_bias", .{.expert}, .{ .expert = .replicated }),
            .num_experts_per_tok = num_experts_per_tok,
            .routed_scaling_factor = routed_scaling_factor,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Router)) void {
        self.gate.weight.deinit();
        self.router_bias.deinit();
    }

    /// Returns (topk weights, topk indices) with shapes {.b, .s, .topk}.
    /// Matches `Step3p5MoEMLP.router_bias_func`:
    ///   logits = x @ W^T
    ///   probs  = sigmoid(logits)        (in fp32)
    ///   biased = probs + router_bias    (only used to choose indices)
    ///   ids    = topk(biased)
    ///   wts    = gather(probs, ids)     (from UNBIASED probs)
    ///   if renormalize: wts /= sum(wts) + 1e-20
    pub fn forward(self: Router, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        // Match HF reference: always run gate matmul + sigmoid in fp32.
        const tagged = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });
        const input = tagged.convert(.f32);

        const gate_f32: zml.nn.Linear = .init(
            self.gate.weight.convert(.f32),
            null,
            .d,
        );

        // calculate raw logits
        const logits = gate_f32.forward(input);

        // calculate scores of how strongly a token matches an expert
        const expert_scores = logits.sigmoid();

        // add bias to influence how experts are distributed
        const expert_scores_with_bias = expert_scores.add(self.router_bias.convert(.f32).broad(expert_scores.shape()));

        // with N=288 total experts, pick top_k (k=8)
        const topk = expert_scores_with_bias.topK(.{ .topk = .expert }, self.num_experts_per_tok, .{});
        const topk_ids = topk.indices.convert(.i32);

        // Gather from UNBIASED probs — bias is only used to pick indices.
        const router_scores = expert_scores.gather(.{ .expert = topk_ids }, .{});

        // Renormalize so the topk weights sum to 1. Scaling by
        // routed_scaling_factor happens in the caller (Moe.forward), matching
        // the HF reference where `route()` returns unscaled weights.
        const denom = router_scores.sum(.topk).addConstant(1e-20);
        const normalized = router_scores.div(denom);

        return .{ normalized, topk_ids };
    }
};

// Router
pub const Router = struct {
    gate: zml.nn.Linear, // no bias inside the Linear; router_bias is applied post-sigmoid
    router_bias: zml.Tensor,
    num_experts_per_tok: u32,
    routed_scaling_factor: f32,

    // k = num_experts_per_tok
    // `store` is the view at the parent `...moe` prefix; HF layout is:
    //   <prefix>.gate.weight
    //   <prefix>.router_bias   (optional)
    pub fn init(
        store: zml.io.TensorStore.View,
        num_experts_per_tok: u32,
        routed_scaling_factor: f32,
    ) Router {
        return .{
            .gate = .init(
                store.createTensor("gate.weight", .{ .expert, .d }, .{ .expert = .replicated, .d = .replicated }),
                null,
                .d,
            ),
            .router_bias = store.createTensor("router_bias", .{.expert}, .{ .expert = .replicated }),
            .num_experts_per_tok = num_experts_per_tok,
            .routed_scaling_factor = routed_scaling_factor,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Router)) void {
        self.gate.weight.deinit();
        self.router_bias.deinit();
    }

    /// Returns (topk weights, topk indices) with shapes {.b, .s, .topk}.
    /// Matches `Step3p5MoEMLP.router_bias_func`:
    ///   logits = x @ W^T
    ///   probs  = sigmoid(logits)        (in fp32)
    ///   biased = probs + router_bias    (only used to choose indices)
    ///   ids    = topk(biased)
    ///   wts    = gather(probs, ids)     (from UNBIASED probs)
    ///   if renormalize: wts /= sum(wts) + 1e-20
<<<<<<< HEAD
    pub fn forward(self: Router, x: zml.Tensor, renormalize: bool) struct { zml.Tensor, zml.Tensor } {
        const input = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });
        const orig_dtype = input.dtype();
        const logits = self.gate.forward(input).convert(.f32); // {.b, .s, .expert}
        const probs = logits.sigmoid();
=======
    pub fn forward(self: Router, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        // Match HF reference: always run gate matmul + sigmoid in fp32.
        const tagged = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });
        const input = tagged.convert(.f32);
>>>>>>> 844fa04b (examples/llm: wip router forward)

        const gate_f32: zml.nn.Linear = .init(
            self.gate.weight.convert(.f32),
            null,
            .d,
        );

        // calculate raw logits
        const logits = gate_f32.forward(input);

        // calculate scores of how strongly a token matches an expert
        const expert_scores = logits.sigmoid();

        // add bias to influence how experts are distributed
        const expert_scores_with_bias = expert_scores.add(self.router_bias.convert(.f32).broad(expert_scores.shape()));

        // with N=288 total experts, pick top_k (k=8)
        const topk = expert_scores_with_bias.topK(.{ .topk = .expert }, self.num_experts_per_tok, .{});
        const topk_ids = topk.indices.convert(.i32);

<<<<<<< HEAD
        var router_scores = probs.gather(.{ .expert = topk_ids }, .{});
        if (renormalize) {
            const denom = router_scores.sum(.topk).addConstant(1e-20);
            router_scores = router_scores.div(denom.broad(router_scores.shape()));
        }
        return .{ router_scores.convert(orig_dtype), topk_ids };
=======
        // Gather from UNBIASED probs — bias is only used to pick indices.
        const router_scores = expert_scores.gather(.{ .expert = topk_ids }, .{});

        // Renormalize so the topk weights sum to 1. Scaling by
        // routed_scaling_factor happens in the caller (Moe.forward), matching
        // the HF reference where `route()` returns unscaled weights.
        const denom = router_scores.sum(.topk).addConstant(1e-20);
        const normalized = router_scores.div(denom);

        expert_scores.print("zml_gate_prob");
        expert_scores_with_bias.print("zml_gate_prob_with_bias");
        router_scores.print("zml_topk_prob");
        normalized.print("zml_renormalized");

        return .{ normalized, topk_ids };
>>>>>>> 844fa04b (examples/llm: wip router forward)
    }
};

// Moe
<<<<<<< HEAD
pub const Moe = struct {
    up_proj: zml.Tensor,
    gate_proj: zml.Tensor,
    down_proj: zml.Tensor,
    router: Router,
    layer_idx: usize,
    limit: ?f32,

    pub fn init(store: zml.io.TensorStore.View, layer_idx: usize) !Moe {
        // init the up, gate, down tensors

        const up_proj_tensor = store.createTensor(
            "up_proj.weight",
            .{ .expert, .dout, .d },
            .{ .expert = .replicated, .dout = .replicated, .d = .replicated },
        );

        const gate_proj_tensor = store.createTensor(
            "gate_proj.weight",
            .{ .expert, .dout, .d },
            .{ .expert = .replicated, .dout = .replicated, .d = .replicated },
        );

        const down_proj_tensor = store.createTensor(
            "down_proj.weight",
            .{ .expert, .dout, .d },
            .{ .expert = .replicated, .dout = .replicated, .d = .replicated },
        );

        return .{
            // swiglu limit temporarily hardcoded to 0
            // .shared_expert = .init(store.parent().withPrefix("share_expert"), 0),
            .up_proj = up_proj_tensor,
            .gate_proj = gate_proj_tensor,
            .down_proj = down_proj_tensor,
            .router = .init(store, 8, 3.0),
<<<<<<< HEAD
            .layer_idx = layer_idx,
            .limit = switch (layer_idx) {
                43, 44 => 7.0,
                else => null,
            },
=======
>>>>>>> 844fa04b (examples/llm: wip router forward)
        };
    }

    pub fn deinit(self: *zml.Bufferized(Moe)) void {
        self.up_proj.deinit();
        self.gate_proj.deinit();
        self.down_proj.deinit();
    }

    pub fn forward(self: Moe, x: zml.Tensor) zml.Tensor {
        if (self.layer_idx >= 42 and self.layer_idx <= 44) {
            return self.forwardLoop(x);
        }
        return self.forwardTriton(x);
    }

<<<<<<< HEAD
    fn forwardTriton(self: Moe, x: zml.Tensor) zml.Tensor {
        const input = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });

=======
>>>>>>> 844fa04b (examples/llm: wip router forward)
        // collect topk weights and indices (renormalized, unscaled)
        const routing_scores, const topk_ids = self.router.forward(input);

        // apply routed_scaling_factor here, matching reference's
        // `routing_weights = routing_weights * self.routed_scaling_factor`
        const scaled = routing_scores.scale(self.router.routed_scaling_factor);
<<<<<<< HEAD

        // Triton MoE backend expects the top-k axis tagged as `.top_expert`.
        const topk_ids_tensor = topk_ids.rename(.{ .topk = .top_expert });
        const scaled_tensor = scaled.rename(.{ .topk = .top_expert });
<<<<<<< HEAD
=======
>>>>>>> 844fa04b (examples/llm: wip router forward)
=======
>>>>>>> 0b9d05c9 (examples/llm: extract .top_expert as tensor)

        // concat the gate and up
        const gate_up_proj = zml.Tensor.concatenate(&.{ self.gate_proj, self.up_proj }, .dout).rename(.{ .dout = .out, .d = .in });

        // we must rename down proj as well
        const down_proj = self.down_proj.rename(.{ .dout = .out, .d = .in });

        // hardcoded zml.moe.metadata, zml.moe.parameters
        const moe_metadata = zml.moe.Metadata.init(.{ .triton = .{} });
        const moe_parameters = zml.moe.Parameters.init(.{ .triton = .{ .num_experts_per_tok = self.router.num_experts_per_tok, .activation = .silu } });

        // in Moe.forward, before forwardMoe
        topk_ids.print("zml_topk_ids");

        topk_ids.print("zml_topk_ids"); // → check min/max ≥ 256
        scaled.print("zml_topk_weights"); // → compare to Python routing_weights
        input.print("zml_moe_input"); // → compare to layer 42 in.0 fixture

        // get all expert outputs as tensor via fused triton kernel instead of Python loop
        // NOTE: swiglu limit not considered. may have to edit
        const moe_output = zml.moe.forwardMoe(
            input,
            topk_ids_tensor,
            scaled_tensor,
            gate_up_proj,
            null,
            null,
            down_proj,
            null,
            null,
            moe_metadata,
            moe_parameters,
        ) catch |err| stdx.debug.panic("moe backend failed: {}", .{err});

        return moe_output;
    }

    fn forwardLoop(self: Moe, x: zml.Tensor) zml.Tensor {
        const input = if (x.shape().isFullyTagged()) x else x.withTags(.{ .b, .s, .d });

        const routing_scores, const topk_ids = self.router.forward(input);
        const routing_scaled = routing_scores.scale(self.router.routed_scaling_factor);

        const x_f32 = input.convert(.f32);
        const gate_w = self.gate_proj.convert(.f32);
        const up_w = self.up_proj.convert(.f32);
        const down_w = self.down_proj.convert(.f32);

        const gate_all = x_f32.dot(gate_w, .d);
        const up_all = x_f32.dot(up_w, .d);

        var gate_act = gate_all.silu();
        var up_act = up_all;
        if (self.limit) |lim| {
            gate_act = gate_act.minimum(.scalar(lim, .f32));
            up_act = up_act.clamp(.scalar(-lim, .f32), .scalar(lim, .f32));
        }
        const act_d = gate_act.mul(up_act).rename(.{ .dout = .d });
        const down_all = act_d.dot(down_w, .d).transpose(.{ .b, .s, .expert, .dout });
        const down_topk = down_all.gather(.{ .expert = topk_ids }, .{});
        const routing = routing_scaled.convert(.f32).broad(down_topk.shape());
        const weighted = down_topk.mul(routing).convert(x.dtype());
        const sort_order = topk_ids.argsort(.topk, .{ .descending = false })
            .rename(.{ .topk = .topk_sorted });
        const weighted_sorted = weighted
            .gather(.{ .topk = sort_order }, .{})
            .rename(.{ .topk_sorted = .topk });

        var acc = weighted_sorted.slice1d(.topk, .single(0));
        inline for (1..8) |i| {
            const contrib = weighted_sorted.slice1d(.topk, .single(@as(i64, @intCast(i))));
            acc = acc.add(contrib);
        }
        return acc;
    }
};
=======
// const Moe = struct {
//     shared_expert: Mlp,
//     shared_expert_gate: zml.nn.Linear,
//     gate_up_proj: zml.Tensor,
//     down_proj: zml.Tensor,
//     router: Router,
// };
>>>>>>> 2ba32fe2 (examples/llm: Step 3.5 Flash router)
// hidden size
// intermediate size
// router bias
// routed scaling factor

// gating
<<<<<<< HEAD
>>>>>>> c856b0f1 (examples/llm: clean up tests and model annotations)
=======
>>>>>>> 2ba32fe2 (examples/llm: Step 3.5 Flash router)

// LoadedModel
pub const LoadedModel = struct {
    inner: Model,
    parsed_config: std.json.Parsed(Config),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
        generation: common.GenerationOptions,
    ) LoadedModel {
        _ = store; // autofix
        const parsed_config = try common.parseConfig(Config, allocator, io, repo);
        errdefer parsed_config.deinit();

        const options: Options = .{
            .sampling_strategy = generation.sampling_strategy,
            .max_seq_len = parsed_config.value.max_position_embeddings,
        };
        _ = options; // autofix

        return .{
            .inner = try .init(),
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
            std.log.scoped(.step3p5flash).info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });
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
        Step3p5Flash.unloadBuffers(&buffers.replicated, allocator);
    }

    // pub fn compile(
    //     self: *const LoadedModel,
    //     allocator: std.mem.Allocator,
    //     io: std.Io,
    //     platform: *const zml.Platform,
    //     backend: zml.attention.Backend,
    //     shardings: common.Shardings,
    //     seqlen: usize,
    //     progress: *std.Progress.Node,
    // ) !inference.CompiledModel {
    //     const params = inference.CompilationParameters.init(self.inner, self.parsed_config.value, @intCast(seqlen), backend, shardings);
    //     return inference.CompiledModel.init(allocator, io, platform, self, self.inner, params, progress);
    // }
};

// Buffers
pub const Buffers = zml.Bufferized(Model);

const Model = struct {
    lm_head: ?zml.nn.Linear,
    // model: Step3p5Flash,

    gen_opts: zml.nn.SamplingStrategy = .{},
    config: Config,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, options: Options) !Model {
        const lm_head: ?zml.nn.Linear = if (store.withPrefix("lm_head").maybeCreateTensor(
            "weight",
            .{ .dout, .d },
            .{ .dout = .replicated, .d = .replicated },
        )) |weight|
            .init(weight, null, .d)
        else
            null;

        return .{
            .lm_head = lm_head,
            .replicated = try .init(allocator, store.withPrefix("model"), config),
            .gen_opts = options.sampling_strategy orelse .{},
            .config = config,
        };
    }
};

// Step3p5Flash
const Step3p5Flash = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    norm: RmsNorm,
    layers: []TransformerLayer,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Step3p5Flash {
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = try .init(store.withPrefix("layers").withLayer(i), config);
        }

        return .{
            .embed_tokens = .{ .weight = store.createTensor(
                "embed_tokens.weight",
                .{ .voc, .d },
                .{ .voc = .replicated, .d = .replicated },
            ) },
            .norm = .{
                .weight = store.withPrefix("norm").createTensor("weight", .{.d}, .{ .d = .replicated }),
                .eps = config.rms_norm_eps,
            },
            .layers = layers,
        };
    }
};

// TransformerLayer
pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    // self_attn: SelfAttn,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,

    pub fn init(store: zml.io.TensorStore.View, config: Config) !TransformerLayer {
        return .{
            .input_layernorm = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps),
            .self_attn = try .init(store.withPrefix("self_attn"), config),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm")),
            .mlp = .init(store.withPrefix("mlp")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        // SelfAttn.unloadBuffers(&self.self_attn);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
        Mlp.unloadBuffers(&self.mlp);
    }

    pub fn forward(
        self: TransformerLayer,
        x0: zml.Tensor,
        token_index: zml.Tensor,
        // kv_cache: KvCache,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { zml.Tensor } {
        // ) struct { zml.Tensor, KvCache } {
        // Self Attention
        //log.debug("TransformerLayer({f}) -> {f}", .{ x0, self.input_layernorm.forward(x0) });
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {f}", .{x0});

        // Keep the residual stream replicated to avoid repeated gathers before q/k/v.
        const x0_replicated = x0.withPartitioning(.{ .d = .replicated });
        const x0_normalized = self.input_layernorm.forward(x0_replicated);
        const delta0, const updated_kv_cache = self.self_attn.forward(
            x0_normalized,
            token_index,
            // kv_cache,
            attention_metadata,
            attention_parameters,
        );

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

// RmsNorm
pub const RmsNorm = struct {
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

    /// L2 normalization of input tensor along .d axis
    /// Step 3.5 Flash uses offset-style RMSNorm: the effective scale is `1 + weight`,
    /// not just `weight`.
    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const normalized = zml.nn.rmsNorm(x, .d, self.eps);
        const scale = self.weight.convert(.f32).addConstant(1).convert(x.dtype()).withTags(.{.d});
        std.log.warn("normalized={f} scale={f}", .{ normalized.shape(), scale.shape() });
        return normalized.mul(scale.broad(normalized.shape()));
    }
};

pub const Mlp = struct {
    up_proj: zml.nn.Linear, // (dim -> hidden_dim)
    gate_proj: zml.nn.Linear, // (dim -> hidden_dim)
    down_proj: zml.nn.Linear, // (hidden_dim -> dim)
    limit: ?i32,

    pub fn init(store: zml.io.TensorStore.View, swiglu_limit: ?i32) Mlp {
        return .{
            .up_proj = .init(
                store.createTensor("up_proj.weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }),
                null,
                .d,
            ),
            .gate_proj = .init(
                store.createTensor("gate_proj.weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }),
                null,
                .d,
            ),
            .down_proj = .init(
                store.createTensor("down_proj.weight", .{ .d, .dout }, .{ .d = .replicated, .dout = .replicated }),
                null,
                .dout,
            ),
            .limit = swiglu_limit,
        };
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        // Add tags to input before providing to our layer.
        const input = x.withTags(.{ .b, .s, .d });

        var up_proj = self.up_proj.forward(input);
        var gate = self.gate_proj.forward(input);
        gate = gate.silu();

        // Step 3.5 Flash clamps gate projection asymmetrically
        if (self.limit) |limit| {
            if (limit != 0) {
                const lim_f = @as(f32, @floatFromInt(limit));
                const max_t = zml.Tensor.scalar(lim_f, gate.dtype());
                const min_t = zml.Tensor.scalar(-lim_f, gate.dtype());

                // Step 3.5 Flash has asymmetric clamping of gate projection
                gate = gate.minimum(max_t);
                up_proj = up_proj.clamp(min_t, max_t);
            }
        }
        return self.down_proj.forward(gate.mul(up_proj));
    }
};

// SwAttn

// SelfAttn

// KvCache
